"""
Script to convert the .txt and .ann files from a brat (https://brat.nlplab.org)
annotation to the input format for dygiepp.

Assumes the input .ann files correspond to the format described in
https://brat.nlplab.org/standoff.html

Supports conversion of .ann files including entities, binary relations,
equivalence relations, and events. Does NOT support formatting attribute
and modification annotations, normalization annotations, or note annotations.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, splitext
from os import listdir
from glob import glob
import json
import spacy
from annotated_doc import AnnotatedDoc


def format_annotated_document(fname_pair, dataset_name, nlp, coref):
    """
    Align the character indices with tokens to get a dygiepp formatted json.

    parameters:
        fname_pair, tuple of str: names of .ann and .txt files to use
        dataset_name, str: name of dataset used for prediction downstream
        nlp, spacy nlp object: model to use for tokenization
        coref, bool: whether or not to treat equivalence relations as corefs

    returns:
        res, json dict: formatted data
        dropped_totals, dict: numbers of original and dropped entities,
            binary and equivalence relations, and events for the document
    """
    # Make annotated doc object
    annotated_doc = AnnotatedDoc.parse_ann(fname_pair[0], fname_pair[1], nlp,
                                           dataset_name, coref)

    # Do the character to token conversion
    annotated_doc.char_to_token()

    # Do the dygiepp conversion
    res = annotated_doc.format_dygiepp()

    # Get the numbers of dropped entities and relations for this document
    dropped_totals = {
        'dropped_ents': annotated_doc.dropped_ents,
        'total_original_ents': annotated_doc.total_original_ents,
        'dropped_rels': annotated_doc.dropped_rels,
        'total_original_rels': annotated_doc.total_original_rels,
        'dropped_equiv_rels': annotated_doc.dropped_equiv_rels,
        'total_original_equiv_rels': annotated_doc.total_original_equiv_rels,
        'dropped_events': annotated_doc.dropped_events,
        'total_original_events': annotated_doc.total_original_events
    }

    return res, dropped_totals


def get_paired_files(all_files):
    """
    Check that there is both a .txt and .ann file for each filename, and return
    a list of tuples of the form ("myfile.txt", "myfile.ann"). Triggers an
    excpetion if one of the two files is missing, ignores any files that don't
    have either a .txt or .ann extension.

    parameters:
        all_files, list of str: list of all filenames in directory

    returns:
        paired_files, list of tuple: list of file pairs
    """
    paired_files = []

    # Get a set of all filenames without extensions
    basenames = set([splitext(name)[0] for name in all_files])

    # Check that there are two files with the right extenstions and put in list
    for name in basenames:

        # Get files with the same name
        matching_filenames = glob(f"{name}.*")

        # Check that both .txt and .ann are present
        txt_present = True if f"{name}.txt" in matching_filenames else False
        ann_present = True if f"{name}.ann" in matching_filenames else False

        # Put in list or raise exception
        if txt_present and ann_present:
            paired_files.append((f"{name}.txt", f"{name}.ann"))
        elif txt_present and not ann_present:
            raise ValueError("The .ann file is missing "
                             f"for the basename {name}. Please fix or delete.")
        elif ann_present and not txt_present:
            raise ValueError("The .txt file is missing "
                             f"for the basename {name}. Please fix or delete.")

    return paired_files


def format_labeled_dataset(data_directory, output_file, dataset_name,
                           use_scispacy, coref):

    # Get model to use for tokenization
    nlp_name = "en_core_sci_sm" if use_scispacy else "en_core_web_sm"
    nlp = spacy.load(nlp_name)

    # Get .txt/.ann file pairs
    all_files = [
        f"{data_directory}/{name}" for name in listdir(data_directory)
    ]
    paired_files = get_paired_files(all_files)

    # Format doc file pairs
    overall_dropped_totals = {
        'dropped_ents': 0,
        'total_original_ents': 0,
        'dropped_rels': 0,
        'total_original_rels': 0,
        'dropped_equiv_rels': 0,
        'total_original_equiv_rels': 0,
        'dropped_events': 0,
        'total_original_events': 0
    }
    res = []
    for fname_pair in paired_files:
        r, dropped_totals = format_annotated_document(fname_pair, dataset_name,
                                                      nlp, coref)
        res.append(r)
        overall_dropped_totals = {
            k: v + dropped_totals[k]
            for k, v in overall_dropped_totals.items()
        }

    print(
        '\n\nCompleted conversion for entire dataset! '
        f'{overall_dropped_totals["dropped_ents"]} of '
        f'{overall_dropped_totals["total_original_ents"]} original entities '
        'were dropped due to tokenization mismatches. As a result, '
        f'{overall_dropped_totals["dropped_rels"]} of '
        f'{overall_dropped_totals["total_original_rels"]} original relations, '
        f'{overall_dropped_totals["dropped_equiv_rels"]} of '
        f'{overall_dropped_totals["total_original_equiv_rels"]} coreference '
        f'clusters, and {overall_dropped_totals["dropped_events"]} of '
        f'{overall_dropped_totals["total_original_events"]} events '
        'were dropped.')

    # Write out doc dictionaries
    with open(output_file, "w") as f:
        for doc in res:
            print(json.dumps(doc), file=f)


def get_args():
    description = "Format labeled dataset from brat standoff"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("data_directory",
                        type=str,
                        help="A directory with input .txt and .ann files, "
                        "one .txt and one .ann for each file name.")
    parser.add_argument("output_file",
                        type=str,
                        help="The output file, .jsonl extension reccomended.")
    parser.add_argument("dataset_name",
                        type=str,
                        help="The name of the dataset. Should match the name "
                        "of the model you'll use for prediction.")
    parser.add_argument(
        "--use-scispacy",
        action="store_true",
        help="If provided, use scispacy to do the tokenization.")
    parser.add_argument("--coref",
                        action="store_true",
                        help="If provided, treat equivalence relations as "
                        "coreference clusters.")

    args = parser.parse_args()

    args.data_directory = abspath(args.data_directory)
    args.output_file = abspath(args.output_file)

    return args


def main():
    args = get_args()
    format_labeled_dataset(**vars(args))


if __name__ == "__main__":
    main()
