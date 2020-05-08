# Courtesy of Jiechen Chen https://www.linkedin.com/in/jiechen-chen/.

import os
import json
import csv

import spacy

DIRECTORY = "data/chemprot"
PROCESSED_SUBDIRECTORY = "/processed_data/"
nlp = spacy.load("en_core_sci_sm")


def read_abstract(file_name):
    '''
    Reads file and creates a dictionary to retrieve abstract information.
    :param file_name: name of file that contains abstract info within the subdirectory
    :return: abstracts_dict: a map of file_id to 'title' and 'abstract' inside that file
    '''
    abstracts_dict = {}
    with open(file_name) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            abstracts_dict[int(row[0])] = {
                'title': row[1],
                'abstract': row[2]
            }
    return abstracts_dict


def save_abstract_info(abstracts_dict):
    '''
    Takes initial abstract information and parses the document id, sentences, and tokens from it. Gets the token index
    and line index of each token.
    :param abstracts_dict: raw dictionary from the training file that maps file_id to title and abstract
    :return: results: dictionary that will store all the information for each file, populated with doc_key, sentences,
    and some meta info that will be used for later processing
    '''
    results = {}
    for file_id in abstracts_dict:
        file_result = dict()
        sentence_lists = []
        token_dict = {}
        full_text = abstracts_dict[file_id].get('title') + " " + abstracts_dict[file_id].get('abstract')

        doc = nlp(full_text)

        sentence_index = 0
        for sentence in doc.sents:
            # USED FOR MODEL: Create a list of sentences, each with a sublist of tokens
            tokens_list = [token.text for token in list(sentence)]
            sentence_lists.append(tokens_list)  # Add list of tokens to overall list that contains all sentences

            # USED FOR LOGIC: Create a token lookup by character index, used in the next section
            previous_char_index = 0
            for token in list(sentence):
                token_dict[token.idx] = {  # Store token by character index
                    'text': token.text,
                    'token_index': token.i,
                    'line_index': sentence_index,
                }
                token_dict[previous_char_index]['next_char_index'] = token.idx
                previous_char_index = token.idx
            sentence_index += 1

        # Store all elements in overall dictionary
        file_result["doc_key"] = file_id
        file_result["sentences"] = sentence_lists
        file_result["token_dict"] = token_dict
        results[file_id] = file_result
    return results


def read_entities(file_name):
    entities_dict = {}
    with open(file_name) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if int(row[0]) not in entities_dict:
                entities_dict[int(row[0])] = []
            entities_dict.get(int(row[0])).append({
                'term': row[1],
                'type': row[2],
                'start_char': int(row[3]),
                'end_char': int(row[4]),
                'text': row[5]
            })
    return entities_dict


def save_entities_info(entities_dict, results):
    special_case = 0  # Spacy did not successfully tokenize this sentence
    regular_case = 0  # Spacy did successfully tokenize this sentence
    merged_case = 0  # Entity token is a substring of Spacy token.
    for file_id in results:
        ner = [[] for i in range(len(results[file_id]['sentences']))]  # Create a list of lists equal to number of setences in text
        term_location = {}  # Create a map of term number of start index, end index, and line number of that term
        token_dict = results[file_id]['token_dict']  # Map of character offset to text, token index, and line index
        for token in entities_dict.get(file_id) or []:
            start_token_info = token_dict.get(token['start_char'])
            if start_token_info:  # If this entity's start char lines up with Spacy's tokenizer's starting char for a token
                start_index = start_token_info['token_index']
                tentative_end_char_index = token['start_char']
                next_start_char_index = start_token_info.get('next_char_index')
                while next_start_char_index and token['end_char'] > next_start_char_index:
                    tentative_end_char_index = next_start_char_index
                    next_start_char_index = token_dict[tentative_end_char_index].get('next_char_index')

                end_token_info = token_dict[tentative_end_char_index]
                end_index = end_token_info['token_index']
                ner[end_token_info['line_index']].append([start_index, end_index, token['type']])
                term_location[token['term']] = {
                    'start_index': start_index,
                    'end_index': end_index,
                    'line_index': end_token_info['line_index']
                }
                if token["text"] != start_token_info["text"]:
                    print(token["text"])
                    print(start_token_info["text"])
                    print()
                    merged_case += 1
                else:
                    regular_case += 1
            else:
                special_case += 1
        #ner = sorted(ner, key=lambda x: x[0])
        results[file_id]['ner'] = ner
        results[file_id]['term_location'] = term_location

    total = special_case + regular_case + merged_case
    frac_discarded = special_case / total
    frac_merged = merged_case / total
    # Throw out cases where the token didn't line up with an entity boundary.
    print(f"Fraction entities discarded due to entity boundary / token index mismatch: {frac_discarded:0.4f}")
    print(f"Fraction entities where entity token is substring of Spacy token: {frac_merged:0.4f}")


def read_relations(file_name):
    relations_dict = {}
    with open(file_name) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if int(row[0]) not in relations_dict:
                relations_dict[int(row[0])] = []
            relations_dict.get(int(row[0])).append({
                'relationship': row[3],
                'arg1': row[4][5:],
                'arg2': row[5][5:],
            })
    return relations_dict


def save_relations(relations_dict, results):
    different_lines = 0
    same_lines = 0
    for file_id in results:
        relation = [[] for i in range(len(results[file_id]['sentences']))]  # Create a list of lists equal to number of setences in text
        term_location_dict = results[file_id]['term_location']
        for relation_entry in relations_dict.get(file_id) or []:
            arg1_location = term_location_dict.get(relation_entry['arg1'])
            arg2_location = term_location_dict.get(relation_entry['arg2'])
            if arg1_location and arg2_location:  # Only if we have term location information for both terms
                if arg1_location['line_index'] == arg2_location['line_index']:
                    relation[arg1_location['line_index']].append([
                        arg1_location['start_index'],
                        arg1_location['end_index'],
                        arg2_location['start_index'],
                        arg2_location['end_index'],
                        relation_entry['relationship'],
                    ])
                    same_lines += 1
                else:
                    different_lines += 1
        #relation = sorted(relation, key=lambda x: x[0])
        results[file_id]['relations'] = relation
    frac_cross_sent = different_lines / (different_lines + same_lines)

    # Remove relations that cross sentence boundaries.
    print(f"Fraction cross-sentence relations (discarded): {frac_cross_sent:0.4f}")


def process_fold(fold):
    print(f"Processing fold {fold}.")
    raw_subdirectory = f"/raw_data/ChemProt_Corpus/chemprot_{fold}/"
    abstracts_dict = read_abstract(DIRECTORY + raw_subdirectory + f'chemprot_{fold}_abstracts.tsv')
    results = save_abstract_info(abstracts_dict)
    entities_dict = read_entities(DIRECTORY + raw_subdirectory + f'chemprot_{fold}_entities.tsv')
    save_entities_info(entities_dict, results)
    relations_dict = read_relations(DIRECTORY + raw_subdirectory + f'chemprot_{fold}_relations.tsv')
    save_relations(relations_dict, results)

    with open(DIRECTORY + PROCESSED_SUBDIRECTORY + f'{fold}.jsonl', 'w') as outfile:
        for file_id in results:
            print(json.dumps({
                'doc_key': results[file_id].get('doc_key'),
                'sentences': results[file_id].get('sentences'),
                'ner': results[file_id].get('ner'),
                'relations': results[file_id].get('relations'),
            }), file=outfile)


def main():
    for fold in ["training", "development", "test"]:
        process_fold(fold)


if __name__ == "__main__":
    main()
