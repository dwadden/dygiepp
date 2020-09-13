import spacy
import pandas as pd
from collections import Counter
from tqdm import tqdm
import json


nlp = spacy.load("en_core_sci_sm")


####################

# Process entities and relations for a given abstract.

def get_entities_in_sent(sent, entities):
    start, end = sent.start_char, sent.end_char
    start_ok = entities["char_start"] >= start
    end_ok = entities["char_end"] <= end
    keep = start_ok & end_ok
    res = entities[keep]
    return res


def align_one(sent, row):
    # Don't distinguish b/w genes that can and can't be looked up in database.
    lookup = {"GENE-Y": "GENE",
              "GENE-N": "GENE",
              "CHEMICAL": "CHEMICAL"}

    start_tok = None
    end_tok = None

    for tok in sent:
        if tok.idx == row["char_start"]:
            start_tok = tok
        if tok.idx + len(tok) == row["char_end"]:
            end_tok = tok

    if start_tok is None or end_tok is None:
        return None
    else:
        expected = sent[start_tok.i - sent.start:end_tok.i - sent.start + 1]
        if expected.text != row.text:
            raise Exception("Entity mismatch")

        return (start_tok.i, end_tok.i, lookup[row["label"]])


def align_entities(sent, entities_sent):
    aligned_entities = {}
    missed_entities = {}
    for _, row in entities_sent.iterrows():
        aligned = align_one(sent, row)
        if aligned is not None:
            aligned_entities[row["entity_id"]] = aligned
        else:
            missed_entities[row["entity_id"]] = None

    return aligned_entities, missed_entities


def format_relations(relations):
    # Convert to dict.
    res = {}
    for _, row in relations.iterrows():
        ent1 = row["arg1"].replace("Arg1:", "")
        ent2 = row["arg2"].replace("Arg2:", "")
        key = (ent1, ent2)
        res[key] = row["label"]

    return res


def get_relations_in_sent(aligned, relations):
    res = []
    keys = set()
    # Loop over the relations, and keep the ones relating entities in this sentences.
    for ents, label in relations.items():
        if ents[0] in aligned and ents[1] in aligned:
            keys.add(ents)
            ent1 = aligned[ents[0]]
            ent2 = aligned[ents[1]]
            to_append = ent1[:2] + ent2[:2] + (label,)
            res.append(to_append)

    return res, keys


####################

# Manage a single document and a single fold.

def one_abstract(row, df_entities, df_relations):
    doc = row["title"] + " " + row["abstract"]
    doc_key = row["doc_key"]
    entities = df_entities.query(f"doc_key == '{doc_key}'")
    relations = format_relations(df_relations.query(f"doc_key == '{doc_key}'"))

    processed = nlp(doc)

    entities_seen = set()
    entities_alignment = set()
    entities_no_alignment = set()
    relations_found = set()

    scierc_format = {"doc_key": doc_key, "dataset": "chemprot", "sentences": [], "ner": [],
                     "relations": []}

    for sent in processed.sents:
        # Get the tokens.
        toks = [tok.text for tok in sent]

        # Align entities.
        entities_sent = get_entities_in_sent(sent, entities)
        aligned, missed = align_entities(sent, entities_sent)

        # Align relations.
        relations_sent, keys_found = get_relations_in_sent(aligned, relations)

        # Append to result list
        scierc_format["sentences"].append(toks)
        entities_to_scierc = [list(x) for x in aligned.values()]
        scierc_format["ner"].append(entities_to_scierc)
        scierc_format["relations"].append(relations_sent)

        # Keep track of which entities and relations we've found and which we haven't.
        entities_seen |= set(entities_sent["entity_id"])
        entities_alignment |= set(aligned.keys())
        entities_no_alignment |= set(missed.keys())
        relations_found |= keys_found

    # Update counts.
    entities_missed = set(entities["entity_id"]) - entities_seen
    relations_missed = set(relations.keys()) - relations_found

    COUNTS["entities_correct"] += len(entities_alignment)
    COUNTS["entities_misaligned"] += len(entities_no_alignment)
    COUNTS["entities_missed"] += len(entities_missed)
    COUNTS["entities_total"] += len(entities)
    COUNTS["relations_found"] += len(relations_found)
    COUNTS["relations_missed"] += len(relations_missed)
    COUNTS['relations_total'] += len(relations)

    return scierc_format


def one_fold(fold):
    directory = "data/chemprot"
    print(f"Processing fold {fold}.")
    raw_subdirectory = "raw_data/ChemProt_Corpus"
    df_abstracts = pd.read_table(f"{directory}/{raw_subdirectory}/chemprot_{fold}/chemprot_{fold}_abstracts.tsv",
                                 header=None, keep_default_na=False,
                                 names=["doc_key", "title", "abstract"])
    df_entities = pd.read_table(f"{directory}/{raw_subdirectory}/chemprot_{fold}/chemprot_{fold}_entities.tsv",
                                header=None, keep_default_na=False,
                                names=["doc_key", "entity_id", "label", "char_start", "char_end", "text"])
    df_relations = pd.read_table(f"{directory}/{raw_subdirectory}/chemprot_{fold}/chemprot_{fold}_relations.tsv",
                                 header=None, keep_default_na=False,
                                 names=["doc_key", "cpr_group", "eval_type", "label", "arg1", "arg2"])

    res = []
    for _, abstract in tqdm(df_abstracts.iterrows(), total=len(df_abstracts)):
        to_append = one_abstract(abstract, df_entities, df_relations)
        res.append(to_append)

    # Write to file.
    name_out = f"{directory}/processed_data/{fold}.jsonl"
    with open(name_out, "w") as f_out:
        for line in res:
            print(json.dumps(line), file=f_out)


####################

# Driver

COUNTS = Counter()

for fold in ["training", "development", "test"]:
    one_fold(fold)

counts = pd.Series(COUNTS)
print()
print("Some entities were missed due to tokenization choices in SciSpacy. Here are the stats:")
print(counts)
