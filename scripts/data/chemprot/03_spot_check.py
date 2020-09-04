"""
Spot-check results to make sure we kept most entities and relationships.

Some will be dropped due to tokenization / sentence splitting.
"""

import pandas as pd
from collections import Counter

from dygie.data.dataset_readers.document import Dataset


def spot_check_fold(fold):
    print(f"Checking {fold}.")
    fname = f"data/chemprot/processed_data/{fold}.jsonl"
    data = Dataset.from_jsonl(fname)

    f_entity = f"data/chemprot/raw_data/ChemProt_Corpus/chemprot_{fold}/chemprot_{fold}_entities.tsv"
    entities = pd.read_table(f_entity, header=None)
    entities.columns = ["doc_key", "entity_id", "label", "start_char", "end_char", "text"]

    f_relation = f"data/chemprot/raw_data/ChemProt_Corpus/chemprot_{fold}/chemprot_{fold}_relations.tsv"
    relations = pd.read_table(f_relation, header=None)
    relations.columns = ["doc_key", "rel_category", "is_task", "label", "arg1", "arg2"]

    res = []

    for entry in data:
        counts = Counter()
        expected_entities = entities.query(f"doc_key == {entry.doc_key}")
        expected_relations = relations.query(f"doc_key == {entry.doc_key}")
        for sent in entry:
            counts["found_entities"] += len(sent.ner)
            counts["found_relations"] += len(sent.relations)

        counts["expected_entities"] = len(expected_entities)
        counts["expected_relations"] = len(expected_relations)

        counts["doc_key"] = entry.doc_key
        res.append(counts)

    res = pd.DataFrame(res).set_index("doc_key")

    frac_entities = res["found_entities"].sum() / res["expected_entities"].sum()
    frac_relations = res["found_relations"].sum() / res["expected_relations"].sum()

    print(f"Fraction of entities preserved from original file: {frac_entities:0.2f}")
    print(f"Fraction of relations preserved from original file: {frac_relations:0.2f}")


def main():
    for fold in ["training", "development", "test"]:
        spot_check_fold(fold)


if __name__ == "__main__":
    main()
