#!/usr/bin/env python3
'''
Merge SENTiVENt event annotations with NER and relations produced by the pretrained DYGIE++ ACE05-Event model.
Requires: a dir with output predictions + parsed gold standard

merge_ace05_predictions.py in dygiepp
8/11/20 Copyright (c) Gilles Jacobs
'''
import json
from pathlib import Path

if __name__ == "__main__":

    parsed_fp = "/home/gilles/repos/dygiepp/data/sentivent/json/all.jsonl"
    pred_fp = "/home/gilles/repos/dygiepp/predictions/sentivent-all.jsonl"

    splits = {"train": (0, 228), "dev": (228, 258), "test": (258, 288)}

    with open(pred_fp, "rt") as pred_in:
        preds = [json.loads(l) for l in pred_in.readlines()]

    with open(parsed_fp, "rt") as parsed_in:
        parses = [json.loads(l) for l in parsed_in.readlines()]

    merges = []
    for (parse, pred) in zip(parses, preds):
        assert parse["doc_key"] == pred["doc_key"]
        merge = parse.copy()
        merge["ner"] = pred["predicted_ner"]
        merges.append(merge)

    opt_dir = Path("/home/gilles/repos/dygiepp/data/sentivent/ner/")
    with open(opt_dir / "all.jsonl", "wt") as all_out:
        for d in merges:
            all_out.write(json.dumps(d) + "\n")

    # split
    for split_name, (start, end) in splits.items():
        with open(opt_dir / f"{split_name}.jsonl", "wt") as spl_out:
            split_data = merges[start:end]
            for d in split_data:
                spl_out.write(json.dumps(d) + "\n")