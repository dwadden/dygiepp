#!/usr/bin/env python3
'''
Dataset inspection script for processed ace05 dataset.

read_aceevent_dataset.py in dygiepp
8/5/20 Copyright (c) Gilles Jacobs
'''
from dygie.data.dataset_readers.data_structures import Dataset
import spacy

data = Dataset("/home/gilles/repos/dygiepp/data/sentivent/ner/all.jsonl")
print(data[0])  # Print the first document.
print(data[0][1].ner)  # Print the named entities in the second sentence of the first document.
all_argument_roles = set()
for doc in data:
    for sen in doc:
        for ev in sen.events:
            roles = [arg.role for arg in ev.arguments]
            all_argument_roles.update(roles)

print(f"{len(all_argument_roles)} roles in train: {all_argument_roles}")
