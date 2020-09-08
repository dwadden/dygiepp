#!/usr/bin/env python3
'''
Dataset inspection script for processed ace05 dataset.

read_event_dataset.py in dygiepp
8/5/20 Copyright (c) Gilles Jacobs
'''
from dygie.data.dataset_readers.data_structures import Dataset
import spacy
from collections import Counter
from pathlib import Path

def count_events(dset):
    counts = {"sentences": 0,
              "events": 0,
              "arguments": 0,
              "entities": 0,
    }
    for doc in dset:
        for sen in doc.sentences:
            counts["sentences"] += 1
            for ev in sen.events:
                counts["events"] += 1
                for arg in ev.arguments:
                    counts["arguments"] += 1
            for ne in sen.ner:
                counts["entities"] += 1
    return counts


base_fp = Path("/home/gilles/repos/dygiepp/data/sentivent/ner_with_subtype_args/")
train = Dataset(base_fp / "train.jsonl")
dev = Dataset(base_fp / "dev.jsonl")
test = Dataset(base_fp / "test.jsonl")

base_fp = Path("/home/gilles/repos/dygiepp/data/ace-event/processed-data/default-settings/json")
train_ace = Dataset(base_fp / "train.json")
dev_ace = Dataset(base_fp / "dev.json")
test_ace = Dataset(base_fp / "test.json")

train_c = count_events(train)
dev_c = count_events(dev)
test_c = count_events(test)
train_ace_c = count_events(train_ace)
dev_ace_c = count_events(dev_ace)
test_ace_c = count_events(test_ace)

# data = Dataset("/home/gilles/repos/dygiepp/data/sentivent/ner/all.jsonl")
data_sentivent = train.documents + dev.documents + test.documents
data_ace = train_ace.documents + dev_ace.documents + test_ace.documents
data = data_ace
# print(data[0])  # Print the first document.
# print(data[0][1].ner)  # Print the named entities in the second sentence of the first document.
all_argument_roles = set()
all_arg_event = dict() # all arguments and events counts
all_event_arg = dict() # all events with arguments
events = set()
for doc in data:
    for sen in doc:
        events.update(sen.events)
        for ev in sen.events:
            roles = [arg.role for arg in ev.arguments]
            all_argument_roles.update(roles)
            all_event_arg.setdefault(ev.trigger.label, Counter()).update(roles)
            for role in roles:
                all_arg_event.setdefault(role, Counter()).update([ev.trigger.label])


print(f"{len(all_argument_roles)} roles in train: {all_argument_roles}")

# print types
all_arg = ["CAPITAL", "TIME", "PLACE"]
for_type_table = sorted([(k, sorted([arg for arg in v.keys() if arg not in all_arg])) for k, v in all_event_arg.items()])
for type, args in for_type_table:
    print(type)
    print(f" $\\rightarrow$ {', '.join(args)}")
