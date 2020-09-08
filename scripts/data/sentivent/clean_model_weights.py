#!/usr/bin/env python3
'''
Script to clean model weights of unwanted experiment runs.
!!REMOVES intermediate epoch weights of every training run.!!
!!REMOVES every final model.tar.gz for every training dir except those in KEEP!!

clean_model_weights.py in dygiepp
8/11/20 Copyright (c) Gilles Jacobs
'''
from pathlib import Path

model_dirp = Path("/home/gilles/repos/dygiepp/models")
KEEP = [
    "sentivent-event-nonerforargs",
    "sentivent-event_args_use_ner_labels:false-loss_weights_events.trigger:1.0-events_context_window:3-loss_weights.ner:0.0001",
    "ace05-event_args_use_ner_labels:false",
    "sentivent-finbert-finetune",
    "sentivent-bert-finetune",
    "sentivent-event-finbert"
]

for weight_fp in model_dirp.rglob("*.th"):
    print("Removed weights", weight_fp)
    weight_fp.unlink()

for model_fp in model_dirp.rglob("model.tar.gz"):
    if model_fp.parts[-2] not in KEEP:
        print("Removed model", model_fp)
        model_fp.unlink()
