"""
Download pretrained SciBERT model and puts the result in ./pretrained/scibert_scivocab_cased.

Usage: python scripts/pretrained/get_scibert.py
"""

import os
import subprocess


# Check that we don't have it already.
if os.path.exists("./pretrained/scibert_scivocab_cased.tar"):
    print("SciBERT already downloaded.")
    quit()

# Make directory, download file, unzip.
os.makedirs("./pretrained", exist_ok=True)
scibert_url = ("https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models"
               "/scibert_scivocab_cased.tar")
subprocess.run(["wget", scibert_url, "--directory-prefix=./pretrained"])
subprocess.run(["tar", "-xvf", "./pretrained/scibert_scivocab_cased.tar",
                "--directory", "./pretrained"])
