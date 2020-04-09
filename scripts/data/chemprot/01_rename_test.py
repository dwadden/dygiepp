# Strip the `_gs` suffix from the test data to make looping easier.

import os
import re


wkdir = "data/chemprot/raw_data/ChemProt_Corpus/chemprot_test_gs"
newdir = wkdir.replace("_gs", "")

os.rename(wkdir, newdir)

for oldfile in os.listdir(newdir):
    if "_gs" in oldfile:
        newfile = oldfile.replace("_gs", "")
        os.rename(f"{newdir}/{oldfile}", f"{newdir}/{newfile}")
