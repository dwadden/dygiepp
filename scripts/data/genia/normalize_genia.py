"""
Collate the genia data.
"""

import os
import subprocess

in_dir = "data/genia/processed-data"
out_dir = "data/genia/normalized-data"


# For the coref data, normalize by adding datset name and getting rid of empty strings.
for name in ["json-coref-all", "json-coref-ident-only", "json-ner"]:
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)
    cmd = ["python",
           "scripts/data/shared/normalize.py",
           f"{in_dir}/{name}",
           f"{out_dir}/{name}",
           "--file_extension=json",
           "--max_tokens_per_doc=0",
           "--dataset=genia"]
    subprocess.run(cmd)


# For ner-only data, collate as well.
name = "json-ner"
collate_dir = "data/genia/collated-data"
os.makedirs(f"{collate_dir}/{name}", exist_ok=True)
cmd = ["python",
       "scripts/data/shared/collate.py",
       f"{in_dir}/{name}",
       f"{collate_dir}/{name}",
       "--file_extension=json",
       "--dataset=genia"]
subprocess.run(cmd)
