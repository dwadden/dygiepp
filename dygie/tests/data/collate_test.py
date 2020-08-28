"""
Test that a dataset doesn't change when it's collated and then de-collated.
"""

import unittest
import json
import os
import shutil
import sys
from pathlib import Path


# Since the collating code isn't inside the `dygie` package, I need to do a little work to import
# it.
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
common_root = current_dir.parent.parent.parent
collate_dir = f"{common_root}/scripts/data/shared"
sys.path.append(collate_dir)

# Now import the code
import collate
import uncollate


# Utility function.
def load_jsonl(fname):
    with open(fname) as f:
        return [json.loads(x) for x in f]


# The actual tests.
class TestCollate(unittest.TestCase):
    def setUp(self):
        self.collated_dir = "tmp/collated"
        self.uncollated_dir = "tmp/uncollated"
        os.makedirs(self.collated_dir, exist_ok=True)
        os.makedirs(self.uncollated_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree("tmp")

    @staticmethod
    def is_same(x1, x2):
        "Compare the fields in two dicts loaded from json."
        # Check if keys are same.
        if sorted(x1.keys()) != sorted(x2.keys()):
            return False

        # Loop over all fields. If not same, return False.
        for key in x1:
            if x1[key] != x2[key]:
                return False

        # If we get to the end, they're the same.
        return True

    def files_same(self, f1, f2):
        "Check that contests of two files are the same."
        data1 = load_jsonl(f1)
        data2 = load_jsonl(f2)

        # Ignore these in the comparison; `dataset` gets added, while `sentence_start` and
        # `clusters` get removed.
        fields_to_ignore = ["dataset", "sentence_start", "clusters"]
        for data in [data1, data2]:
            for entry in data:
                # Since the input data doesn't have a `dataset` field, we don't want to compare on
                # this.
                for field_to_ignore in fields_to_ignore:
                    if field_to_ignore in entry:
                        del entry[field_to_ignore]

        if len(data1) != len(data2):
            return False

        for entry1, entry2 in zip(data1, data2):
            if not self.is_same(entry1, entry2):
                return False

        return True

    def check_collate(self, dirname):
        input_dir = f"fixtures/collate/{dirname}"

        # Make the collator.
        collator_args = collate.get_args([input_dir, self.collated_dir, "--file_extension=json",
                                          f"--dataset={dirname}"])
        collator_runner = collate.CollateRunner(**vars(collator_args))

        # Make the uncollator.
        uncollator_args = uncollate.get_args(
            [self.collated_dir, self.uncollated_dir, f"--order_like_directory={input_dir}",
             "--file_extension=json"])
        uncollator_runner = uncollate.UnCollateRunner(**vars(uncollator_args))

        # Run both.
        collator_runner.run()
        uncollator_runner.run()

        for name in ["train", "dev", "test"]:
            assert self.files_same(f"{input_dir}/{name}.json", f"{self.uncollated_dir}/{name}.json")

    def test_collate(self):
        "Make sure that our Document class can read and write data without changing it."
        for dirname in ["ace-event", "scierc"]:
            self.check_collate(dirname)


if __name__ == "__main__":
    unittest.main()
