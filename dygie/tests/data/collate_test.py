"""
Test that a dataset doesn't change when it's collated and then de-collated.
"""

import unittest
import json
import os
import shutil


from dygie.data.collate import collate, uncollate


class TestCollate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = "tmp"
        os.makedirs(self.tmpdir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @staticmethod
    def is_same(x1, x2):
        "Compare the fields in two dicts loaded from json."
        # Check if keys are same.
        if x1.keys() != x2.keys():
            return False

        # Loop over all fields. If not same, return False.
        for key in x1:
            if x1[key] != x2[key]:
                return False

        # If we get to the end, they're the same.
        return True

    def check_collate(self, document_name):
        # Load the original file.
        # TODO

    def test_collate(self):
        "Make sure that our Document class can read and write data without changing it."
        for document_name in ["ace_event_article", "scierc_article"]:
            self.check_collate(document_name)


if __name__ == "__main__":
    unittest.main()
