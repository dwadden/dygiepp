"""
Spot-checks for the Document class.
"""

import unittest
import json
import os
import shutil

from dygie.data import Document


class TestDocument(unittest.TestCase):
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

    def check_document(self, document_name):
        # Load the original file.
        with open(f"fixtures/{document_name}.json") as f:
            js = json.load(f)
        doc = Document.from_json(js)

        # Dump to file.
        tmpfile = f"{self.tmpdir}/{document_name}.json"
        dumped = doc.to_json()
        with open(tmpfile, "w") as f:
            json.dump(dumped, f)

        # Reload and compare.
        with open(tmpfile) as f:
            reloaded = json.load(f)
        assert self.is_same(js, reloaded)

    def test_document(self):
        "Make sure that our Document class can read and write data without changing it."
        for document_name in ["ace_event_article", "scierc_article", "ace_event_coref_article"]:
            self.check_document(document_name)


if __name__ == "__main__":
    unittest.main()
