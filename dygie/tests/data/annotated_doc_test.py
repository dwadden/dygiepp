"""
Spot checks for the classes defined in annotated_doc.py.

Uses the example provided in data.md, with index modifications to account for
the fact that spacy tokenizes contracted words into two tokens.

Author: Serena G. Lotreck
"""
import unittest
import os
import shutil
import sys

sys.path.append('../../../scripts/new-dataset')

import annotated_doc as ad
import spacy


class TestEnt(unittest.TestCase):
    def setUp(self):

        # Set up tempdir
        self.tmpdir = "tmp"
        os.makedirs(self.tmpdir, exist_ok=True)

        # Set up document text
        nlp = spacy.load("en_core_web_sm")
        dataset = 'scierc'
        text = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
                "She was elected in 2017.")
        text_path = f'{self.tmpdir}/myfile.txt'
        with open(text_path, 'w') as f:
            f.write(text)
        ann = ("T1\tCity 0 7\tSeattle\n"
               "T2\tPerson 25 37\tJenny Durkan\n"
               "T3\tCity 41 51\tthe city's\n"
               "T4\tPerson 59 62\tShe\n"
               "T5\tPersonnel.Election 67 74\telected\n"
               "T6\tYear 78 82\t2017\n"
               "R1\tMayor-Of Arg1:T2 Arg2:T3\n"
               "E1\tPersonnel.Election:T5 Person:T4 Year:T6\n"
               "*\tEQUIV T1 T3\n"
               "*\tEQUIV T2 T4\n")
        ann_path = f'{self.tmpdir}/myfile.ann'
        with open(ann_path, 'w') as f:
            f.write(ann)
        self.sent_idx_tups = [(0, 6), (6, 14), (14, 19)]
        # NOTE: spacy tokenizes words with apostrophes into separate words.

        # Set up annotated_doc object
        self.annotated_doc = ad.AnnotatedDoc.parse_ann(text_path,
                                                       ann_path,
                                                       nlp,
                                                       dataset,
                                                       coref=True)
        self.annotated_doc.char_to_token()

        # Right answer
        self.ner = [[[0, 0, "City"]], [[6, 7, "Person"], [9, 11, "City"]],
                    [[14, 14, "Person"], [16, 16, "Personnel.Election"],
                     [18, 18, "Year"]]]

    def tearDown(self):

        shutil.rmtree(self.tmpdir)

    def test_format_ner_dygiepp(self):

        ner = ad.Ent.format_ner_dygiepp(self.annotated_doc.ents,
                                        self.sent_idx_tups)

        self.assertEqual(ner, self.ner)


class TestBinRel(unittest.TestCase):
    def setUp(self):

        # Set up tempdir
        self.tmpdir = "tmp"
        os.makedirs(self.tmpdir, exist_ok=True)

        # Set up document text
        nlp = spacy.load("en_core_web_sm")
        dataset = 'scierc'
        text = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
                "She was elected in 2017.")
        text_path = f'{self.tmpdir}/myfile.txt'
        with open(text_path, 'w') as f:
            f.write(text)
        ann = ("T1\tCity 0 7\tSeattle\n"
               "T2\tPerson 25 37\tJenny Durkan\n"
               "T3\tCity 41 51\tthe city's\n"
               "T4\tPerson 59 62\tShe\n"
               "T5\tPersonnel.Election 67 74\telected\n"
               "T6\tYear 78 82\t2017\n"
               "R1\tMayor-Of Arg1:T2 Arg2:T3\n"
               "E1\tPersonnel.Election:T5 Person:T4 Year:T6\n"
               "*\tEQUIV T1 T3\n"
               "*\tEQUIV T2 T4\n")
        ann_path = f'{self.tmpdir}/myfile.ann'
        with open(ann_path, 'w') as f:
            f.write(ann)
        self.sent_idx_tups = [(0, 6), (6, 14), (14, 19)]
        # NOTE: spacy tokenizes words with apostrophes into separate words.

        # Set up annotated_doc object
        self.annotated_doc = ad.AnnotatedDoc.parse_ann(text_path,
                                                       ann_path,
                                                       nlp,
                                                       dataset,
                                                       coref=True)
        self.annotated_doc.char_to_token()

        # Set up relation
        self.rel1 = ad.BinRel("R1\tMayor-Of Arg1:T2 Arg2:T3".split())

        # Right answer
        self.relations = [[], [[6, 7, 9, 11, "Mayor-Of"]], []]

    def tearDown(self):

        shutil.rmtree(self.tmpdir)

    def test_set_arg_objects(self):

        self.rel1.set_arg_objects(self.annotated_doc.ents)

        self.assertEqual(self.rel1.arg1, self.annotated_doc.ents[1])
        self.assertEqual(self.rel1.arg2, self.annotated_doc.ents[2])

    def test_format_bin_rels_dygiepp(self):

        self.rel1.set_arg_objects(self.annotated_doc.ents)
        relations = ad.BinRel.format_bin_rels_dygiepp([self.rel1],
                                                      self.sent_idx_tups)

        self.assertEqual(relations, self.relations)


class TestEvent(unittest.TestCase):
    def setUp(self):

        # Set up tempdir
        self.tmpdir = "tmp"
        os.makedirs(self.tmpdir, exist_ok=True)

        # Set up document text
        nlp = spacy.load("en_core_web_sm")
        dataset = 'scierc'
        text = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
                "She was elected in 2017.")
        text_path = f'{self.tmpdir}/myfile.txt'
        with open(text_path, 'w') as f:
            f.write(text)
        ann = ("T1\tCity 0 7\tSeattle\n"
               "T2\tPerson 25 37\tJenny Durkan\n"
               "T3\tCity 41 51\tthe city's\n"
               "T4\tPerson 59 62\tShe\n"
               "T5\tPersonnel.Election 67 74\telected\n"
               "T6\tYear 78 82\t2017\n"
               "R1\tMayor-Of Arg1:T2 Arg2:T3\n"
               "E1\tPersonnel.Election:T5 Person:T4 Year:T6\n"
               "*\tEQUIV T1 T3\n"
               "*\tEQUIV T2 T4\n")
        ann_path = f'{self.tmpdir}/myfile.ann'
        with open(ann_path, 'w') as f:
            f.write(ann)
        self.sent_idx_tups = [(0, 6), (6, 14), (14, 19)]
        # NOTE: spacy tokenizes words with apostrophes into separate words.

        # Set up annotated_doc object
        self.annotated_doc = ad.AnnotatedDoc.parse_ann(text_path,
                                                       ann_path,
                                                       nlp,
                                                       dataset,
                                                       coref=True)
        self.annotated_doc.char_to_token()

        # Set up events
        self.event1 = ad.Event(
            "E1\tPersonnel.Election:T5 Person:T4 Year:T6".split())

        # Right answer
        self.events = [[], [],
                       [[[16, "Personnel.Election"], [14, 14, "Person"],
                         [18, 18, "Year"]]]]

    def tearDown(self):

        shutil.rmtree(self.tmpdir)

    def test_set_arg_objects(self):

        self.event1.set_arg_objects(self.annotated_doc.ents)

        self.assertEqual(self.event1.trigger, self.annotated_doc.ents[4])
        self.assertEqual(
            self.event1.args,
            [self.annotated_doc.ents[3], self.annotated_doc.ents[5]])

    def test_format_events_dygiepp(self):

        self.event1.set_arg_objects(self.annotated_doc.ents)
        events = ad.Event.format_events_dygiepp([self.event1],
                                                self.sent_idx_tups)

        self.assertEqual(events, self.events)


class TestEquivRel(unittest.TestCase):
    def setUp(self):

        # Set up tempdir
        self.tmpdir = "tmp"
        os.makedirs(self.tmpdir, exist_ok=True)

        # Set up document text
        nlp = spacy.load("en_core_web_sm")
        dataset = 'scierc'
        text = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
                "She was elected in 2017.")
        text_path = f'{self.tmpdir}/myfile.txt'
        with open(text_path, 'w') as f:
            f.write(text)
        ann = ("T1\tCity 0 7\tSeattle\n"
               "T2\tPerson 25 37\tJenny Durkan\n"
               "T3\tCity 41 51\tthe city's\n"
               "T4\tPerson 59 62\tShe\n"
               "T5\tPersonnel.Election 67 74\telected\n"
               "T6\tYear 78 82\t2017\n"
               "R1\tMayor-Of Arg1:T2 Arg2:T3\n"
               "E1\tPersonnel.Election:T5 Person:T4 Year:T6\n"
               "*\tEQUIV T1 T3\n"
               "*\tEQUIV T2 T4\n")
        ann_path = f'{self.tmpdir}/myfile.ann'
        with open(ann_path, 'w') as f:
            f.write(ann)

        # Set up annotated_doc object
        self.annotated_doc = ad.AnnotatedDoc.parse_ann(text_path,
                                                       ann_path,
                                                       nlp,
                                                       dataset,
                                                       coref=True)
        self.annotated_doc.char_to_token()

        # Set up equivalence relations
        self.equivrel1 = ad.EquivRel("*\tEQUIV T1 T3".split())
        self.equivrel2 = ad.EquivRel("*\tEQUIV T2 T4".split())

        # The dygiepp-formatted correct answer
        self.corefs = [[[0, 0], [9, 11]], [[6, 7], [14, 14]]]

    def tearDown(self):

        shutil.rmtree(self.tmpdir)

    def test_set_arg_objects(self):

        self.equivrel1.set_arg_objects(self.annotated_doc.ents)
        self.equivrel2.set_arg_objects(self.annotated_doc.ents)

        self.assertEqual(
            self.equivrel1.args,
            [self.annotated_doc.ents[0], self.annotated_doc.ents[2]])
        self.assertEqual(
            self.equivrel2.args,
            [self.annotated_doc.ents[1], self.annotated_doc.ents[3]])

    def test_format_corefs_dygiepp(self):

        self.equivrel1.set_arg_objects(self.annotated_doc.ents)
        self.equivrel2.set_arg_objects(self.annotated_doc.ents)
        corefs = ad.EquivRel.format_corefs_dygiepp(
            [self.equivrel1, self.equivrel2])

        self.assertEqual(corefs, self.corefs)


class TestAnnotatedDoc(unittest.TestCase):
    """
    Tests the functionality of char_to_token and format_dygiepp.
    """
    def setUp(self):

        # Set up temp dir and test docs
        self.tmpdir = "tmp"
        os.makedirs(self.tmpdir, exist_ok=True)

        txt = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
               "She was elected in 2017.")

        self.txt = f'{self.tmpdir}/myfile.txt'
        with open(self.txt, 'w') as f:
            f.write(txt)

        ann = ("T1\tCity 0 7\tSeattle\n"
               "T2\tPerson 25 37\tJenny Durkan\n"
               "T3\tCity 41 51\tthe city's\n"
               "T4\tPerson 59 62\tShe\n"
               "T5\tPersonnel.Election 67 74\telected\n"
               "T6\tYear 78 82\t2017\n"
               "R1\tMayor-Of Arg1:T2 Arg2:T3\n"
               "E1\tPersonnel.Election:T5 Person:T4 Year:T6\n"
               "*\tEQUIV T1 T3\n"
               "*\tEQUIV T2 T4\n")

        self.ann = f'{self.tmpdir}/myfile.ann'
        with open(self.ann, 'w') as f:
            f.write(ann)

        # Define other attributes
        self.nlp = spacy.load("en_core_web_sm")
        self.dataset = 'scierc'

        # Define right answer
        self.dygiepp_dict = {
            "doc_key":
            "myfile",
            "dataset":
            self.dataset,
            "sentences":
            [[tok.text for tok in sent] for sent in self.nlp(txt).sents],
            "ner": [[[0, 0, "City"]], [[6, 7, "Person"], [9, 11, "City"]],
                    [[14, 14, "Person"], [16, 16, "Personnel.Election"],
                     [18, 18, "Year"]]],
            "relations": [[], [[6, 7, 9, 11, "Mayor-Of"]], []],
            "clusters": [[[0, 0], [9, 11]], [[6, 7], [14, 14]]],
            "events": [[], [],
                       [[[16, "Personnel.Election"], [14, 14, "Person"],
                         [18, 18, "Year"]]]]
        }

    def tearDown(self):

        shutil.rmtree(self.tmpdir)

    def test_char_to_token(self):

        annotated_doc = ad.AnnotatedDoc.parse_ann(self.txt,
                                                  self.ann,
                                                  self.nlp,
                                                  self.dataset,
                                                  coref=True)
        annotated_doc.char_to_token()

        self.assertEqual(annotated_doc.ents[0].tok_start, 0)
        self.assertEqual(annotated_doc.ents[1].tok_start, 6)
        self.assertEqual(annotated_doc.ents[2].tok_start, 9)

        self.assertEqual(annotated_doc.ents[0].tok_end, 0)
        self.assertEqual(annotated_doc.ents[1].tok_end, 7)
        self.assertEqual(annotated_doc.ents[2].tok_end, 11)

    def test_format_dygiepp(self):

        annotated_doc = ad.AnnotatedDoc.parse_ann(self.txt,
                                                  self.ann,
                                                  self.nlp,
                                                  self.dataset,
                                                  coref=True)
        annotated_doc.char_to_token()
        res = annotated_doc.format_dygiepp()

        self.assertEqual(res, self.dygiepp_dict)


if __name__ == "__main__":
    unittest.main()
