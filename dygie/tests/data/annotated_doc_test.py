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

verboseprint = print


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
               "T7\tCity 0 7;19 23\tSeattle city\n"
               "R1\tMayor-Of Arg1:T2 Arg2:T3\n"
               "R2\tis Arg1:T7 Arg2:T1\n"
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

        # Set up relation
        self.rel1 = ad.BinRel("R1\tMayor-Of Arg1:T2 Arg2:T3".split())
        self.rel2 = ad.BinRel("R2\tis Arg1:T7 Arg2:T1".split())

        # Right answer
        self.relations = [[], [[6, 7, 9, 11, "Mayor-Of"]], []]

        # Missing entity annotations
        missing_ann = ("T1\tCity 0 7\tSeattle\n"
                       "T2\tPerson 22 37\tJenny Durkan\n"
                       "T3\tCity 41 51\tthe city's\n"
                       "T4\tPerson 59 62\tShe\n"
                       "T5\tPersonnel.Election 67 74\telected\n"
                       "T6\tYear 78 82\t2017\n"
                       "R1\tMayor-Of Arg1:T2 Arg2:T3\n"
                       "E1\tPersonnel.Election:T5 Person:T4 Year:T6\n"
                       "*\tEQUIV T1 T3\n"
                       "*\tEQUIV T2 T4\n")
        missing_ann_path = f'{self.tmpdir}/missing_myfile.ann'
        with open(missing_ann_path, 'w') as f:
            f.write(missing_ann)

        # Set up annotated_doc object
        self.missing_annotated_doc = ad.AnnotatedDoc.parse_ann(
            text_path, missing_ann_path, nlp, dataset, coref=True)

        # Right answer
        self.missing_relations = [[], [], []]

    def tearDown(self):

        shutil.rmtree(self.tmpdir)

    # set_arg_objects is always called *before* char_to_token
    # They will fail if run in the opposite order with entities that get
    # dropped, but if they are only used with brat_to_input.py, the order is
    # baked in and therefore safe
    def test_set_arg_objects(self):

        self.rel1.set_arg_objects(self.annotated_doc.ents)

        self.assertEqual(self.rel1.arg1, self.annotated_doc.ents[1])
        self.assertEqual(self.rel1.arg2, self.annotated_doc.ents[2])

    def test_set_arg_objects_disjoint(self):

        self.rel2.set_arg_objects(self.annotated_doc.ents)

        self.assertEqual(self.rel2.arg1, None)
        self.assertEqual(self.rel2.arg2, self.annotated_doc.ents[0])

    def test_set_arg_objects_missing_arg(self):

        self.rel1.set_arg_objects(self.missing_annotated_doc.ents)

        self.assertEqual(self.rel1.arg1, self.missing_annotated_doc.ents[1])
        self.assertEqual(self.rel1.arg2, self.missing_annotated_doc.ents[2])

    def test_format_bin_rels_dygiepp(self):

        self.rel1.set_arg_objects(self.annotated_doc.ents)
        self.rel2.set_arg_objects(self.annotated_doc.ents)
        self.annotated_doc.char_to_token()
        relations, dropped_rels = ad.BinRel.format_bin_rels_dygiepp(
            [self.rel1, self.rel2], self.sent_idx_tups)

        self.assertEqual(relations, self.relations)

    def test_format_bin_rels_dygiepp_missing_arg(self):

        self.rel1.set_arg_objects(self.missing_annotated_doc.ents)
        self.missing_annotated_doc.char_to_token()
        relations, dropped_rels = ad.BinRel.format_bin_rels_dygiepp(
            [self.rel1], self.sent_idx_tups)

        self.assertEqual(relations, self.missing_relations)


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

        # Set up events
        self.event1 = ad.Event(
            "E1\tPersonnel.Election:T5 Person:T4 Year:T6".split())

        # Right answer
        self.events = [[], [],
                       [[[16, "Personnel.Election"], [14, 14, "Person"],
                         [18, 18, "Year"]]]]

        # Missing entity annotations
        missing_ann = ("T1\tCity 0 7\tSeattle\n"
                       "T2\tPerson 22 37\tJenny Durkan\n"
                       "T3\tCity 41 51\tthe city's\n"
                       "T4\tPerson 59 62\tShe\n"
                       "T5\tPersonnel.Election 63 74\telected\n"
                       "T6\tYear 78 82\t2017\n"
                       "R1\tMayor-Of Arg1:T2 Arg2:T3\n"
                       "E1\tPersonnel.Election:T5 Person:T4 Year:T6\n"
                       "*\tEQUIV T1 T3\n"
                       "*\tEQUIV T2 T4\n")
        missing_ann_path = f'{self.tmpdir}/missing_myfile.ann'
        with open(missing_ann_path, 'w') as f:
            f.write(missing_ann)

        # Set up annotated_doc object
        self.missing_annotated_doc = ad.AnnotatedDoc.parse_ann(
            text_path, missing_ann_path, nlp, dataset, coref=True)

        # Right answer
        self.missing_events = [[], [], []]

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
        self.annotated_doc.char_to_token()
        events, dropped_events = ad.Event.format_events_dygiepp(
            [self.event1], self.sent_idx_tups)

        self.assertEqual(events, self.events)

    def test_set_arg_objects_missing_ann(self):

        self.event1.set_arg_objects(self.missing_annotated_doc.ents)

        self.assertEqual(self.event1.trigger,
                         self.missing_annotated_doc.ents[4])
        self.assertEqual(self.event1.args, [
            self.missing_annotated_doc.ents[3],
            self.missing_annotated_doc.ents[5]
        ])

    def test_format_events_dygiepp_missing_ann(self):

        self.event1.set_arg_objects(self.missing_annotated_doc.ents)
        self.missing_annotated_doc.char_to_token()
        events, dropped_events = ad.Event.format_events_dygiepp(
            [self.event1], self.sent_idx_tups)

        self.assertEqual(events, self.missing_events)


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

        # Set up equivalence relations
        self.equivrel1 = ad.EquivRel("*\tEQUIV T1 T3".split())
        self.equivrel2 = ad.EquivRel("*\tEQUIV T2 T4".split())

        # The dygiepp-formatted correct answer
        self.corefs = [[[0, 0], [9, 11]], [[6, 7], [14, 14]]]

        # Missing entity annotations
        missing_ann = ("T1\tCity 0 7\tSeattle\n"
                       "T2\tPerson 22 37\tJenny Durkan\n"
                       "T3\tCity 41 51\tthe city's\n"
                       "T4\tPerson 59 62\tShe\n"
                       "T5\tPersonnel.Election 67 74\telected\n"
                       "T6\tYear 78 82\t2017\n"
                       "R1\tMayor-Of Arg1:T2 Arg2:T3\n"
                       "E1\tPersonnel.Election:T5 Person:T4 Year:T6\n"
                       "*\tEQUIV T1 T3\n"
                       "*\tEQUIV T2 T4\n")
        missing_ann_path = f'{self.tmpdir}/missing_myfile.ann'
        with open(missing_ann_path, 'w') as f:
            f.write(missing_ann)

        # Set up annotated_doc object
        self.missing_annotated_doc = ad.AnnotatedDoc.parse_ann(
            text_path, missing_ann_path, nlp, dataset, coref=True)

        # The dygiepp-formatted correct answer
        self.missing_corefs = [[[0, 0], [9, 11]]]

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
        self.annotated_doc.char_to_token()
        corefs, dropped_equiv_rels = ad.EquivRel.format_corefs_dygiepp(
            [self.equivrel1, self.equivrel2])

        self.assertEqual(corefs, self.corefs)

    def test_set_arg_objects_missing_ann(self):

        self.equivrel1.set_arg_objects(self.missing_annotated_doc.ents)
        self.equivrel2.set_arg_objects(self.missing_annotated_doc.ents)

        self.assertEqual(self.equivrel1.args, [
            self.missing_annotated_doc.ents[0],
            self.missing_annotated_doc.ents[2]
        ])
        self.assertEqual(self.equivrel2.args, [
            self.missing_annotated_doc.ents[1],
            self.missing_annotated_doc.ents[3]
        ])

    def test_format_corefs_dygiepp_missing_ann(self):

        self.equivrel1.set_arg_objects(self.missing_annotated_doc.ents)
        self.equivrel2.set_arg_objects(self.missing_annotated_doc.ents)
        self.missing_annotated_doc.char_to_token()
        corefs, dropped_equiv_rels = ad.EquivRel.format_corefs_dygiepp(
            [self.equivrel1, self.equivrel2])

        self.assertEqual(corefs, self.missing_corefs)


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


class TestQualityCheck(unittest.TestCase):
    """
    Tests quality_check_sent_splits and its helper, merge_mult_sent_splits.
    """
    def setUp(self):

        self.maxDiff = None

        # Helper input/output
        self.no_merge_no_multi = [(1, 2), (5, 6)]
        self.no_merge_no_multi_answer = [(1, 2), (5, 6)]
        self.no_merge_yes_multi = [(1, 4), (5, 6)]
        self.no_merge_yes_multi_answer = [(1, 4), (5, 6)]
        self.yes_merge_no_multi = [(2, 3), (3, 4), (4, 5), (6, 7), (8, 9),
                                   (9, 10)]
        self.yes_merge_no_multi_answer = [(2, 5), (6, 7), (8, 10)]
        self.yes_merge_yes_multi_no_overlap = [(2, 5), (6, 7), (8, 9), (9, 10)]
        self.yes_merge_yes_multi_no_overlap_answer = [(2, 5), (6, 7), (8, 10)]
        self.yes_merge_yes_multi_yes_overlap = [(2, 9), (8, 9), (9, 10)]
        self.only_one_merge_pair = [(5, 6), (6, 7)]
        self.only_one_merge_pair_answer = [(5, 7)]

        # Main func input/output
        # I'm using real instances for these, may not cover all possible instances
        # but I've done my best to account for all of those in the helper
        self.bioinfer_single = {
            "doc_key":
            "BioInfer.d70.s0",
            "dataset":
            "bioinfer_ppi",
            "sentences":
            [[
                "Aprotinin", "inhibited", "platelet", "aggregation", "induced",
                "by", "thrombin", "(", "0.25", "U.ml-1", ")", "with", "IC50",
                "200", "kIU.ml-1", ",", "and", "inhibited", "the", "rise",
                "of", "cytosolic", "free", "calcium", "concentration", "in",
                "platelets", "stimulated", "by", "thrombin", "(", "0.1",
                "U.ml-1", ")", "in", "the", "absence", "and", "in", "the",
                "presence", "of", "Ca2", "+", "0.5", "mmol", "."
            ],
             [
                 "L-1", "(", "IC50", "117", "and", "50", "kIU.ml-1", ",",
                 "respectively", ")", ",", "but", "had", "no", "effect", "on",
                 "the", "amounts", "of", "actin", "and", "myosin", "heavy",
                 "chain", "associated", "with", "cytoskeletons", "."
             ]],
            "ner": [[[29, 29, "Individual_protein"],
                     [0, 0, "Individual_protein"],
                     [6, 6, "Individual_protein"]],
                    [[68, 70, "Individual_protein"],
                     [66, 66, "Individual_protein"]]],
            "relations": [[[29, 29, 0, 0, "PPI"], [0, 0, 66, 66, "PPI"]],
                          [[68, 70, 0, 0, "PPI"]]]
        }
        self.bioinfer_single_answer = {
            "doc_key":
            "BioInfer.d70.s0",
            "dataset":
            "bioinfer_ppi",
            "sentences": [[
                "Aprotinin", "inhibited", "platelet", "aggregation", "induced",
                "by", "thrombin", "(", "0.25", "U.ml-1", ")", "with", "IC50",
                "200", "kIU.ml-1", ",", "and", "inhibited", "the", "rise",
                "of", "cytosolic", "free", "calcium", "concentration", "in",
                "platelets", "stimulated", "by", "thrombin", "(", "0.1",
                "U.ml-1", ")", "in", "the", "absence", "and", "in", "the",
                "presence", "of", "Ca2", "+", "0.5", "mmol", ".", "L-1", "(",
                "IC50", "117", "and", "50", "kIU.ml-1", ",", "respectively",
                ")", ",", "but", "had", "no", "effect", "on", "the", "amounts",
                "of", "actin", "and", "myosin", "heavy", "chain", "associated",
                "with", "cytoskeletons", "."
            ]],
            "ner": [[[29, 29, "Individual_protein"],
                     [0, 0, "Individual_protein"],
                     [6, 6, "Individual_protein"],
                     [68, 70, "Individual_protein"],
                     [66, 66, "Individual_protein"]]],
            "relations": [[[29, 29, 0, 0, "PPI"], [0, 0, 66, 66, "PPI"],
                           [68, 70, 0, 0, "PPI"]]]
        }

        self.pickle_mult_no_overlaps_or_merges = {
            "doc_key":
            "PMID12825696_abstract",
            "dataset":
            "pickle",
            "sentences":
            [[
                "Ca2", "+", "and", "calmodulin", "(", "CaM", ")", ",", "a",
                "key", "Ca2", "+", "sensor", "in", "all", "eukaryotes", ",",
                "have", "been", "implicated", "in", "defense", "responses",
                "in", "plants", "."
            ],
             [
                 "To", "elucidate", "the", "role", "of", "Ca2", "+", "and",
                 "CaM", "in", "defense", "signaling", ",", "we", "used",
                 "35S-labeled", "CaM", "to", "screen", "expression",
                 "libraries", "prepared", "from", "tissues", "that", "were",
                 "either", "treated", "with", "an", "elicitor", "derived",
                 "from", "Phytophthora", "megasperma", "or", "infected",
                 "with", "Pseudomonas", "syringae", "pv", "."
             ], ["tabaci", "."],
             [
                 "Nineteen", "cDNAs", "that", "encode", "the", "same",
                 "protein", ",", "pathogen-induced", "CaM-binding", "protein",
                 "(", "PICBP", ")", ",", "were", "isolated", "."
             ],
             [
                 "The", "PICBP", "fusion", "proteins", "bound", "35S-CaM", ",",
                 "horseradish", "peroxidase-labeled", "CaM", "and",
                 "CaM-Sepharose", "in", "the", "presence", "of", "Ca2", "+",
                 "whereas", "EGTA", ",", "a", "Ca2", "+", "chelator", ",",
                 "abolished", "binding", ",", "confirming", "that", "PICBP",
                 "binds", "CaM", "in", "a", "Ca2", "+", "-dependent", "manner",
                 "."
             ],
             [
                 "Using", "a", "series", "of", "bacterially", "expressed",
                 "truncated", "versions", "of", "PICBP", ",", "four",
                 "CaM-binding", "domains", ",", "with", "a", "potential",
                 "CaM-binding", "consensus", "sequence", "of",
                 "WSNLKKVILLKRFVKSL", ",", "were", "identified", "."
             ],
             [
                 "The", "deduced", "PICBP", "protein", "sequence", "is",
                 "rich", "in", "leucine", "residues", "and", "contains",
                 "three", "classes", "of", "repeats", "."
             ],
             [
                 "The", "PICBP", "gene", "is", "differentially", "expressed",
                 "in", "tissues", "with", "the", "highest", "expression", "in",
                 "stem", "."
             ],
             [
                 "The", "expression", "of", "PICBP", "in", "Arabidopsis",
                 "was", "induced", "in", "response", "to", "avirulent",
                 "Pseudomonas", "syringae", "pv", "."
             ], ["tomato", "carrying", "avrRpm1", "."],
             [
                 "Furthermore", ",", "PICBP", "is", "constitutively",
                 "expressed", "in", "the", "Arabidopsis", "accelerated",
                 "cell", "death2", "-", "2", "mutant", "."
             ],
             [
                 "The", "expression", "of", "PICBP", "in", "bean", "leaves",
                 "was", "also", "induced", "after", "inoculation", "with",
                 "avirulent", "and", "non-pathogenic", "bacterial", "strains",
                 "."
             ],
             [
                 "In", "addition", ",", "the", "hrp1", "mutant", "of",
                 "Pseudomonas", "syringae", "pv", "."
             ],
             [
                 "tabaci", "and", "inducers", "of", "plant", "defense", "such",
                 "as", "salicylic", "acid", ",", "hydrogen", "peroxide", "and",
                 "a", "fungal", "elicitor", "induced", "PICBP", "expression",
                 "in", "bean", "."
             ],
             [
                 "Our", "data", "suggest", "a", "role", "for", "PICBP", "in",
                 "Ca2", "+", "-mediated", "defense", "signaling", "and",
                 "cell-death", "."
             ],
             [
                 "Furthermore", ",", "PICBP", "is", "the", "first",
                 "identified", "CBP", "in", "eukaryotes", "with", "four",
                 "Ca2", "+", "-dependent", "CaM-binding", "domains", "."
             ]],
            "ner": [[[3, 3, "Protein"], [5, 5, "Protein"], [0, 1, "Element"],
                     [10, 12, "Protein"]],
                    [[34, 34, "Protein"], [59, 60, "Multicellular_organism"],
                     [31, 32, "Element"], [41, 42, "Protein"],
                     [64, 68, "Unicellular_organism"]], [],
                    [[82, 82, "Protein"], [78, 80, "Protein"]],
                    [[107, 107, "Organic_compound_other"], [93, 93, "Protein"],
                     [95, 97, "Protein"], [99, 99, "Protein"],
                     [104, 105, "Element"], [119, 119, "Protein"],
                     [121, 121, "Protein"], [89, 91, "Protein"],
                     [110, 112, "Organic_compound_other"]],
                    [[138, 138, "Protein"], [141, 142, "Peptide"],
                     [151, 151, "Peptide"], [147, 149, "Peptide"]],
                    [[164, 165, "Amino_acid_monomer"], [158, 159, "Protein"]],
                    [[174, 175, "DNA"]],
                    [[193, 193, "Multicellular_organism"], [191, 191, "DNA"],
                     [199, 204, "Unicellular_organism"]], [[206, 206, "DNA"]],
                    [[210, 210, "Protein"],
                     [216, 222, "Multicellular_organism"]],
                    [[229, 230, "Plant_region"], [227, 227, "DNA"]],
                    [[247, 248, "Unicellular_organism"],
                     [250, 254, "Unicellular_organism"]],
                    [[265, 266, "Inorganic_compound_other"],
                     [262, 263, "Plant_hormone"],
                     [272, 273, "Biochemical_process"],
                     [275, 275, "Multicellular_organism"]],
                    [[285, 291, "Biochemical_process"], [283, 283, "DNA"]],
                    [[295, 295, "Protein"], [305, 309, "Peptide"],
                     [300, 300, "Protein"]]],
            "relations": [[], [], [], [],
                          [[119, 119, 121, 121, "interacts"],
                           [89, 91, 93, 93, "interacts"],
                           [89, 91, 95, 97, "interacts"],
                           [89, 91, 99, 99, "interacts"],
                           [89, 91, 104, 105, "interacts"],
                           [107, 107, 89, 91, "inhibits"],
                           [107, 107, 104, 105, "inhibits"]], [],
                          [[164, 165, 158, 159, "is-in"]], [],
                          [[191, 191, 193, 193, "is-in"],
                           [199, 204, 191, 191, "activates"]],
                          [[206, 206, 199, 204, "is-in"]],
                          [[210, 210, 216, 222, "is-in"]],
                          [[227, 227, 229, 230, "is-in"]],
                          [[247, 248, 272, 273, "activates"]],
                          [[262, 263, 272, 273, "activates"],
                           [265, 266, 272, 273, "activates"],
                           [272, 273, 275, 275, "is-in"]],
                          [[283, 283, 285, 291, "is-in"]],
                          [[305, 309, 295, 295, "is-in"]]]
        }

        self.pickle_mult_no_overlaps_or_merges_answer = {
            "doc_key":
            "PMID12825696_abstract",
            "dataset":
            "pickle",
            "sentences":
            [[
                "Ca2", "+", "and", "calmodulin", "(", "CaM", ")", ",", "a",
                "key", "Ca2", "+", "sensor", "in", "all", "eukaryotes", ",",
                "have", "been", "implicated", "in", "defense", "responses",
                "in", "plants", "."
            ],
             [
                 "To", "elucidate", "the", "role", "of", "Ca2", "+", "and",
                 "CaM", "in", "defense", "signaling", ",", "we", "used",
                 "35S-labeled", "CaM", "to", "screen", "expression",
                 "libraries", "prepared", "from", "tissues", "that", "were",
                 "either", "treated", "with", "an", "elicitor", "derived",
                 "from", "Phytophthora", "megasperma", "or", "infected",
                 "with", "Pseudomonas", "syringae", "pv", ".", "tabaci", "."
             ],
             [
                 "Nineteen", "cDNAs", "that", "encode", "the", "same",
                 "protein", ",", "pathogen-induced", "CaM-binding", "protein",
                 "(", "PICBP", ")", ",", "were", "isolated", "."
             ],
             [
                 "The", "PICBP", "fusion", "proteins", "bound", "35S-CaM", ",",
                 "horseradish", "peroxidase-labeled", "CaM", "and",
                 "CaM-Sepharose", "in", "the", "presence", "of", "Ca2", "+",
                 "whereas", "EGTA", ",", "a", "Ca2", "+", "chelator", ",",
                 "abolished", "binding", ",", "confirming", "that", "PICBP",
                 "binds", "CaM", "in", "a", "Ca2", "+", "-dependent", "manner",
                 "."
             ],
             [
                 "Using", "a", "series", "of", "bacterially", "expressed",
                 "truncated", "versions", "of", "PICBP", ",", "four",
                 "CaM-binding", "domains", ",", "with", "a", "potential",
                 "CaM-binding", "consensus", "sequence", "of",
                 "WSNLKKVILLKRFVKSL", ",", "were", "identified", "."
             ],
             [
                 "The", "deduced", "PICBP", "protein", "sequence", "is",
                 "rich", "in", "leucine", "residues", "and", "contains",
                 "three", "classes", "of", "repeats", "."
             ],
             [
                 "The", "PICBP", "gene", "is", "differentially", "expressed",
                 "in", "tissues", "with", "the", "highest", "expression", "in",
                 "stem", "."
             ],
             [
                 "The", "expression", "of", "PICBP", "in", "Arabidopsis",
                 "was", "induced", "in", "response", "to", "avirulent",
                 "Pseudomonas", "syringae", "pv", ".", "tomato", "carrying",
                 "avrRpm1", "."
             ],
             [
                 "Furthermore", ",", "PICBP", "is", "constitutively",
                 "expressed", "in", "the", "Arabidopsis", "accelerated",
                 "cell", "death2", "-", "2", "mutant", "."
             ],
             [
                 "The", "expression", "of", "PICBP", "in", "bean", "leaves",
                 "was", "also", "induced", "after", "inoculation", "with",
                 "avirulent", "and", "non-pathogenic", "bacterial", "strains",
                 "."
             ],
             [
                 "In", "addition", ",", "the", "hrp1", "mutant", "of",
                 "Pseudomonas", "syringae", "pv", ".", "tabaci", "and",
                 "inducers", "of", "plant", "defense", "such", "as",
                 "salicylic", "acid", ",", "hydrogen", "peroxide", "and", "a",
                 "fungal", "elicitor", "induced", "PICBP", "expression", "in",
                 "bean", "."
             ],
             [
                 "Our", "data", "suggest", "a", "role", "for", "PICBP", "in",
                 "Ca2", "+", "-mediated", "defense", "signaling", "and",
                 "cell-death", "."
             ],
             [
                 "Furthermore", ",", "PICBP", "is", "the", "first",
                 "identified", "CBP", "in", "eukaryotes", "with", "four",
                 "Ca2", "+", "-dependent", "CaM-binding", "domains", "."
             ]],
            "ner": [[[3, 3, "Protein"], [5, 5, "Protein"], [0, 1, "Element"],
                     [10, 12, "Protein"]],
                    [[34, 34, "Protein"], [59, 60, "Multicellular_organism"],
                     [31, 32, "Element"], [41, 42, "Protein"],
                     [64, 68, "Unicellular_organism"]],
                    [[82, 82, "Protein"], [78, 80, "Protein"]],
                    [[107, 107, "Organic_compound_other"], [93, 93, "Protein"],
                     [95, 97, "Protein"], [99, 99, "Protein"],
                     [104, 105, "Element"], [119, 119, "Protein"],
                     [121, 121, "Protein"], [89, 91, "Protein"],
                     [110, 112, "Organic_compound_other"]],
                    [[138, 138, "Protein"], [141, 142, "Peptide"],
                     [151, 151, "Peptide"], [147, 149, "Peptide"]],
                    [[164, 165, "Amino_acid_monomer"], [158, 159, "Protein"]],
                    [[174, 175, "DNA"]],
                    [[193, 193, "Multicellular_organism"], [191, 191, "DNA"],
                     [199, 204, "Unicellular_organism"], [206, 206, "DNA"]],
                    [[210, 210, "Protein"],
                     [216, 222, "Multicellular_organism"]],
                    [[229, 230, "Plant_region"], [227, 227, "DNA"]],
                    [[247, 248, "Unicellular_organism"],
                     [250, 254, "Unicellular_organism"],
                     [265, 266, "Inorganic_compound_other"],
                     [262, 263, "Plant_hormone"],
                     [272, 273, "Biochemical_process"],
                     [275, 275, "Multicellular_organism"]],
                    [[285, 291, "Biochemical_process"], [283, 283, "DNA"]],
                    [[295, 295, "Protein"], [305, 309, "Peptide"],
                     [300, 300, "Protein"]]],
            "relations": [[], [], [],
                          [[119, 119, 121, 121, "interacts"],
                           [89, 91, 93, 93, "interacts"],
                           [89, 91, 95, 97, "interacts"],
                           [89, 91, 99, 99, "interacts"],
                           [89, 91, 104, 105, "interacts"],
                           [107, 107, 89, 91, "inhibits"],
                           [107, 107, 104, 105, "inhibits"]], [],
                          [[164, 165, 158, 159, "is-in"]], [],
                          [[191, 191, 193, 193, "is-in"],
                           [199, 204, 191, 191, "activates"],
                           [206, 206, 199, 204, "is-in"]],
                          [[210, 210, 216, 222, "is-in"]],
                          [[227, 227, 229, 230, "is-in"]],
                          [[247, 248, 272, 273, "activates"],
                           [262, 263, 272, 273, "activates"],
                           [265, 266, 272, 273, "activates"],
                           [272, 273, 275, 275, "is-in"]],
                          [[283, 283, 285, 291, "is-in"]],
                          [[305, 309, 295, 295, "is-in"]]]
        }

        self.pickle_mult_subsequent_merges = {
            "doc_key":
            "PMID28911019_abstract",
            "dataset":
            "pickle",
            "sentences":
            [[
                "BACKGROUND", "AND", "AIMS", ":", "Selected", "beneficial",
                "Pseudomonas", "spp", ".", "strains", "have", "the", "ability",
                "to", "influence", "root", "architecture", "in", "Arabidopsis",
                "thaliana", "by", "inhibiting", "primary", "root",
                "elongation", "and", "promoting", "lateral", "root", "and",
                "root", "hair", "formation", "."
            ],
             [
                 "A", "crucial", "role", "for", "auxin", "in", "this",
                 "long-term", "(", "1week", ")", ",", "long-distance",
                 "plant-microbe", "interaction", "has", "been", "demonstrated",
                 "."
             ],
             [
                 "METHODS", ":", "Arabidopsis", "seedlings", "were",
                 "cultivated", "in", "vitro", "on", "vertical", "plates",
                 "and", "inoculated", "with", "pathogenic", "strains",
                 "Pseudomonas", "syringae", "pv", "."
             ],
             [
                 "maculicola", "(", "Psm", ")", "and", "P.", "syringae", "pv",
                 "."
             ],
             [
                 "tomato", "DC3000", "(", "Pst", ")", ",", "as", "well", "as",
                 "Agrobacterium", "tumefaciens", "(", "Atu", ")", "and",
                 "Escherichia", "coli", "(", "Eco", ")", "."
             ],
             [
                 "Root", "hair", "lengths", "were", "measured", "after", "24",
                 "and", "48h", "of", "direct", "exposure", "to", "each",
                 "bacterial", "strain", "."
             ],
             [
                 "Several", "Arabidopsis", "mutants", "with", "impaired",
                 "responses", "to", "pathogens", ",", "impaired", "ethylene",
                 "perception", "and", "defects", "in", "the", "exocyst",
                 "vesicle", "tethering", "complex", "that", "is", "involved",
                 "in", "secretion", "were", "also", "analysed", "."
             ],
             [
                 "KEY", "RESULTS", ":", "Arabidopsis", "seedling", "roots",
                 "infected", "with", "Psm", "or", "Pst", "responded",
                 "similarly", "to", "when", "infected", "with", "plant",
                 "growth-promoting", "rhizobacteria", ";", "root", "hair",
                 "growth", "was", "stimulated", "and", "primary", "root",
                 "growth", "was", "inhibited", "."
             ],
             [
                 "Other", "plant-", "and", "soil-adapted", "bacteria",
                 "induced", "similar", "root", "hair", "responses", "."
             ],
             [
                 "The", "most", "compromised", "root", "hair", "growth",
                 "stimulation", "response", "was", "found", "for", "the",
                 "knockout", "mutants", "exo70A1", "and", "ein2", "."
             ],
             [
                 "The", "single", "immune", "pathways", "dependent", "on",
                 "salicylic", "acid", ",", "jasmonic", "acid", "and", "PAD4",
                 "are", "not", "directly", "involved", "in", "root", "hair",
                 "growth", "stimulation", ";", "however", ",", "in", "the",
                 "mutual", "cross-talk", "with", "ethylene", ",", "they",
                 "indirectly", "modify", "the", "extent", "of", "the",
                 "stimulation", "of", "root", "hair", "growth", "."
             ],
             [
                 "The", "Flg22", "peptide", "does", "not", "initiate", "root",
                 "hair", "stimulation", "as", "intact", "bacteria", "do", ",",
                 "but", "pretreatment", "with", "Flg22", "prior", "to", "Psm",
                 "inoculation", "abolished", "root", "hair", "growth",
                 "stimulation", "in", "an", "FLS2", "receptor",
                 "kinase-dependent", "manner", "."
             ],
             [
                 "These", "early", "response", "phenomena", "are", "not",
                 "associated", "with", "changes", "in", "auxin", "levels", ",",
                 "as", "monitored", "with", "the", "pDR5::GUS", "auxin",
                 "reporter", "."
             ],
             [
                 "CONCLUSIONS", ":", "Early", "stimulation", "of", "root",
                 "hair", "growth", "is", "an", "effect", "of", "an",
                 "unidentified", "component", "of", "living", "plant",
                 "pathogenic", "bacteria", "."
             ],
             [
                 "The", "root", "hair", "growth", "response", "is",
                 "triggered", "in", "the", "range", "of", "hours", "after",
                 "bacterial", "contact", "with", "roots", "and", "can", "be",
                 "modulated", "by", "FLS2", "signalling", "."
             ],
             [
                 "Bacterial", "stimulation", "of", "root", "hair", "growth",
                 "requires", "functional", "ethylene", "signalling", "and",
                 "an", "efficient", "exocyst-dependent", "secretory",
                 "machinery", "."
             ]],
            "ner": [[[6, 9, "Unicellular_organism"],
                     [18, 19, "Multicellular_organism"]],
                    [[38, 38, "Plant_hormone"]],
                    [[55, 56, "Multicellular_organism"],
                     [69, 73, "Unicellular_organism"]],
                    [[75, 75, "Unicellular_organism"],
                     [78, 83, "Unicellular_organism"]],
                    [[85, 85, "Unicellular_organism"],
                     [91, 92, "Unicellular_organism"],
                     [94, 94, "Unicellular_organism"],
                     [97, 98, "Unicellular_organism"],
                     [100, 100, "Unicellular_organism"]], [],
                    [[121, 122, "Multicellular_organism"],
                     [130, 131, "Biochemical_process"]],
                    [[152, 154, "Plant_region"],
                     [157, 157, "Unicellular_organism"],
                     [159, 159, "Unicellular_organism"]], [],
                    [[207, 207, "Multicellular_organism"],
                     [209, 209, "Multicellular_organism"]],
                    [[217, 218, "Plant_hormone"], [220, 221, "Plant_hormone"],
                     [223, 223, "Protein"], [241, 241, "Plant_hormone"]],
                    [[257, 258, "Peptide"], [273, 273, "Peptide"],
                     [276, 276, "Unicellular_organism"]],
                    [[300, 300, "Plant_hormone"], [307, 309, "DNA"]], [],
                    [[354, 355, "Biochemical_process"]],
                    [[365, 366, "Biochemical_process"]]],
            "relations": [[[6, 9, 18, 19, "interacts"]], [], [], [], [], [],
                          [],
                          [[157, 157, 152, 154, "interacts"],
                           [159, 159, 152, 154, "interacts"]], [], [],
                          [[217, 218, 241, 241, "interacts"],
                           [220, 221, 241, 241, "interacts"],
                           [223, 223, 241, 241, "interacts"]], [], [], [], [],
                          []]
        }

        self.pickle_mult_subsequent_merges_answer = {
            "doc_key":
            "PMID28911019_abstract",
            "dataset":
            "pickle",
            "sentences":
            [[
                "BACKGROUND", "AND", "AIMS", ":", "Selected", "beneficial",
                "Pseudomonas", "spp", ".", "strains", "have", "the", "ability",
                "to", "influence", "root", "architecture", "in", "Arabidopsis",
                "thaliana", "by", "inhibiting", "primary", "root",
                "elongation", "and", "promoting", "lateral", "root", "and",
                "root", "hair", "formation", "."
            ],
             [
                 "A", "crucial", "role", "for", "auxin", "in", "this",
                 "long-term", "(", "1week", ")", ",", "long-distance",
                 "plant-microbe", "interaction", "has", "been", "demonstrated",
                 "."
             ],
             [
                 "METHODS", ":", "Arabidopsis", "seedlings", "were",
                 "cultivated", "in", "vitro", "on", "vertical", "plates",
                 "and", "inoculated", "with", "pathogenic", "strains",
                 "Pseudomonas", "syringae", "pv", ".", "maculicola", "(",
                 "Psm", ")", "and", "P.", "syringae", "pv", ".", "tomato",
                 "DC3000", "(", "Pst", ")", ",", "as", "well", "as",
                 "Agrobacterium", "tumefaciens", "(", "Atu", ")", "and",
                 "Escherichia", "coli", "(", "Eco", ")", "."
             ],
             [
                 "Root", "hair", "lengths", "were", "measured", "after", "24",
                 "and", "48h", "of", "direct", "exposure", "to", "each",
                 "bacterial", "strain", "."
             ],
             [
                 "Several", "Arabidopsis", "mutants", "with", "impaired",
                 "responses", "to", "pathogens", ",", "impaired", "ethylene",
                 "perception", "and", "defects", "in", "the", "exocyst",
                 "vesicle", "tethering", "complex", "that", "is", "involved",
                 "in", "secretion", "were", "also", "analysed", "."
             ],
             [
                 "KEY", "RESULTS", ":", "Arabidopsis", "seedling", "roots",
                 "infected", "with", "Psm", "or", "Pst", "responded",
                 "similarly", "to", "when", "infected", "with", "plant",
                 "growth-promoting", "rhizobacteria", ";", "root", "hair",
                 "growth", "was", "stimulated", "and", "primary", "root",
                 "growth", "was", "inhibited", "."
             ],
             [
                 "Other", "plant-", "and", "soil-adapted", "bacteria",
                 "induced", "similar", "root", "hair", "responses", "."
             ],
             [
                 "The", "most", "compromised", "root", "hair", "growth",
                 "stimulation", "response", "was", "found", "for", "the",
                 "knockout", "mutants", "exo70A1", "and", "ein2", "."
             ],
             [
                 "The", "single", "immune", "pathways", "dependent", "on",
                 "salicylic", "acid", ",", "jasmonic", "acid", "and", "PAD4",
                 "are", "not", "directly", "involved", "in", "root", "hair",
                 "growth", "stimulation", ";", "however", ",", "in", "the",
                 "mutual", "cross-talk", "with", "ethylene", ",", "they",
                 "indirectly", "modify", "the", "extent", "of", "the",
                 "stimulation", "of", "root", "hair", "growth", "."
             ],
             [
                 "The", "Flg22", "peptide", "does", "not", "initiate", "root",
                 "hair", "stimulation", "as", "intact", "bacteria", "do", ",",
                 "but", "pretreatment", "with", "Flg22", "prior", "to", "Psm",
                 "inoculation", "abolished", "root", "hair", "growth",
                 "stimulation", "in", "an", "FLS2", "receptor",
                 "kinase-dependent", "manner", "."
             ],
             [
                 "These", "early", "response", "phenomena", "are", "not",
                 "associated", "with", "changes", "in", "auxin", "levels", ",",
                 "as", "monitored", "with", "the", "pDR5::GUS", "auxin",
                 "reporter", "."
             ],
             [
                 "CONCLUSIONS", ":", "Early", "stimulation", "of", "root",
                 "hair", "growth", "is", "an", "effect", "of", "an",
                 "unidentified", "component", "of", "living", "plant",
                 "pathogenic", "bacteria", "."
             ],
             [
                 "The", "root", "hair", "growth", "response", "is",
                 "triggered", "in", "the", "range", "of", "hours", "after",
                 "bacterial", "contact", "with", "roots", "and", "can", "be",
                 "modulated", "by", "FLS2", "signalling", "."
             ],
             [
                 "Bacterial", "stimulation", "of", "root", "hair", "growth",
                 "requires", "functional", "ethylene", "signalling", "and",
                 "an", "efficient", "exocyst-dependent", "secretory",
                 "machinery", "."
             ]],
            "ner": [[[6, 9, "Unicellular_organism"],
                     [18, 19, "Multicellular_organism"]],
                    [[38, 38, "Plant_hormone"]],
                    [[55, 56, "Multicellular_organism"],
                     [69, 73, "Unicellular_organism"],
                     [75, 75, "Unicellular_organism"],
                     [78, 83, "Unicellular_organism"],
                     [85, 85, "Unicellular_organism"],
                     [91, 92, "Unicellular_organism"],
                     [94, 94, "Unicellular_organism"],
                     [97, 98, "Unicellular_organism"],
                     [100, 100, "Unicellular_organism"]], [],
                    [[121, 122, "Multicellular_organism"],
                     [130, 131, "Biochemical_process"]],
                    [[152, 154, "Plant_region"],
                     [157, 157, "Unicellular_organism"],
                     [159, 159, "Unicellular_organism"]], [],
                    [[207, 207, "Multicellular_organism"],
                     [209, 209, "Multicellular_organism"]],
                    [[217, 218, "Plant_hormone"], [220, 221, "Plant_hormone"],
                     [223, 223, "Protein"], [241, 241, "Plant_hormone"]],
                    [[257, 258, "Peptide"], [273, 273, "Peptide"],
                     [276, 276, "Unicellular_organism"]],
                    [[300, 300, "Plant_hormone"], [307, 309, "DNA"]], [],
                    [[354, 355, "Biochemical_process"]],
                    [[365, 366, "Biochemical_process"]]],
            "relations": [[[6, 9, 18, 19, "interacts"]], [], [], [], [],
                          [[157, 157, 152, 154, "interacts"],
                           [159, 159, 152, 154, "interacts"]], [], [],
                          [[217, 218, 241, 241, "interacts"],
                           [220, 221, 241, 241, "interacts"],
                           [223, 223, 241, 241, "interacts"]], [], [], [], [],
                          []]
        }

        self.seedev_true_cross_sent_rel = {
            'doc_key':
            'SeeDev-binary-14701918-5',
            'dataset':
            'seedev',
            'sentences':
            [[
                'Developmental', 'Regulation', 'of', 'MUM4', 'during', 'Seed',
                'Coat', 'Secretory', 'Cell', 'Differentiation', 'by', 'AP2',
                ',', 'TTG1', ',', 'and', 'GL2', 'MUM4', 'transcript',
                'increases', 'in', 'differentiating', 'siliques', 'at', 'the',
                'time', 'of', 'mucilage', 'production', '.'
            ],
             [
                 'Two', 'lines', 'of', 'evidence', 'suggest', 'that', 'this',
                 'up-regulation', 'occurs', 'in', 'the', 'seed', 'coat',
                 'epidermis', 'to', 'support', 'mucilage', 'biosynthesis', '.'
             ],
             [
                 'First', ',', 'the', 'only', 'obvious', 'phenotypic',
                 'defect', 'in', 'mum4', 'plants', 'occurs', 'in', 'the',
                 'seed', 'coat', 'epidermis', '.'
             ],
             [
                 'Second', ',', 'MUM4', 'expression', 'is', 'severely',
                 'attenuated', 'in', 'siliques', 'of', 'ap2', 'mutants',
                 'that', 'fail', 'to', 'differentiate', 'the', 'outer', 'two',
                 'layers', 'of', 'the', 'seed', 'coat', '.'
             ],
             [
                 'Such', 'a', 'specific', 'up-regulation', 'of', 'a',
                 'putative', 'NDP-l-Rha', 'synthase', 'may', 'be', 'required',
                 'to', 'provide', 'extra', 'Rha', 'for', 'the', 'production',
                 'of', 'the', 'large', 'quantity', 'of', 'RGI', 'required',
                 'for', 'mucilage', 'synthesis', '.'
             ],
             [
                 'If', 'so', ',', 'the', 'amount', 'of', 'this', 'enzyme',
                 'must', 'be', 'the', 'limiting', 'factor', 'in', 'Rha',
                 'biosynthesis', 'and', 'the', 'amount', 'of', 'Rha', 'a',
                 'limiting', 'factor', 'in', 'RGI', 'biosynthesis', '.'
             ]],
            'ner':
            [[[3, 3, 'Gene'], [5, 8, 'Tissue'], [5, 9, 'Development_Phase'],
              [11, 11, 'Protein'], [13, 13, 'Protein'], [16, 16, 'Protein'],
              [17, 17, 'Gene'], [17, 18, 'RNA'], [21, 22, 'Development_Phase'],
              [22, 22, 'Tissue'], [27, 28, 'Pathway']],
             [[41, 43, 'Tissue'], [46, 47, 'Pathway']],
             [[57, 58, 'Genotype'], [62, 64, 'Tissue']],
             [[68, 68, 'Gene'], [74, 74, 'Tissue'], [76, 76, 'Genotype'],
              [81, 89, 'Regulatory_Network'], [83, 89, 'Tissue'],
              [88, 89, 'Tissue']],
             [[98, 99, 'Protein_Family'], [109, 115, 'Pathway'],
              [118, 119, 'Pathway']],
             [[135, 136, 'Pathway'], [146, 147, 'Pathway']]],
            'relations':
            [[[13, 13, 3, 3, 'Regulates_Expression'],
              [11, 11, 3, 3, 'Regulates_Expression'],
              [16, 16, 3, 3, 'Regulates_Expression'],
              [17, 18, 21, 22, 'Exists_At_Stage'],
              [13, 13, 5, 9, 'Exists_At_Stage'],
              [11, 11, 5, 9, 'Exists_At_Stage'],
              [16, 16, 5, 9, 'Exists_At_Stage'],
              [5, 9, 3, 3, 'Regulates_Expression']],
             [[46, 47, 41, 43, 'Is_Localized_In']], [],
             [[74, 74, 68, 68, 'Regulates_Expression'],
              [76, 76, 81, 89, 'Regulates_Process'],
              [76, 76, 68, 68, 'Regulates_Expression']],
             [[98, 99, 135, 136, 'Regulates_Process'],
              [109, 115, 118, 119, 'Regulates_Process'],
              [98, 99, 109, 115,
               'Regulates_Process']],
             [[135, 136, 146, 147, 'Regulates_Process']],
             ]
        }

        self.seedev_true_cross_sent_rel_answer = {
            'doc_key':
            'SeeDev-binary-14701918-5',
            'dataset':
            'seedev',
            'sentences':
            [[
                'Developmental', 'Regulation', 'of', 'MUM4', 'during', 'Seed',
                'Coat', 'Secretory', 'Cell', 'Differentiation', 'by', 'AP2',
                ',', 'TTG1', ',', 'and', 'GL2', 'MUM4', 'transcript',
                'increases', 'in', 'differentiating', 'siliques', 'at', 'the',
                'time', 'of', 'mucilage', 'production', '.'
            ],
             [
                 'Two', 'lines', 'of', 'evidence', 'suggest', 'that', 'this',
                 'up-regulation', 'occurs', 'in', 'the', 'seed', 'coat',
                 'epidermis', 'to', 'support', 'mucilage', 'biosynthesis', '.'
             ],
             [
                 'First', ',', 'the', 'only', 'obvious', 'phenotypic',
                 'defect', 'in', 'mum4', 'plants', 'occurs', 'in', 'the',
                 'seed', 'coat', 'epidermis', '.'
             ],
             [
                 'Second', ',', 'MUM4', 'expression', 'is', 'severely',
                 'attenuated', 'in', 'siliques', 'of', 'ap2', 'mutants',
                 'that', 'fail', 'to', 'differentiate', 'the', 'outer', 'two',
                 'layers', 'of', 'the', 'seed', 'coat', '.'
             ],
             [
                 'Such', 'a', 'specific', 'up-regulation', 'of', 'a',
                 'putative', 'NDP-l-Rha', 'synthase', 'may', 'be', 'required',
                 'to', 'provide', 'extra', 'Rha', 'for', 'the', 'production',
                 'of', 'the', 'large', 'quantity', 'of', 'RGI', 'required',
                 'for', 'mucilage', 'synthesis', '.', 'If', 'so', ',', 'the', 'amount', 'of', 'this', 'enzyme',
                 'must', 'be', 'the', 'limiting', 'factor', 'in', 'Rha',
                 'biosynthesis', 'and', 'the', 'amount', 'of', 'Rha', 'a',
                 'limiting', 'factor', 'in', 'RGI', 'biosynthesis', '.'
             ]],
            'ner':
            [[[3, 3, 'Gene'], [5, 8, 'Tissue'], [5, 9, 'Development_Phase'],
              [11, 11, 'Protein'], [13, 13, 'Protein'], [16, 16, 'Protein'],
              [17, 17, 'Gene'], [17, 18, 'RNA'], [21, 22, 'Development_Phase'],
              [22, 22, 'Tissue'], [27, 28, 'Pathway']],
             [[41, 43, 'Tissue'], [46, 47, 'Pathway']],
             [[57, 58, 'Genotype'], [62, 64, 'Tissue']],
             [[68, 68, 'Gene'], [74, 74, 'Tissue'], [76, 76, 'Genotype'],
              [81, 89, 'Regulatory_Network'], [83, 89, 'Tissue'],
              [88, 89, 'Tissue']],
             [[98, 99, 'Protein_Family'], [109, 115, 'Pathway'],
              [118, 119, 'Pathway'], [135, 136, 'Pathway'], [146, 147, 'Pathway']]],
            'relations':
            [[[13, 13, 3, 3, 'Regulates_Expression'],
              [11, 11, 3, 3, 'Regulates_Expression'],
              [16, 16, 3, 3, 'Regulates_Expression'],
              [17, 18, 21, 22, 'Exists_At_Stage'],
              [13, 13, 5, 9, 'Exists_At_Stage'],
              [11, 11, 5, 9, 'Exists_At_Stage'],
              [16, 16, 5, 9, 'Exists_At_Stage'],
              [5, 9, 3, 3, 'Regulates_Expression']],
             [[46, 47, 41, 43, 'Is_Localized_In']], [],
             [[74, 74, 68, 68, 'Regulates_Expression'],
              [76, 76, 81, 89, 'Regulates_Process'],
              [76, 76, 68, 68, 'Regulates_Expression']],
             [[98, 99, 135, 136, 'Regulates_Process'],
              [109, 115, 118, 119, 'Regulates_Process'],
              [98, 99, 109, 115,
               'Regulates_Process'], [135, 136, 146, 147, 'Regulates_Process']],
             ]
        }


    def test_merge_mult_splits_no_merge_no_multi(self):

        result = ad.AnnotatedDoc.merge_mult_splits(self.no_merge_no_multi)

        self.assertEqual(result, self.no_merge_no_multi_answer)

    def test_merge_mult_splits_no_merge_yes_multi(self):

        result = ad.AnnotatedDoc.merge_mult_splits(self.no_merge_yes_multi)

        self.assertEqual(result, self.no_merge_yes_multi_answer)

    def test_merge_mult_splits_yes_merge_no_multi(self):

        result = ad.AnnotatedDoc.merge_mult_splits(self.yes_merge_no_multi)

        self.assertEqual(result, self.yes_merge_no_multi_answer)

    def test_merge_mult_splits_yes_merge_yes_multi_no_overlap(self):

        result = ad.AnnotatedDoc.merge_mult_splits(
            self.yes_merge_yes_multi_no_overlap)

        self.assertEqual(result, self.yes_merge_yes_multi_no_overlap_answer)

    def test_merge_mult_splits_yes_merge_yes_multi_yes_overlap(self):

        self.assertRaises(AssertionError, ad.AnnotatedDoc.merge_mult_splits,
                          self.yes_merge_yes_multi_yes_overlap)

    def test_merge_mult_splits_only_one_merge_pair(self):

        result = ad.AnnotatedDoc.merge_mult_splits(self.only_one_merge_pair)

        self.assertEqual(result, self.only_one_merge_pair_answer)

    def test_quality_check_bioinfer_single(self):

        result = ad.AnnotatedDoc.quality_check_sent_splits(
            self.bioinfer_single)

        self.assertEqual(result, self.bioinfer_single_answer)

    def test_quality_check_pickle_mult_no_overlaps_or_merges(self):

        result = ad.AnnotatedDoc.quality_check_sent_splits(
            self.pickle_mult_no_overlaps_or_merges)

        self.assertEqual(result, self.pickle_mult_no_overlaps_or_merges_answer)

    def test_quality_check_pickle_mult_subsequent_merges(self):

        result = ad.AnnotatedDoc.quality_check_sent_splits(
            self.pickle_mult_subsequent_merges)

        self.assertEqual(result, self.pickle_mult_subsequent_merges_answer)

    def test_quality_check_seedev_true_cross_sent_rel(self):

        result = ad.AnnotatedDoc.quality_check_sent_splits(self.seedev_true_cross_sent_rel)

        self.assertEqual(result, self.seedev_true_cross_sent_rel_answer)


class TestDropCounters(unittest.TestCase):
    """
    Tests the functionality of the entity and relation counters in the
    AnnotatedDoc class.
    """
    def setUp(self):

        # Set up temp dir and test docs
        self.tmpdir = "tmp"
        os.makedirs(self.tmpdir, exist_ok=True)

        simple_txt = (
            "Seattle is a rainy city. Jenny Durkan is the city's mayor. "
            "She was elected in 2017.")

        self.simple_txt = f'{self.tmpdir}/mysimplefile.txt'
        with open(self.simple_txt, 'w') as f:
            f.write(simple_txt)

        simple_ann = ("T1\tCity 0 7;13 23\tSeattle\n"
                      "T2\tPerson 25 37\tJenny Durkan\n"
                      "T3\tCity 41 51\tthe city's\n"
                      "T4\tPerson 59 62\tShe\n"
                      "T5\tPersonnel.Election 67 74\telected\n"
                      "T6\tYear 78 82\t2017\n"
                      "T7\tCity 13 23\trainy city\n"
                      "R1\tIs-A Arg1:T1 Arg2:T7\n"
                      "R2\tMayor-Of Arg1:T2 Arg2:T3\n"
                      "E1\tPersonnel.Election:T5 Person:T4 Year:T6\n"
                      "*\tEQUIV T1 T3\n"
                      "*\tEQUIV T2 T4\n")

        self.simple_ann = f'{self.tmpdir}/mysimplefile.ann'
        with open(self.simple_ann, 'w') as f:
            f.write(simple_ann)

        complex_txt = (
            "Global target profile of the kinase inhibitor bosutinib "
            "in primary chronic myeloid leukemia cells.\n"
            "The detailed molecular mechanism of action of second-generation "
            "BCR-ABL tyrosine kinase inhibitors, including perturbed targets and "
            "pathways, should contribute to rationalized therapy in chronic "
            "myeloid leukemia (CML) or in other affected diseases. Here, we "
            "characterized the target profile of the dual SRC/ABL inhibitor "
            "bosutinib employing a two-tiered approach using chemical proteomics "
            "to identify natural binders in whole cell lysates of primary CML "
            "and K562 cells in parallel to in vitro kinase assays against a large "
            "recombinant kinase panel. The combined strategy resulted in a global "
            "survey of bosutinib targets comprised of over 45 novel tyrosine "
            "and serine/threonine kinases. We have found clear differences in the "
            "target patterns of bosutinib in primary CML cells versus the K562 "
            "cell line. A comparison of bosutinib with dasatinib across the "
            "whole kinase panel revealed overlapping, but distinct, inhibition "
            "profiles. Common among those were the SRC, ABL and TEC family "
            "kinases. Bosutinib did not inhibit KIT or platelet-derived growth "
            "factor receptor, but prominently targeted the apoptosis-linked "
            "STE20 kinases. Although in vivo bosutinib is inactive against ABL "
            "T315I, we found this clinically important mutant to be "
            "enzymatically inhibited in the mid-nanomolar range. Finally, "
            "bosutinib is the first kinase inhibitor shown to target CAMK2G, "
            "recently implicated in myeloid leukemia cell proliferation.")

        self.complex_txt = f'{self.tmpdir}/mycomplexfile.txt'
        with open(self.complex_txt, 'w') as f:
            f.write(complex_txt)

        complex_ann = (
            "T10\tCHEMICAL 932 941\tdasatinib\n"
            "T11\tCHEMICAL 1090 1099\tBosutinib\n"
            "T12\tCHEMICAL 46 55\tbosutinib\n"
            "T13\tGENE-Y 1116 1119\tKIT\n"
            "T14\tGENE-N 1123 1162\tplatelet-derived growth factor receptor\n"
            "T15\tGENE-N 1210 1223\tSTE20 kinases\n"
            "T16\tGENE-Y 1272 1275\tABL\n"
            "T17\tGENE-N 1276 1281\tT315I\n"
            "T18\tGENE-N 1415 1421\tkinase\n"
            "T19\tGENE-Y 1448 1454\tCAMK2G\n"
            "T1\tCHEMICAL 1242 1251\tbosutinib\n"
            "T20\tGENE-Y 402 405\tSRC\n"
            "T21\tGENE-Y 406 409\tABL\n"
            "T22\tGENE-N 592 598\tkinase\n"
            "T23\tGENE-N 634 640\tkinase\n"
            "T24\tGENE-Y 163 166\tBCR\n"
            "T25\tGENE-N 746 783\ttyrosine and serine/threonine kinases\n"
            "T26\tGENE-Y 167 170\tABL\n"
            "T27\tGENE-N 171 186\ttyrosine kinase\n"
            "T28\tGENE-N 959 965\tkinase\n"
            "T29\tGENE-Y 1057 1060\tSRC\n"
            "T2\tCHEMICAL 1392 1401\tbosutinib\n"
            "T30\tGENE-Y 1062 1065\tABL\n"
            "T31\tGENE-Y 1070 1073\tTEC\n"
            "T32\tGENE-N 1081 1088\tkinases\n"
            "T33\tGENE-N 29 35\tkinase\n"
            "T3\tCHEMICAL 420 429\tbosutinib\n"
            "T4\tCHEMICAL 701 710\tbosutinib\n"
            "T5\tCHEMICAL 746 754\ttyrosine\n"
            "T6\tCHEMICAL 759 765\tserine\n"
            "T7\tCHEMICAL 766 775\tthreonine\n"
            "T8\tCHEMICAL 843 852\tbosutinib\n"
            "T9\tCHEMICAL 917 926\tbosutinib\n"
            "R0\tCPR:10 Arg1:T11 Arg2:T13\n"
            "R1\tCPR:10 Arg1:T11 Arg2:T14\n"
            "R2\tCPR:10 Arg1:T1 Arg2:T16\n"
            "R3\tCPR:10 Arg1:T1 Arg2:T17\n"
            "R4\tCPR:2 Arg1:T11 Arg2:T15\n"
            "R5\tCPR:4 Arg1:T10 Arg2:T28\n"
            "R6\tCPR:4 Arg1:T12 Arg2:T33\n"
            "R7\tCPR:4 Arg1:T2 Arg2:T18\n"
            "R8\tCPR:4 Arg1:T2 Arg2:T19\n"
            "R9\tCPR:4 Arg1:T3 Arg2:T20\n"
            "R10\tCPR:4 Arg1:T3 Arg2:T21\n"
            "R11\tCPR:4 Arg1:T9 Arg2:T28\n")

        self.complex_ann = f'{self.tmpdir}/mycomplexfile.ann'
        with open(self.complex_ann, 'w') as f:
            f.write(complex_ann)

        # Define other attributes
        self.nlp = spacy.load("en_core_web_sm")
        self.scinlp = spacy.load("en_core_sci_sm")
        self.dataset = 'scierc'

    def test_entity_counters_simple(self):

        annotated_doc = ad.AnnotatedDoc.parse_ann(self.simple_txt,
                                                  self.simple_ann,
                                                  self.nlp,
                                                  self.dataset,
                                                  coref=True)
        annotated_doc.char_to_token()
        res = annotated_doc.format_dygiepp()
        self.assertEqual(annotated_doc.total_original_ents, 7)
        self.assertEqual(annotated_doc.dropped_ents, 1)

    def test_relation_counters_simple(self):

        annotated_doc = ad.AnnotatedDoc.parse_ann(self.simple_txt,
                                                  self.simple_ann,
                                                  self.nlp,
                                                  self.dataset,
                                                  coref=True)
        annotated_doc.char_to_token()
        res = annotated_doc.format_dygiepp()
        self.assertEqual(annotated_doc.total_original_rels, 2)
        self.assertEqual(annotated_doc.dropped_rels, 1)

    def test_entity_counters_complex(self):

        annotated_doc = ad.AnnotatedDoc.parse_ann(self.complex_txt,
                                                  self.complex_ann,
                                                  self.scinlp,
                                                  self.dataset,
                                                  coref=True)
        annotated_doc.char_to_token()
        res = annotated_doc.format_dygiepp()
        self.assertEqual(annotated_doc.total_original_ents, 33)
        self.assertEqual(annotated_doc.dropped_ents, 6)

    def test_relation_counters_complex(self):

        annotated_doc = ad.AnnotatedDoc.parse_ann(self.complex_txt,
                                                  self.complex_ann,
                                                  self.scinlp,
                                                  self.dataset,
                                                  coref=True)
        annotated_doc.char_to_token()
        res = annotated_doc.format_dygiepp()
        self.assertEqual(annotated_doc.total_original_rels, 12)
        self.assertEqual(annotated_doc.dropped_rels, 2)


if __name__ == "__main__":
    unittest.main()
