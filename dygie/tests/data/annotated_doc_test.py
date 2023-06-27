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

class TestDropCounters(unittest.TestCase):
    """
    Tests the functionality of the entity and relation counters in the
    AnnotatedDoc class..
    """
    def setUp(self):

        # Set up temp dir and test docs
        self.tmpdir = "tmp"
        os.makedirs(self.tmpdir, exist_ok=True)

        simple_txt = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
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

        complex_txt = ("Global target profile of the kinase inhibitor bosutinib "
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

        complex_ann = ("T10\tCHEMICAL 932 941\tdasatinib\n"
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
