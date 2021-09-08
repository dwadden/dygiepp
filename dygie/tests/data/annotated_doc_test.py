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

        # Set up document text 
        self.nlp = spacy.load("en_core_web_sm") 
        self.text = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
                    "She was elected in 2017.")
        self.sents = [[tok.text for tok in sent] for sent in self.nlp(self.text).sents]
        self.sent_idx_tups = [(0, 6), (6, 14), (14, 19)]
        # NOTE: spacy tokenizes words with apostrophes into separate words.
        
        # Set up entities 
        self.ent_list = [ad.Ent("T1\tCity 0 7\tSeattle".split()),
                        ad.Ent("T2\tPerson 25 37\tJenny Durkan".split()),
                        ad.Ent("T3\tCity 41 51\tthe city's".split())]
        
        # Right answer
        self.ner = [[[0, 0, "City"]],[[6, 7, "Person"], [9, 11, "City"]],[]]


    def test_char_to_token(self):

        ad.Ent.char_to_token(self.ent_list, self.sents, self.nlp)
        
        self.assertEqual(self.ent_list[0].start, 0)
        self.assertEqual(self.ent_list[1].start, 6)
        self.assertEqual(self.ent_list[2].start, 9)

        self.assertEqual(self.ent_list[0].end, 0)
        self.assertEqual(self.ent_list[1].end, 7)
        self.assertEqual(self.ent_list[2].end, 11)


    def test_format_ner_dygiepp(self):

        ad.Ent.char_to_token(self.ent_list, self.sents, self.nlp)
        
        ner = ad.Ent.format_ner_dygiepp(self.ent_list, self.sent_idx_tups)

        self.assertEqual(ner, self.ner)


class TestBinRel(unittest.TestCase):

    def setUp(self):

        # Set up document text 
        self.nlp = spacy.load("en_core_web_sm") 
        self.text = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
                    "She was elected in 2017.")
        self.sents = [[tok.text for tok in sent] for sent in self.nlp(self.text).sents]
        self.sent_idx_tups = [(0, 6), (6, 14), (14, 19)]
        # NOTE: spacy tokenizes words with apostrophes into separate words.
       
        # Set up entities 
        self.ents = [ad.Ent("T1\tCity 0 7\tSeattle".split()),
                    ad.Ent("T2\tPerson 25 37\tJenny Durkan".split()),
                    ad.Ent("T3\tCity 41 51\tthe city's".split())]
        ad.Ent.char_to_token(self.ents, self.sents, self.nlp)
        
        self.rel1 = ad.BinRel("R1\tMayor-Of Arg1:T2 Arg2:T3".split())

        self.relations = [[], [[6, 7, 9, 11, "Mayor-Of"]],[]]


    def test_set_arg_objects(self):
        
        self.rel1.set_arg_objects(self.ents)

        self.assertEqual(self.rel1.arg1, self.ents[1])
        self.assertEqual(self.rel1.arg2, self.ents[2])


    def test_format_bin_rels_dygiepp(self):

        self.rel1.set_arg_objects(self.ents)
        relations = ad.BinRel.format_bin_rels_dygiepp([self.rel1], self.sent_idx_tups)

        self.assertEqual(relations, self.relations)


class TestEvent(unittest.TestCase):

    def setUp(self):
        
        # Set up document text 
        self.nlp = spacy.load("en_core_web_sm") 
        self.text = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
                    "She was elected in 2017.")
        self.sents = [[tok.text for tok in sent] for sent in self.nlp(self.text).sents]
        self.sent_idx_tups = [(0, 6), (6, 14), (14, 19)]
        # NOTE: spacy tokenizes words with apostrophes into separate words.
       
        # Set up entities 
        self.ents = [ad.Ent("T4\tPerson 59 62\tShe".split()),
                     ad.Ent("T5\tPersonnel.Election 67 74\telected".split()),
                     ad.Ent("T6\tYear 78 82\t2017".split())]
        ad.Ent.char_to_token(self.ents, self.sents, self.nlp)

        # Set up events 
        self.event1 = ad.Event("E1\tPersonnel.Election:T5 Person:T4 Year:T6".split())

        # Right answer 
        self.events = [[],[],
                [[[16, "Personnel.Election"],[14, 14, "Person"],[18, 18, "Year"]]]]


    def test_set_arg_objects(self):
        
        self.event1.set_arg_objects(self.ents)

        self.assertEqual(self.event1.trigger, self.ents[1])
        self.assertEqual(self.event1.args, [self.ents[0], self.ents[2]])


    def test_format_events_dygiepp(self):

        self.event1.set_arg_objects(self.ents)
        events = ad.Event.format_events_dygiepp([self.event1], self.sent_idx_tups)

        self.assertEqual(events, self.events)


class TestEquivRel(unittest.TestCase):

    def setUp(self):

        # Set up document text 
        self.nlp = spacy.load("en_core_web_sm") 
        self.text = ("Seattle is a rainy city. Jenny Durkan is the city's mayor. "
                    "She was elected in 2017.")
        self.sents = [[tok.text for tok in sent] for sent in self.nlp(self.text).sents]
        self.sent_idx_tups = [(0, 6), (6, 14), (14, 19)]
        # NOTE: spacy tokenizes words with apostrophes into separate words.

        # Set up entities 
        self.ents = [ad.Ent("T1\tCity 0 7\tSeattle".split()),
                     ad.Ent("T2\tPerson 25 37\tJenny Durkan".split()),
                     ad.Ent("T3\tCity 41 51\tthe city's".split()),
                     ad.Ent("T4\tPerson 59 62\tShe".split()),
                     ad.Ent("T5\tPersonnel.Election 67 74\telected".split()),
                     ad.Ent("T6\tYear 78 82\t2017".split())]
        ad.Ent.char_to_token(self.ents, self.sents, self.nlp)

        # Set up equivalence relations 
        self.equivrel1 = ad.EquivRel("*\tEQUIV T1 T3".split())
        self.equivrel2 = ad.EquivRel("*\tEQUIV T2 T4".split())

        # The dygiepp-formatted correct answer 
        self.corefs = [[[0,0],[9,11]],[[6,7],[14,14]]]


    def test_set_arg_objects(self):

        self.equivrel1.set_arg_objects(self.ents)
        self.equivrel2.set_arg_objects(self.ents)

        self.assertEqual(self.equivrel1.args, [self.ents[0], self.ents[2]])
        self.assertEqual(self.equivrel2.args, [self.ents[1], self.ents[3]])


    def test_format_corefs_dygiepp(self):

        self.equivrel1.set_arg_objects(self.ents)
        self.equivrel2.set_arg_objects(self.ents)
        corefs = ad.EquivRel.format_corefs_dygiepp([self.equivrel1, self.equivrel2])

        self.assertEqual(corefs, self.corefs)


class TestAnnotatedDoc:
    """
    Only tests the integration functionality of format_dygiepp, since 
    set_annotation_objects and char_to_token just call methods from other 
    classes that have already been unit tested.
    """
    def setUp(self):
        
        # Set up temp dir and test docs 
        self.tmpdir = "tmp"
        os.makedirs(self.tmpdir)
     
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
        dygiepp_dict = {"doc_id":"myfile",
                        "dataset":self.dataset,
                        "sentences":[[tok.text for tok in sent] for sent in self.nlp(txt).sents],
                        "ner":[
                                [[0, 0, "City"]],
                                [[6, 7, "Person"], [9, 11, "City"]],
                                [[14, 14, "Person"], [16, 16, "Personnel.Election"], [18, 18, "2017"]]
                              ],
                        "relations":[[], [[6, 7, 9, 11, "Mayor-Of"]],[]],
                        "events":[[],[],
                                [[[16, "Personnel.Election"],[14, 14, "Person"],[18, 18, "Year"]]]],
                        "clusters":[[[0,0],[9,11]],[[6,7],[14,14]]]}


    def tearDown(self):

        shutil.rmtree(self.tmpdir)


    def test_format_dygiepp(self):
        
        annotated_doc = ad.AnnotatedDoc.parse_ann(self.txt, self.ann, self.nlp, self.dataset)
        annotated_doc.set_annotation_objects()
        annotated_doc.char_to_token()
        res = annotated_doc.format_dygiepp()

        self.assertEqual(res, self.dygiepp_dict)


if __name__ == "__main__":
    unittest.main()
