"""
Spot checks for the classes defined in annotated_doc.py.

Author: Serena G. Lotreck 
"""
import unittest 

from dygie.scripts.new-dataset import annotated_doc as ad 


class TestEnt(unittest.TestCase):

    def setUp(self):

        self.sents = [["Seattle", "is", "a", "rainy", "city","."],
                    ["Jenny", "Durkan", "is", "the", "city's", "mayor", "."],
                    ["She", "was", "elected", "in", "2017", "."]]

        self.sent_idx_tups = [(0, 6), (6, 13), (13, 19)]
        self.one_token = ad.Ent("T1\tCity 0 7\tSeattle".split())
        self.two_token_noPunct = ad.Ent("T2\tPerson 25 37\tJenny Durkan".split())
        self.two_token_punct = ad.Ent("T3\tCity 41 51\tthe city's".split())
        self.ent_list = [self.one_token, self.two_token_noPunct, self.two_token_punct]
        self.ner = [[[0, 0, "City"]],[[6, 7, "Person"], [9, 10, "City"]],[]]
    

    def test_char_to_token(self):

        ad.Ent.char_to_token(self.ent_list, self.sents)

        self.assertEqual(self.ent_list[0].start, 0)
        self.assertEqual(self.ent_list[1].start, 6)
        self.assertEqual(self.ent_list[2].start, 9)

        self.assertEqual(self.ent_list[0].end, 0)
        self.assertEqual(self.ent_list[1].end, 7)
        self.assertEqual(self.ent_list[2].end, 10)


    def test_format_ner_dygiepp(self):

        ner = ad.Ent.format_ner_dygiepp(self.ent_list, self.sent_idx_tups)

        self.assertEqual(ner, self.ner)


class TestBinRel(unittest.TestCase):

    def setUp(self):

        self.sents = [["Seattle", "is", "a", "rainy", "city","."],
                    ["Jenny", "Durkan", "is", "the", "city's", "mayor", "."],
                    ["She", "was", "elected", "in", "2017", "."]]

        self.sent_idx_tups = [(0, 6), (6, 13), (13, 19)]
        
        self.ents = [ad.Ent("T1\tCity 0 7\tSeattle".split()),
                    ad.Ent("T2\tPerson 25 37\tJenny Durkan".split()),
                    ad.Ent("T3\tCity 41 51\tthe city's".split())]
        ad.Ent.char_to_token(selt.ents, self.sents)
        
        self.rel1 = ad.BinRel("R1\tMayor-of Arg1:T2 Arg2:T3".split())

        self.relations = [[], [[6, 7, 9, 10, "Mayor-Of"]],[]]


    def test_set_arg_objects(self):
        
        self.rel1.set_arg_objects(self.ents)

        self.assertEqual(self.rel1.arg1, self.ents[1])
        self.assertEqual(self.rel1.arg2, self.ents[2])


    def test_format_bin_rels_dygiepp(self):

        relations = ad.BinRel.format_bin_rel_dygiepp([self.rel1], self.sents)

        self.assertEqual(relations, self.relations)


class TestEvent(unittest.TestCase)
