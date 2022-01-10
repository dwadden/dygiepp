import unittest
from dygie.spacy_interface.spacy_interface import prepare_spacy_doc
import spacy

class TestSpacyInterface(unittest.TestCase):
    
    def setUp(self) -> None:
        nlp = spacy.load('en_core_web_sm')
        text = "Title: VocGAN: A High-Fidelity Real-time Vocoder with a Hierarchically-nested Adversarial Network\nSection:"
        doc = nlp(text)
        sentences = [[tok.text for tok in sent] for sent in doc.sents]
        self.prediction = {'doc_key': 'test',
                            'dataset': 'scierc',
                            'sentences': sentences,
                            'predicted_ner': [[[2, 2, 'Method', 15.5283, 1.0],
                            [5, 11, 'Method', 3.0847, 0.9563],
                            [6, 11, 'Method', 3.8185, 0.9672],
                            [14, 18, 'Method', 3.4321, 0.9686],
                            [15, 18, 'Method', 11.8431, 1.0],
                            [19, 19, 'Generic', 4.7359, 0.7531]]],
                            'predicted_relations': [[[2, 2, 6, 11, 'HYPONYM-OF', 2.0108, 0.8819],
                            [19, 19, 19, 19, 'USED-FOR', 0.8034, 0.2309]]]}
        self.doc = doc
        return super().setUp()

    def test_relation(self):
        doc = prepare_spacy_doc(self.doc, self.prediction)
        # number of sentences
        self.assertEqual(len(doc._.rels),1)
        # number of relations
        self.assertEqual(len(doc._.rels[0]),2)
        # type of relations
        self.assertEqual(doc._.rels[0][0][2], 'HYPONYM-OF')
        self.assertEqual(doc._.rels[0][1][2], 'USED-FOR')
        

    def test_span_based_entity(self):
        doc = prepare_spacy_doc(self.doc, self.prediction)
        # number of sentences
        self.assertEqual(len(doc._.span_ents),1)
        # number of span based entities 
        self.assertEqual(len(doc._.span_ents[0]),6)

    def test_spacy_entity(self):
        doc = prepare_spacy_doc(self.doc, self.prediction)
        # number of proned merged entities
        self.assertEqual(len(doc.ents),4)







if __name__ == '__main__':
    unittest.main()