from allennlp.data import Batch
from dygie.models.dygie import DyGIE
from dygie.data.dataset_readers.dygie import DyGIEReader
from allennlp.models.archival import load_archive
from allennlp.nn import util
from spacy.language import Language
from spacy.tokens import Span
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

Doc.set_extension("rels", default=[], force=True)
Span.set_extension("rels", default=[], force=True)



class DygieppPipe:
    name = 'dygiepp'
    def __init__(self,
        nlp: Language,
        pretrained_filepath: str = "./pretrained/scierc-lightweight.tar.gz"
    ) -> None:
        """spacy factory class for adding information to spacy document. For now just entities and relations.
        It adds entities to doc.ents and relations to doc._.rels: List[List[Token,Token,str]] which is a list of relations
        as  entity1, entity2, relation name

        Args:
            nlp (Language): Spacy Language instance
            name (str, optional): Pipe name. Defaults to "dygiepp".
            pretrained_filepath (str, optional): Address of pre-trained model to extract information. Defaults to "./pretrained/scierc-lightweight.tar.gz".
        """
        # TODO add events and cluster information to spacy doc too
        archive = load_archive(pretrained_filepath)
        self._model = archive.model
        archive.config['dataset_reader'].pop('type') # it's stupid but was necessary!
        self._dataset_reader = DyGIEReader.from_params(archive.config['dataset_reader'])
    def __call__(self, doc: Doc) -> Doc:
        cuda_device = self._model._get_prediction_device()
        sentences = [[tok.text for tok in sent] for sent in doc.sents]
        ins = self._dataset_reader.text_to_instance({"sentences": sentences, "doc_key": "test", "dataset": "scierc"})
        dataset = Batch([ins])
        dataset.index_instances(self._model.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        prediction = self._model.make_output_human_readable(self._model(**model_input)).to_json()
        # store the relations to doc._.rels of spacy extension
        doc_rels = []
        for rels,ds in zip(prediction['predicted_relations'],doc.sents):
            sent_rels = []
            for rel in rels:
                e1 = doc[rel[0]:rel[1]+1]
                e2 = doc[rel[2]:rel[3]+1]
                tag = rel[4]
                sent_rels.append((e1,e2,tag))
            doc_rels.append(sent_rels)
            ds._.rels = sent_rels
        doc._.rels = doc_rels
        # store entities to doc.ents of spacy
        preds = [p for r in prediction['predicted_ner'] for p in r]
        # because spacy can't support the overlapped entities we have to merge overlapped entities
        # to the longest ones.
        dist_ents = []
        prc = []
        for i,p1 in enumerate(preds):
            t=[p1]
            if i in prc:
                continue
            for j,p2 in enumerate(preds[i+1:]):
                if p2[0]<=p1[1]:
                    t.append(p1)
                    prc.append(j+i+1)
            dist_ents.append(t)
        res = []
        for t in dist_ents:
            if len(t) == 1:
                res.append(t[0])
            elif len(t)>1:
                mn = t[0][0]
                mx = t[0][1]
                for p in t[1:]:
                    if p[0]<mn:
                        mn = p[0]
                    if p[1]>mx:
                        mx=p[1]
                res.append([mn,mx,t[0][2],t[0][3],t[0][4]])
        sel_ents = []
        for ent in res:
            try:
                d = doc[ent[0]:ent[1]+1]
                s = doc.char_span(d.start_char, d.end_char, label=ent[2])
                if s:
                    sel_ents.append(s)
            except Exception as e:
                print("error in spacy span", e)
                raise e
        doc.ents = sel_ents
        return doc

