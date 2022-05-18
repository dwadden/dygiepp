from typing import Dict, List
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
Doc.set_extension("span_ents", default=[], force=True)
Span.set_extension("label_", default=[], force=True)
Doc.set_extension("events", default=[], force=True)
Span.set_extension("events", default=[], force=True)


def prepare_spacy_doc(doc: Doc, prediction: Dict) -> Doc:
    doc_rels = []
    doc_evs = []
    # store events as relations. include confidence scores in the relation tuple (TODO: add relation property)
    for evs, ds in zip(prediction.get("predicted_events", []), doc.sents):
        sent_evs = []
        for ev in evs:
            if len(ev)>=3:
                trig = [r for r in ev if r[1]=="TRIGGER"]
                arg0s = [r for r in ev if r[2]=="ARG0"]
                #example arg0s: [[40, 43, 'ARG0', 12.1145, 1.0], [45, 45, 'ARG0', 11.3498, 1.0]]
                arg1s = [r for r in ev if r[2]=="ARG1"]
                e_trig = doc[trig[0][0]:trig[0][0]+1]
                for arg0 in arg0s:
                    e_arg0 = doc[arg0[0] : arg0[1] + 1]
                    for arg1 in arg1s:
                        e_arg1 = doc[arg1[0] : arg1[1] + 1]
                        #here confidence is set as the minimum among {trigger,args}, as a conservative measure.
                        sent_evs.append({"ARG0":e_arg0,"ARG1":e_arg1,"RELATION_TRIGGER":e_trig,"CONF":min([arg0[4],arg1[4],trig[0][3]])})
                        
        doc_evs.append(sent_evs)
        ds._.events = sent_evs
    doc._.events = doc_evs
    #TODO add doc._.span_ents too. 

    for rels, ds in zip(prediction.get("predicted_relations", []), doc.sents):
        sent_rels = []
        for rel in rels:
            e1 = doc[rel[0] : rel[1] + 1]
            e2 = doc[rel[2] : rel[3] + 1]
            tag = rel[4]
            sent_rels.append((e1, e2, tag))
        doc_rels.append(sent_rels)
        ds._.rels = sent_rels
    doc._.rels = doc_rels
    if "predicted_ner" not in prediction:
        return doc
    preds = [p for r in prediction.get("predicted_ner", []) for p in r]
    # storing all span based entitis to doc._.span_ents
    span_ents = []
    for sent in prediction["predicted_ner"]:
        ent_sent = []
        for ent in sent:
            d = doc[ent[0] : ent[1] + 1]
            d._.label_ = ent[2]
            ent_sent.append(d)
        span_ents.append(ent_sent)
    doc._.span_ents = span_ents
    # store entities to doc.ents of spacy
    # because spacy can't support the overlapped entities we have to merge overlapped entities
    # to the longest ones.
    dist_ents = []
    prc = []
    for i, p1 in enumerate(preds):
        t = [p1]
        if i in prc:
            continue
        for j, p2 in enumerate(preds[i + 1 :]):
            if p2[0] <= p1[1]:
                t.append(p1)
                prc.append(j + i + 1)
        dist_ents.append(t)
    res = []
    for t in dist_ents:
        if len(t) == 1:
            res.append(t[0])
        elif len(t) > 1:
            mn = t[0][0]
            mx = t[0][1]
            for p in t[1:]:
                if p[0] < mn:
                    mn = p[0]
                if p[1] > mx:
                    mx = p[1]
            res.append([mn, mx, t[0][2], t[0][3], t[0][4]])
    sel_ents = []
    for ent in res:
        try:
            d = doc[ent[0] : ent[1] + 1]
            s = doc.char_span(d.start_char, d.end_char, label=ent[2])
            if s:
                sel_ents.append(s)
        except Exception as e:
            print("error in spacy span", e)
            raise e
    doc.ents = sel_ents
    return doc


class DygieppPipe:
    name = "dygiepp"

    def __init__(
        self,
        nlp: Language,
        pretrained_filepath: str = "./pretrained/scierc-lightweight.tar.gz",
        dataset_name: str = "scierc",
    ) -> None:
        """spacy factory class for adding information to spacy document. For now just entities and relations.
        It adds entities to doc.ents and relations to doc._.rels: List[List[Token,Token,str]] which is a list of relations
        as  entity1, entity2, relation name

        Args:
            nlp (Language): Spacy Language instance
            name (str, optional): Pipe name. Defaults to "dygiepp".
            pretrained_filepath (str, optional): Address of pre-trained model to extract information. Defaults to "./pretrained/scierc-lightweight.tar.gz".
            dataset_name (str, optional): Dataset name used for model. Defaults to "scierc".
        """
        # TODO add events and cluster information to spacy doc too
        archive = load_archive(pretrained_filepath)
        self._model = archive.model
        self._model.eval()
        archive.config["dataset_reader"].pop("type")  # it's stupid but was necessary!
        self._dataset_reader = DyGIEReader.from_params(archive.config["dataset_reader"])
        self.dataset_name = dataset_name

    def __call__(self, doc: Doc) -> Doc:
        cuda_device = self._model._get_prediction_device()
        sentences = [[tok.text for tok in sent] for sent in doc.sents]
        ins = self._dataset_reader.text_to_instance(
            {"sentences": sentences, "doc_key": "test", "dataset": self.dataset_name}
        )
        dataset = Batch([ins])
        dataset.index_instances(self._model.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        prediction = self._model.make_output_human_readable(
            self._model(**model_input)
        ).to_json()
        # prepare and store ent/relation information to spacy Doc
        return prepare_spacy_doc(doc, prediction)
