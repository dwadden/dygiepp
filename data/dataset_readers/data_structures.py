import json
from dygie.models.shared import fields_to_batches
import numpy as np


class Dataset:
    def __init__(self, json_file):
        with open(json_file) as f:
            self.js = [json.loads(line) for line in f]
        self.documents = [Document(js) for js in self.js]

    def __getitem__(self, ix):
        return self.documents[ix]


class Document:
    def __init__(self, js):
        self._doc_key = js["doc_key"]
        entries = fields_to_batches(js, ["doc_key"])
        sentence_lengths = [len(entry["sentences"]) for entry in entries]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0
        self.sentences = [Sentence(entry, sentence_start)
                          for (entry, sentence_start)
                          in zip(entries, sentence_starts)]

    def __repr__(self):
        return "\n".join([str(i) + ": " + " ".join(sent.text) for i, sent in enumerate(self.sentences)])

    def __getitem__(self, ix):
        return self.sentences[ix]


class Sentence:
    def __init__(self, entry, sentence_start):
        self.sentence_start = sentence_start
        self.text = entry["sentences"]
        # Gold
        self.ner = [NER(this_ner, self.text, sentence_start)
                    for this_ner in entry["ner"]]
        self.relations = [Relation(this_relation, self.text, sentence_start) for
                          this_relation in entry["relations"]]
        self.events = Events(entry["events"], self.text, sentence_start)

        # Predicted
        self.predicted_ner = [NER(this_ner, self.text, sentence_start) for
                              this_ner in entry["predicted_ner"]]
        self.predicted_relations = [Relation(this_relation, self.text, sentence_start) for
                                    this_relation in entry["predicted_relations"]]
        self.predicted_events = Events(entry["predicted_events"], self.text, sentence_start)

    def __repr__(self):
        the_text = " ".join(self.text)
        the_lengths = np.array([len(x) for x in self.text])
        tok_ixs = ""
        for i, offset in enumerate(the_lengths):
            true_offset = offset if i < 10 else offset - 1
            tok_ixs += str(i)
            tok_ixs += " " * true_offset

        return the_text + "\n" + tok_ixs


class Span:
    def __init__(self, start, end, text, sentence_start):
        self.start_doc = start
        self.end_doc = end
        self.span_doc = (self.start_doc, self.end_doc)
        self.start_sent = start - sentence_start
        self.end_sent = end - sentence_start
        self.span_sent = (self.start_sent, self.end_sent)
        self.text = text[self.start_sent:self.end_sent + 1]

    def __repr__(self):
        return str((self.start_sent, self.end_sent, self.text))

    def __eq__(self, other):
        return (self.span_doc == other.span_doc and
                self.span_sent == other.span_sent and
                self.text == other.text)

    def __hash__(self):
        tup = self.span_doc + self.span_sent + (" ".join(self.text),)
        return hash(tup)


class Token:
    def __init__(self, ix, text, sentence_start):
        self.ix_doc = ix
        self.ix_sent = ix - sentence_start
        self.text = text[self.ix_sent]

    def __repr__(self):
        return str((self.ix_sent, self.text))


class Trigger:
    def __init__(self, token, label):
        self.token = token
        self.label = label

    def __repr__(self):
        return self.token.__repr__()[:-1] + ", " + self.label + ")"


class Argument:
    def __init__(self, span, role, event_type):
        self.span = span
        self.role = role
        self.event_type = event_type

    def __repr__(self):
        return self.span.__repr__()[:-1] + ", " + self.event_type + ", " + self.role + ")"

    def __eq__(self, other):
        return (self.span == other.span and
                self.role == other.role and
                self.event_type == other.event_type)

    def __hash__(self):
        return self.span.__hash__() + hash((self.role, self.event_type))


class NER:
    def __init__(self, ner, text, sentence_start):
        self.span = Span(ner[0], ner[1], text, sentence_start)
        self.label = ner[2]

    def __repr__(self):
        return self.span.__repr__() + ": " + self.label


class Relation:
    def __init__(self, relation, text, sentence_start):
        start1, end1 = relation[0], relation[1]
        start2, end2 = relation[2], relation[3]
        label = relation[4]
        span1 = Span(start1, end1, text, sentence_start)
        span2 = Span(start2, end2, text, sentence_start)
        self.pair = (span1, span2)
        self.label = label

    def __repr__(self):
        return self.pair[0].__repr__() + ", " + self.pair[1].__repr__() + ": " + self.label


class Event:
    def __init__(self, event, text, sentence_start):
        trig = event[0]
        args = event[1:]
        trigger_token = Token(trig[0], text, sentence_start)
        self.trigger = Trigger(trigger_token, trig[1])

        self.arguments = []
        for arg in args:
            span = Span(arg[0], arg[1], text, sentence_start)
            self.arguments.append(Argument(span, arg[2], self.trigger.label))

    def __repr__(self):
        res = "<"
        res += self.trigger.__repr__() + ": "
        for arg in self.arguments:
            res += arg.__repr__() + "; "
        res = res[:-2] + ">"
        return res


class Events:
    def __init__(self, events_json, text, sentence_start):
        self.event_list = [Event(this_event, text, sentence_start) for this_event in events_json]
        self.triggers = set([event.trigger for event in self.event_list])
        self.arguments = set([arg for event in self.event_list for arg in event.arguments])

    def __len__(self):
        return len(self.event_list)

    def __getitem__(self, i):
       return self.event_list[i]

    def __repr__(self):
        return "\n".join([event.__repr__() for event in self.event_list])

    def span_matches(self, argument):
        return set([candidate for candidate in self.arguments
                    if candidate.span.span_sent == argument.span.span_sent])

    def event_type_matches(self, argument):
        return set([candidate for candidate in self.span_matches(argument)
                    if candidate.event_type == argument.event_type])

    def matches_except_event_type(self, argument):
        matched = [candidate for candidate in self.span_matches(argument)
                   if candidate.event_type != argument.event_type
                   and candidate.role == argument.role]
        return set(matched)

    def exact_match(self, argument):
        for candidate in self.arguments:
            if candidate == argument:
                return True
        return False
