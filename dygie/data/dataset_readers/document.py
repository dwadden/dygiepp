from dygie.models.shared import fields_to_batches
import numpy as np


def get_sentence_of_span(span, sentence_starts, doc_tokens):
    """
    Return the index of the sentence that the span is part of.
    """
    # Inclusive sentence ends
    sentence_ends = [x - 1 for x in sentence_starts[1:]] + [doc_tokens - 1]
    in_between = [span[0] >= start and span[1] <= end
                  for start, end in zip(sentence_starts, sentence_ends)]
    assert sum(in_between) == 1
    the_sentence = in_between.index(True)
    return the_sentence


class Document:
    def __init__(self, js):
        self.doc_key = js["doc_key"]
        self.dataset = js.get("dataset")
        entries = fields_to_batches(js, ["doc_key", "clusters"])
        sentence_lengths = [len(entry["sentences"]) for entry in entries]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0
        self.sentence_starts = sentence_starts
        self.sentences = [Sentence(entry, sentence_start, sentence_ix, self)
                          for sentence_ix, (entry, sentence_start)
                          in enumerate(zip(entries, sentence_starts))]
        # Store cofereference annotations.
        if "clusters" in js:
            self.clusters = [Cluster(entry, i, self)
                             for i, entry in enumerate(js["clusters"])]
        else:
            self.clusters = None
        self._update_sentences_with_clusters()

    def _update_sentences_with_clusters(self):
        "Add cluster dictionary to each sentence, if there are coreference clusters."
        for sent in self.sentences:
            sent.cluster_dict = {} if self.clusters is not None else None

        if self.clusters is None:
            return

        for clust in self.clusters:
            for member in clust.members:
                sent = member.sentence
                sent.cluster_dict[member.span.span_sent] = member.cluster_id

    def __repr__(self):
        return "\n".join([str(i) + ": " + " ".join(sent.text) for i, sent in enumerate(self.sentences)])

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)

    def print_plaintext(self):
        for sent in self:
            print(" ".join(sent.text))

    def find_cluster(self, entity):
        """
        Search through coreference clusters and return the one containing the query entity, if it's
        part of a cluster. If we don't find a match, return None.
        """
        for clust in self.clusters:
            for entry in clust:
                if entry.span == entity.span:
                    return clust

        return None

    @property
    def n_tokens(self):
        return sum([len(sent) for sent in self.sentences])


class Sentence:
    def __init__(self, entry, sentence_start, sentence_ix, doc):
        self.sentence_start = sentence_start
        self.text = entry["sentences"]
        self.sentence_ix = sentence_ix
        self.doc = doc

        # Store events.
        if "ner" in entry:
            self.ner = [NER(this_ner, self.text, sentence_start)
                        for this_ner in entry["ner"]]
            self.ner_dict = {entry.span.span_sent: entry.label for entry in self.ner}
        else:
            self.ner = None
            self.ner_dict = None

        # Store relations.
        if "relations" in entry:
            self.relations = [Relation(this_relation, self.text, sentence_start) for
                              this_relation in entry["relations"]]
            relation_dict = {}
            for rel in self.relations:
                key = (rel.pair[0].span_sent, rel.pair[1].span_sent)
                relation_dict[key] = rel.label
            self.relation_dict = relation_dict
        else:
            self.relations = None
            self.relation_dict = None

        # Store events.
        if "events" in entry:
            self.events = Events(entry["events"], self.text, sentence_start)
        else:
            self.events = None

    def __repr__(self):
        the_text = " ".join(self.text)
        the_lengths = np.array([len(x) for x in self.text])
        tok_ixs = ""
        for i, offset in enumerate(the_lengths):
            true_offset = offset if i < 10 else offset - 1
            tok_ixs += str(i)
            tok_ixs += " " * true_offset

        return the_text + "\n" + tok_ixs

    def __len__(self):
        return len(self.text)


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

    def __eq__(self, other):
        return (self.span == other.span and
                self.label == other.label)


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

    def __eq__(self, other):
        return (self.pair == other.pair) and (self.label == other.label)


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
        res += self.trigger.__repr__() + ":\n"
        for arg in self.arguments:
            res += 6 * " " + arg.__repr__() + ";\n"
        res = res[:-2] + ">"
        return res


class Events:
    def __init__(self, events_json, text, sentence_start):
        self.event_list = [Event(this_event, text, sentence_start) for this_event in events_json]
        self.triggers = set([event.trigger for event in self.event_list])
        self.arguments = set([arg for event in self.event_list for arg in event.arguments])

        # Store trigger and argument dictionaries.
        trigger_dict = {}
        argument_dict = {}
        for event in self.event_list:
            trigger_key = event.trigger.token.ix_sent  # integer index
            trigger_val = event.trigger.label          # trigger label
            trigger_dict[trigger_key] = trigger_val
            for argument in event.arguments:
                arg_key = (trigger_key, argument.span.span_sent)  # (trigger_ix, (arg_start, arg_end))
                arg_value = argument.role                         # argument label
                argument_dict[arg_key] = arg_value

        self.trigger_dict = trigger_dict
        self.argument_dict = argument_dict

    def __len__(self):
        return len(self.event_list)

    def __getitem__(self, i):
        return self.event_list[i]

    def __repr__(self):
        return "\n\n".join([event.__repr__() for event in self.event_list])

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


class Cluster:
    def __init__(self, cluster, cluster_id, document):
        # Make sure the cluster ID is an int.
        if not isinstance(cluster_id, int):
            raise TypeError("Coreference cluster ID's must be ints.")

        members = []
        for entry in cluster:
            sentence_ix = get_sentence_of_span(entry, document.sentence_starts, document.n_tokens)
            sentence = document[sentence_ix]
            span = Span(entry[0], entry[1], sentence.text, sentence.sentence_start)
            # If we're doing predicted clusters, use the predicted entities.
            ners = [x for x in sentence.ner if x.span == span]
            assert len(ners) <= 1
            ner = ners[0] if len(ners) == 1 else None
            to_append = ClusterMember(span, ner, sentence, cluster_id)
            members.append(to_append)

        self.members = members
        self.cluster_id = cluster_id

    def __repr__(self):
        return f"{self.cluster_id}: " + self.members.__repr__()

    def __getitem__(self, ix):
        return self.members[ix]


class ClusterMember:
    def __init__(self, span, ner, sentence, cluster_id):
        self.span = span
        self.ner = ner
        self.sentence = sentence
        self.cluster_id = cluster_id

    def __repr__(self):
        return f"<{self.sentence.sentence_ix}> " + self.span.__repr__()
