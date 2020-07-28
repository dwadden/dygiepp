from dygie.models.shared import fields_to_batches, batches_to_fields
import copy
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


def update_sentences_with_clusters(sentences, clusters):
    "Add cluster dictionary to each sentence, if there are coreference clusters."
    for sent in sentences:
        sent.cluster_dict = {} if clusters is not None else None

    if clusters is None:
        return sentences

    for clust in clusters:
        for member in clust.members:
            sent = member.sentence
            sent.cluster_dict[member.span.span_sent] = member.cluster_id

    return sentences


class Document:
    def __init__(self, doc_key, dataset, sentences, clusters):
        self.doc_key = doc_key
        self.dataset = dataset
        self.sentences = sentences
        self.clusters = clusters

    @classmethod
    def from_json(cls, js):
        "Read in from json-loaded dict."
        doc_key = js["doc_key"]
        dataset = js.get("dataset")
        entries = fields_to_batches(js, ["doc_key", "dataset", "clusters"])
        sentence_lengths = [len(entry["sentences"]) for entry in entries]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0
        sentence_starts = sentence_starts.tolist()
        sentences = [Sentence(entry, sentence_start, sentence_ix)
                     for sentence_ix, (entry, sentence_start)
                     in enumerate(zip(entries, sentence_starts))]
        # Store cofereference annotations.
        if "clusters" in js:
            clusters = [Cluster(entry, i, sentences, sentence_starts)
                        for i, entry in enumerate(js["clusters"])]
        else:
            clusters = None

        # Update the sentences with coreference cluster labels.
        sentences = update_sentences_with_clusters(sentences, clusters)

        return cls(doc_key, dataset, sentences, clusters)

    def to_json(self):
        "Write to json dict."
        res = {"doc_key": self.doc_key,
               "dataset": self.dataset}
        sents_json = [sent.to_json() for sent in self]
        fields_json = batches_to_fields(sents_json)
        res.update(fields_json)
        if self.clusters is not None:
            res["clusters"] = [cluster.to_json() for cluster in self.clusters]
        # TODO(dwadden) Don't use hasattr.
        if hasattr(self, "predicted_clusters"):
            res["predicted_clusters"] = [cluster.to_json() for cluster in self.predicted_clusters]

        return res

    # TODO(dwadden) Write a unit test to make sure this does the correct thing.
    def split(self, max_tokens_per_doc):
        """
        Greedily split a long document into smaller documents, each shorter than
        `max_tokens_per_doc`
        """
        # TODO(dwadden) Implement splitting when there's coref annotations. This is more difficult
        # because coreference clusters have to be split across documents.
        if self.clusters is not None:
            raise NotImplementedError("Splitting documents with coreference annotations not implemented.")

        # If the document is already short enough, return it as a list with a single item.
        if self.n_tokens <= max_tokens_per_doc:
            return [self]

        sentences = copy.deepcopy(self.sentences)

        sentence_groups = []
        current_group = []
        group_length = 0
        sentence_tok_offset = 0
        sentence_ix_offset = 0
        for sentence in sentences:
            # Can't deal with single sentences longer than the limit.
            if len(sentence) > max_tokens_per_doc:
                msg = f"Sentence \"{''.join(sentence.text)}\" has more than {max_tokens_per_doc} tokens. Please split this sentence."
                raise ValueError(msg)

            if group_length + len(sentence) <= max_tokens_per_doc:
                # If we're not at the limit, add it to the current sentence group.
                sentence.sentence_start -= sentence_tok_offset
                sentence.sentence_ix -= sentence_ix_offset
                current_group.append(sentence)
                group_length += len(sentence)
            else:
                # Otherwise, start a new sentence group and adjust sentence offsets.
                sentence_groups.append(current_group)
                sentence_tok_offset = sentence.sentence_start
                sentence_ix_offset = sentence.sentence_ix
                sentence.sentence_start -= sentence_tok_offset
                sentence.sentence_ix -= sentence_ix_offset
                current_group = [sentence]
                group_length = len(sentence)

        # Add the final sentence group.
        sentence_groups.append(current_group)

        # Create a separate document for each sentence group.
        doc_keys = [f"{self.doc_key}_SPLIT_{i}" for i in range(len(sentence_groups))]
        res = [self.__class__(doc_key, self.dataset, sentence_group, self.clusters)
               for doc_key, sentence_group in zip(doc_keys, sentence_groups)]

        return res

    def __repr__(self):
        return "\n".join([str(i) + ": " + " ".join(sent.text) for i, sent in enumerate(self.sentences)])

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)

    def print_plaintext(self):
        for sent in self:
            print(" ".join(sent.text))

    @property
    def n_tokens(self):
        return sum([len(sent) for sent in self.sentences])

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
    def __init__(self, entry, sentence_start, sentence_ix):
        self.sentence_start = sentence_start
        self.sentence_ix = sentence_ix
        self.text = entry["sentences"]

        # Store events.
        if "ner" in entry:
            self.ner = [NER(this_ner, self)
                        for this_ner in entry["ner"]]
            self.ner_dict = {entry.span.span_sent: entry.label for entry in self.ner}
        else:
            self.ner = None
            self.ner_dict = None

        # Store relations.
        if "relations" in entry:
            self.relations = [Relation(this_relation, self) for
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
            self.events = Events(entry["events"], self)
        else:
            self.events = None

    def to_json(self):
        res = {"sentences": self.text}
        if self.ner is not None:
            res["ner"] = [entry.to_json() for entry in self.ner]
        # TODO(dwadden) Don't treat predicted and gold data differently.
        if hasattr(self, "predicted_ner"):
            res["predicted_ner"] = [entry.to_json() for entry in self.predicted_ner]
        if self.relations is not None:
            res["relations"] = [entry.to_json() for entry in self.relations]
        if hasattr(self, "predicted_relations"):
            res["predicted_relations"] = [entry.to_json() for entry in self.predicted_relations]
        if self.events is not None:
            res["events"] = self.events.to_json()
        if hasattr(self, "predicted_events"):
            res["predicted_events"] = self.predicted_events.to_json()

        return res

    def __repr__(self):
        the_text = " ".join(self.text)
        the_lengths = [len(x) for x in self.text]
        tok_ixs = ""
        for i, offset in enumerate(the_lengths):
            true_offset = offset if i < 10 else offset - 1
            tok_ixs += str(i)
            tok_ixs += " " * true_offset

        return the_text + "\n" + tok_ixs

    def __len__(self):
        return len(self.text)


class Span:
    def __init__(self, start, end, sentence, sentence_offsets=False):
        # The `start` and `end` are relative to the document. We convert them to be relative to the
        # sentence.
        self.sentence = sentence
        # Need to store the sentence text to make span objects hashable.
        self.sentence_text = " ".join(sentence.text)
        self.start_sent = start if sentence_offsets else start - sentence.sentence_start
        self.end_sent = end if sentence_offsets else end - sentence.sentence_start

    @property
    def start_doc(self):
        return self.start_sent + self.sentence.sentence_start

    @property
    def end_doc(self):
        return self.end_sent + self.sentence.sentence_start

    @property
    def span_doc(self):
        return (self.start_doc, self.end_doc)

    @property
    def span_sent(self):
        return (self.start_sent, self.end_sent)

    @property
    def text(self):
        return self.sentence.text[self.start_sent:self.end_sent + 1]

    def __repr__(self):
        return str((self.start_sent, self.end_sent, self.text))

    def __eq__(self, other):
        return (self.span_doc == other.span_doc and
                self.span_sent == other.span_sent and
                self.sentence == other.sentence)

    def __hash__(self):
        tup = self.span_sent + (self.sentence_text,)
        return hash(tup)


class Token:
    def __init__(self, ix, sentence, sentence_offsets=False):
        self.sentence = sentence
        self.ix_sent = ix if sentence_offsets else ix - sentence.sentence_start

    @property
    def ix_doc(self):
        return self.ix_sent + self.sentence.sentence_start

    @property
    def text(self):
        return self.sentence.text[self.ix_sent]

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
    def __init__(self, ner, sentence, sentence_offsets=False):
        self.span = Span(ner[0], ner[1], sentence, sentence_offsets)
        self.label = ner[2]

    def __repr__(self):
        return self.span.__repr__() + ": " + self.label

    def __eq__(self, other):
        return (self.span == other.span and
                self.label == other.label)

    def to_json(self):
        return list(self.span.span_doc) + [self.label]


class PredictedNER:
    def __init__(self, ner, sentence, sentence_offsets=False):
        self.span = Span(ner[0], ner[1], sentence, sentence_offsets)
        self.label = ner[2]
        self.raw_score = ner[3]
        self.softmax_score = ner[4]

    def __repr__(self):
        return f"{self.span.__repr__()}: {self.label} with confidence {self.softmax_score}"

    def to_json(self):
        return list(self.span.span_doc) + [self.label, self.raw_score, self.softmax_score]


class Relation:
    def __init__(self, relation, sentence, sentence_offsets=False):
        start1, end1 = relation[0], relation[1]
        start2, end2 = relation[2], relation[3]
        label = relation[4]
        span1 = Span(start1, end1, sentence, sentence_offsets)
        span2 = Span(start2, end2, sentence, sentence_offsets)
        self.pair = (span1, span2)
        self.label = label

    def __repr__(self):
        return self.pair[0].__repr__() + ", " + self.pair[1].__repr__() + ": " + self.label

    def __eq__(self, other):
        return (self.pair == other.pair) and (self.label == other.label)

    def to_json(self):
        return list(self.pair[0].span_doc) + list(self.pair[1].span_doc) + [self.label]


class Event:
    def __init__(self, event, sentence, sentence_offsets=False):
        trig = event[0]
        args = event[1:]
        trigger_token = Token(trig[0], sentence, sentence_offsets)
        self.trigger = Trigger(trigger_token, trig[1])

        self.arguments = []
        for arg in args:
            span = Span(arg[0], arg[1], sentence, sentence_offsets)
            self.arguments.append(Argument(span, arg[2], self.trigger.label))

    def to_json(self):
        trig_json = [self.trigger.token.ix_doc, self.trigger.label]
        arg_json = []
        for arg in self.arguments:
            arg_entry = list(arg.span.span_doc) + [arg.role]
            arg_json.append(arg_entry)
        res = [trig_json] + arg_json
        return res

    def __repr__(self):
        res = "<"
        res += self.trigger.__repr__() + ":\n"
        for arg in self.arguments:
            res += 6 * " " + arg.__repr__() + ";\n"
        res = res[:-2] + ">"
        return res


class Events:
    def __init__(self, events_json, sentence, sentence_offsets=False):
        self.event_list = [Event(this_event, sentence, sentence_offsets)
                           for this_event in events_json]
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

    def to_json(self):
        return [event.to_json() for event in self]

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
    def __init__(self, cluster, cluster_id, sentences, sentence_starts):
        # Make sure the cluster ID is an int.
        if not isinstance(cluster_id, int):
            raise TypeError("Coreference cluster ID's must be ints.")

        n_tokens = sum([len(x) for x in sentences])

        members = []
        for entry in cluster:
            sentence_ix = get_sentence_of_span(entry, sentence_starts, n_tokens)
            sentence = sentences[sentence_ix]
            span = Span(entry[0], entry[1], sentence)
            ners = [x for x in sentence.ner if x.span == span]
            assert len(ners) <= 1
            ner = ners[0] if len(ners) == 1 else None
            to_append = ClusterMember(span, ner, sentence, cluster_id)
            members.append(to_append)

        self.members = members
        self.cluster_id = cluster_id

    def to_json(self):
        return [list(member.span.span_doc) for member in self.members]

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
