"""
Convert ACE data to our json format.
"""

import xml.etree.ElementTree as ET
import json
from os import path
import os
import re
import argparse
from dataclasses import dataclass
from typing import List
import spacy
from spacy.symbols import ORTH
import numpy as np


class AceException(Exception):
    pass


class CrossSentenceException(AceException):
    pass


class MultiTokenTrigerException(AceException):
    pass


def in_between(ix, pair):
    assert ix != pair[0] and ix != pair[1]
    return ix > pair[0] and ix < pair[1]

@dataclass
class TokSpan:
    # Note that end chars are inclusive.
    start_char: int
    end_char: int
    text_string: str

    def align(self, sent):
        self.span_doc = get_token_indices(self, sent)
        self.span_sentence = get_token_indices(self, sent.as_doc())
        self.adjusted_span_sentence = get_token_indices(self, sent.as_doc())
        self.adjusted_text_string = str(self.text_string)

    def adjust(self, tok):
        if in_between(tok.i, self.span_sentence):
            assert tok.text == "\n" or tok.text == " "  # Either a newline or an occasional whitespace.
            self.adjusted_text_string = self.adjusted_text_string.replace("\n", " ")
            self.adjusted_span_sentence = (self.adjusted_span_sentence[0],
                                           self.adjusted_span_sentence[1] - 1)
        elif tok.i < self.span_sentence[0]:
            self.adjusted_span_sentence = tuple([x - 1 for x in self.adjusted_span_sentence])

    def adjust_spans_doc(self, entry_start):
        self.adjusted_span_doc = tuple([x + entry_start for x in self.adjusted_span_sentence])


@dataclass
class Entity(TokSpan):
    mention_id: str
    mention_type: str
    flavor: str

    def to_json(self):
        return [*self.adjusted_span_doc, self.mention_type]


@dataclass
class RelationArgument(TokSpan):
    argument_id: str
    relation_role: str


@dataclass
class Relation:
    relation_type: str
    arg1: RelationArgument
    arg2: RelationArgument

    def align(self, sent):
        self.arg1.align(sent)
        self.arg2.align(sent)

    def adjust(self, tok):
        self.arg1.adjust(tok)
        self.arg2.adjust(tok)

    def adjust_spans_doc(self, entry_start):
        self.arg1.adjust_spans_doc(entry_start)
        self.arg2.adjust_spans_doc(entry_start)

    def to_json(self):
        return [*self.arg1.adjusted_span_doc, *self.arg2.adjusted_span_doc, self.relation_type]


@dataclass
class EventTrigger(TokSpan):
    trigger_id: str
    trigger_type: str


@dataclass
class EventArgument(TokSpan):
    argument_id: str
    argument_role: str


@dataclass
class Event:
    trigger: EventTrigger
    arguments: List[EventArgument]

    def align(self, sent):
        self.trigger.align(sent)
        for arg in self.arguments:
            arg.align(sent)

    def adjust(self, tok):
        self.trigger.adjust(tok)
        for arg in self.arguments:
            arg.adjust(tok)

    def adjust_spans_doc(self, entry_start):
        self.trigger.adjust_spans_doc(entry_start)
        for arg in self.arguments:
            arg.adjust_spans_doc(entry_start)

    def to_json(self):
        trigger_span = self.trigger.adjusted_span_doc
        assert trigger_span[0] == trigger_span[1]
        trigger = [[trigger_span[0], self.trigger.trigger_type]]
        args = []
        for arg in self.arguments:
            # Collapse time argument roles following Bishan.
            arg_role = "Time" if "Time" in arg.argument_role else arg.argument_role
            args.append([*arg.adjusted_span_doc, arg_role])
        res = trigger + sorted(args)
        return res


@dataclass
class Entry:
    sent: spacy.tokens.span.Span
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]

    def align(self):
        for entity in self.entities:
            entity.align(self.sent)
        for relation in self.relations:
            relation.align(self.sent)
        for event in self.events:
            event.align(self.sent)

    def remove_whitespace(self):
        final_toks = []
        self.align()
        for tok in self.sent.as_doc():
            if tok.is_space:
                self.adjust(tok)
            else:
                final_toks.append(tok)
        self.final_toks = final_toks

    def adjust(self, tok):
        for entity in self.entities:
            entity.adjust(tok)
        for relation in self.relations:
            relation.adjust(tok)
        for event in self.events:
            event.adjust(tok)

    def adjust_spans_doc(self, entry_start):
        self.adjusted_start = entry_start
        for entity in self.entities:
            entity.adjust_spans_doc(entry_start)
        for relation in self.relations:
            relation.adjust_spans_doc(entry_start)
        for event in self.events:
            event.adjust_spans_doc(entry_start)

    def to_json(self):
        self.entities = sorted(self.entities, key=lambda x: x.span_sentence)
        ner = [entity.to_json() for entity in self.entities]
        ner_flavors = [entity.flavor for entity in self.entities]
        relations = sorted([relation.to_json() for relation in self.relations])
        events = sorted([event.to_json() for event in self.events])
        sentences = [tok.text for tok in self.final_toks]
        return dict(sentences=sentences, ner=ner, relations=relations, events=events,
                    sentence_start=self.adjusted_start, ner_flavor=ner_flavors)

    def is_real(self):
        # If no tokens, make sure it's got no entities or anything.
        n_toks = len(self.final_toks)
        # Get rid of empty sentences
        n_entities = len(self.entities)
        n_relations = len(self.relations)
        n_events = len(self.events)
        if n_toks == 0:
            assert n_entities == n_relations == n_events == 0
            return False
        else:
            return True


class Doc:
    def __init__(self, entries, doc_key):
        self.entries = entries
        self.doc_key = doc_key

    def remove_whitespace(self):
        for entry in self.entries:
            entry.remove_whitespace()
        self.entries = [entry for entry in self.entries if entry.is_real()]

    def adjust_spans_doc(self):
        # Get the token starts of the sentence
        entry_lengths = [len(entry.final_toks) for entry in self.entries]
        entry_starts = np.cumsum(entry_lengths)
        entry_starts = np.roll(entry_starts, 1)
        entry_starts[0] = 0
        for entry, start in zip(self.entries, entry_starts):
            entry.adjust_spans_doc(start)

    def to_json(self):
        self.remove_whitespace()
        self.adjust_spans_doc()
        by_entry = [entry.to_json() for entry in self.entries]
        res = {}
        for field in ["sentences", "ner", "relations", "events", "sentence_start"]:
            res[field] = [entry[field] for entry in by_entry]
        res["doc_key"] = self.doc_key
        return res


def debug_if(cond):
    if cond:
        import ipdb; ipdb.set_trace()


def get_token_indices(entity, sent):
    start_token = [tok for tok in sent if tok.idx == entity.start_char]
    debug_if(len(start_token) != 1)
    start_token = start_token[0]
    end_token = [tok for tok in sent if tok.idx + len(tok) - 1 == entity.end_char]
    debug_if(len(end_token) != 1)
    end_token = end_token[0]
    start_ix = start_token.i
    end_ix = end_token.i
    return start_ix, end_ix


def get_token_of(doc, char):
    'Given a document and a character in the document, get the token that the char lives in.'
    for tok in doc:
        if char >= tok.idx and char < tok.idx + len(tok):
            return doc[tok.i]
    raise Exception('Should not get here.')


# Copied over from Heng Ji's student's code.

class Document:
    def __init__(self, annotation_path, text_path, doc_key, fold, heads_only=True,
                 real_entities_only=True, include_pronouns=False):
        '''
        A base class for ACE xml annotation
        :param annotation_path:
        :param text_path:
        '''
        self._heads_only = heads_only
        self._real_entities_only = real_entities_only
        self._doc_key = doc_key
        self._annotation_path = annotation_path
        self._annotation_xml = ET.parse(self._annotation_path)
        self._text_path = text_path
        self._text = self._load_text(text_path)
        self.doc = self._make_nlp(self._text)
        assert self.doc.text == self._text
        self.entity_list, self.entity_ids = self._populate_entity_list()
        self.entity_lookup = self._populate_entity_lookup()
        if self._real_entities_only:
            self._allowed_flavors = ["entity", "pronoun"] if include_pronouns else ["entity"]
            self.entity_list = [x for x in self.entity_list if x.flavor in self._allowed_flavors]
        else:
            self._allowed_flavors = None
        self.event_list = self._populate_event_list()
        self.relation_list = self._populate_relation_list()
        self._fold = fold

    def _make_nlp(self, text):
        '''
        Add a few special cases to spacy tokenizer so it works with ACe mistakes.
        '''
        # Prevent edge case where there are sentence breaks in bad places
        def custom_seg(doc):
            for index, token in enumerate(doc):
                if self._doc_key == "AFP_ENG_20030417.0307":
                    if token.text == "Ivanov":
                        token.sent_start = False
                if '--' in token.text:
                    doc[index].sent_start = False
                    doc[index + 1].sent_start = False
                if token.text == "things" and doc[index + 1].text == "their":
                    doc[index + 1].sent_start = False
                if (token.text == "Explosions" and
                    token.i < len(doc) and
                    doc[index - 1].text == "." and
                    doc[index - 2].text == "Baghdad"):
                    token.sent_start = True
                # Comma followed by whitespace doesn't end a sentence.
                if token.text == "," and doc[index + 1].is_space:
                    doc[index + 2].sent_start = False
                # "And" only starts a sentence if preceded by period or question mark.
                if token.text in ["and", "but"] and doc[index - 1].text not in [".", "?", "!"]:
                    doc[index].sent_start = False
                if (not ((token.is_punct and token.text not in [",", "_", ";", "...", ":", "(", ")", '"']) or token.is_space)
                    and index < len(doc) - 1):
                    doc[index + 1].sent_start = False
                if "\n" in token.text:
                    if index + 1 < len(doc):
                        next_token = doc[index + 1]
                        if len(token) > 1:
                            next_token.sent_start = True
                        else:
                            next_token.sent_start = False
                if token.text == "-":
                    before = doc[index - 1]
                    after = doc[index + 1]
                    if not (before.is_space or before.is_punct or after.is_space or after.is_punct):
                        after.sent_start = False
            return doc

        nlp = spacy.load('en')
        nlp.add_pipe(custom_seg, before='parser')

        single_tokens = ['sgt.',
                         'sen.',
                         'col.',
                         'brig.',
                         'gen.',
                         'maj.',
                         'sr.',
                         'lt.',
                         'cmdr.',
                         'u.s.',
                         'mr.',
                         'p.o.w.',
                         'u.k.',
                         'u.n.',
                         'ft.',
                         'dr.',
                         'd.c.',
                         'mt.',
                         'st.',
                         'snr.',
                         'rep.',
                         'ms.',
                         'capt.',
                         'sq.',
                         'jr.',
                         'ave.']
        for special_case in single_tokens:
            nlp.tokenizer.add_special_case(special_case, [dict(ORTH=special_case)])
            upped = special_case.upper()
            nlp.tokenizer.add_special_case(upped, [dict(ORTH=upped)])
            capped = special_case.capitalize()
            nlp.tokenizer.add_special_case(capped, [dict(ORTH=capped)])

        doc = nlp(text)
        assert doc.text == text
        return doc

    def _load_text(self, text_path):
        '''
        Load in text and strip out tags.
        '''
        with open(text_path, "r") as f:
            text_data = f.read()

        # Get rid of XML tags.
        remove_tags = re.compile('<.*?>', re.DOTALL)  # Also match expressions with a newline in the middle.
        text_data = remove_tags.sub("", text_data)

        # Fix errors in ACE.
        text_data = text_data.replace("dr. germ. the", "dr. germ, the")
        text_data = text_data.replace("arms inspectors. 300 miles west",
                                      "arms inspectors, 300 miles west")

        if self._doc_key in["APW_ENG_20030327.0376", "APW_ENG_20030519.0367"]:
            text_data = text_data.replace("_", "-")

        return text_data

    def _get_chars(self, start_char, end_char, trigger=False):
        the_text = self.doc.char_span(start_char, end_char + 1)
        start_tok = get_token_of(self.doc, start_char)
        end_tok = get_token_of(self.doc, end_char)
        if trigger and start_tok != end_tok:
            raise MultiTokenTrigerException()
            # # If the trigger is multiple words, get the highest token in the dependency parse.
            # the_root = self.doc[start_tok.i:end_tok.i + 1].root
            # start_char = the_root.idx
            # end_char = start_char + len(the_root) - 1
            # the_text = the_root.text
        elif the_text is None:
            # Otherwise, just take all spans containing the entity.
            start_char = start_tok.idx
            end_char = end_tok.idx + len(end_tok) - 1
            the_text = self.doc.char_span(start_char, end_char + 1)

        return start_char, end_char, the_text

    def _populate_entity_list(self):
        entity_ids = []
        res = []
        xml_root = self._annotation_xml.getroot()
        field_to_find = "head" if self._heads_only else "extent"
        for one_entity in xml_root[0].findall('entity'):
            entity_id = one_entity.attrib["ID"]
            entity_ids.append(entity_id)
            for one_entity_mention in one_entity.findall('entity_mention'):
                mention_id = one_entity_mention.attrib['ID']
                mention_type = one_entity.attrib['TYPE']
                # Others have only looked at the head.
                tentative_start = int(one_entity_mention.find(field_to_find)[0].attrib['START'])
                tentative_end = int(one_entity_mention.find(field_to_find)[0].attrib['END'])

                start_char, end_char, text_string = self._get_chars(tentative_start, tentative_end)

                # Parser chokes on the space.
                if (self._doc_key == "soc.history.war.world-war-ii_20050127.2403" and
                    text_string.text == "lesliemills2002@netscape. net"):
                    continue

                # Keep option to ignore pronouns.
                flavor = "pronoun" if one_entity_mention.attrib["TYPE"] == "PRO" else "entity"

                entry = Entity(start_char, end_char, text_string, mention_id=mention_id,
                               mention_type=mention_type, flavor=flavor)
                res.append(entry)

        # Values. Values don't have heads.
        field_to_find = "extent"
        for one_value in xml_root[0].findall('value'):
            value_id = one_value.attrib["ID"]
            entity_ids.append(value_id)
            for one_value_mention in one_value.findall('value_mention'):
                mention_id = one_value_mention.attrib['ID']
                # In the AAAI 2019 paper, they lump all the values together into one label.
                mention_type = 'VALUE'

                tentative_start = int(one_value_mention.find(field_to_find)[0].attrib['START'])
                tentative_end = int(one_value_mention.find(field_to_find)[0].attrib['END'])
                start_char, end_char, text_string = self._get_chars(tentative_start, tentative_end)

                # Parser chokes on the space.
                if (self._doc_key == "soc.history.war.world-war-ii_20050127.2403" and
                    text_string.text == "lesliemills2002@netscape. net"):
                    continue

                entry = Entity(start_char, end_char, text_string, mention_id=mention_id,
                               mention_type=mention_type, flavor="value")
                res.append(entry)

        # Also timex2. These also don't have heads.
        field_to_find = "extent"
        for one_timex2 in xml_root[0].findall('timex2'):
            timex2_id = one_timex2.attrib["ID"]
            entity_ids.append(timex2_id)
            for one_timex2_mention in one_timex2.findall('timex2_mention'):
                mention_id = one_timex2_mention.attrib['ID']
                mention_type = 'TIMEX2'
                # Others have only looked at the head.
                tentative_start = int(one_timex2_mention.find(field_to_find)[0].attrib['START'])
                tentative_end = int(one_timex2_mention.find(field_to_find)[0].attrib['END'])
                start_char, end_char, text_string = self._get_chars(tentative_start, tentative_end)

                # Crosses a sentence boundary.
                if self._doc_key == "CNN_ENG_20030508_210555.5" and start_char == 1316 and end_char == 1335:
                    continue
                # This is just ridiculous.
                weird_times = set(["BACONSREBELLION_20050127.1017", "MARKBACKER_20041103.1300"])
                if self._doc_key in weird_times and "????" in text_string.text:
                    continue

                entry = Entity(start_char, end_char, text_string, mention_id=mention_id,
                               mention_type=mention_type, flavor="timex2")
                res.append(entry)

        return res, entity_ids

    def _populate_entity_lookup(self):
        return {entry.mention_id: entry for entry in self.entity_list}

    def _populate_event_list(self):
        res = []
        xml_root = self._annotation_xml.getroot()
        for one_event in xml_root[0].findall('event'):
            for one_event_mention in one_event.findall('event_mention'):
                include = True
                trigger_id = one_event_mention.attrib['ID']
                trigger_type = '%s.%s' % (one_event.attrib['TYPE'], one_event.attrib['SUBTYPE'])
                trigger_tag = one_event_mention.find('anchor')
                try:
                    start_char, end_char, text_string = self._get_chars(
                        int(trigger_tag[0].attrib['START']),
                        int(trigger_tag[0].attrib['END']),
                        trigger=True)
                # If we hit a multi-token trigger, skip the event mention.
                except MultiTokenTrigerException:
                    continue
                # Buggy event. Crosses sentence. Skip it.
                if self._doc_key == "APW_ENG_20030308.0314" and start_char == 3263 and end_char == 3270:
                    continue
                if self._doc_key == "soc.history.what-if_20050129.1404" and start_char == 554 and end_char == 556:
                    continue
                event_trigger = EventTrigger(start_char, end_char, text_string, trigger_id,
                                             trigger_type)
                argument_list = []
                for one_event_mention_argument in one_event_mention.findall('event_mention_argument'):
                    argument_id = one_event_mention_argument.attrib['REFID']
                    if self._heads_only:
                        assert argument_id in self.entity_lookup
                        this_entity = self.entity_lookup[argument_id]
                        # If we're only doing real entities and this isn't one, don't append.
                        if self._real_entities_only and this_entity.flavor not in self._allowed_flavors:
                            continue
                        start_char, end_char, text_string = (this_entity.start_char,
                                                             this_entity.end_char,
                                                             this_entity.text_string)
                    else:
                        event_mention_argument_tag = one_event_mention_argument.find('extent')
                        relation_mention_argument_tag = one_event_mention_argument.find('extent')
                        start_char, end_char, text_string = self._get_chars(
                            int(event_mention_argument_tag[0].attrib['START']),
                            int(event_mention_argument_tag[0].attrib['END']))

                    # Check that we've seen the entity. If it's a value or timex, just skip it as an
                    # argument.
                    entity_id = "-".join(argument_id.split("-")[:-1])
                    assert entity_id in self.entity_ids

                    argument_role = one_event_mention_argument.attrib['ROLE']
                    to_append = EventArgument(start_char, end_char, text_string, argument_id,
                                              argument_role)
                    argument_list.append(to_append)
                if include:
                    res.append(Event(event_trigger, argument_list))
        return res

    def _populate_relation_list(self):
        res = []
        xml_root = self._annotation_xml.getroot()
        for one_relation in xml_root[0].findall('relation'):
            for one_relation_mention in one_relation.findall('relation_mention'):
                include = True
                relation_type = '%s.%s' % (one_relation.attrib['TYPE'], one_relation.attrib['SUBTYPE'])
                argument_dict = {}
                for one_relation_mention_argument in one_relation_mention.findall("relation_mention_argument"):
                    argument_id = one_relation_mention_argument.attrib['REFID']
                    # If doing heads only, get the span by looking up the entity and getting its span.
                    if self._heads_only:
                        assert argument_id in self.entity_lookup
                        this_entity = self.entity_lookup[argument_id]
                        start_char, end_char, text_string = (this_entity.start_char,
                                                             this_entity.end_char,
                                                             this_entity.text_string)
                    else:
                        relation_mention_argument_tag = one_relation_mention_argument.find('extent')
                        start_char, end_char, text_string = self._get_chars(
                            int(relation_mention_argument_tag[0].attrib['START']),
                            int(relation_mention_argument_tag[0].attrib['END']))

                    # Check that we've seen the entity. If it's a value or timex, skip the event.
                    entity_id = "-".join(argument_id.split("-")[:-1])
                    assert entity_id in self.entity_ids

                    relation_role = one_relation_mention_argument.attrib['ROLE']
                    this_argument = RelationArgument(
                        start_char, end_char, text_string, argument_id, relation_role)

                    # Skip if not a real entity and we're only keeping real entities.
                    if self._heads_only and self._real_entities_only:
                        this_entity = self.entity_lookup[this_argument.argument_id]
                        if this_entity.flavor not in self._allowed_flavors:
                            include = False

                    if this_argument.relation_role == "Arg-1":
                        argument_dict["arg1"] = this_argument
                    elif this_argument.relation_role == "Arg-2":
                        # This is a mis-annotated relation. Ignore it.
                        if (self._doc_key == 'CNN_ENG_20030430_093016.0' and
                            text_string.text == "the school in an\nunderprivileged rural area"):
                            include = False
                        if (self._doc_key == "CNN_ENG_20030430_093016.0" and
                            start_char == 3091 and end_char == 3096):
                            include = False
                        # Crosses a sentence boundary.
                        if (self._doc_key == "rec.travel.cruises_20050222.0313" and
                            start_char == 1435 and end_char == 1442):
                            include = False
                        if (self._doc_key == "rec.travel.cruises_20050222.0313" and
                            start_char == 1456 and end_char == 1458):
                            include = False

                        argument_dict["arg2"] = this_argument
                    else:
                        include = False
                if include:
                    relation = Relation(relation_type, argument_dict["arg1"], argument_dict["arg2"])
                    # There are some examples where the identical relation mention shows up twice,
                    # for instance "young men and women in this country" in
                    # CNN_CF_20030304.1900.04.apf.xml. When this occurs, ignore it.
                    if relation in res:
                        continue
                    else:
                        res.append(relation)
        return res

    @staticmethod
    def _check_in_range(span, sent):
        # The end character inequality must be string. since end character for spans are inclusive
        # and end characters for sentences are exclusive.
        # Raise an exception if the span crosses a sentence boundary.
        if span.start_char >= sent.start_char and span.end_char < sent.end_char:
            return True
        if span.end_char <= sent.start_char:
            return False
        if span.start_char >= sent.end_char:
            return False
        else:
            raise CrossSentenceException

    def _sentence_get_ner(self, sent):
        entities = []
        to_remove = []  # Only relevant for full extents.
        for entity in self.entity_list:
            try:
                in_range = self._check_in_range(entity, sent)
            # If the entity crosses a sentence boundary
            except CrossSentenceException as e:
                # This shouldn't happen if we're only using entity heads; raise an exception.
                if self._heads_only:
                    raise e
                # With full extents this may happen; notify user and skip this example.
                else:
                    # Add to list of entities that will be removed.
                    to_remove.append(entity)
                    msg = f'Entity "{entity.text_string}" crosses sentence boundary. Skipping.'
                    print(msg)
                    continue
            if in_range:
                debug_if(entity in self._seen_so_far['entity'])
                self._seen_so_far["entity"].append(entity)
                entities.append(entity)
        # If doing full entity extents, remove entities that crossed sentence boundaries.
        for failure in to_remove:
            self.entity_list.remove(failure)

        return entities

    def _sentence_get_relations(self, sent):
        def in_range(candidate):
            each_one = [self._check_in_range(entry, sent) for entry in [candidate.arg1, candidate.arg2]]
            if all(each_one):
                debug_if(candidate in self._seen_so_far['relation'])
                return True
            if all([not entry for entry in each_one]):
                return False
            else:
                import ipdb; ipdb.set_trace()

        relations = []
        for relation in self.relation_list:
            # This is an annotation mistake and crosses sentence boundaries. Just ignore it.
            if in_range(relation):
                self._seen_so_far["relation"].append(relation)
                relations.append(relation)
        return relations

    def _sentence_get_events(self, sent):
        def in_range(candidate):
            each_one = ([self._check_in_range(candidate.trigger, sent)] +
                        [self._check_in_range(entry, sent) for entry in candidate.arguments])
            if all(each_one):
                debug_if(candidate in self._seen_so_far['event'])
                return True
            if all([not entry for entry in each_one]):
                return False
            else:
                import ipdb; ipdb.set_trace()

        events = []
        for event in self.event_list:
            # Event that crosses sentence.
            if in_range(event):
                self._seen_so_far["event"].append(event)
                trigger_span = get_token_indices(event.trigger, sent)
                debug_if(trigger_span[0] != trigger_span[1])
                events.append(event)
        return events

    def _get_entry(self, sent):
        toks = [tok for tok in sent]
        ner = self._sentence_get_ner(sent)
        rel = self._sentence_get_relations(sent)
        events = self._sentence_get_events(sent)
        return Entry(sent=sent, entities=ner, relations=rel, events=events)

    def _check_all_seen(self):
        assert len(self._seen_so_far["entity"]) == len(self.entity_list)
        assert len(self._seen_so_far["relation"]) == len(self.relation_list)
        assert len(self._seen_so_far["event"]) == len(self.event_list)

    def to_json(self):
        self._seen_so_far = dict(entity=[], relation=[], event=[])
        entries = [self._get_entry(sent) for sent in self.doc.sents]
        doc = Doc(entries, self._doc_key)
        self._check_all_seen()
        js = doc.to_json()
        return js


####################

# Main function.


def one_fold(fold, output_dir, heads_only=True, real_entities_only=True, include_pronouns=False):
    doc_path = "./data/ace-event/raw-data"
    split_path = "./scripts/data/ace-event/event-split"

    doc_keys = []
    with open(path.join(split_path, fold + ".filelist")) as f:
        for line in f:
            doc_keys.append(line.strip())

    with open(path.join(output_dir, fold + ".json"), "w") as g:
        for doc_key in doc_keys:
            annotation_path = path.join(doc_path, doc_key + ".apf.xml")
            text_path = path.join(doc_path, doc_key + ".sgm")
            document = Document(annotation_path, text_path, doc_key, fold, heads_only,
                                real_entities_only, include_pronouns)
            js = document.to_json()
            g.write(json.dumps(js, default=int) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ACE event data.")
    parser.add_argument("output_name", help="Name for output directory.")
    parser.add_argument("--use_span_extent", action="store_true",
                        help="Use full extent of entity mentions instead of just heads.")
    parser.add_argument("--include_times_and_values", action="store_true",
                        help="Treat times and values as entities and include them as event arguments.")
    parser.add_argument("--include_pronouns", action="store_true",
                        help="Include pronouns as entities and include them as event arguments.")
    args = parser.parse_args()

    output_dir = f"./data/ace-event/processed-data/{args.output_name}/json"
    os.makedirs(output_dir, exist_ok=True)

    for fold in ["train", "dev", "test"]:
        msg = f"Parsing {fold} set."
        print(msg)
        one_fold(fold,
                 output_dir,
                 heads_only=(not args.use_span_extent),
                 real_entities_only=(not args.include_times_and_values),
                 include_pronouns=args.include_pronouns)


if __name__ == "__main__":
    main()
