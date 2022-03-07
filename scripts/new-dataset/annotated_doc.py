"""
Defines the classes used in brat_to_input.py.

Author: Serena G. Lotreck
"""
from os.path import basename, splitext
import warnings


class AnnotatedDocError(Exception):
    """
    General class for errors arising during document preprocessing.
    """
    pass


class AnnotatedDoc:
    def __init__(self, text, sents, ents, bin_rels, events, equiv_rels,
                 doc_key, dataset, coref, nlp, total_original_ents):
        """
        Provides dual functionality for class construction. If this function is
        used, be sure that the ents, bin_rels, events, and equiv_rels are
        objects of the corresponding classes.
        """
        self.text = text
        self.sents = sents
        self.ents = ents
        self.bin_rels = bin_rels
        self.events = events
        self.equiv_rels = equiv_rels
        self.doc_key = doc_key
        self.dataset = dataset
        self.coref = coref  # True if EquivRels should be treated as corefs
        self.nlp = nlp
        self.dropped_ents = 0
        self.total_original_ents = total_original_ents


    @classmethod
    def parse_ann(cls, txt, ann, nlp, dataset, coref):
        """
        Parses .ann file and creates a new AnnotatedDoc instance.

        parameters:
            txt, str: path to .txt file that corresponds to the .ann file
            ann, str: path to .ann file to parse
            nlp, spacy nlp object: nlp object to use for tokenization
            dataset, str: name of the dataset that will be used in prediction
            coref, bool: whether or not to treat equivalence rels as corefs

        return:
            annotated_doc, instance of AnnotatedDoc for this .ann file
        """
        # Get doc_id
        doc_key = splitext(basename(ann))[0]

        # Get text as one string and tokenized sents
        with open(txt) as myf:
            text = myf.read()

        doc = nlp(text)
        sents = [[tok.text for tok in sent] for sent in doc.sents]

        # Read in the lines from the file, each row is an annotation
        with open(ann) as myf:
            lines = myf.readlines()

        # Drop discontinuous ents by looking for semicolons before second \t
        lines_continuous = []
        for line in lines:
            if line[0] == 'T':
                second_tab = line.rfind('\t')
                if ';' in line[:second_tab]:
                    idx = line[:line.index("\t")]
                    warnings.warn(f'Entity "{line[second_tab:]}" (ID: '
                          f'{idx}) is disjoint, and will be dropped.')
                else:
                    lines_continuous.append(line)
            else:
                lines_continuous.append(line)

        # Split on whitespace to get the separate elements of the annotation
        split_lines = [line.split() for line in lines_continuous]

        # Make class instances for the different annotation types
        ents = []
        bin_rels = []
        events = []
        equiv_rels = []
        total_original_ents = 0
        for line in split_lines:

            # The first character of the first element in the annotation
            # is the annotation type: T = entity, R = relation, E = event,
            # * = equivalence relation
            if line[0][0] == 'T':
                ents.append(Ent(line))
                total_original_ents += 1

            elif line[0][0] == 'R':
                bin_rels.append(BinRel(line))

            elif line[0][0] == 'E':
                events.append(Event(line))

            elif line[0][0] == '*' and coref:
                equiv_rels.append(EquivRel(line))

        annotated_doc = AnnotatedDoc(text, sents, ents, bin_rels, events,
                                     equiv_rels, doc_key, dataset, coref, nlp,
                                     total_original_ents)
        annotated_doc.set_annotation_objects()

        return annotated_doc


    def set_annotation_objects(self):
        """
        For each type of annotation, replace the string IDs with the
        corresponding entity objects, using each class' respective method.
        """
        [bin_rel.set_arg_objects(self.ents) for bin_rel in self.bin_rels]
        [event.set_arg_objects(self.ents) for event in self.events]
        [equiv_rel.set_arg_objects(self.ents) for equiv_rel in self.equiv_rels]


    def format_dygiepp(self):
        """
        Creates a dygiepp-formatted json for the doc, using each class'
        formatting method.
        """
        # Get the token start and end indices for each sentence
        sent_idx_tups = []
        last_end_tok_plus_one = 0
        for sent in self.sents:

            start_tok = last_end_tok_plus_one
            last_end_tok_plus_one = start_tok + len(
                sent)  # End index of sentence is non-inclusive

            sent_idx_tups.append((start_tok, last_end_tok_plus_one))

        # Format data
        ner = Ent.format_ner_dygiepp(self.ents, sent_idx_tups)
        bin_rels = BinRel.format_bin_rels_dygiepp(self.bin_rels, sent_idx_tups)
        if len(self.equiv_rels
               ) > 0 and self.coref:  # Some datasets don't have coreferences
            corefs = EquivRel.format_corefs_dygiepp(self.equiv_rels)
        if len(self.events) > 0:  # Some datasets don't have events
            events = Event.format_events_dygiepp(self.events, sent_idx_tups)

        # Make dict
        res = {
            "doc_key": self.doc_key,
            "dataset": self.dataset,
            "sentences": self.sents,
            "ner": ner,
            "relations": bin_rels
        }

        if len(self.equiv_rels
               ) > 0 and self.coref:  # Some datasets don't have coreferences
            res["clusters"] = corefs
        if len(self.events) > 0:  # Some datasets don't have events
            res["events"] = events

        return res


    def char_to_token(self):
        """
        Does the heavy lifting for converting brat format to dygiepp format.
        Gets the token start and end indices for entities.  Raises a warning
        if no alignment can be found for an entity, as the entity will be
        dropped.

        NOTE: End character indices from brat are non-inclusive, like the
        indexing in python. This is different from DyGIE++'s token indexing,
        where the end indices are inclusive.
        """
        # Tokenize the text with spacy
        tok_text = self.nlp(self.text)

        # Get the alignment for each entity
        ent_list_tokens = []
        for ent in self.ents:

            # Find the start token
            start_tok = [tok for tok in tok_text if tok.idx == ent.char_start]

            if len(start_tok) == 0:

                # If the entity can't be found because there isn't an exact
                # match in the list, warn that it will be dropped
                warnings.warn(f'The entity {ent.text} (ID: {ent.ID}) cannot '
                      'be aligned to the tokenization, and will be dropped.')
                self.dropped_ents += 1

            else:

                # Get token start index
                ent_tok_start = start_tok[0].i

                # Get the number of tokens in ent
                processed_ent = self.nlp(ent.text)
                num_tok = len(processed_ent)
                if num_tok > 1:
                    ent_tok_end = ent_tok_start + num_tok - 1
                else:
                    ent_tok_end = ent_tok_start

                # Double-check that the tokens from the annotation file match up
                # with the tokens in the source text.
                ent_tok_text = [tok.text.lower() for tok in processed_ent]
                doc_tok_text = [tok.text.lower() for i, tok in enumerate(tok_text)
                                if i >= ent_tok_start and i <= ent_tok_end]
                if ent_tok_text != doc_tok_text:
                    msg = ('The annotation file and source document disagree '
                            f'on the tokens for entity {ent.text} (ID: '
                           f'{ent.ID}). This entity will be dropped.')
                    warnings.warn(msg)
                    self.dropped_ents += 1
                    continue

                # Set the token start and end chars
                ent.set_tok_start_end(ent_tok_start, ent_tok_end)

                # Append to list to keep
                ent_list_tokens.append(ent)

        # Set the list of entities that had token matches as ents for doc
        self.ents = ent_list_tokens

        print(f'Completed doc {self.doc_key}. {self.dropped_ents} of '
                f'{self.total_original_ents} entities '
                'were dropped due to tokenization mismatches.')


class Ent:
    def __init__(self, line):
        """
        Does not account for discontinuous annotations, these should have
        been removed before creating entity objects.
        """
        self.ID = line[0]
        self.label = line[1]

        # Since disjoint entities have been dropped, start and end indices will
        # always be at the same indices in the list

        self.char_start = int(line[2])
        self.char_end = int(line[3])
        self.tok_start = None
        self.tok_end = None
        self.text = " ".join(line[4:])

    def set_tok_start_end(self, tok_start, tok_end):
        """
        Set the start_end_tups attribute. To be used to change out character
        indices for token indices.

        parameters:
            start_end_tups, list of tuples: list of start and end token indices

        returns: None
        """
        self.tok_start = tok_start
        self.tok_end = tok_end

    @staticmethod
    def format_ner_dygiepp(ent_list, sent_idx_tups):
        """
        Take a list of start and end tokens for entities and format them for
        dygiepp. Assumes all entities are annotated within sentence boundaries
        and that entity indices have been converted to tokens.

        parameters:
            ent_list, list of Ent obj: list of entities to format
            sent_idx_tups, list of tuple: start, end indices for each sentence.
                End indices are non-inclusive.

        returns:
            ner, list of list: dygiepp formatted ner
        """
        ner = []
        # Go through each sentence to get the entities in that sentence
        for sent_start, sent_end in sent_idx_tups:

            # Check all entities to see if they're in this sentence
            sent_ents = []
            for ent in ent_list:

                if sent_start <= ent.tok_start < sent_end:
                    sent_ents.append([ent.tok_start, ent.tok_end, ent.label])

            ner.append(sent_ents)

        return ner


class BinRel:
    def __init__(self, line):

        self.ID = line[0]
        self.label = line[1]
        self.arg1 = line[2][line[2].index(':') +
                            1:]  # ID of arg is after semicolon
        self.arg2 = line[3][line[3].index(':') + 1:]

    def set_arg_objects(self, arg_list):
        """
        Given a list of Ent objects, replaces the string ID for arg1 and arg2
        taken from the original annotation with the Ent object instance that
        represents that entity.

        parameters:
            arg_list, list of Ent instances: entities from the same .ann file

        returns: None
        """
        for ent in arg_list:

            if ent.ID == self.arg1:
                self.arg1 = ent

            elif ent.ID == self.arg2:
                self.arg2 = ent

    @staticmethod
    def format_bin_rels_dygiepp(rel_list, sent_idx_tups):
        """
        Take a list of relations and format them for dygiepp. Assumes all
        realtions are annotated within sentence boundaries and that entity
        indices have been converted to tokens.

        parameters:
            rel_list, list of BinRel objects: list of relations to format
            sent_idx_tups, list of tuple: start, end indices for each sentence.
                End indices are non-inclusive.

        returns:
            bin_rels, list of list: dygiepp formatted relations
        """
        bin_rels = []
        # Go through each sentence to get the relations in that sentence
        for sent_start, sent_end in sent_idx_tups:

            # Check first entity to see if relation is in this sentence
            sent_rels = []
            for rel in rel_list:
                rel_start = rel.arg1.tok_start
                if sent_start <= rel_start < sent_end:
                    sent_rels.append([
                        rel.arg1.tok_start, rel.arg1.tok_end,
                        rel.arg2.tok_start, rel.arg2.tok_end, rel.label
                    ])

            bin_rels.append(sent_rels)

        return bin_rels


class Event:
    def __init__(self, line):

        self.ID = line[0]
        # ID of arg is after semicolon
        self.trigger = line[1][line[1].index(':') + 1:]
        # Type of trigger is before semicolon
        self.trigger_type = line[1][:line[1].index(':')]
        self.args = []
        for arg in line[2:]:
            arg_ID = arg[arg.index(':') + 1:]
            arg_label = arg[:arg.index(':')]
            self.args.append((arg_ID, arg_label))

    def set_arg_objects(self, arg_list):
        """
        Given a list of entity objects, replaces the string ID for the trigger,
        arg1 and arg2 taken from the original annotation with the Ent object
        instance that represents that entity.

        parameters:
            arg_list, list of Ent instances: entities from the same .ann file

        returns: None
        """
        # Format a dict with arg ID as key, Ent obj as value
        # for more efficient lookup
        ent_dict = {ent.ID: ent for ent in arg_list}

        # Replace trigger
        self.trigger = ent_dict[self.trigger]

        # Replace args
        arg_objs = []
        for arg_ID, arg_label in self.args:

            # Get the arg from the ent list by ID
            arg_obj = ent_dict[arg_ID]

            # Add back to list with object in place of ID
            arg_objs.append(arg_obj)

        self.args = arg_objs

    @staticmethod
    def format_events_dygiepp(event_list, sent_idx_tups):
        """
        Take a list of events and format them for dygiepp. Assumes all
        events are annotated within sentence boundaries and that entity
        indices have been converted to tokens.

        NOTE: In ACE, triggers can only be one token long. As the specified
        format in data.md only includes events with single token triggers,
        this function will use only the first token (with a warning) if there
        are multiple-token triggers in the dataset.

        parameters:
            event_list, list of Event objects: events to format
            sent_idx_tups, list of tuple: start, end indices for each sentence.
                End indices are non-inclusive.

        returns:
            events, list of list: dygiepp formatted events
        """
        events = []
        # Go through each sentence to get the relations in that sentence
        for sent_start, sent_end in sent_idx_tups:

            # Check trigger to see if event is in this sentence and format
            sent_events = []
            for event in event_list:

                # Check if event is in sentence
                trigger_start = event.trigger.tok_start

                if sent_start <= trigger_start < sent_end:

                    formatted_event = []
                    # Format trigger
                    # TODO: Check if triggers can be more than one token for not ACE
                    trigger_end = event.trigger.tok_end
                    if trigger_start != trigger_end:

                        print(f'Warning! Trigger "{event.trigger.text}" (ID: '
                              f'{event.trigger.ID}) has multiple tokens. Only '
                              'the first token will be used.')

                        trigger = [trigger_start, event.trigger_type]
                        formatted_event.append(trigger)

                    else:
                        trigger = [trigger_start, event.trigger_type]
                        formatted_event.append(trigger)

                    # Format args
                    for arg_obj in event.args:

                        arg_start = arg_obj.tok_start
                        arg_end = arg_obj.tok_end
                        arg_label = arg_obj.label

                        arg = [arg_start, arg_end, arg_label]
                        formatted_event.append(arg)

                    sent_events.append(formatted_event)

            events.append(sent_events)

        return events


class EquivRel:
    def __init__(self, line):

        self.label = line[1]
        self.args = line[2:]

    def set_arg_objects(self, arg_list):
        """
        Given a list of entity objects, replaces the string ID for all args
        taken from the original annotation with the Ent object
        instance that represents that entity.

        parameters:
            arg_list, list of Ent instances: entities from the same .ann file

        returns: None
        """
        ent_objs = []
        for ent in arg_list:
            if ent.ID in self.args:
                ent_objs.append(ent)

        self.args = ent_objs

    @staticmethod
    def format_corefs_dygiepp(equiv_rels_list):
        """
        Format coreferences for dygiepp. Assumes that entity indices have been
        converted to tokens. Coreferences can be annotated across sentence
        boundaries.

        parameters:
            equiv_rels_list, list of EquivRel objects: coref clusters to format

        returns:
            corefs, list of list: dygiepp formatted coreference clusters
        """
        corefs = []
        for equiv_rel in equiv_rels_list:
            cluster = [[arg.tok_start, arg.tok_end] for arg in equiv_rel.args]
            corefs.append(cluster)

        return corefs
