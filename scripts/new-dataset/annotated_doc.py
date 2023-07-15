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
                 doc_key, dataset, coref, nlp, dropped_ents, total_original_ents,
                 total_original_rels, total_original_equiv_rels,
                 total_original_events):
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
        self.dropped_ents = dropped_ents
        self.dropped_rels = 0
        self.dropped_equiv_rels = 0
        self.dropped_events = 0
        self.total_original_ents = total_original_ents
        self.total_original_rels = total_original_rels
        self.total_original_equiv_rels = total_original_equiv_rels
        self.total_original_events = total_original_events

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
        discont_dropped = 0
        for line in lines:
            if line[0] == 'T':
                second_tab = line.rfind('\t')
                if ';' in line[:second_tab]:
                    idx = line[:line.index("\t")]
                    discont_dropped += 1
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
        total_original_ents = discont_dropped
        total_original_rels = 0
        total_original_equiv_rels = 0
        total_original_events = 0
        for line in split_lines:

            # The first character of the first element in the annotation
            # is the annotation type: T = entity, R = relation, E = event,
            # * = equivalence relation
            if line[0][0] == 'T':
                ents.append(Ent(line))
                total_original_ents += 1

            elif line[0][0] == 'R':
                bin_rels.append(BinRel(line))
                total_original_rels += 1

            elif line[0][0] == 'E':
                events.append(Event(line))
                total_original_events += 1

            elif line[0][0] == '*' and coref:
                equiv_rels.append(EquivRel(line))
                total_original_equiv_rels += 1

        annotated_doc = AnnotatedDoc(text, sents, ents, bin_rels, events,
                                     equiv_rels, doc_key, dataset, coref, nlp,
                                     discont_dropped, total_original_ents,
                                     total_original_rels, total_original_equiv_rels,
                                     total_original_events)
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
        bin_rels, self.dropped_rels = BinRel.format_bin_rels_dygiepp(
            self.bin_rels, sent_idx_tups)
        print(
            f'Completed relation formatting for {self.doc_key}. {self.dropped_rels} of '
            f'{self.total_original_rels} relations were dropped due to tokenization mismatches.'
        )
        if len(self.equiv_rels
               ) > 0 and self.coref:  # Some datasets don't have coreferences
            corefs, self.dropped_equiv_rels = EquivRel.format_corefs_dygiepp(
                self.equiv_rels)
            print(f'Completed coreference formatting for {self.doc_key}. '
                  f'{self.dropped_equiv_rels} of '
                  f'{self.total_original_equiv_rels} were dropped due to '
                  'tokenization mismatches.')
        if len(self.events) > 0:  # Some datasets don't have events
            events, self.dropped_events = Event.format_events_dygiepp(
                self.events, sent_idx_tups)
            print(f'Completed event formatting for {self.doc_key}. '
                  f'{self.dropped_events} of '
                  f'{self.total_original_events} were dropped due to '
                  'tokenization mismatches.')

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

        res = AnnotatedDoc.quality_check_sent_splits(res)

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
                warnings.warn(
                    f'The entity {ent.text} (ID: {ent.ID}) cannot '
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
                doc_tok_text = [
                    tok.text.lower() for i, tok in enumerate(tok_text)
                    if i >= ent_tok_start and i <= ent_tok_end
                ]
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

        print(
            f'Completed character to token conversion for doc {self.doc_key}. '
            f'{self.dropped_ents} of {self.total_original_ents} entities '
            'were dropped due to tokenization mismatches.')

    @staticmethod
    def quality_check_sent_splits(doc_dict):
        """
        Function to detect and correct incorrect sentence splits in a dygiepp-
        formatted doc dictionary.

        This function relies on the assumption that a cross-sentence entity or
        relation in a dygiepp-formatted doc is a result of an incorrect sentence
        split on the part of the tokenizer, rather than intentional. If a
        cross-sentence entity or relation is found, all sentences between the
        sentences containing the two joined entities or entity parts will be
        combined into one.
        
        Example: BioInfer.d70 is one sentence only, with two relations. However,
        the conversion to jsonl results in the following doc dictionary:
        
        {"doc_key": "BioInfer.d70",
        "dataset": "bioinfer",
        "sentences": [["Aprotinin", "inhibited", "platelet", "aggregation", "induced",
                "by", "thrombin", "(", "0.25", "U.ml-1", ")", "with", "IC50", "200",
                "kIU.ml-1", ",", "and", "inhibited", "the", "rise", "of", "cytosolic",
                "free", "calcium", "concentration", "in", "platelets", "stimulated", "by",
                "thrombin", "(", "0.1", "U.ml-1", ")", "in", "the", "absence", "and", "in",
                "the", "presence", "of", "Ca2", "+", "0.5", "mmol", "."],
            ["L-1", "(","IC50", "117", "and", "50", "kIU.ml-1", ",", "respectively",
                ")", ",", "but", "had", "no", "effect", "on", "the", "amounts", "of",
                "actin", "and", "myosin", "heavy", "chain", "associated", "with",
                "cytoskeletons", "."]],
        "ner": [[[29, 29, "Individual_protein"], [0, 0, "Individual_protein"],
            [6, 6, "Individual_protein"]],
            [[68, 70, "Individual_protein"], [66, 66, "Individual_protein"]]],
        "relations": [[[29, 29, 0, 0, "PPI"], [0, 0, 66, 66, "PPI"]],
            [[68, 70, 0, 0, "PPI"]]]}

        parameters:
            doc_dict, dict: dygiepp-formatted doc

        returns:
            doc_dict_corrected, dict: dict with sentence splits corrected
        """
        # Get the sentence start and end indices
        sent_idxs = []
        for i, sent in enumerate(doc_dict['sentences']):
            if i == 0:
                sent_start = 0
            else:
                sent_start = sent_idxs[i-1][1] + 1
            sent_end = sent_start + len(sent)  - 1
            sent_idxs.append((sent_start, sent_end))

        # For each entity and relation, check if it crosses sentence boundaries
        sents_to_join = []
        for i in range(len(doc_dict['sentences'])):
            
            for ent in doc_dict['ner'][i]:
                e_start = ent[0]
                e_end = ent[1]
                ent_sent_mems = []
                for j, sent in enumerate(sent_idxs):
                    if sent[0] <= e_start <= sent[1]:
                        ent_sent_mems.append(j)
                    if sent[0] <= e_end <= sent[1]:
                        ent_sent_mems.append(j)
                if ent_sent_mems[0] != ent_sent_mems[1]:
                    ent_sent_mems = tuple(sorted(ent_sent_mems))
                    sents_to_join.append(ent_sent_mems)

            for rel in doc_dict['relations'][i]:
                e1_start = rel[0]
                e2_start = rel[2]
                rel_sent_mems = []
                for j, sent in enumerate(sent_idxs):
                    if sent[0] <= e1_start <= sent[1]:
                        rel_sent_mems.append(j)
                    if sent[0] <= e2_start <= sent[1]:
                        rel_sent_mems.append(j)
                if rel_sent_mems[0] != rel_sent_mems[1]:
                    rel_sent_mems = tuple(sorted(rel_sent_mems))
                    sents_to_join.append(rel_sent_mems)
            
        sents_to_join = list(set([tuple(pair) for pair in sents_to_join]))

    
        # Join sentences that need it
        if len(sents_to_join) == 0:
            doc_dict_corrected = doc_dict
            return doc_dict_corrected
        else:
            doc_dict_corrected = {'doc_key': doc_dict['doc_key'], 'dataset': doc_dict['dataset']}
            for key in ['sentences', 'ner', 'relations']:

                # If there are multiples, we need to do some extra processing
                if len(sents_to_join) > 1:

                    # Merge continuous joins
                    sents_to_join = AnnotatedDoc.merge_mult_splits(sents_to_join)

                joined = []
                for i, pair in enumerate(sents_to_join):
                    # Add all sentences before the first to join
                    first_idx = min(pair)
                    if i == 0:
                        add_cand = doc_dict[key][:first_idx]
                        if len(add_cand) > 0:
                            joined.extend(add_cand)
    
                    # Join the group and add
                    last_idx = max(pair)
                    sent_list = doc_dict[key][first_idx:last_idx+1]
                    sents_merged = [tok for sent in sent_list for tok in sent]
                    sents_merged = [sents_merged]
                    joined.extend(sents_merged)
    
                    # If it's the last merge, add the rest
                    if (not i == len(doc_dict[key]) - 1) and (i == len(sents_to_join) - 1):
                        add_cand = doc_dict[key][last_idx + 1:]
                        if len(add_cand) > 0:
                            joined.extend(add_cand)

                    # If it's not and there's a gap before next merge, add those
                    elif (last_idx + 1 != sents_to_join[i+1][0]):
                        add_cand = doc_dict[key][last_idx + 1: sents_to_join[i+1][0]]
                        if len(add_cand) > 0:
                            joined.extend(add_cand)
    
                # Add to new doc
                doc_dict_corrected[key] = joined

            if len(sents_to_join) >= 1:
                print(f'{len(sents_to_join)} sentence joins were performed '
                            f'to fix erroneous sentence splits in doc {doc_dict["doc_key"]}')
    
            return doc_dict_corrected

    @staticmethod
    def merge_mult_splits(sents_to_join):
        """
        Given a list of sentence index pairs, determine if any represent multi-
        joins (a sentence that was split into multiple fragments), and get the
        first and last indices of the continuous split to join.

        parameters:
            sents_to_join, list of tuples: pairs of sentence indices

        returns:
            final_pairings, list of tuples: first and last indices of
                continuous splits
        """
        # First sort by the first index
        srtd = sorted(sents_to_join, key=lambda x: x[0])

        # Do a common-sense check that the end indices don't overlap
        end_overlaps = [True if srtd[i][1] > srtd[i+1][0] else False
                            for i in range(len(srtd) - 1)]
        assert not any(end_overlaps), ('One or more pairs of sentences to join '
                                        'overlaps another')

        # Then get the indices where continuous joins stop
        break_idxs = []
        for i in range(len(srtd)-1):
            if srtd[i][1] != srtd[i+1][0]:
                break_idxs.append(i)
        break_idxs = [-1] + break_idxs 

        # If the only break is at 0, we can just return the list
        if break_idxs == [-1] and len(sents_to_join) == 1:
            final_pairings = srtd
            return final_pairings
        else:
            final_pairings = []

        # Use break indices to get the start and end indices of continuous joins
        for i in range(len(break_idxs)):
            if i == len(break_idxs) - 1:
                cont_join = srtd[break_idxs[i]+1:]
                cont_join = (cont_join[0][0], cont_join[-1][1])
            else:
                cont_join = srtd[break_idxs[i]+1: break_idxs[i+1]+1]
                cont_join = (cont_join[0][0], cont_join[-1][1])

            final_pairings.append(cont_join)

        return final_pairings


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
        represents that entity. Replaces the string with None if no argument is
        found.

        parameters:
            arg_list, list of Ent instances: entities from the same .ann file

        returns: None
        """
        found_arg1 = False
        found_arg2 = False
        for ent in arg_list:

            if ent.ID == self.arg1:
                self.arg1 = ent
                found_arg1 = True

            elif ent.ID == self.arg2:
                self.arg2 = ent
                found_arg2 = True

        # Replace any args that weren't found with None
        if not found_arg1:
            self.arg1 = None
        if not found_arg2:
            self.arg2 = None

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
            dropped_rels, int: number of relations that were dropped due to
                entity token mismatches
        """
        bin_rels = []
        dropped_rels_list = []
        dropped_rel_warnings = []
        num_its = 0

        # Go through each sentence to get the relations in that sentence
        for sent_start, sent_end in sent_idx_tups:

            # Check first entity to see if relation is in this sentence
            sent_rels = []
            for rel in rel_list:
                # Check to see if either entity as dropped because disjoint
                if rel.arg1 is None or rel.arg2 is None:
                    dropped_rel_warnings.append(
                            'One or more of the argument entities for '
                            f'relation {rel.ID} was dropped because it was '
                            'disjoint. This relation will also be dropped as '
                            'a result.')
                    dropped_rels_list.append(rel)
                    continue
                # Check to make sure both entities actually have token starts
                if rel.arg1.tok_start == None or rel.arg2.tok_start == None:
                    dropped_rel_warnings.append(
                        'Either the start or end token for relation '
                        f'{rel.arg1.text} -- {rel.label} -- {rel.arg2.text} '
                        f'(ID: {rel.ID}) was dropped due to tokenization '
                        'mismatches. This relation will also be dropped '
                        'as a result.')
                    dropped_rels_list.append(rel)
                    continue
                rel_start = rel.arg1.tok_start
                if sent_start <= rel_start < sent_end:
                    sent_rels.append([
                        rel.arg1.tok_start, rel.arg1.tok_end,
                        rel.arg2.tok_start, rel.arg2.tok_end, rel.label
                    ])

            bin_rels.append(sent_rels)

        dropped_rels = len(list(set(dropped_rels_list)))

        unique_warnings = list(set(dropped_rel_warnings))
        for wa in unique_warnings:
            warnings.warn(wa)

        return bin_rels, dropped_rels


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
            dropped_ents, int: number of events dropped due to entity token
                mismatches
        """
        events = []
        dropped_events_list = []
        dropped_event_warnings = []
        # Go through each sentence to get the relations in that sentence
        for sent_start, sent_end in sent_idx_tups:

            # Check trigger to see if event is in this sentence and format
            sent_events = []
            for event in event_list:

                # Check to make sure the entities involved in the event all
                # have token starts
                # First, check the trigger
                if event.trigger.tok_start == None or event.trigger.tok_end == None:
                    dropped_event_warnings.append(
                        f'The trigger for event ID: {event.ID} '
                        f'(trigger: {event.trigger.text} was dropped due '
                        'to tokenization mismatches. This event will be '
                        'dropped as a result.')
                    dropped_events_list.append(event)
                    continue
                # Then check all the arguments in the event
                any_missing_arg = False
                for arg_obj in event.args:
                    if arg_obj.tok_start == None or arg_obj.tok_end == None:
                        any_missing_arg = True
                if any_missing_arg:
                    dropped_event_warnings.append(
                        f'One or more arguments for event ID: '
                        f'{event.ID} were dropped due to tokenization mismatches. '
                        'This event will be dropped as a result.')
                    dropped_events_list.append(event)
                    continue

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
        
        dropped_events = len(list(set(dropped_events_list)))

        unique_warnings = list(set(dropped_event_warnings))
        for wa in unique_warnings:
            warnings.warn(wa)

        return events, dropped_events


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
            dropped_equiv_rels, int: number of equivalence relations dropped
                due to entity tokenization mistmatches
        """
        corefs = []
        dropped_equiv_rels = 0
        for equiv_rel in equiv_rels_list:

            # Check that both entities exist
            any_missing_args = False
            for arg in equiv_rel.args:
                if arg.tok_start == None or arg.tok_end == None:
                    any_missing_args = True
            if any_missing_args:
                arg_texts = [arg.text for arg in equiv_rel.args]
                warnings.warn(
                    'One or more arguments in the coreference '
                    f'cluster {equiv_rel.label, arg_texts} was dropped '
                    'Due to entity tokenization mismatches. This '
                    'coreference will also be dropped as a reult.')
                dropped_equiv_rels += 1
                continue

            cluster = [[arg.tok_start, arg.tok_end] for arg in equiv_rel.args]
            corefs.append(cluster)

        return corefs, dropped_equiv_rels
