"""
Defines the classes used in brat_to_input.py.

Author: Serena G. Lotreck 
"""
from os.path import basename, splitext
import operator 


class AnnotatedDoc:

    def __init__(self, text, sents, ents, bin_rels, events, equiv_rels, doc_key, 
            dataset, coref, nlp):
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
        self.coref = coref # True if EquivRels should be treated as coreference clusters 
        self.nlp = nlp

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

        # Drop discontinuous entities by looking for semicolons before second \t
        lines_continuous = []
        for line in lines:
            if line[0] == 'T':
                second_tab = line.rfind('\t')
                if ';' in line[:second_tab]:
                    idx = line[:line.index("\t")]
                    print(f'Warning! Entity "{line[second_tab:]}" (ID: '
                            f'{idx}) is disjoint, and '
                            'will be dropped.')
                else: lines_continuous.append(line)
            else: lines_continuous.append(line)

        # Split on whitespace to get the separate elements of the annotation
        split_lines = [line.split() for line in lines_continuous]

        # Make class instances for the different annotation types 
        ents = []
        bin_rels = []
        events = []
        equiv_rels = []
        for line in split_lines:
            
            # The first character of the first element in the annotation
            # is the annotation type: T = entity, R = relation, E = event, 
            # * = equivalence relation
            if line[0][0] == 'T':
                ents.append(Ent(line))

            elif line[0][0] == 'R':
                bin_rels.append(BinRel(line))

            elif line[0][0] == 'E':
                events.append(Event(line))

            elif line[0][0] == '*':
                equiv_rels.append(EquivRel(line, coref))

        annotated_doc = AnnotatedDoc(text, sents, ents, bin_rels, events, 
                equiv_rels, doc_key, dataset, coref, nlp)
        annotated_doc.set_annotation_objects()
        
        return annotated_doc
        

    def set_annotation_objects(self):
        """
        For each type of annotation, replace the string IDs with the 
        corresponding entity objects, using each class' respective method.
        """
        self.bin_rels = [bin_rel.set_arg_objects(self.ents) for bin_rel in self.bin_rels]
        self.events = [event.set_arg_objects(self.ents) for event in self.events]
        self.equiv_rels = [equiv_rel.set_arg_objs(self.ents) for equiv_rel in self.equiv_rels]


    def char_to_token(self):
        """
        Calls the static method of the Ent class that does the heavy lifting
        converting character indices to tokens. 
        """
        self.ents = Ent.char_to_token(self.ents, self.sents, self.nlp)


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
            last_end_tok_plus_one = start_tok + len(sent) # End index of sentence is non-inclusive 

            sent_idx_tups.append((start_tok, last_end_tok_plus_one))

        # Format data 
        ner = Ent.format_ner_dygiepp(self.ents, sent_idx_tups)
        bin_rels = BinRel.format_bin_rels_dygiepp(self.bin_rels, sent_idx_tups)
        if len(self.equiv_rels) > 0 and coref: # Some datasets don't have coreferences
            corefs = EquivRel.format_coreferences_dygiepp(self.equiv_rels)
        if len(self.events) > 0: # Some datasets don't have events 
            events = Event.format_events_dygiepp(self.events, sent_idx_tups)

        # Make dict
        res = {"doc_key": self.doc_key,
               "dataset": self.dataset,
               "sentences": self.sents,
               "ner": ner,
               "relations": bin_rels}

        if len(self.equiv_rels) > 0 and coref: # Some datasets don't have coreferences
            res["clusters"] = corefs
        if len(self.events) > 0: # Some datasets don't have events 
            res["events"] = events

        return res 


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

        self.start = int(line[2])
        self.end = int(line[3])
        self.text = " ".join(line[4:])


    def set_start_end(self, start, end):
        """
        Set the start_end_tups attribute. To be used to change out character
        indices for token indices.

        parameters:
            start_end_tups, list of tuples: list of start and end token indices
        
        returns: None
        """
        self.start = start
        self.end = end 
        

    @staticmethod
    def char_to_token(ent_list, sentences, nlp):
        """
        Does the heavy lifting for converting brat format to dygiepp format.
        Replaces the start and end attributes for entities with their corresponding 
        token indices. Raises a warning if no alignment can be found for an entity,
        as the entity will be dropped.
        
        I would appreciate feedback on this alignment approach! 

        My initial thought was to do a cumulative sum of the lengths of the
        tokens in the list in order to convert char indices to token indices.
        However, because there usually aren't spaces before punctuation like 
        periods and commas, but there can be before hyphens, I wasn't sure how 
        to make sure that I didn't make an asusmption about where to insert
        spaces when doing the cumulative sum, because even one mismatch (e.g.
        if I assumed there shouldn't be a space between a hyphen and the 
        last token, but then a text uses a hypen as an en dash) would cause
        all of the rest of the entities in the doc to be thrown away. 

        So the approach I settled on was to order the entities by their 
        character indices, and then go through the entities, using the text 
        attribute that comes from the .ann file to find the corresponding
        token using the index function. To account for the fact that many 
        entities can occur multiple times in an annotation, having the entities
        in start-to-finish order means that I can start searching for the next
        entity at the start index of the previous one (which allows for 
        overlapping annotations). To account for the fact that entities can 
        have multiple tokens, I use the original tokenizer on the entity text
        from the .ann file and search for the first token.
        
        I would also love feedback on where this method is placed; I had originally
        placed it in the AnnotatedDoc class, but while writing unittests realized
        that seemed sort of wrong, since it only operates on the Ent class, so 
        I put it here and changed the one in AnnotatedDoc to just call this one 
        -- but neither approach seems ideal to me. 

        parameters:
            ent_list, list of Ent objects: entities to convert 
            sentences, list of tokens: sentences in the doc 
            nlp, spacy NLP object: tokenizer to use

        returns: ent_list_toks, list of Ent objects with token indices 
        """
        ent_list_toks = []

        # Get sentences as one tokenized list
        # Because dygiepp token indices are with respect to the doc
        tokenized_doc = [tok for sent in sentences for tok in sent]

        # Order the entities by their start indices 
        sorted_ents = sorted(ent_list, key=operator.attrgetter('start'))
        
        # Get alignment for each entity
        last_tok = 0  # Index of the first token of the last entity 
        for ent in sorted_ents:
           
            # Tokenize the entity text 
            ent_tokens_text = [tok.text for tok in nlp(ent.text)]
            
            # Search for the text of the entity 
            try:
               
                # Start seach at the start token index of the last entity 
                start_tok = tokenized_doc.index(ent_tokens_text[0], last_tok)
                last_tok = start_tok

                # Since entities have to be continuous, add len(ent) to get end
                # Subtract 1 because end index is inclusive
                end_tok = start_tok + len(ent_tokens_text) - 1  
                
                # Update this entity's index list with token indices 
                ent.set_start_end(start_tok, end_tok)

                # Add entity to list to keep 
                ent_list_toks.append(ent)

            except ValueError:
        
                # If the entity can't be found because there isn't an exact 
                # match in the list, warn that it will be dropped
                print(f'Warning! The entity {ent.text} (ID: {ent.ID}) cannot '
                        'be aligned to the tokenization, and will be dropped.')    
        
        return ent_list_toks

    @staticmethod
    def format_ner_dygiepp(ent_list, sent_idx_tups):
        """
        Take a list of start and end tokens for entities and format them for 
        dygiepp. Assumes all entities are annotated within sentence boundaries
        and that entity indices have been converted to tokens.

        parameters:
            ent_list, list of Ent obj: list of entities to format
            sent_idx_tups, list of tuple: start and end indices for each sentence 

        returns:
            ner, list of list: dygiepp formatted ner 
        """
        ner = []
        # Go through each sentence to get the entities belonging to that sentence 
        for sent_start, sent_end in sent_idx_tups:
            
            # Check all entities to see if they're in this sentence 
            sent_ents = []
            for ent in ent_list:

                if sent_start <= ent.start < sent_end: # Because the end idx is non-inclusive 
                    sent_ents.append([ent.start, ent.end, ent.label])

            ner.append(sent_ents)

        return ner 


class BinRel:

    def __init__(self, line):

        self.ID = line[0]
        self.label = line[1]
        self.arg1 = line[2][line[2].index(':')+1:] # ID of arg is after semicolon
        self.arg2 = line[3][line[3].index(':')+1:]


    def set_arg_objects(self, arg_list):
        """
        Given a list of entity objects, replaces the string ID for arg1 and arg2
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
            sent_idx_tups, list of tuple: start and end indices for each sentence 

        returns:
            bin_rels, list of list: dygiepp formatted relations
        """
        bin_rels = []
        # Go through each sentence to get the relations belonging to that sentence 
        for sent_start, sent_end in sent_idx_tups:
            
            # Check first entity to see if relation is in this sentence 
            sent_rels = []
            for rel in rel_list:
                rel_start = rel.arg1.start
                if sent_start <= rel_start < sent_end:
                    sent_rels.append([rel.arg1.start, 
                                        rel.arg1.end,
                                        rel.arg2.start, 
                                        rel.arg2.end,
                                        rel.label])
                    
            bin_rels.append(sent_rels)

        return bin_rels


class Event:

    def __init__(self, line):

        self.ID = line[0]
        self.trigger = line[1][line[1].index(':')+1:] # ID of arg is after semicolon
        self.trigger_type = line[1][:line[1].index(':')] # Type of trigger is before semicolon
        
        self.args = [] 
        for arg in line[2:]:
            arg_ID = arg[arg.index(':')+1:]
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
        ent_dict = {ent.ID : ent for ent in arg_list}
        
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
            sent_idx_tups, list of tuple: start and end indices for each sentence 

        returns:
            events, list of list: dygiepp formatted events
        """
        events = []
        # Go through each sentence to get the relations belonging to that sentence 
        for sent_start, sent_end in sent_idx_tups:
            
            # Check trigger to see if event is in this sentence and format
            sent_events = []
            for event in event_list:
                
                # Check if event is in sentence 
                trigger_start = event.trigger.start
                
                if sent_start <= trigger_start < sent_end:
                    
                    formatted_event = []
                    # Format trigger 
                    ## TODO: Check if triggers can be more than one token for not ACE
                    trigger_end = event.trigger.end
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
                        
                        arg_start = arg_obj.start
                        arg_end = arg_obj.end
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
            cluster = [[arg.start, arg.end] for arg in equiv_rel.args]
            corefs.append(cluster)

        return corefs 
