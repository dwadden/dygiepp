"""
Defines the classes used in brat_to_input.py.

Author: Serena G. Lotreck 
"""
from os.path import basename, splitext
import operator 


class AnnotatedDoc:

    def __init__(self, text, sents, ents, bin_rels, events, equiv_rels, doc_key, 
            dataset):
        """
        Provides dual functionality for class construction. If this funciton is
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

    @classmethod
    def parse_ann(cls, txt, ann, nlp, dataset):
        """
        Parses .ann file and creates a new AnnotatedDoc instance.
        
        parameters:
            txt, str: path to .txt file that corresponds to the .ann file
            ann, str: path to .ann file to parse
            nlp, spacy nlp object: nlp object to use for tokenization
            dataset, str: name of the dataset that will be used in prediction

        return:
            annotated_doc, instance of AnnotatedDoc for this .ann file
        """
        # Get doc_id
        doc_id = splitext(basename(ann))[0] 

        # Get text as one string and tokenized sents 
        with open(txt) as myf:
            text = myf.read()

        doc = nlp(text)
        sents = [[tok.text for tok in sent] for sent in doc.sents]

        # Read in the lines from the file, each row is an annotation
        with open(ann) as myf:
            lines = myf.readlines()

        # Split on whitespace to get the separate elements of the annotation
        split_lines = [line.split() for line in lines]

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
                equiv_rels.append(EquivRel(line))

        annotated_doc = AnnotatedDoc(text, sents, ents, bin_rels, events, 
                equiv_rels, doc_key, dataset)
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
        overlapping annotations). 
        """
        # Get sentences as one tokenized list
        # Because dygiepp token indices are with respect to the doc
        tokenized_doc = [tok for sent in sents for tok in sent]

        # Order the entities by their start indices 
        sorted_ents = sorted(x, key=operator.attrgetter('first_start'))

        # Get alignment for each entity
        last_tok = 0  # Index of the first token of the last entity 
        for ent in sorted_ents:
            
            # Search for the text of the entity 
            try:
                
                ent_tok_tups = []
                for part in ent.start_end_tups:

                    # Get character indices 
                    start_char = part[0]
                    end_char = part[1]

                    # Start search at the start token index of the last entity
                    start_tok = tokenized_doc.index(ent.text, last_tok)
                    last_tok = start_tok # Update counter 

                    # Start search for end tok at the start token
                    words = ent.text.split()
                    end_tok = tokenized_doc.index(words[-1], start_tok)

                    # Add to this entity's list of start and end token indices
                    ent_tok_tups.append((start_tok, end_tok))

                # Update this entity's index list with token indices 
                ent.start_ent_tups = ent_tok_tups

            except ValueError:
                
                # If the entity can't be found because there isn't an exact 
                # match in the list, warn that it will be dropped
                print(f'Warning! The entity {ent.text} (ID: {ent.ID}) cannot '
                        'be aligned to the tokenization, and will be dropped.')    


        def format_dygiepp(self):
            """
            Creates a dygiepp-formatted json for the doc, using each class' 
            formatting method.
            """
            # Get the token start and end indices for each sentence 
            sent_idx_tups = []
            last_end_tok = 0
            for sent in self.sents:
                
                start_tok = last_end_tok_plus_one
                last_end_tok = start_tok + len(sent) # End index of sentence is non-inclusive 

                sent_idx_tups.append((start_tok, last_end_tok))

            # Format data 
            ner = Ent.format_ner_dygiepp(self, sent_idx_tups)
            bin_rels = BinRel.format_bin_rels_dygiepp(self, sent_idx_tups)
            ## TODO: EquivRels?
            if len(self.events) > 0: # Some datasets don't have events
                events = Event.format_events_dygiepp(self, sent_idx_tups)

                # Make dict
                res = {"doc_key": self.doc_key,
                       "dataset": self.dataset,
                       "sentences": self.sents,
                       "ner": ner,
                       "relations": bin_rels,
                       "events": events}
            else:

                # Make dict
                res = {"doc_key": self.doc_key,
                       "dataset": self.dataset,
                       "sentences": self.sents,
                       "ner": ner,
                       "relations": bin_rels}

            return res 


class Ent:

    __init__(self, line):
        """
        Accounts for discontinuous annotations.
        Discontinuous annotation indices are split by semicolon, e.g.
        'T1 label start1 end1;start2 end2 text'
        """
        self.ID = line[0]
        self.label = line[1]
        
        ent_# str = ' '.join(line)
        self.start_end_tups = find_start_end_tups(ent_str, [])
        self.first_start = self.start_end_tups[0][0] # Use to sort ents easily 

        num_start_end = len(self.start_end_tups)
        self.text = line[1+(num_start_end*2):]


    @staticmethod
    def find_start_end_tups(ent_str, start_end_tups):
        """
        Finds the start and end tuples for a string from a brat .ann file 
        representing an entity. Accounts for disjoint annotations that have
        an arbitrary number of start and end tuples.

        parameters:
            ent_str, str: string with character offsets, where first character is 
                the start of the first offset (e.g. no tab or space beginning)
            start_end_tups, list: list of two-tuples, where each tuple is 
                (start, end). On the first call this will be empty.
        
        returns: 
            start_end_tups, list of two-tuples: list of (start, end)
        """
        # Base case 
        if ';' not in ent_str:
            start_offset = ent_str[:ent_str.index(' ')]
            end_offset = ent_str[ent_str.index(' ')+1:ent_str.index('\t')]
            offsets.append((start_offset, end_offset))
            return offsets
        
        # Recursive case 
        else:
            start_offset = ent_str[:ent_str.index(' ')]
            end_offset = ent_str[ent_str.index(' ')+1:ent_str.index(';')]
            offsets.append((start_offset, end_offset))
            return get_offsets(ent_str[ent_str.index(';')+1:], offsets)


    def set_start_end_tups(self, start_end_tups):
        """
        Set the start_end_tups attribute. To be used to change out character
        indices for token indices.

        parameters:
            start_end_tups, list of tuples: list of start and end token indices
        
        returns: None
        """
        self.start_end_tups = start_end_tups
        

    @staticmethod
    def format_ner_dygiepp(annotated_doc, sent_idx_tups):
        """
        Take a list of start and end tokens for entities and format them for 
        dygiepp. Assumes all entitiesa re annotated within sentence boundaries.

        Would also appreciate here, to see if there's a way to do fewer loops!

        parameters:
            annotated_doc, AnnotataedDoc instance: the doc to be formatted 
            sent_idx_tups, list of tuple: start and end indices for each sentence 

        returns:
            ner, list of list: dygiepp formatted ner 
        """
        ner = []
        # Go through each sentence to get the entities belonging to that sentence 
        for sent_start, sent_end in sent_idx_tups:
            
            # Check all entities to see if they're in this sentence 
            sent_ents = []
            for ent in annotated_doc.ents:
                ent_start = ent.start_end_tups[0][0]
                if sent_start <= ent_start < sent_end: # Because the end idx is non-inclusive 
                    ## TODO: check if/how disjoint entities are formatted for dygiepp
                    sent_ents.append([ent_start, ent_end, ent.label])

            ner.append(sent_ents)

        return ner 


class BinRel:

    __init__(self, line):

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


    def format_bin_rels_dygiepp(annotated_doc, sent_idx_tups):
        """
        Take a list of relations and format them for dygiepp. Assumes all 
        realtions are annotated within sentence boundaries.

        parameters:
            annotated_doc, AnnotataedDoc instance: the doc to be formatted 
            sent_idx_tups, list of tuple: start and end indices for each sentence 

        returns:
            bin_rels, list of list: dygiepp formatted relations
        """
        bin_rels = []
        # Go through each sentence to get the relations belonging to that sentence 
        for sent_start, sent_end in sent_idx_tups:
            
            # Check all entities to see if they're in this sentence 
            sent_rels = []
            for rel in annotated_doc.bin_rels:
                rel_start = rel.arg1.start_end_tups[0][0]
                if sent_start <= rel_start < sent_end:
                    ## TODO: check if dygiepp supports disjoint ents, if so, how to format relations?
                    sent_rels.append([rel.arg1.start_end_tups[0][0], 
                                        rel.arg1.start_end_tups[0][1],
                                        rel.arg2.start_end_tups[0][0], 
                                        rel.arg2.start_end_tups[0][2],
                                        rel.label])
                    
            bin_rels.append(sent_ents)

        return bin_rels


class Event:

    __init__(self, line):

        ## TODO: fix this to account for the fact that there can be more than two triggers 
        self.ID = line[0]
        self.trigger = line[1][line[1].index(':')+1:] # ID of arg is after semicolon
        self.trigger_type = line[1][:line[1].index(':')] # Type of trigger is before semicolon
        self.arg1 = line[2][line[2].index(':')+1:]
        self.arg1_role = line[2][:line[2].index(':')] # Role of arg is before semicolon
        self.arg2 = line[3][line[3].index(':')+1:]
        self.arg2_role = line[3][:line[3].index(':')]


    def set_arg_objects(self, arg_list):
        """
        Given a list of entity objects, replaces the string ID for the trigger,
        arg1 and arg2 taken from the original annotation with the Ent object 
        instance that represents that entity.

        parameters:
            arg_list, list of Ent instances: entities from the same .ann file

        returns: None
        """
        for ent in arg_list:

            if ent.ID == self.arg1:
                self.arg1 = ent

            elif ent.ID == self.arg2:
                self.arg2 = ent


    def format_events_dygiepp(annotated_doc, sent_idx_tups):
        """
        Take a list of events and format them for dygiepp. Assumes all 
        events are annotated within sentence boundaries.

        parameters:
            annotated_doc, AnnotataedDoc instance: the doc to be formatted 
            sent_idx_tups, list of tuple: start and end indices for each sentence 

        returns:
            events, list of list: dygiepp formatted events
        """
        ## TODO
        pass 


## TODO: check if you need this, may not 
class EquivRel:
    
    __init__(self, line):

        self.label = line[1]
        self.equivalent_ents = line[2:]


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
            if ent.ID in self.equivalent_ents:
                ent_objs.append(ent)

