"""
Defines the classes used in brat_to_input.py.

Author: Serena G. Lotreck 
"""
from os.path import basename, splitext

class AnnotatedDoc:

    def __init__(self, text, sents, ents, bin_rels, events, equiv_rels, doc_id):
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
        self.doc_id = doc_id


    @classmethod
    def parse_ann(cls, txt, ann, nlp):
        """
        Parses .ann file and creates a new AnnotatedDoc instance.
        
        parameters:
            txt, str: path to .txt file that corresponds to the .ann file
            ann, str: path to .ann file to parse
            nlp, spacy nlp object: nlp object to use for tokenization

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

        annotated_doc = AnnotatedDoc(text, sents, ents, bin_rels, events, equiv_rels, doc_id)
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
        token indices.
        
        Uses the cumulative sum of the token string lengths plus spaces (except before
        punctuation)
        """
        # Get sentences as one tokenized list
        # Because dygiepp token indices are with respect to the doc
        tokenized_doc = [tok for sent in sents for tok in sent]

        # Get alignment for each entity 
        for ent in self.ents:
            
            cum_char_sum = 0
            for tok in tokenized_doc:
                # Need to figure out how to determine if there should be a space 
                # before/after punctuation in order to keep the two annotation
                # schemes aligned -- adding even one extra space would throw off 
                # the entire thing

                if ent.start == 


        


class Ent:

    __init__(self, line):

        self.ID = line[0]
        self.label = line[1]
        self.start = line[2]
        self.end = line[3]
        self.text = " ".join(line[4:])


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


class Event:

    __init__(self, line):

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

