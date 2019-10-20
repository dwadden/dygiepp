# -*- coding: utf-8 -*-
"""
23 Jan 2017
To convert GENIA XML annotation file into line-by-line format easier to be processed
The output file will be in the following format:
    For tokenized files (.tok):
        The first line is the tokenized text, separated by spaces
        The second line is the POS tags (post-processed, oversplit tokens have * as POS)
        The third line is the list of annotations, in token offset
        The fourth line is blank
    For non-tokenized files (.span):
        The first line is the original text
        The second line is the list of annotations, in character offset
        The third line is blank
For both types of output files, the annotation is in the following format separated by space:
    First token is the list of indices in the format <start>,<end>(+<start>,<end>)*
    Second token is the type of the entity (e.g., G#protein_molecule)
"""

# Import statements
from bs4 import BeautifulSoup as BS
import sys
import os
from os import path
import re
import math
import pandas as pd

class Token(object):
    def __init__(self, text, orig_text, start, end, after, before, postag, orig_postag):
        '''Token object to faithfully represent a token

        To be represented faithfully, a token needs to hold:
        - text: The text it is covering, might be normalized
        - orig_text: The original text it is covering, found in the original text
        - start: The start index in the sentence it appears in
        - end: The end index in the sentence it appears in
        - after: The string that appear after this token, but before the next token
        - before: The string that appear before this token, but after the previous token
        - postag: The POS tag of this token, might be adjusted due to oversplitting
        - orig_postag: The POS tag of the original token this token comes from
        Inspired by CoreLabel in Stanford CoreNLP
        '''
        self.text = text
        self.orig_text = orig_text
        self.start = start
        self.end = end
        self.after = after
        self.before = before
        self.postag = postag
        self.orig_postag = orig_postag

class Span(object):
    def __init__(self, start, end):
        '''Span object represents any span with start and end indices'''
        self.start = start
        self.end = end

    def get_text(self, text):
        return text[self.start:self.end]

    def contains(self, span2):
        return self.start <= span2.start and self.end >= span2.end

    def overlaps(self, span2):
        if ((self.start >= span2.start and self.start < span2.end) or
                (span2.start >= self.start and span2.start < self.end)):
            return True
        return False

    def equals(self, span2):
        return self.start == span2.start and self.end == span2.end

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '{},{}'.format(self.start, self.end)

class Annotation(object):
    def __init__(self, spans, label, text=None, parent=None):
        '''Annotation object defines an annotation by a list of spans and its label.

        Optionally, this object can hold the containing text, so that the text of this annotation can be recovered by calling get_text method.

        If this annotation is discontiguous (more than one spans), the parent specifies the annotation that contains all the discontiguous entities in the same coordinated expression
        '''
        self.spans = spans
        self.label = label
        self.text = text
        self.parent = parent

    def get_text(self):
        return ' ... '.join(span.get_text(self.text) for span in self.spans)

    def overlaps(self, ann2):
        for span in self.spans:
            for span2 in ann2.spans:
                if span.overlaps(span2):
                    return True
        return False

    def contains(self, ann2):
        for span2 in ann2.spans:
            this_span_is_contained = False
            for span in self.spans:
                if span.contains(span2):
                    this_span_is_contained = True
                    break
            if not this_span_is_contained:
                return False
        return True

    def equals(self, ann2):
        if ann2 is None:
            return False
        for span, span2 in zip(self.spans, ann2.spans):
            if not span.equals(span2):
                return False
        if self.label != ann2.label:
            return False
        return True

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '{} {}'.format('+'.join(str(span) for span in self.spans), self.label)

class Sentence(object):
    def __init__(self, sentence_xml):
        self.text = sentence_xml.get_text()
        self.tokens, self.orig_tokens = self.get_tokens(sentence_xml)
        self.span_annotations, self.token_annotations = self.get_annotations(sentence_xml)

    def get_tokens(self, sentence_xml):
        '''Returns the list of tokens from a sentence

        This method oversplits the tokens, so that, as much as possible, all entities can be composed by joining the tokens (so hopefully no entity that spans only part of a token)
        '''
        text = self.text
        tokens = []
        orig_tokens = []
        idx = 0
        orig_idx = 0
        for orig_token in sentence_xml.find_all('w'):
            token_txt = orig_token.get_text()
            postag = orig_token['c']
            token_orig_idx = text.find(token_txt, orig_idx)
            before = text[orig_idx:token_orig_idx]
            if before.strip() != '':
                print('Missing <w> element for: {} ({})'.format(before, text))
            if len(orig_tokens) >= 1:
                orig_tokens[-1].after = before
            orig_tokens.append(Token(token_txt, token_txt, token_orig_idx, token_orig_idx+len(token_txt), '', before, postag, postag))
            orig_idx = token_orig_idx + len(token_txt)

            # Do oversplitting of the tokens, so that instances like the token "IL-2-mediated"
            # as tokenized originally where only "IL-2" is an entity, can be represented as a
            # list of tokens instead of having the entity covering only part of a token
            for token in re.split('([-/,.+])', token_txt):
                if token is None or len(token) == 0:
                    continue
                token_idx = text.find(token, idx)
                before = text[idx:token_idx]
                if len(tokens) >= 1:
                    tokens[-1].after = before
                # The tokens resulting from an oversplitting with be given the POS tag '*'
                # Only the last token will retain the original POS tag
                tokens.append(Token(token, token, token_idx, token_idx+len(token), '', before, '*', postag))
                idx = token_idx + len(token)
            tokens[-1].postag = tokens[-1].orig_postag # This makes the last token retains the original POS tag
            tok_idx = len(tokens)-2
            if postag != '*':
                while tok_idx >= 0 and tokens[tok_idx].postag == '*':
                    tokens[tok_idx].orig_postag = postag
                    tok_idx -= 1
        return tokens, orig_tokens

    def get_annotations(self, sentence_xml):
        '''Extracts annotations from a sentence annotation'''
        text = self.text
        tokens = self.tokens
        span_annotations = []
        Sentence.process_annotation(sentence_xml, text, span_annotations, 0)
        token_annotations = []
        # Converts the character-based annotations into token-based annotations
        for annotation in span_annotations:
            span_tokens = []
            for span in annotation.spans:
                span_tokens.append(Sentence.span_to_token(span, tokens))
            token_annotation = Annotation(span_tokens, annotation.label, annotation.text, annotation.parent)
            token_annotations.append(token_annotation)
        return span_annotations, token_annotations

    @staticmethod
    def span_to_token(span, tokens):
        '''Returns the list of tokens that covers the given list of character spans'''
        start = -1
        end = -1
        for idx, token in enumerate(tokens):
            if span.start < token.end and start == -1:
                start = idx
            if token.end <= span.end:
                end = idx + 1
        return Span(start, end)

    @staticmethod
    def normalize_lex(lex):
        return lex.replace('-_','-').replace('_-','-').replace('__','_').replace('*_','*').replace('\\*','*').strip('_')

    @staticmethod
    def process_annotation(parent_annotation, text, span_annotations, idx):
        '''The method that processes the children of a BeautifulSoup tag
        '''
        for annotation in parent_annotation.find_all('cons', recursive=False):
            ann_txt = annotation.get_text()
            ann_idx = text.find(ann_txt, idx)
            if ann_idx == -1:
                raise('1:Cannot find {} in {} at {} ({})'.format(annotation.get_text(), text[idx:], idx, text))
            Sentence.process_annotation(annotation, text, span_annotations, ann_idx)
            try:
                ann_lex = annotation['lex']
            except:
                ann_lex = annotation.get_text().replace(' ','_')
            ann_lex = Sentence.normalize_lex(ann_lex)
            try:
                ann_sem = annotation['sem']
            except:
                # No sem means this is part of discontiguous entity, should have been handled by the discontiguous entity handler below when it processes the parent
                idx = ann_idx + len(ann_txt)
                continue
            if not ann_sem.startswith('('):
                # This is a contiguous entity
                # Just add it into the list of annotations
                span_annotations.append(Annotation([Span(ann_idx, ann_idx+len(ann_txt))], ann_sem, text))
            # Find all possible constituents of the discontiguous entities
            sub_cons = annotation.find_all('cons', recursive=True)
            if len(sub_cons) > 1 and (ann_sem.startswith('(') or ann_lex.startswith('*') or ann_lex.endswith('*')):
                # This contains a discontiguous entity
                # We need to find the spans of each discontiguous entity
                combined_ann = Annotation([Span(ann_idx, ann_idx+len(ann_txt))], ann_sem, text)
                sub_anns = Sentence.parse_lex(ann_lex, ann_sem)
                sub_cons_ann = []
                # Find the character span of each constituent
                for sub_con in sub_cons:
                    sub_con_txt = sub_con.get_text()
                    sub_con_idx = text.find(sub_con_txt, idx)
                    if sub_con['lex'].startswith('(') or '*' not in sub_con['lex']:
                        # This is contiguous entity, should have been handled by case 1 above
                        continue
                    if sub_con_idx == -1:
                        # This means a constituent cannot be found in its parent constituent, a bug in this script
                        print(sub_cons_ann)
                        raise Exception('2:Cannot find {} in {} at {} ({})'.format(sub_con_txt, text[idx:], idx, text))
                    sub_cons_ann.append((Sentence.normalize_lex(sub_con['lex']), Span(sub_con_idx, sub_con_idx+len(sub_con_txt))))
                    idx = sub_con_idx + len(sub_con_txt)
                # Map each entity to its character span(s)
                for sub_lex, sub_sem in sub_anns:
                    spans = Sentence.find_spans(sub_lex, text, sub_cons_ann)
                    span_annotations.append(Annotation(spans, sub_sem, text, combined_ann))
            idx = ann_idx + len(ann_txt)

    @staticmethod
    def parse_lex(lex, sem):
        result = []
        for sub_lex, sub_sem in zip(Sentence.split_lex(lex), Sentence.split_lex(sem)):
            if '#' in sub_sem:
                if sub_lex == 'amino-terminal_(729-766)_region':
                    # Special case, since the text is:
                    # "Deletions of a relatively short amino- (729-766) or carboxy- terminal (940-984) region"
                    #
                    sub_lex = 'amino-(729-766)_terminal_region'
                result.append((Sentence.normalize_lex(sub_lex), sub_sem))
        return result

    @staticmethod
    def split_lex(lex, idx=None):
        '''Parses a lex attribute (might be nested) into a list of basic lex form (i.e., no nested lex)
        '''
        if idx is None:
            idx = [0]
        result = []
        if idx[0] == len(lex):
            return result
        if lex[idx[0]] == '(' and re.match('^\\((AND|OR|BUT_NOT|AS_WELL_AS|VERSUS|TO|NOT_ONLY_BUT_ALSO|NEITHER_NOR|THAN) .+', lex):
            idx[0] = lex.find(' ', idx[0])
            while idx[0] < len(lex) and lex[idx[0]] == ' ':
                idx[0] += 1
                result.extend(Sentence.split_lex(lex, idx))
        else:
            open_bracket_count = 0
            end_of_lex = -1
            for pos, char in enumerate(lex[idx[0]:]):
                if char == '(':
                    open_bracket_count += 1
                elif char == ')':
                    if open_bracket_count > 0:
                        open_bracket_count -= 1
                    else:
                        end_of_lex = pos+idx[0]
                        break
                elif char == ' ':
                    end_of_lex = pos+idx[0]
                    break
            if end_of_lex == -1:
                end_of_lex = len(lex)
            result.append(lex[idx[0]:end_of_lex])
            idx[0] = end_of_lex
        return result

    @staticmethod
    def find_spans(lex, text, cons):
        '''Given an entity from the lex attribute of cons, and the list of possible constituents (with their character spans), return the list of character spans that forms the entity

        The search of the components of the discontiguous entities depends on the lex of the original discontiguous entity (e.g., in (AND class_I_interferon class_II_interferon)) by trying to match the string with the lex of the constituents found inside that tags (e.g., "class*", "*I*", "*II*", "*interferon" for the above case).

        Various checks have been in place to ensure that the entire string is found, while allowing some minor differences.
        '''
        spans = []
        lex_idx = 0
        prev_lex_idx = -1
        lex = lex.lower().strip('*')
        for con_lex, con_span in cons:
            con_lex = con_lex.strip('*').lower()
            con_lex_idx = lex.find(con_lex, lex_idx)
            if con_lex_idx - lex_idx >= 2:
                # Ensure that we don't skip over too many characters
                con_lex_idx = -1
            if con_lex_idx - lex_idx == 1:
                # Skipping one character might be permissible, given that it's not an important character
                if lex[lex_idx] not in ' -_/*':
                    con_lex_idx = -1
            if con_lex_idx == -1:
                # We didn't find this constituent in the parent string
                # Normally we would just skip this and continue checking the next constituent,
                # However, in some cases, a constituent is a prefix of the next constituent.
                # In that case, we might need to back-off the previous match, and try the longer constituent.
                # For example, when trying to match "class_II_interferon", we might match "*I*" to the first "I" of "class_II_interferon", which is incorrect, as it should be matched with "*II*" which comes after "*I*". So the following is the backing-off mechanism to try to match that.
                # Since this theoretically not 100% accurate, each back-off action is logged, and we need to check whether the back-off was correct.
                # For GENIA dataset, there are 53 cases of backing-off, and all of them have been verified correct
                con_lex_idx = lex.find(con_lex, prev_lex_idx)
                if con_lex_idx != -1 and con_lex_idx < lex_idx and len(con_lex) > spans[-1].end-spans[-1].start:
                    print('Found {} from backing off from {} for {}, please check ({}) {}'.format(con_lex, spans[-1].get_text(text), lex, text, cons))
                    del(spans[len(spans)-1])
                    spans.append(Span(con_span.start, con_span.end))
                    prev_lex_idx = lex_idx
                    lex_idx = con_lex_idx + len(con_lex)
                    if con_lex.endswith('-'):
                        lex_idx -= 1
                else:
                    continue
            else:
                spans.append(Span(con_span.start, con_span.end))
                prev_lex_idx = lex_idx
                lex_idx = con_lex_idx + len(con_lex)
                if con_lex.endswith('-'):
                    lex_idx -= 1
        diff = abs(lex_idx-len(lex.rstrip('*')))
        if diff >= 1:
            # To check whether the entity is completed
            print('Cons: {}'.format(cons))
            if diff == 1:
                print('WARNING: differ by one: "{}", found: "{}"'.format(lex, lex[:lex_idx]))
            else:
                print('\n===\nCannot find complete mention of "{}" in "{}", found only "{}"\n===\n'.format(lex, text, lex[:lex_idx]))
        for idx in range(len(spans)-1, 0, -1):
            if spans[idx].start == spans[idx-1].end or text[spans[idx-1].end:spans[idx].start] == ' ':
                spans[idx-1].end = spans[idx].end
                del(spans[idx])
        return spans


class Article(object):
    def __init__(self, article_xml):
        self.text = article_xml.get_text()
        self.sentences = self.get_sentences(article_xml)
        self.doc_key = self.get_doc_key(article_xml)

    @staticmethod
    def get_sentences(article_xml):
        sentences = article_xml.find_all('sentence')
        sentences_obj = []
        for sentence in sentences:
            sentences_obj.append(Sentence(sentence))
        return sentences_obj

    @staticmethod
    def get_doc_key(article_xml):
        elem = article_xml.find_all('bibliomisc')
        if not len(elem) == 1:
            raise Exception('Wrong number of document IDs.')
        return elem[0].text.replace('MEDLINE:', '')


def split_train_dev_test(sentences, train_pct=0.8, dev_pct=0.1, test_pct=0.1):
    count = len(sentences)
    train_count = int(train_pct*count)
    dev_count = int(math.ceil(dev_pct*count))
    test_count = int(math.ceil(test_pct*count))
    train_count -= train_count + dev_count + test_count - count
    start_test_idx = train_count+dev_count
    return sentences[:train_count], sentences[train_count:start_test_idx], sentences[start_test_idx:]

def filter_annotations(anns, remove_disc, remove_over, use_five_types):
    result = []
    for ann in anns:
        ann = Annotation(ann.spans[:], ann.label, ann.text, ann.parent)
        if use_five_types:
            if not re.match('G#(DNA|RNA|cell_line|cell_type|protein).*', ann.label):
                continue
            if ann.label not in ['G#cell_line', 'G#cell_type']:
                ann.label = ann.label[:ann.label.find('_')]
        if remove_disc and len(ann.spans) > 1:
            ann.spans = [Span(ann.spans[0].start, ann.spans[-1].end)]
        if remove_over or (remove_disc and ann.parent is not None):
            need_to_be_removed = False
            for idx in reversed(range(len(result))):
                ann2 = result[idx]
                if not remove_over and remove_disc and ann.parent is not None and ann.parent != ann2.parent:
                    continue
                if ann2.overlaps(ann):
                    if ann2.contains(ann):
                        need_to_be_removed = True
                    elif ann.contains(ann2):
                        del(result[idx])
                    else:
                        # Neither is contained within the other, not nested! Remove one arbitrarily, easier to remove the latter
                        need_to_be_removed = True
            if need_to_be_removed:
                continue
        result.append(ann)
    return result

def main():
    if len(sys.argv) < 3:
        print('Usage: python {} <path_to_GENIA_POS_Corpus> <output_dir> (1|0:output original POS tag instead of *)'.format(sys.argv[0]))
        sys.exit(0)
    xml_path = sys.argv[1]
    output_dir = sys.argv[2]
    use_orig_postag = True
    if len(sys.argv) >= 4:
        if sys.argv[3] == '1':
            use_orig_postag = True
        else:
            use_orig_postag = False
    with open(xml_path, 'r') as infile:
        soup = BS(infile.read(), 'lxml')

    articles = soup.find_all('article')
    articles_obj = []

    for article in articles:
        articles_obj.append(Article(article))

    doc_keys = pd.Series([entry.doc_key for entry in articles_obj])
    doc_keys.name = "doc_order"
    doc_keys.to_csv(path.join(path.dirname(output_dir), "doc_order.csv"), index=False, header=False)

    for tokenized in [False, True]:
        if tokenized:
            tokenized_str = '.tok'
        else:
            tokenized_str = '.span'
        for level in ['all', 'no_disc', 'no_disc_no_over']:
            if level == 'no_disc':
                remove_disc = True
                remove_over = False
            elif level == 'no_disc_no_over':
                remove_disc = True
                remove_over = True
            else:
                remove_disc = False
                remove_over = False
            for use_five_types in [True, False]:
                if use_five_types:
                    filtered = '5types'
                else:
                    filtered = '36types'

                for article in articles_obj:
                    filename = '{}{}.{}.{}.data'.format(article.doc_key, tokenized_str, filtered, level)
                    with open('{}/{}'.format(output_dir, filename), 'w') as outfile:
                        for sentence in article.sentences:
                            if tokenized:
                                token_anns = filter_annotations(sentence.token_annotations, remove_disc, remove_over, use_five_types)
                                outfile.write('{}\n'.format(' '.join(token.text for token in sentence.tokens)))
                                if use_orig_postag:
                                    outfile.write('{}\n'.format(' '.join(token.orig_postag for token in sentence.tokens)))
                                else:
                                    outfile.write('{}\n'.format(' '.join(token.postag for token in sentence.tokens)))
                                outfile.write('{}\n'.format('|'.join(str(ann) for ann in token_anns)))
                                outfile.write('\n')
                            else:
                                span_anns = filter_annotations(sentence.span_annotations, remove_disc, remove_over, use_five_types)
                                outfile.write('{}\n'.format(sentence.text))
                                outfile.write('{}\n'.format('|'.join(str(ann) for ann in span_anns)))
                                outfile.write('\n')

if __name__ == '__main__':
    main()
