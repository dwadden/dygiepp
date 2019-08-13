"""
The article identifiers for the coreference and NER data sets don't match up.
I'll align them by matching on title.
"""

from bs4 import BeautifulSoup as BS
import os
from os import path
import glob
import pickle as pkl

import pandas as pd
import numpy as np
from Levenshtein.StringMatcher import StringMatcher

def make_lookups():
  """
  Need to match the ner data and the coref data by sentence. Create
  dictionaries where the keys are the document ID's and the values are the
  sentences. Save as .pkl files to match.
  """
  xml_path = "/data/dave/proj/scierc_coref/data/genia/GENIAcorpus3.02p/GENIAcorpus3.02.merged.xml"
  coref_path = "/data/dave/proj/scierc_coref/data/genia/GENIA_MedCo_coreference_corpus_1.0"

  with open(xml_path, 'r') as infile:
    soup = BS(infile.read().decode('utf-8'), 'lxml')

  articles = soup.find_all('article')
  articles_obj = []

  def get_ner_info(article):
    medline_id = article.find("bibliomisc").text
    title = article.find("title").text.strip()
    return (medline_id, title)

  ner_info = dict([get_ner_info(entry) for entry in articles])

  with open("/data/dave/proj/scierc_coref/data/genia/align/ner_ids.pkl", "wb") as f:
    pkl.dump(ner_info, f, protocol=-1)

  coref_files = glob.glob(path.join(coref_path, "*.xml"))

  def get_coref_info(name):
    with open(name, "r") as f:
      this_soup = BS(f.read().decode('utf-8'), 'lxml')
      pmid = this_soup.find("pmid").text
      title = this_soup.find("articletitle").text.strip()
      return (pmid, title)

  coref_info = dict([get_coref_info(entry) for entry in coref_files])

  with open("/data/dave/proj/scierc_coref/data/genia/align/coref_ids.pkl", "wb") as f:
    pkl.dump(coref_info, f, protocol=-1)


def create_matches():
  """
  Create a dataset mapping the ner id's to the coref id's and vice versa.
  """
  def make_comparator():
    matcher = StringMatcher()

    def compare(str1, str2):
      matcher.set_seqs(str1, str2)
      return matcher.distance()
    return compare


  with open("/data/dave/proj/scierc_coref/data/genia/align/coref_ids.pkl", "rb") as f:
    coref = pkl.load(f).items()
  with open("/data/dave/proj/scierc_coref/data/genia/align/ner_ids.pkl", "rb") as f:
    ner = pkl.load(f).items()

  compare = make_comparator()

  matched = set()
  matches = []
  # Loop over all the coref entries. Find the ner entry that's closest in edit
  # distance. If it's not 0 in edit distance, print it. Nonzero can happen
  # because of minor differences in editing.
  for article in coref:
    sentence = article[1]
    distance = [compare(sentence, entry[1]) for entry in ner]
    the_match = np.argmin(distance)
    match_article = ner[the_match]
    match_distance = distance[the_match]
    # Make sure we're not double-matching.
    if the_match in matched:
      raise Exception("Already matched.")
    if match_distance:
      print sentence
      print match_article[1]
    matches.append(dict(ner=match_article[0].replace("MEDLINE:", ""),
                        coref=article[0]))
    matched.add(the_match)

  matches = pd.DataFrame(matches)
  outfile = "/data/dave/proj/scierc_coref/data/genia/align/alignment.csv"
  matches.to_csv(outfile, index=False)
  print "All done."


def main():
  create_matches()


if __name__ == '__main__':
  main()
