"""
A couple spot-checks that the GENIA processing
"""

from __future__ import print_function

import os
from os import path
from bs4 import BeautifulSoup as BS
import random


def check_against_xml_source():
    "Make sure my processed documents are consistent with the source XML"
    processed_dir = "/data/dave/proj/scierc_coref/data/genia/sutd-article/final"
    xml_file = "/data/dave/proj/scierc_coref/data/genia/GENIAcorpus3.02p/GENIAcorpus3.02.merged.xml"

    with open(xml_file, 'r') as infile:
        soup = BS(infile.read(), 'lxml')
    articles = soup.find_all('article')

    # Check 5 random files

    files = os.listdir(processed_dir)
    random.shuffle(files)
    files = files[:5]

    for processed_file in files:
        with open(path.join(processed_dir, processed_file), 'r') as f:
            lines = f.readlines()
            art_text = "".join(x for x in lines[::4]).replace(" ,", "")

        found_one = False
        for article in articles:
            this_text = article.get_text()
            if art_text[:50] in this_text or art_text[200:250] in this_text:
                found_one = True
                print("Got one.")
                print(this_text)
                print()
                print(art_text)
                print()
        if not found_one:
            print("Didn't find one.")
            print(art_text)
        print(40 * "#")


def check_against_sutd_processed():
    """
    Grab one of my processed documents. Make sure I can find a match in one of
    the SUTD processed files.
    """
    theirs_file = "/data/dave/proj/scierc_coref/data/genia/sutd/test.data"
    with open(theirs_file, "r") as f:
        theirs = f.readlines()

    mine_file = "/data/dave/proj/scierc_coref/data/genia/sutd-article/final/90244434.tok.5types.no_disc.data"
    with open(mine_file, "r") as f:
        mine = f.readlines()

    my_first = mine[0]
    foo = [entry == my_first for entry in theirs]
    ix = [i for i, entry in enumerate(foo) if entry][0]
    theirs_match = theirs[ix:ix + len(mine)]

    same = [x == y for x, y in zip(mine, theirs_match)]
    print(sum(same))
    print(len(mine))
