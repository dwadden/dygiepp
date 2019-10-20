"""
Add coreference information to the GENIA data set.

Also remove blacklisted documents, for which the merge didn't go correctly and off-by-one-errors
were introduced.
"""

from collections import defaultdict

from os import path
import os
import glob
import re
import json
from bs4 import BeautifulSoup as BS
import pandas as pd
import argparse

import shared


genia_base = "./data/genia"
genia_raw = f"{genia_base}/raw-data"
genia_processed = f"{genia_base}/processed-data"

json_dir = f"{genia_processed}/json-ner"
coref_dir = f"{genia_raw}/GENIA_MedCo_coreference_corpus_1.0"
alignment_file = f"{genia_raw}/align/alignment.csv"

alignment = pd.read_csv(alignment_file).set_index("ner")


def get_coref_types():
    """
    Get the different types of coref in the GENIA data set, and save to file. Once this is done
    once, just hard-code the list of coref types.
    """
    coref_files = glob.glob(path.join(coref_dir, "*xml"))
    coref_counts = defaultdict(lambda: 0)
    for i, coref_file in enumerate(coref_files):
        with open(coref_file, "r") as f:
            soup = BS(f.read().decode("utf-8"), "lxml")
            corefs = soup.find_all("coref")
            for coref in corefs:
                coref_type = coref.attrs["type"] if "type" in coref.attrs else "NONE"
                coref_counts[coref_type] += 1

    res = pd.DataFrame(coref_counts.items())
    res.columns = ["label", "count"]
    res = res.sort_values("count", ascending=False)
    out_file = "/data/dave/proj/scierc_coref/data/genia/stats/coref-counts.csv"
    res.to_csv(out_file, index=False)


class Coref(object):
    """Represents a single coreference."""

    def __init__(self, xml, soup_text, sents):
        "Get the text, id, referrent, and text span."
        self.xml = xml
        self.text = xml.text
        self.tokens = [tok for tok in re.split('([ -/,.+])', self.text)
                       if tok not in ["", " "]]
        self.id = xml.attrs["id"]
        # A very small number of corefs have two parents. I'm going to just take the first parent.
        # TODO(dwadden) If time, go back and fix this.
        self.ref = xml.attrs["ref"].split(" ")[0] if "ref" in xml.attrs else None
        self.span = self._get_span(sents, soup_text)
        self.type = xml.attrs["type"] if "type" in xml.attrs else None

    def _get_span(self, sents, soup_text):
        """Get text span of coref. We have inclusive endpoints."""

        spans = shared.find_sub_lists(self.tokens, sents)
        n_matches = len(spans)
        # Case 1: If can't match the span, record and return. This doesn't happen
        # much.
        if n_matches == 0:
            stats["no_matches"] += 1
            return None
        # Case 2: IF there are multiple span matches, go back and look the original
        # XML tag to determine which match we want.
        elif n_matches > 1:
            xml_tag = self.xml.__repr__()
            tmp_ixs = shared.find_sub_lists(list(self.text), list(soup_text))
            text_ixs = []
            # Last character of the match must be a dash or char after must be an
            # escape, else we're not at end of token.
            text_ixs = [ixs for ixs in tmp_ixs if
                        soup_text[ixs[1] + 1] in '([ -/,.+])<' or soup_text[ixs[1]] == "-"]
            if len(text_ixs) != n_matches:
                # If the number of xml tag matches doesn't equal the number of span
                # matches, record and return.
                stats["different_num_matches"] += 1
                return None
            tag_ix = shared.find_sub_lists(list(xml_tag), list(soup_text))
            assert len(tag_ix) == 1
            tag_ix = tag_ix[0]
            text_inside = [x[0] >= tag_ix[0] and x[1] <= tag_ix[1] for x in text_ixs]
            assert sum(text_inside) == 1
            match_ix = text_inside.index(True)
        else:
            match_ix = 0
        stats["successful_matches"] += 1
        span = spans[match_ix]
        return span


class Corefs(object):
    """Holds all corefs and represents relations between them."""

    def __init__(self, soup, sents_flat, coref_types):
        self.coref_types = coref_types
        coref_items = soup.find_all("coref")
        corefs = [Coref(item, soup.__repr__(), sents_flat) for item in coref_items]
        # Put the cluster exemplars first.
        corefs = sorted(corefs, key=lambda coref: coref.ref is None, reverse=True)
        coref_ids = [coref.id for coref in corefs]
        corefs = self._assign_parent_indices(corefs, coref_ids)
        clusters = self._get_coref_clusters(corefs)
        clusters = self._cleanup_coref_clusters(corefs, clusters)
        cluster_spans = self._make_cluster_spans(clusters)
        self.corefs = corefs
        self.clusters = clusters
        self.cluster_spans = cluster_spans

    @staticmethod
    def _assign_parent_indices(corefs, coref_ids):
        """Give each coref the index of it parent in the list of corefs."""
        for coref in corefs:
            if coref.ref is None:
                coref.parent_ix = None
            else:
                coref.parent_ix = coref_ids.index(coref.ref)
        return corefs

    @staticmethod
    def _get_coref_clusters(corefs):
        def get_cluster_assignment(coref):
            ids_so_far = set()
            this_coref = coref
            while this_coref.ref is not None:
                # Condition to prevent self-loops.
                if this_coref.id in ids_so_far or this_coref.id == this_coref.ref:
                    return None
                ids_so_far.add(this_coref.id)
                parent = corefs[this_coref.parent_ix]
                this_coref = parent
            return this_coref.id

        clusters = {None: set()}
        for coref in corefs:
            if coref.ref is None:
                # It's a cluster exemplar
                coref.cluster_assignment = coref.id
                clusters[coref.id] = set([coref])
            else:
                cluster_assignment = get_cluster_assignment(coref)
                coref.cluster_assignment = cluster_assignment
                clusters[cluster_assignment].add(coref)
        return clusters

    def _cleanup_coref_clusters(self, corefs, clusters):
        """
        Remove items that didn't get spans, don't have an allowed coref type, or
        weren't assigned a cluster
        """
        # Remove unassigned corefs.
        _ = clusters.pop(None)
        for coref in corefs:
            # If the referent entity didn't get a span match, remove the cluster.
            if coref.ref is None:
                if coref.span is None:
                    _ = clusters.pop(coref.id)
            # If a referring coref didn't have a span or isn't the right coref type, remove it.
            else:
                if coref.type not in self.coref_types or coref.span is None:
                    # Check to make sure the cluster wasn't already removed.
                    if coref.cluster_assignment in clusters:
                        clusters[coref.cluster_assignment].remove(coref)
        # Now remove singleton clusters.
        # Need to make it a list to avoid `dictionary size changed iteration` error."
        for key in list(clusters.keys()):
            if len(clusters[key]) == 1:
                _ = clusters.pop(key)
        return clusters

    @staticmethod
    def _make_cluster_spans(clusters):
        """Convert to nested list of cluster spans, as in scierc data."""
        res = []
        for key, cluster in clusters.items():
            cluster_spans = []
            for coref in cluster:
                cluster_spans.append(list(coref.span))
            res.append(sorted(cluster_spans))
        return res


def get_excluded():
    "Get list of files that had random off-by-1-errors and will be excluded."
    current_path = os.path.dirname(os.path.realpath(__file__))
    excluded = pd.read_table(f"{current_path}/exclude.txt", header=None, squeeze=True).values
    return excluded


def one_fold(fold, coref_types, out_dir, keep_excluded):
    """Add coref field to json, one fold."""
    print("Running fold {0}.".format(fold))
    excluded = get_excluded()
    with open(path.join(json_dir, "{0}.json".format(fold))) as f_json:
        with open(path.join(out_dir, "{0}.json".format(fold)), "w") as f_out:
            for counter, line in enumerate(f_json):
                doc = json.loads(line)
                pmid = int(doc["doc_key"].split("_")[0])
                medline_id = alignment.loc[pmid][0]
                xml_file = path.join(coref_dir, str(medline_id) + ".xml")
                sents_flat = shared.flatten(doc["sentences"])
                with open(xml_file, "r") as f_xml:
                    soup = BS(f_xml.read(), "lxml")
                    corefs = Corefs(soup, sents_flat, coref_types)
                doc["clusters"] = corefs.cluster_spans
                # Save unless it's bad and we're excluding bad documents.
                if keep_excluded or doc["doc_key"] not in excluded:
                    f_out.write(json.dumps(doc) + "\n")


def get_clusters(coref_types, out_dir, keep_excluded):
    """Add coref to json, filtering to only keep coref roots and `coref_types`."""
    global stats
    stats = dict(no_matches=0, successful_matches=0, different_num_matches=0)
    for fold in ["train", "dev", "test"]:
        one_fold(fold, coref_types, out_dir, keep_excluded)
    print(stats)


def main():
    # get_coref_types()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ident-only", action="store_true",
                        help="If true, only do `IDENT` coreferences.")
    parser.add_argument("--keep-excluded", action="store_true",
                        help="If true, keep training docs that were excluded due to off-by-1 errors.")
    args = parser.parse_args()
    if args.ident_only:
        coref_types = ["IDENT"]
    else:
        coref_types = [
            "IDENT",
            "NONE",
            "RELAT",
            "PRON",
            "APPOS",
            "OTHER",
            "PART-WHOLE",
            "WHOLE-PART"
        ]

    coref_type_name = "ident-only" if args.ident_only else "all"
    out_dir = f"{genia_processed}/json-coref-{coref_type_name}"

    if not path.exists(out_dir):
        os.mkdir(out_dir)

    get_clusters(coref_types, out_dir, args.keep_excluded)


if __name__ == '__main__':
    main()
