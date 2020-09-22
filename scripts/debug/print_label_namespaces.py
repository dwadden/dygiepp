"""
Print the label namespaces available for a given model.
"""

import argparse
from textwrap import dedent
import tarfile


def get_args():
    desc = """
    Print out the label namespaces available for a pretrained model.
    Usage example: python scripts/debug/print_label_namespaces.py pretrained/scierc.tar.gz
    """
    parser = argparse.ArgumentParser(description=dedent(desc),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("model_archive", type=str,
                        help="The `.tar.gz` archive containing the trained model.")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    model_archive = args.model_archive
    tar = tarfile.open(model_archive)
    members = tar.getmembers()
    print("Available label namespaces:")
    for member in members:
        name = member.name
        if "vocabulary" in name and ".txt" in name and "non_padded_namespaces" not in name:
            print(name.replace("vocabulary/", ""))


if __name__ == "__main__":
    main()
