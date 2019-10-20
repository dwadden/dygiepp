import re

def save_list(xs, name):
    "Save a list as text, one entry per line."
    with open(name, "w") as f:
        for x in xs:
            f.write(str(x) + "\n")


def load_list(name, convert=lambda x: x):
    """
    Make each line of a file into a list entry. Apply a conversion function to
    each line (e.g. convert to int). By default, the conversion does nothing.
    """
    res = []
    with open(name, "r") as f:
        for line in f:
            res.append(convert(line.strip()))
    return res


def flatten(xxs):
    """
    Flatten a nested list into a single list. Note this only works for lists
    nested one deep. For the general case we'd need recursion
    """
    return [x for xs in xxs for x in xs]


# From https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list

def find_sub_list(sl, l):
  sll = len(sl)
  for ind in (i for i, e in enumerate(l) if e == sl[0]):
    if l[ind:ind + sll] == sl:
      return ind, ind + sll - 1
  # If nothing found, return None.
  return None


def find_sub_lists(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results


def fields_to_batches(d):
    """
    The input is a dict whose items are batched tensors. The output is a list of dictionaries - one
    per entry in the batch - with the slices of the tensors for that entry. Here's an example.

    Input:
    d = {"a": [[1, 2], [3,4]], "b": [1, 2]}
    Output:
    res = [{"a": [1, 2], "b": 1}, {"a": [3, 4], "b": 2}].
    """
    # Make sure all input dicts have same length.
    lengths = [len(x) for x in d.values()]
    assert len(set(lengths)) == 1
    length = lengths[0]
    keys = d.keys()
    res = [{k: d[k][i] for k in keys} for i in range(length)]
    return res
