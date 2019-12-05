import sys

words = []
for line in open(sys.argv[1]):
    entry = line.split("\t")
    if len(entry) < 2:
        if len(words) > 0:
            print(" ".join(words))
            words = []
    else:
        words.append(entry[1])
if len(words) > 0:
    print(" ".join(words))
    words = []
