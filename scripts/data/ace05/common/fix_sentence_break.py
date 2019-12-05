# -*- coding: utf-8 -*-
import sys

if len(sys.argv) != 4:
    sys.stderr.write("usage:"+sys.argv[0]+" txt annotation newtxt")
    sys.exit(-1)


doc=[]

for line in open(sys.argv[1]):
    doc.append(line)

doc="".join(doc)

terms = {}
rels = []
for line in open(sys.argv[2]):
    if line.startswith('T'):
        annots = line.rstrip().split("\t", 2)
        typeregion = annots[1].split(" ")
        start = int(typeregion[1])
        end = int(typeregion[2])
        terms[annots[0]] = (start, end)
    else:
        rel = line.rstrip().split("\t")
        args = rel[1].split(" ")
        arg1 = args[1].split(":",2)
        arg2 = args[2].split(":",2)
        rels.append([arg1[1], arg2[1]])

for start,end in terms.values():
    if "\n" in doc[start:end]:
        l = []
        l.append(doc[0:start])
        l.append(doc[start:end].replace("\n", " "))
        l.append(doc[end:])
        doc = "".join(l)
                                                    
for arg in rels:
    arg1 = arg[0]
    arg2 = arg[1]
    start = min(terms[arg1][0], terms[arg2][0])
    end = max(terms[arg1][1], terms[arg2][1])
    if "\n" in doc[start:end]:
        l = []
        l.append(doc[0:start])
        l.append(doc[start:end].replace("\n", " "))
        l.append(doc[end:])
        doc = "".join(l)

out = open(sys.argv[3], 'w')
out.write(doc)
out.close()


    
