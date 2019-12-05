# -*- coding: utf-8 -*-
import sys
import pdb
if len(sys.argv) != 4:
    sys.stderr.write("usage:"+sys.argv[0]+" txt annotation newtxt > aligned_annotation")
    sys.exit(-1)


orig=[]
new=[]

for line in open(sys.argv[1]):
    orig.append(line)

for line in open(sys.argv[3]):
    new.append(line)

original="".join(orig)
newtext ="".join(new)

annotation=[]
terms = {}
ends = {}

for line in open(sys.argv[2]):
    if line.startswith('T'):
        annots = line.rstrip().split("\t", 2)
        typeregion = annots[1].split(" ")
        start = int(typeregion[1])
        end = int(typeregion[2])
        if not start in terms:
            terms[start] = []
        if not end in ends:
            ends[end] = []
        if len(annots) == 3:
            terms[start].append([start, end, annots[0], typeregion[0], annots[2], False])
        else:
            terms[start].append([start, end, annots[0], typeregion[0], "", False])
        ends[end].append(start)            
    else:
        annotation.append(line)

orgidx = 0
newidx = 0
orglen = len(original)
newlen = len(newtext)

while orgidx < orglen and newidx < newlen:
    if original[orgidx] == newtext[newidx]:
        orgidx+=1
        newidx+=1
    elif newtext[newidx] == '\n':
        newidx+=1
    elif original[orgidx] == '\n':
        orgidx+=1
    elif newtext[newidx] == ' ':
        newidx+=1
    elif original[orgidx] == ' ':
        orgidx+=1
    elif newtext[newidx] == '\t':
        newidx+=1
    elif original[orgidx] == '\t':
        orgidx+=1
    elif newtext[newidx] == '.':
        # ignore extra "." for stanford
        newidx+=1
    else:
        assert False, "%d\t$%s$\t$%s$" % (orgidx, original[orgidx:orgidx+20], newtext[newidx:newidx+20])
    if orgidx in terms: 
        for l in terms[orgidx]:
            l[0] = newidx
    if orgidx in ends:
        for start in ends[orgidx]:
            for l in terms[start]:
                if not l[-1] and l[1] == orgidx:
                    l[1] = newidx
                    l[-1] = True
        del ends[orgidx]

for ts in terms.values():
    for term in ts:
        if term[4] == "":
            print("%s\t%s %d %d\t%s" % (term[2], term[3], term[0], term[1], newtext[term[0]:term[1]]))
        else:
            assert newtext[term[0]:term[1]].replace("&AMP;", "&").replace("&amp;", "&").replace(" ", "").replace("\n", "") == term[4].replace(" ", ""), newtext[term[0]:term[1]] + "<=>" + term[4]
            print("%s\t%s %d %d\t%s" % (term[2], term[3], term[0], term[1], newtext[term[0]:term[1]].replace("\n", " ")))

for annot in annotation:
    print(annot, end='')

