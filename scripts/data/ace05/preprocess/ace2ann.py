import xml.etree.ElementTree as ET
import sys, re

if len(sys.argv) < 4:
    sys.stderr.write("Usage: python %s apf.xml sgm txt" % sys.argv[0])
    sys.exit(0)

apf_tree = ET.parse(sys.argv[1])
apf_root = apf_tree.getroot()

named_entities = {}
check_nes = {}
ne_starts={}
ne_ends={}
ne_map = {}
for entity in apf_root.iter('entity'):
    ne_type = entity.attrib["TYPE"]
    for mention in entity.iter('entity_mention'):
        ne_id = mention.attrib["ID"]
        for child in mention:
            if child.tag == 'head':
                for charseq in child:
                    start = int(charseq.attrib["START"])
                    end = int(charseq.attrib["END"])+1
                    text = re.sub(r"\n", r" ", charseq.text)
                    ne_tuple = (ne_type, start, end, text)
                    if ne_tuple in check_nes:
                        sys.stderr.write("duplicated entity %s\n" % (ne_id))
                        ne_map[ne_id] = check_nes[ne_tuple]
                        continue
                    check_nes[ne_tuple] = ne_id
                    named_entities[ne_id] = [ne_id, ne_type, start, end, text]
                    if not start in ne_starts:
                        ne_starts[start] = []
                    ne_starts[start].append(ne_id)
                    if not end in ne_ends:
                        ne_ends[end] = []
                    ne_ends[end].append(ne_id)

rels = {}
check_rels = []
for relation in apf_root.iter('relation'):
    rel_type = relation.attrib["TYPE"]
    for mention in relation.iter('relation_mention'):
        rel_id = mention.attrib["ID"]
        rel = [rel_id, rel_type, "", ""]
        ignore = False
        for arg in mention.iter('relation_mention_argument'):
            arg_id = arg.attrib["REFID"]
            if arg.attrib["ROLE"] != "Arg-1" and arg.attrib["ROLE"] != "Arg-2":
                continue
            if arg_id in ne_map:
                arg_id = ne_map[arg_id]
            rel[int(arg.attrib["ROLE"][-1])+1] = arg_id
            if not arg_id in named_entities:
                ignore = True
                # ignored duplicated entity
        if ignore:
            sys.stderr.write("ignored relation %s\n" % (rel_id))
            continue
        if rel[1:] in check_rels:
            sys.stderr.write("duplicated relation %s\n" % (rel_id))
            continue
        check_rels.append(rel[1:])
        rels[rel_id] = rel

doc = open(sys.argv[2]).read()
doc = re.sub(r"<[^>]+>", "", doc)
doc = re.sub(r"(\S+)\n(\S[^:])", r"\1 \2", doc)

offset = 0
size = len(doc)
current = 0
regions = []
for i in range(size):
    if i in ne_starts or i in ne_ends:
        inc = 0
        if (doc[i-1] != " " and doc[i-1] != "\n") and (doc[i] != " " and doc[i] != "\n"):
            regions.append(doc[current:i])
            inc = 1
            current = i
        if i in ne_starts:
            for ent in ne_starts[i]:
                named_entities[ent][2] += offset + inc
        if i in ne_ends:
            for ent in ne_ends[i]:
                named_entities[ent][3] += offset
        offset+=inc
regions.append(doc[current:])
doc = " ".join(regions)

for ne in named_entities.values():
    if "\n" in doc[int(ne[2]):int(ne[3])]:
        l = []
        l.append(doc[0:int(ne[2])])
        l.append(doc[int(ne[2]):int(ne[3])].replace("\n", " "))
        l.append(doc[int(ne[3]):])
        doc = "".join(l)

for ne in named_entities.values():
    assert doc[int(ne[2]):int(ne[3])].replace("&AMP;", "&").replace("&amp;", "&").replace(" ", "") == ne[4].replace(" ",""), "%s <=> %s" % (doc[int(ne[2]):int(ne[3])], ne[4])

out = open(sys.argv[3], 'w')
out.write(doc)
out.close()


for ne in named_entities.values():
    print("T%s\t%s %d %d\t%s" % tuple(ne))
    
for rel in rels.values():
    print("R%s\t%s Arg1:T%s Arg2:T%s" % tuple(rel))

