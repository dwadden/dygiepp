import pdb
import glob
import json
import numpy as np
lengths = []
def parseAce(annfn, entity_set, rel_set):
    entity_dir = {}
    rel_dir = {}
    for line in open(annfn):
        line = line.rstrip()
        if 'Arg1:' in line:
            tokens = line.split('\t')
            key = tokens[0]
            rels = tokens[1].split()
            rel = rels[0]
            arg1 = rels[1].split(':')[-1]
            arg2 = rels[2].split(':')[-1]
            rel_dir[key] = {'relation':rel, 'arg1':arg1, 'arg2':arg2}
            rel_set.add(rel)
        else:
            tokens = line.split('\t')
            key = tokens[0]
            offsets = tokens[1].split()[1:]
            offset0 = int(offsets[0])
            offset1 = int(offsets[1])
            ner = tokens[1].split()[0]
            keyphrase = tokens[2]
            global lengths
            lengths.append(len(keyphrase.split()))
            entity_dir[key] = {'ner':ner, 'offset': [offset0, offset1], 'keyphrase':keyphrase}
            entity_set.add(ner)

    return entity_dir, rel_dir

def parseStanfordOld(stanfordfn):
    doc = {}
    for line in open(stanfordfn):
        line = line.rstrip()
        tokens = line.split('\t')
        if not line: continue
        if tokens[2].startswith('sentence'):
            sentid = int(tokens[2].split('"')[1][1:])
            offset0 = int(tokens[0])
            offset1 = int(tokens[1])
            doc[sentid] = {'offset':[offset0, offset1], 'tokens':{}}
        else:
            offset0 = int(tokens[0])
            offset1 = int(tokens[1])
            tokid = int(tokens[2].split('"')[1][1:])
            dephead = tokens[2].split('"')[-2]
            if dephead == "ROOT":
                dephead = 0
            else:
                dephead = int(dephead[1:])
            pos = tokens[2].split()[3].split('"')[1]
            deptype = tokens[2].split()[-1].split('=')[0]
            doc[sentid]['tokens'][tokid] = {'dephead':dephead, "pos":pos, "deptype":deptype,"offset":[offset0,offset1]}
    return doc

def parseStanford(stanfordfn):
    doc = {}
    token_dict_offset0 = {}
    token_dict_offset1 = {}
    for line in open(stanfordfn):
        line = line.rstrip()
        tokens = line.split('\t')
        if not line: continue
        if tokens[2].startswith('sentence'):
            sentid = int(tokens[2].split('"')[1][1:])
            offset0 = int(tokens[0])
            offset1 = int(tokens[1])
            doc[sentid] = {'offset':[offset0, offset1], 'tokens':{}}
        else:
            offset0 = int(tokens[0])
            offset1 = int(tokens[1])
            tokid = int(tokens[2].split('"')[1][1:])
            dephead = tokens[2].split('"')[-2]
            if dephead == "ROOT":
                dephead = 0
            else:
                dephead = int(dephead[1:])
            pos = tokens[2].split()[3].split('"')[1]
            deptype = tokens[2].split()[-1].split('=')[0]
            token_dict_offset0[offset0] = {'dephead':dephead, "pos":pos, "deptype":deptype, 'sentid':sentid, 'tokenid':tokid}
            token_dict_offset1[offset1] = {'dephead':dephead, "pos":pos, "deptype":deptype, 'sentid':sentid, 'tokenid':tokid}

    return doc, token_dict_offset0, token_dict_offset1



def Ace2json(entity_dir, rel_dir, token_dict_offset1, token_dict_offset2, txtfn, docs, nercount, relcount, sentcount):
    fn = txtfn.split('/')[-1].replace('.split.txt','')
    print(fn)
    fid = open(txtfn)
    text = fid.read()
    text = text.rstrip()
    sentences = text.split('\n')
    sentences = [line.split() for line in sentences]
    sentcount += len(sentences)
    sentence_ids = []
    i = 0
    for sentence in sentences:
        ids = []
        for word in sentence:
            ids.append(i)
            i += 1
        sentence_ids.append(ids)

    ner = [[] for i in range(len(sentences))]
    relations = [[] for i in range(len(sentences))]

    for entity in entity_dir:
        offsets = tuple(entity_dir[entity]['offset'])
        if offsets[0] in token_dict_offset1:
            offset0 = token_dict_offset1[offsets[0]]
            tokenid0 = offset0['tokenid']
        else:
            pdb.set_trace()
        if offsets[1] in token_dict_offset2:
            offset1 = token_dict_offset2[offsets[1]]
            tokenid1 = offset1['tokenid']
        else:
            pdb.set_trace()
        ner[offset0['sentid']].append([tokenid0,tokenid1,entity_dir[entity]['ner']])
        nercount += 1
    for relation in rel_dir:
        arg1 = rel_dir[relation]['arg1']
        arg2 = rel_dir[relation]['arg2']
        tokid0 = token_dict_offset1[entity_dir[arg1]['offset'][0]]['tokenid']
        tokid1 = token_dict_offset2[entity_dir[arg1]['offset'][1]]['tokenid']
        tokid2 = token_dict_offset1[entity_dir[arg2]['offset'][0]]['tokenid']
        tokid3 = token_dict_offset2[entity_dir[arg2]['offset'][1]]['tokenid']
        relations[token_dict_offset1[entity_dir[arg1]['offset'][0]]['sentid']].append([tokid0, tokid1, tokid2, tokid3, rel_dir[relation]['relation']])
        relcount += 1
    docs.append({"sentences":sentences, "ner":ner, "relations": relations, "clusters":[], "doc_key":fn})
    return nercount, relcount, sentcount

def WriteDocs(docs, outfn):
    with open(outfn,'w') as f:
        for relation in docs:
            f.write(json.dumps(relation))
            f.write('\n')

out_dir="../../../../data/ace05/processed-data/json"

docs = []
entity_set = set()
rel_set = set()
nercount = 0
relcount = 0
sentcount = 0
for stanfordfn in glob.glob('./corpus/dev/' + '*.stanford.so'):
    txtfn = stanfordfn.replace('stanford.so','txt')
    annfn = stanfordfn.replace('stanford.so','ann')
    entity_dir, rel_dir = parseAce(annfn, entity_set, rel_set)
    parses, token_dict_offset1, token_dict_offset2 = parseStanford(stanfordfn)
    nercount, relcount, sentcount = Ace2json(entity_dir, rel_dir, token_dict_offset1, token_dict_offset2, txtfn, docs, nercount, relcount, sentcount)
outfn = f'{out_dir}/dev.json'
WriteDocs(docs, outfn)

docs = []
entity_set = set()
rel_set = set()
nercount = 0
relcount = 0
sentcount = 0
for stanfordfn in glob.glob('./corpus/test/' + '*.stanford.so'):
    txtfn = stanfordfn.replace('stanford.so','txt')
    annfn = stanfordfn.replace('stanford.so','ann')
    entity_dir, rel_dir = parseAce(annfn, entity_set, rel_set)
    parses, token_dict_offset1, token_dict_offset2 = parseStanford(stanfordfn)
    nercount, relcount, sentcount = Ace2json(entity_dir, rel_dir, token_dict_offset1, token_dict_offset2, txtfn, docs, nercount, relcount, sentcount)
outfn = f'{out_dir}/test.json'
WriteDocs(docs, outfn)

docs = []
entity_set = set()
rel_set = set()
nercount = 0
relcount = 0
sentcount = 0
for stanfordfn in glob.glob('./corpus/train/' + '*.stanford.so'):
    txtfn = stanfordfn.replace('stanford.so','txt')
    annfn = stanfordfn.replace('stanford.so','ann')
    entity_dir, rel_dir = parseAce(annfn, entity_set, rel_set)
    parses, token_dict_offset1, token_dict_offset2 = parseStanford(stanfordfn)
    nercount, relcount, sentcount = Ace2json(entity_dir, rel_dir, token_dict_offset1, token_dict_offset2, txtfn, docs, nercount, relcount, sentcount)
outfn = f'{out_dir}/train.json'
WriteDocs(docs, outfn)
