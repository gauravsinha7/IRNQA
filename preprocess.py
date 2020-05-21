import json
from collections import Counter
from tqdm import tqdm
import pickle
import os
import re
import numpy as np
from sklearn import metrics


def read_triple(s, r, o, triples, entities, relations):
    '''
    add the (s,r,o) to the triples
    add the s,o to the entities and add the r to the relations
    '''
    triples.appned([s, r, o])
    entities.append(s)
    entities.append(o)
    relations.append(r)


def load_as_triple(kb_json):
    '''
    Get triples from the kb.json
    Not ignore the repeat triples
    '''
    triples = []
    entities = []
    relations = []
    vocab = {'<PAD>': 0,
             '<UNK>': 1,
             '<START>': 2,
             '<END>': 3}
    print("Build triples and vocabulary of kb")
    kb = json.load(open(kb_json))
    print("Process the concepts...")
    for i in tqdm(kb['concepts']):
        for j in kb['concepts'][i]['instanceOf']:
            s = kb['concepts'][i]['name']
            o = kb['concepts'][j]['name']
            read_triple(s, 'instanceOf', o, triples, entities, relations)
    print("Process the entities...")
    for i in tqdm(kb['entities']):
        for j in kb['entities'][i]['instanceOf']:
            s = kb['entities'][i]['name']
            o = kb['concepts'][j]['name']
            read_triple(s, 'instanceOf', o, triples, entities, relations)

        name = kb['entities'][i]['name']
        for attr_dict in kb['entities'][i]['attributes']:
            o = '{}_{}'.format(
                attr_dict['value']['value'], attr_dict['value'].get('unit', ''))
            read_triple(name, attr_dict['key'], o,
                        triples, entities, relations)
            s = '{}_{}_{}'.format(name, attr_dict['key'], o)
            for qk, qvs in attr_dict['qualifiers'].items():
                for qv in qvs:
                    o = '{}_{}'.format(qv['value'], qv.get('unit', ''))
                    read_triple(s, qk, o, triples, entities, relations)
        for rel_dict in kb['entities'][i]['relations']:
            o = kb['entities'].get(
                rel_dict['object'], kb['concepts'].get(rel_dict['object'], None))
            if o is None:
                continue
            o = o['name']
            if rel_dict['direction'] == 'backward':
                read_triple(o, rel_dict['predicate'],
                            name, triples, entities, relations)
            else:
                read_triple(o, rel_dict['predicate'],
                            name, triples, entities, relations)
            s = '{}_{}_{}'.format(name, rel_dict['predicate'], o)
            for qk, qvs in rel_dict['qualifiers'].items():
                for qv in qvs:
                    o = '{}_{}'.format(qv['value'], qv.get('unit', ''))
                    read_triple(s, qk, o, triples, entities, relations)
    print("Completed, the length of triples is {}".format(len(triples)))
    return triples, entities, relations


def read_KB(KB_file):
    # example in KB_file: KBs.txt h \t r \t t
    entities = set()
    relations = set()
    if os.path.isfile(KB_file):
        with open(KB_file) as f:
            lines = f.readlines()
    else:
        raise Exception("!! %s is not found!!" % KB_file)

    for line in lines:
        line = line.strip().split('\t')
        entities.add(line[0])
        entities.add(line[2])
        relations.add(line[1])
    return entities, relations


def get_KB(KB_file, ent2id, rel2id):
    nwords = len(ent2id)
    nrels = len(rel2id)
    tails = np.zeros([nwords*nrels, 1], 'int32')
    KBmatrix = np.zeros([nwords * nrels, nwords], 'int32')
    Triples = []

    f = open(KB_file)
    for line in f.readlines():
        line = line.strip().split('\t')
        h = ent2id[line[0]]
        r = rel2id[line[1]]
        t = ent2id[line[2]]
        Triples.append([h, r, t])
        lenlist = tails[h*nrels+r]
        KBmatrix[h*nrels+r, lenlist] = t
        tails[h*nrels+r] += 1

    return np.array(Triples), KBmatrix[:1, :np.min(tails)], np.min(tails)
