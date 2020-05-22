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

    return np.array(Triples), KBmatrix[:1, :np.min(tails)], np.min(tails)   //# FIXME


def read_data(data_file):

    # q+'\t'+ans+'\t'+p+'\t'+ansset+'\t'+c+'\t'+sub+'\n'
    # question \t ans(ans1/ans2/) \t e1#r1#e2#r2#e3#<end>#e3
    # question \t  ans  \t  e1#r1#e2#r2#e3#<end>#e3  \t   ans1/ans2/   \t   e1#r1#e2///e2#r2#e3#///s#r#t///s#r#t


    if os.path.isfile(data_file):
        with open(data_file) as f:
            lines = f.readlines()
    else:
        raise Exception("!! %s is not found!!" % data_file)

    words = set()
    data = []
    questions = []
    doc = []

    for line in lines:
        line = line.strip().split('\t')
        qlist = line[0].strip().split()
        k = line[1].find('(')
        if not k == -1:
            if line[1][k-1] == '_':
                k += (line[1][k+1:-1].find('(') + 1)
            asset = line[1][k+1:-1]
            line[1] = line[1][:k]
        else:
            asset = line[3]
        data.append([line[0], line[1], line[2], asset])

        for w in qlist:
            words.add(w)
        questions.append(qlist)

    sentence_size = max(len(i) for i in questions)

    return words, data, sentence_size



def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]



def process_data(KB_file, data_file):
    entities, relations = read_KB(KB_file)
    words, data, sentence_size = read_data(data_file)

    word2id = {}
    ent2id = {}
    rel2id = {}

    word2id['<unk>'] = 0
    rel2id['<end>'] = 0
    ent2id['<unk>'] = 0

    for r in relations:
        # same r_id in rel2id and word2id
        if r not in rel2id.keys():
            rel2id[r] = len(rel2id)
        if r not in word2id.keys():
            word2id[r] = len(word2id)
    for e in entities:
        if e not in ent2id.keys():
            ent2id[e] = len(ent2id)
    for word in words:
        if word not in word2id.keys():
            word2id[word] = len(word2id)

    print('here are %d words in word2id(vocab)' % len(word2id))
    print('here are %d relations in rel2id(rel_vocab)' % len(rel2id))
    print('here are %d entities in ent2id(ent_vocab)' % len(ent2id))

    Triples, KBs, tails_size = get_KB(KB_file, ent2id, rel2id)

    print("The number of records or triples", len(np.nonzero(KBs)[0]))

    Q = []
    QQ = []
    A = []
    AA = []
    P = []
    PP = []
    S = []
    SS = []

    for query, answer, path, answerset in data:
        path = path.strip().split('#')  # path = [s,r1,m,r2,t]
        #answer = path[-1]

        query = query.strip().split()
        ls = max(0, sentence_size-len(query))
        q = [word2id[w] for w in query] + [0] * ls
        Q.append(q)
        QQ.append(query)


        a = np.zeros(len(ent2id))  # if use new ans-vocab, add 0 for 'end'
        a[ent2id[answer]] = 1
        A.append(a)
        AA.append(ent2id[answer])

        p = []
        for i in range(len(path)):
            if i % 2 == 0:
                e = ent2id[path[i]]
                p.append(e)
            else:
                r = rel2id[path[i]]
                p.append(r)

        P.append(p)
        PP.append(path)

        anset = answerset.split('/')
        anset = anset[:-1]
        ass = []
        for a in anset:
            ass.append(ent2id[a])
        S.append(ass)
        SS.append(anset)

    return np.array(Q), np.array(A), np.array(P), np.array(S), Triples, sentence_size, word2id, ent2id, rel2id


def MultiAcc(labels,preds,length):
    #length = path = 2 * hop + 1   (hop == path_l + cons_l + final == path_l * 2 + 1 )
    #compare path and final answer accuracy
    Acc = []
    #Acc=np.asarray(Acc)
    for i in range(length):
        Acc.append(round(metrics.accuracy_score(labels[:,i],preds[:,i]),3))

    batch_size = preds.shape[0]
    correct = 0.0
    for j in range(batch_size):
        k = length - 1
        while(labels[j,k]==0):
            k -= 2
        if(labels[j,k]==preds[j,k]):
            correct += 1.0   #final answer accuracy
    Acc.append(round( correct/batch_size ,3))
    return Acc

def InSet(labels,anset,preds):
    #get accuracy(whether in answer set or not)
    #labels does not matter
    #preds is path-list
    #labels is path-labels
    right = 0.0
    for i in range(len(anset)):
        if type(preds[i]) is np.int64:
            ans_pred = preds[i]
        else:
            ans_pred = preds[i,-1]
            '''
            k = len(labels[0]) - 1
            while(labels[i,k]==0):
                k -= 2
            ans_pred = preds[i,k]
            '''
        if ans_pred in anset[i]:
            right += 1
    return round(right/len(anset), 3)
