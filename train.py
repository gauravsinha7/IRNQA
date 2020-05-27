from IPython import embed
import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm
from preprocess import process_data, InSet, MultiAcc
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from model import IRN

def train(args):
    KB_file = 'data/2H-kb.txt'
    data_file = 'data/2H.txt'
    start = time.time()
    Q,A,P,S,Triples,args.query_size, word2id, ent2id, rel2id = process_data(KB_file, data_file)
    args.path_size = len(P[0])
    args.nhop = args.path_size / 2


    print ("read data cost %f seconds" %(time.time()-start))
    args.nwords = len(word2id)
    args.nrels = len(rel2id)
    args.nents = len(ent2id)

    trainQ, testQ, trainA, testA, trainP, testP, trainS, testS = train_test_split(Q, A, P, S, test_size=.1, random_state=123)
    trainQ, validQ, trainA, validA, trainP, validP, trainS, validS = train_test_split(trainQ, trainA, trainP, trainS, test_size=.11, random_state=0)

    n_train = trainQ.shape[0]
    n_test = testQ.shape[0]
    n_val = validQ.shape[0]
    print(trainQ.shape, trainA.shape,trainP.shape,trainS.shape)
    print(type(trainQ), type(trainA),type(trainP),type(trainS))




    trQ=torch.from_numpy(trainQ)
    trP=torch.from_numpy(trainP)
    trA=torch.from_numpy(trainA)
    #trS=torch.from_numpy(trainS)
    print(type(trQ), type(trP),type(trA))


    train_labels = np.argmax(trainA, axis=1)
    test_labels = np.argmax(testA, axis=1)
    valid_labels = np.argmax(validA, axis=1)
    batches = list(zip(range(0, n_train-args.batch_size, args.batch_size), range(args.batch_size, n_train, args.batch_size)))
    pre_batches = list(zip(range(0, Triples.shape[0]-args.batch_size, args.batch_size), range(args.batch_size, Triples.shape[0], args.batch_size)))


    model = IRN(args)
    optimizer = optim.Adam(model.parameters(), args.init_lr,weight_decay=1e-5)
    pre_val_preds = model.predict(Triples, validQ, validP)
    pre_test_preds = model.predict(Triples, testQ, testP)
    for t in range(args.nepoch):
        np.random.shuffle(batches)
        for i in range(args.inner_nepoch):
            np.random.shuffle(pre_batches)
            pre_total_cost = 0.0
            for s, e in pre_batches:
                pretrain_loss = model.batch_pretrain(
                    Triples[s:e], trainQ[0:args.batch_size],
                    trainA[0:args.batch_size],
                    np.argmax(trainA[0:args.batch_size],
                            axis=1), trainP[0:args.batch_size])
                optimizer.zero_grad()
                pretrain_loss.backward()
                optimizer.step()
        total_cost = 0.0

        for s, e in batches:
            total_cost = model(Triples[s:e], trainQ[s:e], trainA[s:e],
                                        np.argmax(trainA[s:e], axis=1),
                                        trainP[s:e])
            optimizer.zero_grad()
            total_cost.backward()
            optimizer.step()
        if t % 1 == 0:

            train_preds = model.predict(Triples,trainQ,trainP)
            train_acc = MultiAcc(trP,train_preds, model._path_size)
            train_true_acc = InSet(trainP,trainS,train_preds)

            val_preds = model.predict(Triples,validQ, validP)
            val_acc = MultiAcc(validP,val_preds,model._path_size)
            val_true_acc = InSet(validP,validS,val_preds)


            print('-----------------------')
            print('Epoch', t)
            print('Train Accuracy:', train_true_acc)
            print('Validation Accuracy:', val_true_acc)
            print('-----------------------')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--edim', default=50, type=int, help="words vector dimension [50]")
    parser.add_argument('--nhop', default=3, type=int, help="number of hops [2/3+1]")
    parser.add_argument('--batch_size', default=50, type = int, help = "batch size to use during training [50]")
    parser.add_argument('--nepoch', default=5000, type=int, help="number of epoch to use during training [1000]")
    parser.add_argument('--inner_nepoch', default=3, type=int, help="PRN inner loop [5]")
    parser.add_argument('--init_lr', default=0.001, type=float, help="initial learning rate")
    parser.add_argument('--epsilon', default=1e-8, type=float, help="Epsilon value for Adam Optimizer")
    parser.add_argument('--max_grad_norm', default=20, type=int, help="clip gradients to this norm [20]")
    parser.add_argument('--dataset', default="pq2h", type=str, help="pq/pql/wc/")
    parser.add_argument('--checkpoint_dir', default="checkpoint", type=str, help="checkpoint directory")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
