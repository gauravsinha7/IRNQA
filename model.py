import torch
import torch.nn as nn
from utils import position_encoding
import os
import numpy as np
from sklearn import metrics


class IRN(nn.Module):
    def __init__(self, args):
        super(IRN, self).__init__()
        self._margin = 4
        self._batch_size = args.batch_size
        self._vocab_size = args.nwords
        self._rel_size = args.nrels
        self._ent_size = args.nents
        self._sentence_size = args.query_size
        self._embedding_size = args.edim
        self._path_size = args.path_size
        self._hops = int(args.nhop)
        self._max_grad_norm = args.max_grad_norm
        self._name = "IRN"
        self._inner_epochs = args.inner_nepoch
        self._checkpoint_dir = args.checkpoint_dir + '/' + self._name
        self.build_vars()


    def forward(self, KBs, queries, answers, answers_id, paths):
        nexample = queries.shape[0]
        keys = np.repeat(np.reshape(np.arange(self._rel_size), [1, -1]),
                         nexample,
                         axis=0)
        pad = np.arange(nexample)
        ones = np.ones(nexample)
        zeros = np.zeros(nexample)

        loss = torch.Tensor(zeros).unsqueeze(1)
        s_index = torch.Tensor(paths[:, 0]).unsqueeze(1)
        q_emb = self.Q[torch.LongTensor(queries)]
        q = torch.sum(q_emb, dim=1)
        state = self.E[s_index.long()].squeeze(1)
        p = s_index
        for hop in range(self._hops):
            step = 2 * hop
            gate = torch.matmul(q, torch.matmul(self.R, self.Mrq).t())+torch.matmul(state, torch.matmul(self.R, self.Mrs).t())
            rel_logits = gate
            r_index = torch.argmax(rel_logits, dim=1)
            gate = torch.softmax(gate,dim=1)
            real_rel_onehot = torch.Tensor(paths[:, step + 1])

            predict_rel_onehot = torch.nn.functional.one_hot(r_index, num_classes=self._rel_size)
            state = state + torch.matmul(gate, torch.matmul(self.R, self.Mrs))
            critrion = nn.CrossEntropyLoss(reduce=False)
            loss += critrion(rel_logits, real_rel_onehot.long()).unsqueeze(1)

            q = q - torch.matmul(gate, torch.matmul(self.R, self.Mrq))
            value = torch.matmul(state, self.Mse)
            ans = torch.matmul(value, self.E.t())
            t_index = torch.argmax(ans, dim=1).float()
            r_index = r_index.float()
            t_index = r_index/(r_index+1e-15)*t_index + \
                (1-r_index/(r_index+1e-15)) * p[:, -1].float()

            p = torch.cat((p, r_index.float().view(-1, 1)), dim=1)
            p = torch.cat((p, t_index.float().view(-1, 1)), dim=1)

            real_ans_onehot = torch.Tensor(paths[:, step + 2])
            loss += critrion(ans, real_ans_onehot.long()).unsqueeze(1)

        loss = torch.sum(loss)
        self.E.data = self.E.data / (torch.pow(self.E.data, 2).sum(dim=1, keepdim=True))
        self.R.data = self.R.data / (torch.pow(self.R.data, 2).sum(dim=1, keepdim=True))
        self.Q.data = self.Q.data / (torch.pow(self.Q.data, 2).sum(dim=1, keepdim=True))
        return loss

    def build_vars(self):
        nil_word_slot = torch.zeros(1, self._embedding_size)
        nil_rel_slot = torch.zeros(1, self._embedding_size)
        '''
        self.E = nn.Parameter(
            torch.cat(
                (nil_word_slot,
                 nn.init.xavier_normal_(
                     torch.Tensor(self._ent_size - 1, self._embedding_size))),
                dim=0))
        self.Q = nn.Parameter(
            torch.cat((nil_word_slot,
                       nn.init.xavier_normal_(
                           torch.Tensor(self._vocab_size - 1,
                                        self._embedding_size))),
                      dim=0))
        self.R = nn.Parameter(
            torch.cat(
                (nil_rel_slot,
                 nn.init.xavier_normal_(
                     torch.Tensor(self._rel_size - 1, self._embedding_size))),
                dim=0))
        '''
        self.E = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self._ent_size, self._embedding_size)))
        self.Q = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self._vocab_size, self._embedding_size)))
        self.R = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self._rel_size, self._embedding_size)))

        self.Mrq = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(self._embedding_size, self._embedding_size)))
        self.Mrs = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(self._embedding_size, self._embedding_size)))
        self.Mse = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(self._embedding_size, self._embedding_size)))

        self._zeros = torch.zeros(1)

    def match(self):
        Similar = torch.matmul(torch.matmul(self.R, self.Mrq), self.Q.t())
        _, idx = torch.topk(Similar, 5)
        return idx
