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
