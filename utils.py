import os
import math
import random
import numpy as np

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    m_i = sum_j l_ij*A*x_ij /J/d
    l_ij = Jd-jd-iJ+2ij  = ij-Ji/2-jd/2+Jd/4
    return l-matrix-transpose (fixed)
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = (1 + 4 * encoding / embedding_size / sentence_size) / 2
    return np.transpose(encoding)
