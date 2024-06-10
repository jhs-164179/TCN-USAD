import os
import numpy as np


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_seq(data, seq_len):
    seqs = [
        data[i:(i + seq_len)]
        for i in range(len(data) - seq_len + 1)
    ]
    x = np.array([seq[:seq_len] for seq in seqs], dtype='float32')

    return x


def minmax(data, min_val=None, max_val=None, norm=True):
    if norm:
        min_val = np.min(data)
        max_val = np.max(data)
        result = (data - min_val) / (max_val - min_val)
    else:
        result = data * (max_val - min_val) + min_val
    return result
