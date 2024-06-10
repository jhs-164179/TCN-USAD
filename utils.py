import os
import time
import numpy as np
import tensorflow as tf
from keras import losses
from keras.callbacks import Callback


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


def make_dataset(x, y, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(
        batch_size=batch_size
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


class TimeHistory(Callback):
    def __init__(self):
        super(TimeHistory, self).__init__()
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start)


