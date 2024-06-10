import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from .TCN import TCN


class Encoder(Model):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dilations=None, mode='LSTM'):
        super(Encoder, self).__init__()
        self.encoder = Sequential()
        if mode == 'LSTM':
            self.encoder.add(layers.LSTM(output_dim, return_sequences=False))
        elif mode == 'GRU':
            self.encoder.add(layers.GRU(output_dim, return_sequences=False))
        elif mode == 'TCN':
            self.encoder.add(TCN(hidden_dim, kernel_size=2, dilations=dilations, residual=True))
            self.encoder.add(layers.Flatten())
            self.encoder.add(layers.Dense(output_dim))
        else:
            self.encoder.add(layers.Dense(output_dim, activation='relu'))
        self.encoder.add(layers.RepeatVector(input_dim))

    def call(self, x):
        x = self.encoder(x)
        return x


class Decoder(Model):
    def __init__(self, hidden_dim, output_dim, dilations=None, mode='LSTM'):
        super(Decoder, self).__init__()
        self.decoder = Sequential()
        if mode == 'LSTM':
            self.decoder.add(layers.LSTM(hidden_dim, return_sequences=True))
        elif mode == 'GRU':
            self.decoder.add(layers.GRU(hidden_dim, return_sequences=True))
        elif mode == 'TCN':
            self.decoder.add(TCN(hidden_dim, kernel_size=2, dilations=dilations, residual=False))
        else:
            self.decoder.add(layers.Dense(hidden_dim, activation='relu'))
        self.decoder.add(layers.Dense(hidden_dim, activation='relu'))
        self.decoder.add(layers.TimeDistributed(layers.Dense(output_dim)))

    def call(self, x):
        x = self.decoder(x)
        return x


class AE:
    def __init__(
            self, input_dim, z_dim, d_hidden_dim, e_hidden_dim=None, dilations=None, mode='LSTM',
            max_epochs=50, learning_rate=.001
    ):
        self.encoder = Encoder(input_dim, z_dim, e_hidden_dim, dilations, mode)
        self.decoder = Decoder(d_hidden_dim, 1, dilations, mode)
        # self.decoder = Decoder(d_hidden_dim, input_dim, dilations, mode)

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

    def fit(self, train_data, val_data=None):
        train_loss = []
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        loss_fn = tf.keras.losses.MeanSquaredError()

        train_time = 0
        train_times = []
        for epoch in range(1, self.max_epochs + 1):
            t_loss = []
            train_start = time.time()
            for x, y in train_data:
                with tf.GradientTape() as tape:
                    z = self.encoder(x)
                    preds = self.decoder(z)

                    # loss = tf.reduce_mean(tf.square(y - preds))
                    # print(y.shape, preds.shape)

                    loss = loss_fn(y, preds)

                grad = tape.gradient(loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
                optimizer.apply_gradients(
                    zip(grad, self.encoder.trainable_variables + self.decoder.trainable_variables))
                t_loss.append(loss.numpy())
            avg_t = sum(t_loss) / len(t_loss)
            train_loss.append(avg_t)
            tt = (time.time() - train_start)
            train_times.append(tt)
            train_time += tt
            print(f'epoch {epoch} train_loss: {avg_t:.4f} | {tt:.2f} sec')

            if val_data is not None:
                val_loss = []
                val_time = 0
                val_times = []
                v_loss = []
                val_start = time.time()
                for x, y in val_data:
                    z = self.encoder(x)
                    preds = self.decoder(z)

                    # loss = tf.reduce_mean(tf.square(y - preds))
                    loss = loss_fn(y, preds)

                    v_loss.append(loss.numpy())
                avg_v = sum(v_loss) / len(v_loss)
                val_loss.append((avg_v))
                tt = (time.time() - val_start)
                val_times.append(tt)
                val_time += tt
                print(f'epoch {epoch} val_loss: {avg_v:.4f} | {tt:.2f} sec')

        if val_data is not None:
            print(f'Train time: {train_time:.4f} | Validation time: {val_time:.4f}')
        else:
            print(f'Train time: {train_time:.4f}')

        history = {
            'train_loss': train_loss,
            'train_time': train_times
        }
        if val_data is not None:
            history['val_loss'] = val_loss
            history['val_time'] = val_times

        return history

    def predict(self, data):
        # reconstruct
        recons = []
        for x, y in data:
            z = self.encoder(x)
            preds = self.decoder(z)

            recons.extend(preds.numpy())

        return np.squeeze(np.array(recons))

    def save(self, e_path, d_path):
        self.encoder.save_weights(e_path)
        self.decoder.save_weights(d_path)

    def load(self, e_path, d_path):
        self.encoder.load_weights(e_path)
        self.decoder.load_weights(d_path)
