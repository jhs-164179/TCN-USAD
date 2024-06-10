import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from models.v1.TCN import TCN


class Encoder(Model):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dilations=None, mode='LSTM'):
        super(Encoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        self.encoder = Sequential()
        if mode == 'LSTM':
            self.encoder.add(layers.LSTM(output_dim, return_sequences=False))
            # self.encoder.add(layers.RepeatVector(input_dim))
        elif mode == 'GRU':
            self.encoder.add(layers.GRU(output_dim, return_sequences=False))
            # self.encoder.add(layers.RepeatVector(input_dim))
        elif mode == 'TCN':
            # hidden_dims != list
            self.encoder.add(TCN(hidden_dims, kernel_size=2, dilations=dilations))
            self.encoder.add(layers.Flatten())
            self.encoder.add(layers.Dense(output_dim, activation='relu'))
            # self.encoder.add(layers.RepeatVector(input_dim))
        else:
            # hidden_dims == list
            for hidden_dim in hidden_dims:
                self.encoder.add(layers.Dense(hidden_dim, activation='relu'))
            self.encoder.add(layers.Dense(output_dim, activation='relu'))
            self._set_inputs(tf.TensorSpec([None, input_dim], tf.float32, name='input'))

    def call(self, x):
        x = self.encoder(x)
        return x


class Decoder(Model):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dilations=None, mode='LSTM'):
        super(Decoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [output_dim // 4, output_dim // 2]
        self.decoder = Sequential()
        if mode == 'LSTM':
            # hidden_dims != list
            self.decoder.add(layers.RepeatVector(input_dim))
            self.decoder.add(layers.LSTM(hidden_dims, return_sequences=True))
            self.decoder.add(layers.Dense(hidden_dims, activation='relu'))
            self.decoder.add(layers.TimeDistributed(layers.Dense(output_dim, activation='sigmoid')))
        elif mode == 'GRU':
            # hidden_dims != list
            self.decoder.add(layers.RepeatVector(input_dim))
            self.decoder.add(layers.GRU(hidden_dims, return_sequences=True))
            self.decoder.add(layers.Dense(hidden_dims, activation='relu'))
            self.decoder.add(layers.TimeDistributed(layers.Dense(output_dim, activation='sigmoid')))
        elif mode == 'TCN':
            # hidden_dims != list
            self.decoder.add(layers.RepeatVector(input_dim))
            self.decoder.add(TCN(hidden_dims, kernel_size=2, dilations=dilations, residual=False))
            self.decoder.add(layers.Dense(hidden_dims, activation='relu'))
            self.decoder.add(layers.TimeDistributed(layers.Dense(output_dim, activation='sigmoid')))
        else:
            # hidden_dims == list
            for hidden_dim in hidden_dims:
                self.decoder.add(layers.Dense(hidden_dim, activation='relu'))
            self.decoder.add(layers.Dense(output_dim, activation='sigmoid'))

    def call(self, x):
        x = self.decoder(x)
        return x


class USAD:
    def __init__(
            self, input_dim, z_dim, e_hidden_dims, d_hidden_dims, dilations=None, mode='LSTM',
            max_epochs=50, learning_rate=.001
    ):
        self.encoder = Encoder(input_dim, z_dim, e_hidden_dims, mode=mode)
        if mode == 'Dense':
            self.decoder_G = Decoder(input_dim, input_dim, d_hidden_dims, mode=mode)
            self.decoder_D = Decoder(input_dim, input_dim, d_hidden_dims, mode=mode)
        else:
            self.decoder_G = Decoder(input_dim, 1, d_hidden_dims, dilations=dilations, mode=mode)
            self.decoder_D = Decoder(input_dim, 1, d_hidden_dims, dilations=dilations, mode=mode)

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

    def fit(self, train_data, val_data=None):
        train_loss1 = []
        train_loss2 = []
        optimizer1 = tf.keras.optimizers.Adam(self.learning_rate)
        optimizer2 = tf.keras.optimizers.Adam(self.learning_rate)
        train_time = 0
        train_times = []
        for epoch in range(1, self.max_epochs + 1):
            t_loss1 = []
            t_loss2 = []
            train_start = time.time()
            for x, y in train_data:
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    z = self.encoder(x)
                    preds_G = self.decoder_G(z)
                    preds_D = self.decoder_D(z)
                    preds_GD = self.decoder_D(self.encoder(preds_G))

                    # print(y.shape, preds_G.shape, preds_D.shape, preds_GD.shape)

                    loss1 = (1 / epoch) * tf.reduce_mean((y - preds_G) ** 2) + (1 - 1 / epoch) * tf.reduce_mean(
                        (y - preds_GD) ** 2)
                    loss2 = (1 / epoch) * tf.reduce_mean((y - preds_D) ** 2) - (1 - 1 / epoch) * tf.reduce_mean(
                        (y - preds_GD) ** 2)
                grad1 = tape1.gradient(loss1, self.encoder.trainable_variables + self.decoder_G.trainable_variables)
                grad2 = tape2.gradient(loss2, self.encoder.trainable_variables + self.decoder_D.trainable_variables)
                optimizer1.apply_gradients(
                    zip(grad1, self.encoder.trainable_variables + self.decoder_G.trainable_variables))
                optimizer2.apply_gradients(
                    zip(grad2, self.encoder.trainable_variables + self.decoder_D.trainable_variables))
                t_loss1.append(loss1.numpy())
                t_loss2.append(loss2.numpy())
            avg_t1 = sum(t_loss1) / len(t_loss1)
            avg_t2 = sum(t_loss2) / len(t_loss2)
            train_loss1.append(avg_t1)
            train_loss2.append(avg_t2)
            tt = (time.time() - train_start)
            train_times.append(tt)
            train_time += tt
            print(f'epoch {epoch} train1_loss: {avg_t1:.4f} | train2_loss: {avg_t2:.4f} | {tt:.2f} sec')

            if val_data is not None:
                val_loss1 = []
                val_loss2 = []
                val_time = 0
                val_times = []
                v_loss1 = []
                v_loss2 = []
                val_start = time.time()
                for x, y in val_data:
                    z = self.encoder(x)
                    preds_G = self.decoder_G(z)
                    preds_D = self.decoder_D(z)
                    preds_GD = self.decoder_D(self.encoder(preds_G))
                    loss3 = (1 / epoch) * tf.reduce_mean((y - preds_G) ** 2) + (1 - 1 / epoch) * tf.reduce_mean(
                        (y - preds_GD) ** 2)
                    loss4 = (1 / epoch) * tf.reduce_mean((y - preds_D) ** 2) - (1 - 1 / epoch) * tf.reduce_mean(
                        (y - preds_GD) ** 2)
                    v_loss1.append(loss3.numpy())
                    v_loss2.append(loss4.numpy())
                avg_v1 = sum(v_loss1) / len(v_loss1)
                avg_v2 = sum(v_loss2) / len(v_loss2)
                val_loss1.append(avg_v1)
                val_loss2.append(avg_v2)
                tt = (time.time() - val_start)
                val_times.append(tt)
                val_time += tt
                print(f'epoch {epoch} val1_loss: {avg_v1:.4f} | val2_loss: {avg_v2:.4f} | {tt:.2f} sec')

        if val_data is not None:
            print(f'Train time: {train_time:.4f} | Validation time: {val_time:.4f}')
        else:
            print(f'Train time: {train_time:.4f}')

        history = {
            'train_loss1': train_loss1,
            'train_loss2': train_loss2,
            'train_time': train_times
        }
        if val_data is not None:
            history['val_loss1'] = val_loss1
            history['val_loss2'] = val_loss2
            history['val_time'] = val_times

        return history

    def predict(self, data, alpha=1., beta=0.):
        # anomaly score
        scores = []
        for x, y in data:
            z = self.encoder(x)
            preds_G = self.decoder_G(z)
            preds_D = self.decoder_D(z)
            preds_GD = self.decoder_D(self.encoder(preds_G))

            batch_scores = alpha * ((y - preds_G) ** 2) + beta * ((y - preds_GD) ** 2)
            scores.extend(batch_scores.numpy())

        return np.squeeze(np.array(scores))

    def reconstruct(self, data):
        # reconstruct
        recons_G = []
        recons_GD = []
        for x, y in data:
            z = self.encoder(x)
            preds_G = self.decoder_G(z)
            preds_D = self.decoder_D(z)
            preds_GD = self.decoder_D(self.encoder(preds_G))

            recons_G.extend(preds_G.numpy())
            recons_GD.extend(preds_GD.numpy())

        return np.squeeze(np.array(recons_G)), np.squeeze(np.array(recons_GD))

    def save(self, e_path, d_G_path, d_D_path):
        self.encoder.save_weights(e_path)
        self.decoder_G.save_weights(d_G_path)
        self.decoder_D.save_weights(d_D_path)

    def load(self, e_path, d_G_path, d_D_path):
        self.encoder.load_weights(e_path)
        self.decoder_G.load_weights(d_G_path)
        self.decoder_D.load_weights(d_D_path)
