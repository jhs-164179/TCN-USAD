import time
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from .TCN import TCN


class Encoder(Model):
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super(Encoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        self.encoder = Sequential()
        for hidden_dim in hidden_dims:
            self.encoder.add(layers.Dense(hidden_dim, activation='relu'))
        self.encoder.add(layers.Dense(output_dim, activation='relu'))
        self._set_inputs(tf.TensorSpec([None, input_dim], tf.float32, name='input'))

    def call(self, x):
        x = self.encoder(x)
        return x


class Decoder(Model):
    def __init__(self, output_dim, hidden_dims=None):
        super(Decoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [output_dim // 4, output_dim // 2]
        self.decoder = Sequential()
        for hidden_dim in hidden_dims:
            self.decoder.add(layers.Dense(hidden_dim, activation='relu'))
        self.decoder.add(layers.Dense(output_dim, activation='sigmoid'))

    def call(self, x):
        x = self.decoder(x)
        return x


class USAD:
    def __init__(
            self, input_dim, z_dim, e_hidden_dims, d_hidden_dims,
            max_epochs=50, learning_rate=.001
    ):
        self.encoder = Encoder(input_dim, z_dim, e_hidden_dims)
        self.decoder_G = Decoder(input_dim, d_hidden_dims)
        self.decoder_D = Decoder(input_dim, d_hidden_dims)

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
