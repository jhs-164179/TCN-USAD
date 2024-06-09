from tensorflow.keras import Model, Sequential, layers
import tensorflow as tf
import time


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

    def call(self, x, training=False, mask=False):
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

    def call(self, x, training=False, mask=False):
        x = self.decoder(x)
        return x


class USAD:
    def __init__(
            self, input_dim, z_dim, e_hidden_dims, d_hidden_dims,
            max_epochs, learning_rate=.001
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
        for epoch in range(1, self.max_epochs + 1):
            train_start = time.time()
            for x, y in train_data:
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    z = self.encoder(x)
                    preds_G = self.decoder_G(z, training=True)
                    preds_D = self.decoder_D(z, training=True)
                    preds_GD = self.decoder_D(self.encoder(preds_G), training=True)
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

                train_loss1.append(loss1.numpy())
                train_loss2.append(loss2.numpy())

                tt = (time.time() - train_start)
                train_time += tt
            print(f'epoch {epoch} train1_loss: {loss1:.4f} | train2_loss: {loss2:.4f} | {tt:.2f} sec')

            if val_data is not None:
                val_loss1 = []
                val_loss2 = []

                val_time = 0
                val_start = time.time()
                for x, y in val_data:
                    z = self.encoder(x)
                    preds_G = self.decoder_G(z, training=False)
                    preds_D = self.decoder_D(z, training=False)
                    preds_GD = self.decoder_D(self.encoder(preds_G), training=False)
                    loss3 = (1 / epoch) * tf.reduce_mean((y - preds_G) ** 2) + (1 - 1 / epoch) * tf.reduce_mean(
                        (y - preds_GD) ** 2)
                    loss4 = (1 / epoch) * tf.reduce_mean((y - preds_D) ** 2) - (1 - 1 / epoch) * tf.reduce_mean(
                        (y - preds_GD) ** 2)

                    val_loss1.append(loss3.numpy())
                    val_loss2.append(loss4.numpy())

                    tt = (time.time() - val_start)
                    val_time += tt
                print(f'epoch {epoch} val1_loss: {loss3:.4f} | val2_loss: {loss4:.4f} | {tt:.2f} sec')

        if val_data is None:
            history = {
                'train_loss1': train_loss1,
                'train_loss2': train_loss2,
                'train_time': train_time
            }
        else:
            history = {
                'train_loss1': train_loss1,
                'train_loss2': train_loss2,
                'val_loss1': val_loss1,
                'val_loss2': val_loss2,
                'train_time': train_time,
                'valid_time': val_time,
            }
        return history
