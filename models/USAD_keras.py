import tensorflow as tf
from keras import Model, Sequential, layers, metrics, losses
from .TCN import TCN


class Encoder(Model):
    def __init__(self, z_dim, seq_len, hidden_dims=None, dilations=None, mode='USAD'):
        super(Encoder, self).__init__()
        
        if mode == 'USAD':
            # USAD(MLP) | hidden_dims : list
            if hidden_dims is None:
                hidden_dims = [seq_len // 2, seq_len // 4]
            self.encoder = Sequential([
                layers.Dense(hidden_dim, activation='relu')
                for hidden_dim in hidden_dims
            ])
            self.encoder.add(layers.Dense(z_dim, activation='relu'))
        elif mode == 'LSTM':
            # LSTM-USAD
            self.encoder = Sequential([
                layers.LSTM(z_dim, return_sequences=False),
                layers.RepeatVector(seq_len)
            ])
        elif mode == 'GRU':
            # GRU-USAD
            self.encoder = Sequential([
                layers.GRU(z_dim, return_sequences=False),
                layers.RepeatVector(seq_len)
            ])
        else:
            # TCN-USAD | hidden_dims : int
            if dilations is None:
                dilations = [1, 2, 4, 8]
            self.encoder = Sequential([
                TCN(hidden_dims, kernel_size=5, dilations=dilations),
                layers.Flatten(),
                layers.Dense(z_dim),
                layers.RepeatVector(seq_len)
            ])

    def call(self, x):
        x = self.encoder(x)
        return x
    

class Decoder(Model):
    def __init__(self, output_dim, seq_len=24, hidden_dims=None, dilations=None, mode='USAD'):
        super(Decoder, self).__init__()

        if mode == 'USAD':
            # USAD(MLP) | hidden_dims : list
            if hidden_dims is None:
                hidden_dims = [seq_len // 4, seq_len // 2]
            self.decoder = Sequential([
                layers.Dense(hidden_dim, activation='relu')
                for hidden_dim in hidden_dims
            ])
            self.decoder.add(layers.Dense(output_dim, activation='sigmoid'))
        elif mode == 'LSTM':
            # LSTM-USAD
            self.decoder = Sequential([
                layers.LSTM(hidden_dims, return_sequences=True),
                layers.Dense(hidden_dims, activation='relu'),
                layers.TimeDistributed(layers.Dense(output_dim, activation='sigmoid'))
            ])
        elif mode == 'GRU':
            # GRU-USAD
            self.decoder = Sequential([
                layers.GRU(hidden_dims, return_sequences=True),
                layers.Dense(hidden_dims, activation='relu'),
                layers.TimeDistributed(layers.Dense(output_dim, activation='sigmoid'))
            ])
        else:
            # TCN-USAD | hidden_dims : int
            if dilations is None:
                dilations = [1, 2, 4, 8]
            self.decoder = Sequential([
                TCN(hidden_dims, kernel_size=5, dilations=dilations),
                layers.Dense(hidden_dims, activation='relu'),
                layers.TimeDistributed(layers.Dense(output_dim, activation='sigmoid'))
            ])

    def call(self, x):
        x = self.decoder(x)
        return x
    

class USAD_keras(Model):
    def __init__(self, seq_len=24, z_dim=64, output_dim=1, hidden_dims=None, dilations=None, mode='USAD'):
        super(USAD_keras, self).__init__()

        if mode == 'USAD' and hidden_dims is None:
            hidden_dims = [seq_len // 2, seq_len // 4]
        if mode == 'TCN' and dilations is None:
            dilations = [1, 2, 4, 8]
        
        self.encoder = Encoder(z_dim, seq_len, hidden_dims, dilations, mode)
        if mode == 'USAD':
            # USAD(MLP) | hidden_dims : list
            self.decoder_G = Decoder(output_dim, seq_len, hidden_dims[::-1], dilations, mode)
            self.decoder_D = Decoder(output_dim, seq_len, hidden_dims[::-1], dilations, mode)
        else:
            # (LSTM, GRU, TCN)-USAD | hidden_dims : int
            self.decoder_G = Decoder(output_dim, seq_len, hidden_dims, dilations, mode)
            self.decoder_D = Decoder(output_dim, seq_len, hidden_dims, dilations, mode)
        
        self.g_loss_tracker = metrics.Mean(name="loss1(g)")
        self.d_loss_tracker = metrics.Mean(name="loss2(d)")
        self.val_g_loss_tracker = metrics.Mean(name="val_loss1(g)")
        self.val_d_loss_tracker = metrics.Mean(name="val_loss2(d)")

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker]
    
    def compile(self, g_optimizer, d_optimizer):
        super(USAD_keras, self).compile()

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss = USAD_Loss()
        self.d_loss = USAD_Loss()

    def call(self, x):
        z = self.encoder(x)
        preds_G = self.decoder_G(z)
        preds_D = self.decoder_D(z)
        preds_GD = self.decoder_D(self.encoder(preds_G))
        return preds_G, preds_D, preds_GD

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            # z = self.encoder(x)
            # preds_G = self.decoder_G(z)
            # preds_D = self.decoder_D(z)
            # preds_GD = self.decoder_D(self.encoder(preds_G))
            preds_G, preds_D, preds_GD = self(x, training=True)

            g_loss = self.g_loss.call(y, preds_G, preds_GD)
            d_loss = self.d_loss.call(y, preds_D, preds_GD)

        grad_G = tape.gradient(g_loss, self.encoder.trainable_variables + self.decoder_G.trainable_variables)
        grad_D = tape.gradient(d_loss, self.encoder.trainable_variables + self.decoder_D.trainable_variables)

        self.g_optimizer.apply_gradients(zip(grad_G, self.encoder.trainable_variables + self.decoder_G.trainable_variables))
        self.d_optimizer.apply_gradients(zip(grad_D, self.encoder.trainable_variables + self.decoder_D.trainable_variables))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {
            'g_loss': self.g_loss_tracker.result(),
            'd_loss': self.d_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x, y = data

        preds_G, preds_D, preds_GD = self(x, training=False)

        g_loss = self.g_loss.call(y, preds_G, preds_GD)
        d_loss = self.d_loss.call(y, preds_D, preds_GD)

        self.val_g_loss_tracker.update_state(g_loss)
        self.val_d_loss_tracker.update_state(d_loss)
        # return {
        #     'val_g_loss': self.val_g_loss_tracker.result(),
        #     'val_d_loss': self.val_d_loss_tracker.result(),
        # }
        return {
            'g_loss': self.val_g_loss_tracker.result(),
            'd_loss': self.val_d_loss_tracker.result(),
        }


class USAD_Loss(losses.Loss):
    def __init__(self):
        super(USAD_Loss, self).__init__()

    def call(self, y_true, y_pred, preds_GD, alpha=.5, beta=.5):
        a_mse = tf.reduce_mean(tf.square(y_true - y_pred))
        b_mse = tf.reduce_mean(tf.square(y_true - preds_GD))
        loss = alpha * a_mse + beta * b_mse        
        return loss
    