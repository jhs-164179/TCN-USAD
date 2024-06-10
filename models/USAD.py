import time
import numpy as np
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


class USAD(Model):
    def __init__(self, seq_len=24, z_dim=64, output_dim=1, hidden_dims=None, dilations=None, mode='USAD'):
        super(USAD, self).__init__()

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
            # TCN-USAD | hidden_dims : int
            self.decoder_G = Decoder(output_dim, seq_len, hidden_dims, dilations, mode)
            self.decoder_D = Decoder(output_dim, seq_len, hidden_dims, dilations, mode)
        
        self.g_loss_tracker = metrics.Mean(name="loss1(g)")
        self.d_loss_tracker = metrics.Mean(name="loss2(d)")
    
    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker]

    def compile(self, g_optimizer, d_optimizer):
        super(USAD, self).compile()

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        # self.loss1_fn = Loss1()
        # self.loss2_fn = Loss2()

    def call(self, x):
        z = self.encoder(x)
        preds_G = self.decoder_G(z)
        preds_D = self.decoder_D(z)
        preds_GD = self.decoder_D(self.encoder(preds_G))
        return preds_G, preds_D, preds_GD

    def compute_loss(self, y_true, preds_G, preds_D, preds_GD, epoch):
        loss1 = (1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_G)) + (1 - 1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_GD))
        loss2 = (1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_D)) - (1 - 1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_GD))
        return loss1, loss2
    
    def train_step(self, data, epoch):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            preds_G, preds_D, preds_GD = self(x)

            loss1, loss2 = self.compute_loss(y, preds_G, preds_D, preds_GD, epoch)
            # loss1 = self.loss1_fn(y, preds_G, preds_GD, epoch)
            # loss2 = self.loss2_fn(y, preds_D, preds_GD, epoch)

        grad1 = tape.gradient(loss1, self.encoder.trainable_variables + self.decoder_G.trainable_variables)
        grad2 = tape.gradient(loss2, self.encoder.trainable_variables + self.decoder_D.trainable_variables)

        self.g_optimizer.apply_gradients(zip(grad1, self.encoder.trainable_variables + self.decoder_G.trainable_variables))
        self.d_optimizer.apply_gradients(zip(grad2, self.encoder.trainable_variables + self.decoder_D.trainable_variables))

        del tape

        self.g_loss_tracker.update_state(loss1)
        self.d_loss_tracker.update_state(loss2)
        
        return {"loss1(g)": self.g_loss_tracker.result(), "loss2(d)": self.d_loss_tracker.result()}
    
    def fit(self, train_dataset, val_dataset=None, epochs=50):
        history = {
            "loss1(g)": [],
            "loss2(d)": [],
        }
        if val_dataset is not None:
            history["val_loss1(g)"] = []
            history["val_loss2(d)"] = []

        times = []
        for epoch in range(1, epochs + 1):
            start = time.time()
            epoch_gen_loss = []
            epoch_disc_loss = []
            for step, data in enumerate(train_dataset):
                metrics = self.train_step(data, epoch)
                epoch_gen_loss.append(metrics['loss1(g)'].numpy())
                epoch_disc_loss.append(metrics['loss2(d)'].numpy())

            epoch_time = time.time() - start
            
            avg_gen_loss = np.mean(epoch_gen_loss)
            avg_disc_loss = np.mean(epoch_disc_loss)
            history["loss1(g)"].append(avg_gen_loss)
            history["loss2(d)"].append(avg_disc_loss)

            print(f"Epoch {epoch}/{epochs}, G Loss: {avg_gen_loss:.6f}, D Loss: {avg_disc_loss:.6f} | Time: {epoch_time:.2f} sec")
            
            if val_dataset is not None:
                val_gen_loss = []
                val_disc_loss = []
                for step, data in enumerate(val_dataset):
                    preds_G, preds_D, preds_GD = self(data[0])
                    val_loss1, val_loss2 = self.compute_loss(data[1], preds_G, preds_D, preds_GD, epoch)
                    # val_loss1 = self.loss1_fn(data[1], preds_G, preds_GD, epoch)
                    # val_loss2 = self.loss2_fn(data[1], preds_D, preds_GD, epoch)

                    val_gen_loss.append(val_loss1.numpy())
                    val_disc_loss.append(val_loss2.numpy())

                avg_val_gen_loss = np.mean(val_gen_loss)
                avg_val_disc_loss = np.mean(val_disc_loss)
                history["val_loss1(g)"].append(avg_val_gen_loss)
                history["val_loss2(d)"].append(avg_val_disc_loss)

                print(f"Epoch {epoch}/{epochs}, val_G Loss: {avg_val_gen_loss:.6f}, val_D Loss: {avg_val_disc_loss:.6f}")
                
            times.append(time.time() - start)
        
        return times, history


class Loss1(losses.Loss):
    def __init__(self, name="loss1(g)", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, preds_G, preds_GD, epoch):
        loss = (1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_G)) + (1 - 1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_GD))
        return loss
    

class Loss2(losses.Loss):
    def __init__(self, name="loss2(d)", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, preds_D, preds_GD, epoch):
        loss = (1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_D)) - (1 - 1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_GD))
        return loss