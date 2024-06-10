import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
import numpy as np
import time

class Encoder(Model):
    def __init__(self, input_dim, z_dim, hidden_dims=None, mode='Dense'):
        super(Encoder, self).__init__()
        if mode == 'Dense':
            if hidden_dims is None:
                hidden_dims = [input_dim // 2, input_dim // 4]
            self.encoder = Sequential([
                layers.Dense(hidden_dim, activation='relu')
                for hidden_dim in hidden_dims
            ])
            self.encoder.add(layers.Dense(z_dim, activation='relu'))
        else:
            raise ValueError("Unsupported mode. Only 'Dense' mode is supported in this example.")

    def call(self, x):
        return self.encoder(x)

class Decoder(Model):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dilations=None, mode='Dense'):
        super(Decoder, self).__init__()
        if mode == 'Dense':
            if hidden_dims is None:
                hidden_dims = [input_dim // 4, input_dim // 2]
            self.decoder = Sequential([
                layers.Dense(hidden_dim, activation='relu')
                for hidden_dim in hidden_dims
            ])
            self.decoder.add(layers.Dense(output_dim, activation='sigmoid'))
        else:
            raise ValueError("Unsupported mode. Only 'Dense' mode is supported in this example.")

    def call(self, x):
        return self.decoder(x)

class Loss1(tf.keras.losses.Loss):
    def __init__(self, name="loss1", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, preds_G, preds_GD, epoch):
        return (1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_G)) + (1 - 1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_GD))

class Loss2(tf.keras.losses.Loss):
    def __init__(self, name="loss2", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, preds_D, preds_GD, epoch):
        return (1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_D)) - (1 - 1 / epoch) * tf.reduce_mean(tf.square(y_true - preds_GD))

class USAD(Model):
    def __init__(self, input_dim, z_dim, e_hidden_dims, d_hidden_dims, dilations=None, mode='Dense', max_epochs=50, learning_rate=.001):
        super(USAD, self).__init__()
        self.encoder = Encoder(input_dim, z_dim, e_hidden_dims, mode=mode)
        self.decoder_G = Decoder(z_dim, input_dim, d_hidden_dims, mode=mode)
        self.decoder_D = Decoder(z_dim, input_dim, d_hidden_dims, mode=mode)
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, g_optimizer, d_optimizer):
        super(USAD, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def call(self, inputs):
        z = self.encoder(inputs)
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

        grad1 = tape.gradient(loss1, self.encoder.trainable_variables + self.decoder_G.trainable_variables)
        grad2 = tape.gradient(loss2, self.encoder.trainable_variables + self.decoder_D.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grad1, self.encoder.trainable_variables + self.decoder_G.trainable_variables))
        self.d_optimizer.apply_gradients(zip(grad2, self.encoder.trainable_variables + self.decoder_D.trainable_variables))
        
        del tape  # Delete the tape explicitly to free up resources

        self.gen_loss_tracker.update_state(loss1)
        self.disc_loss_tracker.update_state(loss2)
        
        return {"generator_loss": self.gen_loss_tracker.result(), "discriminator_loss": self.disc_loss_tracker.result()}

    def fit(self, train_dataset, val_dataset=None):
        history = {
            "generator_loss": [],
            "discriminator_loss": [],
        }
        if val_dataset is not None:
            history["val_generator_loss"] = []
            history["val_discriminator_loss"] = []

        for epoch in range(1, self.max_epochs + 1):
            print(f"Epoch {epoch}/{self.max_epochs}")
            epoch_start_time = time.time()

            epoch_gen_loss = []
            epoch_disc_loss = []
            for step, data in enumerate(train_dataset):
                metrics = self.train_step(data, epoch)
                epoch_gen_loss.append(metrics['generator_loss'].numpy())
                epoch_disc_loss.append(metrics['discriminator_loss'].numpy())

            avg_gen_loss = np.mean(epoch_gen_loss)
            avg_disc_loss = np.mean(epoch_disc_loss)
            history["generator_loss"].append(avg_gen_loss)
            history["discriminator_loss"].append(avg_disc_loss)

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}, Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}, Time: {epoch_time:.2f} sec")
            
            if val_dataset is not None:
                val_gen_loss = []
                val_disc_loss = []
                for step, data in enumerate(val_dataset):
                    preds_G, preds_D, preds_GD = self(data[0])
                    val_loss1, val_loss2 = self.compute_loss(data[1], preds_G, preds_D, preds_GD, epoch)
                    val_gen_loss.append(val_loss1.numpy())
                    val_disc_loss.append(val_loss2.numpy())
                
                avg_val_gen_loss = np.mean(val_gen_loss)
                avg_val_disc_loss = np.mean(val_disc_loss)
                history["val_generator_loss"].append(avg_val_gen_loss)
                history["val_discriminator_loss"].append(avg_val_disc_loss)
                print(f"Validation - Generator Loss: {avg_val_gen_loss:.4f}, Discriminator Loss: {avg_val_disc_loss:.4f}")
        
        return history

    def predict(self, data, alpha=1., beta=0.):
        scores = []
        for x, y in data:
            preds_G, preds_D, preds_GD = self(x)
            batch_scores = alpha * ((y - preds_G) ** 2) + beta * ((y - preds_GD) ** 2)
            scores.extend(batch_scores.numpy())
        return np.squeeze(np.array(scores))

    def reconstruct(self, data):
        recons_G = []
        recons_GD = []
        for x, y in data:
            preds_G, preds_D, preds_GD = self(x)
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
