from keras import Model, Sequential, layers
from .TCN import TCN


class Encoder(Model):
    def __init__(self, z_dim, seq_len, hidden_dim, dilations=None, mode='LSTM'):
        super(Encoder, self).__init__()

        if mode == 'LSTM':
            # LSTM
            self.encoder = Sequential([
                layers.LSTM(z_dim, return_sequences=False),
            ])
        elif mode == 'GRU':
            # GRU
            self.encoder = Sequential([
                layers.GRU(z_dim, return_sequences=False),
            ])
        else:
            # TCN
            if dilations is None:
                dilations = [1, 2, 4, 8]
            self.encoder = Sequential([
                TCN(hidden_dim, kernel_size=5, dilations=dilations),
                layers.Flatten(),
                layers.Dense(z_dim),
            ])
        self.encoder.add(layers.RepeatVector(seq_len))

    def call(self, x):
        x = self.encoder(x)
        return x


class Decoder(Model):
    def __init__(self, hidden_dim, output_dim, dilations=None, mode='LSTM'):
        super(Decoder, self).__init__()

        if mode == 'LSTM':
            # LSTM
            self.decoder = Sequential([
                layers.LSTM(hidden_dim, return_sequences=True),
            ])
        elif mode == 'GRU':
            # GRU
            self.decoder = Sequential([
                layers.GRU(hidden_dim, return_sequences=True),
            ])
        else:
            # TCN
            if dilations is None:
                dilations = [1, 2, 4, 8]
            self.decoder = Sequential([
                TCN(hidden_dim, kernel_size=5, dilations=dilations),
            ])
        self.decoder.add(layers.Dense(hidden_dim, activation='relu'))
        self.decoder.add(layers.TimeDistributed(layers.Dense(output_dim)))

    def call(self, x):
        x = self.decoder(x)
        return x


class AE(Model):
    def __init__(self, z_dim=64, seq_len=24, hidden_dim=128, output_dim=1, dilations=None, mode='LSTM'):
        super(AE, self).__init__()

        if mode == 'TCN' and dilations is None:
            dilations = [1, 2, 4, 8]
        self.encoder = Encoder(z_dim, seq_len, hidden_dim, dilations, mode)
        self.decoder = Decoder(hidden_dim, output_dim, dilations, mode)

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
