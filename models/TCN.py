from tensorflow.keras import Sequential, layers, activations
from tensorflow_addons.layers import WeightNormalization


class TCN(layers.Layer):
    def __init__(self, output_dim, kernel_size, dilations=None, dropout=0.0, residual=True):
        super(TCN, self).__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        self.tcns = [
            Sequential(
                [
                    WeightNormalization(layers.Conv1D(
                        output_dim, kernel_size, padding='causal', dilation_rate=dilation
                    )),
                    layers.Activation(activations.relu),
                    layers.Dropout(dropout)
                ]
            ) for dilation in dilations
        ]
        self.residual = residual

    def call(self, x):
        if self.residual:
            for tcn in self.tcns:
                x_res = tcn(x)
            x = x + x_res
        else:
            for tcn in self.tcns:
                x = tcn(x)
        return x
