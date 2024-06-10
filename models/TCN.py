from keras import layers


class TCN(layers.Layer):
    def __init__(self, output_dim, kernel_size, dilations):
        super(TCN, self).__init__()
        self.tcns = [
            layers.Conv1D(output_dim, kernel_size, padding='causal', dilation_rate=dilation, activation='relu')
            for dilation in dilations
        ]

    def call(self, x):
        for tcn in self.tcns:
            x = tcn(x)
        return x
