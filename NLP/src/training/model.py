import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn


class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        """
        param mode: rnn type, 'rnn_relu', 'rnn_tanh', 'gru' or 'lstm'
        param vocab_size: vocabulary size
        param num_embed: embedding dimension
        param num_hidden: hidden dimension
        param num_layers: number of stacked RNN layers
        param dropout: between 0 and 1, dropout probability
        param tie_weights: tie weights on input and output embedding layers
        """
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.tie_weights = tie_weights
            self.num_hidden = num_hidden
            self.num_embed = num_embed
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer=mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru" % mode)
            if self.tie_weights:
                self.dim_mapping = nn.Dense(num_embed, in_units=num_hidden)
                self.decoder = nn.Dense(vocab_size, in_units=num_embed,
                                        params=self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units=num_hidden)

    def forward(self, inputs, hidden):
        """
        Forward pass of the network using tensors of input and hidden layers
        """
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        if self.tie_weights:
            output = self.dim_mapping(output.reshape((-1, self.num_hidden)))
        output = self.drop(output)
        if self.tie_weights:
            decoded = self.decoder(output)
        else:
            decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        """
        Returns the begin state of the network
        """
        return self.rnn.begin_state(*args, **kwargs)
