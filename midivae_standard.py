import torch.nn as nn
from decoder_standard import Decoder
from encoder import Encoder


class MidiVAE(nn.Module):
    def __init__(self,
                 n_features,
                 n_hidden,
                 n_encoder_layers,
                 n_decoder_layers,
                 latent_dim,
                 batch_size,
                 dropout=0.5,
                 melody_length=48):

        super(MidiVAE, self).__init__()
        self.dropout = dropout
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.melody_length = melody_length

        self.encoder = Encoder(n_features, n_hidden, n_encoder_layers, latent_dim)

        self.decoder = Decoder(n_features, n_hidden, latent_dim, n_decoder_layers)

        self.decoding = nn.Sigmoid()

    def init_hidden(self, batch_size):
        return self.encoder.init_hidden(batch_size), self.decoder.init_hidden(batch_size)

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    def save(self, training=True):
        self.encoder.save(training)
        self.decoder.save(training)
        print(f'MidiVAE successfuly saved!')

    def load(self):
        self.encoder.load()
        self.decoder.load()
        print(f'MidiVAE successfuly loaded!')

    def forward(self, input_batch, encoder_hidden, decoder_hidden):
        """

        :param input_batch: 3D tensor of dim (batch_size, seq_len=48, n_midi_pitches_repr)
        :param encoder_hidden: tuple of tensor from encoder.init_hidden()
        :param decoder_hidden: tuple of tensor from decoder.init_hidden()
        :return:
        """
        z, mu, logvar, _ = self.encoder(input_batch, encoder_hidden)

        decoded, _ = self.decoder(z, decoder_hidden)

        out = self.decoding(decoded)

        return out, z, mu, logvar
