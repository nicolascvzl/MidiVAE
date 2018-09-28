import torch
import torch.nn as nn
from pathlib import Path
import os

PACKAGE_DIR = Path(os.path.realpath(os.path.dirname(__file__)))
MODELS_DIR = PACKAGE_DIR / 'models'

class Encoder(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers, latent_dim, dropout=0.5, melody_length=48):
        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True, dropout=0.5, bidirectional=True)

        self.hidden_shrinking_layer = nn.Linear(n_hidden*2, n_hidden)
        self.time_shrinking_layer = nn.Linear(melody_length, 1)
        self.mu_decoder = nn.Linear(n_hidden, latent_dim)
        self.logvar_decoder= nn.Linear(n_hidden, latent_dim)

        self.init_weights()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.file_path = str(MODELS_DIR / 'encoder')
        self.latent_dim = latent_dim

    def __repr__(self):
        return (
            f'Encoder('
            f'num_layers={self.n_layers}, '
            f'hidden_dim={self.n_hidden}, '
            f'latent_dim={self.latent_dim}, '
            f'dropout_prob={self.dropout})'
        )

    def init_weights(self):
        init_range = 0.1
        self.hidden_shrinking_layer.bias.data.fill_(0)
        self.time_shrinking_layer.bias.data.fill_(0)
        self.mu_decoder.bias.data.fill_(0)
        self.logvar_decoder.bias.data.fill_(0)
        self.hidden_shrinking_layer.weight.data.uniform_(-init_range, init_range)
        self.time_shrinking_layer.weight.data.uniform_(-init_range, init_range)
        self.mu_decoder.weight.data.uniform_(-init_range, init_range)
        self.logvar_decoder.weight.data.uniform_(-init_range, init_range)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())

        return (weight.new_zeros(self.n_layers * 2, batch_size, self.n_hidden),
                weight.new_zeros(self.n_layers * 2, batch_size, self.n_hidden)
                )

    def save(self, training=True):
        if training:
            torch.save(self.state_dict(), self.file_path + self.__repr__())
        else:
            torch.save(self.state_dict(), self.file_path + self.__repr__() + '_trained')
        print('-' * 50)
        print('Encoder Model Saved!')
        print('-' * 50)

    def load(self):
        try:
            self.load_state_dict(torch.load(self.file_path,
                                            map_location=lambda storage, location: storage
                                            )
                                )
            print('-' * 50)
            print(f'Encoder Model {self.__repr__()} Loaded!)')
            print('-' * 50)
            return True
        except:
            return False

    def forward(self, input, hidden):
        """

        :param input: of size [batch_size, seq_len, n_features=128] of type torch.Tensor (for some dtype to define)
        :return: context of input sequence of size [batch_size, latente_dim]
        """

        lstm_output, hidden = self.lstm(input, hidden)

        lstm_output = torch.stack([self.hidden_shrinking_layer(lstm_output[i, : , :]) for i in range(lstm_output.size(0))])


        lstm_output = lstm_output.transpose(1, 2)


        pre_latent = torch.stack([self.time_shrinking_layer(lstm_output[i, :, :]) for i in range(lstm_output.size(0))])
        pre_latent = pre_latent.transpose(1, 2)
        pre_latent = pre_latent.squeeze()

        mu = self.mu_decoder(pre_latent)
        logvar = self.logvar_decoder(pre_latent)

        epsilon = torch.randn(mu.size())

        z = mu + epsilon.mul(torch.exp(0.5 * logvar))

        return z, mu, logvar, hidden
