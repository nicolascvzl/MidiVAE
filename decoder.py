import torch
import torch.nn as nn
from pathlib import Path
import os
from tools import *

PACKAGE_DIR = Path(os.path.realpath(os.path.dirname(__file__)))
MODELS_DIR = PACKAGE_DIR / 'models'


#TODO: implémenter les deux idées de décodeurs
#TODO: probleme pour méthode 2 on feed avec l'output: il faut override le forward du LSTM...?

class Decoder(nn.Module):
    def __init__(self, n_features, n_hidden, latent_dim, n_layers, dropout=0.5, melody_length=48):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.transition_layer = nn.Linear(latent_dim, latent_dim//2)
        self.transition_activation = nn.Tanh()

        self.lstm = nn.LSTM(input_size=latent_dim//2, hidden_size=n_features, num_layers=n_layers, batch_first=True, dropout=0.5, bidirectional=False)

        self.init_weights()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.file_path = str(MODELS_DIR / 'decoder')
        self.latent_dim = latent_dim
        self.melody_length = melody_length

# TODO : look doc of LSTM weights and memory cells are of dim (n_layers * n_directions, batch, hidden_size) (is it still true if batch_first = True ?)


    def __repr__(self):
        return (
            f'Decoder('
            f'num_layers={self.n_layers}, '
            f'latent_dim={self.latent_dim}, '
            f'dropout_prob={self.dropout})'
        )

    def init_weights(self):
        init_range = 0.1

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param, init_range)



    def init_hidden(self, batch_size):
        weight = next(self.parameters())

        return (weight.new_zeros(self.n_layers, batch_size, self.n_features),
                weight.new_zeros(self.n_layers, batch_size, self.n_features)
                )


    def save(self, training=True):
        if training:
            torch.save(self.state_dict(), self.file_path + self.__repr__())
        else:
            torch.save(self.state_dict(), self.file_path + self.__repr__() + '_trained')
        print('-' * 50)
        print('Decoder Model Saved!')
        print('-' * 50)


    def load(self):
        try:
            self.load_state_dict(torch.load(self.file_path,
                                            map_location=lambda storage, location: storage
                                            )
                                )
            print('-' * 50)
            print(f'Decoder Model {self.__repr__()} Loaded!)')
            print('-' * 50)
            return True
        except:
            return False


    def forward(self, z, hidden):

        if len(z.size()) == 1:
            z = torch.unsqueeze(z, dim=0)
        fake_sequence = torch.stack([z for i in range(self.melody_length)])

        lstm_input = torch.stack([self.transition_layer(fake_sequence[i, :, :]) for i in range(fake_sequence.size(0))])

        lstm_input = self.transition_activation(lstm_input)

        lstm_input = lstm_input.transpose(0, 1)

        lstm_output, hidden = self.lstm(lstm_input, hidden)

        return lstm_output, hidden

