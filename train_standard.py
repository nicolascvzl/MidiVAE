from model_manager_standard import *
from midivae_standard import MidiVAE

SRC_DATA = os.path.join(os.getcwd(), 'data')
PR_DATA = 'pr_100k.pkl'


def train():
    print(f'Loading dataset...')
    dataset = try_to_load_as_pickled_object_or_None(os.path.join(SRC_DATA, PR_DATA))
    if dataset is None:
        print('Unable to load heavy dataset, exiting...')
        raise ValueError
    print('Done!')

    n_features = 72
    n_hidden = 500
    latent_dim = 500
    n_encoder_layers = 1
    n_decoder_layers = 2
    batch_size = 500
    beta = 100

    print('Creating a MidiVAE...', end='', flush=True)
    midivae = MidiVAE(n_features=n_features,
                      n_hidden=n_hidden,
                      n_encoder_layers=n_encoder_layers,
                      n_decoder_layers=n_decoder_layers,
                      latent_dim=latent_dim,
                      batch_size=batch_size)
    print('Done.')
    print('-' * 50)

    batches_per_epoch = 20
    print('Defining a model manager...')
    model_manager = ModelManager(model=midivae,
                                 dataset=dataset,
                                 beta=beta,
                                 loss_name='BCE',
                                 optimizer_name='adam',
                                 batches_per_epoch=batches_per_epoch,
                                 lr=1e-3
                                 )
    print('Done.')
    print('-' * 50)

    model_manager.train_model(batch_size=batch_size,
                              num_epochs=100,
                              )


if __name__ == '__main__':
    train()
