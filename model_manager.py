from itertools import islice
import torch.nn.utils as utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from midivae import MidiVAE
from optimizers import optimizer_from_name
from loss import loss_from_name
from tools import *
from PIL import Image


CUDA_ENABLED = torch.cuda.is_available()


class ModelManager:

    def __init__(self, model: MidiVAE, dataset, beta, batches_per_epoch=100, loss_name: str='BCE', optimizer_name: str='adam', lr=1e-3):
        self.model = model
        if CUDA_ENABLED:
            print('Moving the model to GPU...', end='', flush=True)
            self.model.cuda()
            print('Done.')

        self.optimizer_name = optimizer_name
        self.optimizer = optimizer_from_name(optimizer_name, lr=lr)(self.model.parameters())
        self.lr = lr
        self.beta = beta
        self.loss = loss_from_name(loss_name)
        self.dataset = dataset
        self.batches_per_epoch = batches_per_epoch

        with open(os.path.join(os.getcwd(), 'test_data/piano_roll.pkl'), 'rb') as f:
            samples_list = pickle.load(f)
        samples_list = [wrap_cuda(torch.from_numpy(cut(np.transpose(data.astype('int32')))).type('torch.FloatTensor')) for data in samples_list]
        self.samples_list = samples_list

    def load_if_saved(self):
        self.model.load()

    def save(self):
        self.model.save()

    def sample_update(self, samples_list):
        for data in samples_list:
            input_batch = wrap_cuda(torch.unsqueeze(data, dim=0))

            encoder_hidden, decoder_hidden = self.model.init_hidden(batch_size=1)
            encoder_hidden = wrap_cuda(encoder_hidden)
            decoder_hidden = wrap_cuda(decoder_hidden)

            self.optimizer.zero_grad()
            output, _, mu, logvar = self.model(input_batch, encoder_hidden, decoder_hidden)
            loss, _, _ = self.mean_loss(output, input_batch, mu, logvar)
            loss.backward()
            utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

    def loss_and_acc_on_epoch(self, batch_size, batches_per_epoch, data_loader, train=True):
        mean_loss = 0.
        mean_r_loss = 0.
        mean_kld_loss = 0.
        mean_notes_accuracy = 0.

        for input_batch in tqdm(islice(data_loader, batches_per_epoch)):
            input_batch = wrap_cuda(input_batch)

            encoder_hidden , decoder_hidden = self.model.init_hidden(batch_size=batch_size)
            encoder_hidden = wrap_cuda(encoder_hidden)
            decoder_hidden = wrap_cuda(decoder_hidden)

            self.optimizer.zero_grad()
            output, _, mu, logvar = self.model(input_batch, encoder_hidden, decoder_hidden)

            loss, r_loss, kld_loss = self.mean_loss(output, input_batch, mu, logvar)
            if train:
                loss.backward()
                utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()

            notes_accuracy = self.accuracy(output=output, input_time_slice=input_batch)

            mean_loss += loss
            mean_r_loss += r_loss
            mean_kld_loss += kld_loss
            mean_notes_accuracy += notes_accuracy

        return mean_loss / batches_per_epoch, mean_r_loss / batches_per_epoch, mean_kld_loss / batches_per_epoch, mean_notes_accuracy / batches_per_epoch

    def mean_loss(self, output, input_time_slice, mu, logvar):

        assert output.size() == input_time_slice.size()

        r_loss = 0.
        for i in range(input_time_slice.size(0)):
            r_loss += self.loss(output[i, :, :], input_time_slice[i, :, :])

        r_loss = r_loss / input_time_slice.size(0)

        if input_time_slice.size(0) == 1:
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1)).mean().squeeze()
        else:
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()

        loss = r_loss + self.beta * kld

        return loss, r_loss, kld

    def accuracy(self, output, input_time_slice):
        decoded_output = wrap_cuda((output >= 0.5).type('torch.FloatTensor'))
        acc = (decoded_output == input_time_slice).sum().item() / (output.size()[0] * output.size()[1] * output.size()[2])
        return acc

    def prepare_data(self, batch_size, test_batch_size, **kwargs):

        class Subset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __getitem__(self, index):
                return self.dataset[self.indices[index]]

            def __len__(self):
                return len(self.indices)

        def random_split(dataset, lengths):
            assert sum(lengths) == len(dataset)
            indices = torch.randperm(sum(lengths))
            indices = wrap_cuda(indices)
            list_ = [None] * 3
            offset = 0
            for i, length in enumerate(lengths):
                offset += length
                list_[i] = Subset(dataset, indices[offset - length : offset])
            return list_[0], list_[1], list_[2]

        def collate_batch(batch):
            """

            :param batch: a list of 2D numpy arrays (seq_len, n_midi_pitches=128)
            :return: data: a torch.Tensor of dim (batch_size = len(batch), seq_len, n_midi_pitches_cut=72) -- see tools.cut() for more info on cut pitches
            """
            data = [torch.from_numpy(cut(np.transpose(time_slice.astype('int32')))) for time_slice in batch]
            data = torch.stack(data)
            data = data.float()
            data = wrap_cuda(data)
            return data

        num_melodies = len(self.dataset)
        a = int(85 * num_melodies / 100)
        b = int(10 * num_melodies / 100)
        c = num_melodies - (a + b)

        lengths = [a, b, c]
        train_dataset, validation_dataset, test_dataset = random_split(self.dataset, lengths)

        assert batch_size * self.batches_per_epoch <= len(train_dataset)
        assert test_batch_size * self.batches_per_epoch <= len(validation_dataset)
        assert test_batch_size * self.batches_per_epoch <= len(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch, **kwargs)
        validation_loader = DataLoader(validation_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, collate_fn=collate_batch, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch, **kwargs)

        return train_loader, validation_loader, test_loader

    def train_model(self, batch_size, num_epochs):
        train_loader, validation_loader, test_loader = self.prepare_data(batch_size, test_batch_size=batch_size//2)
        best_loss = 1e3
        train_r_losses = []
        train_kld_losses = []
        val_r_losses = []
        val_kld_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch_index in range(num_epochs):
            self.model.train()
            mean_loss, mean_r_loss, mean_kld_loss, notes_accuracy = self.loss_and_acc_on_epoch(batch_size=batch_size, batches_per_epoch=self.batches_per_epoch, data_loader=train_loader, train=True)

            self.model.eval()
            val_mean_loss, val_mean_r_loss, val_mean_kld_loss, val_notes_accuracy = self.loss_and_acc_on_epoch(batch_size=batch_size//2, batches_per_epoch=self.batches_per_epoch, data_loader=validation_loader, train=False)

            train_r_losses.append(mean_r_loss)
            train_kld_losses.append(mean_kld_loss)
            train_accuracies.append(notes_accuracy)

            val_r_losses.append(val_mean_r_loss)
            val_kld_losses.append(val_mean_kld_loss)
            val_accuracies.append(val_notes_accuracy)

            print(
                f'Train Epoch: {epoch_index}/{num_epochs} \t'
                f'Total Loss: {mean_loss}\t'
                f'Reconstruction Loss: {mean_r_loss}\t'
                f'KLD Loss: {mean_kld_loss}\t'
                f'Notes Accuracy: {notes_accuracy * 100} %\t'
            )

            print(
                f'Total Validation Loss: {val_mean_loss} \t'
                f'Validation Reconstruction Loss: {val_mean_r_loss}\t'
                f'Validation KLD Loss: {val_mean_kld_loss}\t'
                f'Validation Notes Accuracy: {val_notes_accuracy * 100} %\t'
            )
            print('-' * 100)

            if val_mean_loss < best_loss:
                best_loss = val_mean_loss
                self.model.save()

            print('Updating model on test data...')
            self.sample_update(self.samples_list)
            print('Done.')

            print('-' * 50)
            print('Assessing latent structure...')
            sample = self.samples_list[0]
            tolerance = [0.01, 0.1, 1, 10]

            for eps in tolerance:
                sample_st, neighbour_st = self.assess_latent_structure(sample, eps)

                sample_mat = np.array(Converter.as_matrix(sample_st, 24 * 2))
                neighbour_mat = np.array(Converter.as_matrix(neighbour_st, 24 * 2))

                save_as_pickled_object([sample_mat, neighbour_mat, matrix_distance(sample_mat, neighbour_mat)], os.path.join(os.getcwd(), 'tests/latent_structure_mat/' + 'epoch_' + str(epoch_index) + '_tol_' + str(eps) + '.pkl'))


                sample_img = Image.fromarray(sample_mat[::-1], 'L')
                neighbour_img = Image.fromarray(neighbour_mat[:-1], 'L')

                sample_img.save(os.path.join(os.getcwd(), 'tests/latent_structure_png/' + 'sample_' 'epoch_' + str(epoch_index) + '_tol_' + str(eps) + '.png'))
                neighbour_img.save(os.path.join(os.getcwd(), 'tests/latent_structure_png/' + 'neighbour_' 'epoch_' + str(epoch_index) + '_tol_' + str(eps) + '.png'))

            print('Done.')
            print('-' * 50)

            if epoch_index % 5 == 0:
                print('-' * 50)
                print('Interpolation tests...')
                self.interpolation(sample1=self.samples_list[0], sample2=self.samples_list[1], step=0.25, file_name='Layla_unppluged')
                self.interpolation(sample1=self.samples_list[2], sample2=self.samples_list[3], step=0.25, file_name='Layla_electric')
                self.interpolation(sample1=self.samples_list[4], sample2=self.samples_list[5], step=0.25, file_name='Knockin_on')
                self.interpolation(sample1=self.samples_list[1], sample2=self.samples_list[3], step=0.25, file_name='Layla_cross')
                print('Done.')
                print('=' * 50)

        print('End of training.')

        print('Saving Losses')
        save_as_pickled_object([train_r_losses, train_kld_losses, val_r_losses, val_kld_losses], os.path.join(os.getcwd(), 'losses/losses.pkl'))
        print('Done.')

        print('Saving Accuracies')
        save_as_pickled_object([train_accuracies, val_accuracies], os.path.join(os.getcwd(), 'accuracy/accuracies.pkl'))
        print('Done')

    def decode_latent(self, z):
        sigm = nn.Sigmoid()
        hidden = self.model.decoder.init_hidden(1)
        out, _ = self.model.decoder(z, hidden)
        out = sigm(out)
        st = reconstruction(out)
        return st

    def interpolation(self, sample1, sample2, step, file_name: str):
        hidden = self.model.encoder.init_hidden(2)
        batch = torch.stack([sample1, sample2])
        z, _, _, _ = self.model.encoder(batch, hidden)
        z1 = z[0, :]
        z2 = z[1, :]

        for alpha in np.linspace(0, 1, 5):
            tmp = float(alpha) * z1 + (1 - float(alpha)) * z2
            tmp_out = self.decode_latent(tmp)
            arr = np.array(Converter.as_matrix(tmp_out, 48))[::-1]
            img = Image.fromarray(arr, 'L')
            img.save('tests/interpolation_png/' + file_name + '_' + str(alpha) + '.png')
            with open(os.path.join(os.getcwd(), 'tests/interpolation_mat/' + file_name + '_' + str(alpha) + '.png'),
                      'wb') as f:
                pickle.dump(np.array(Converter.as_matrix(tmp_out, 24 * 2)), f)


    def assess_latent_structure(self, sample, eps):
        hidden = self.model.encoder.init_hidden(1)
        input = torch.unsqueeze(sample, dim=0)
        z, _, _, _ = self.model.encoder(input, hidden)

        perturbation = torch.randn(z.size()) / 2
        while torch.dist(z, z + perturbation) > eps:
            perturbation /= 2

        sample_st = self.decode_latent(z)
        neighbour_st = self.decode_latent(z + perturbation)

        return sample_st, neighbour_st