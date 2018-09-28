import random
import time
import math
import pickle
import os
import torch
from music21 import *
from music21.interval import Interval
from fractions import Fraction
from functools import lru_cache
from collections import deque
from ls import spotify_chord as sc
import torch.nn as nn
import numpy as np
import sys
from seq.converter import Converter
from seq.smart_multi_track import SmartMultiTrack
from seq.track import SmartTrack
import png
from numpy import count_nonzero
from numpy.ma import logical_xor
from seq.interval import Interval
from seq.smart_multi_track import SmartMultiTrack
from seq.smart_seq import SmartSequence
from seq.track import SmartTrack

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def load_data(PATH):
    path = PATH + '.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_splits(PATH):
    path = PATH + '.pkl'
    with open(path, 'rb') as f:
        train_set, val_set, test_set = pickle.load(f)
    return train_set, val_set, test_set


def random_choice(list_):
    return list_[random.randint(0, len(list_) - 1)]


def dictionary_length(dictionary):
    l = 0
    for chord in dictionary:
        l += len(dictionary[chord])
    return l


def dictionary_to_list(dictionary):
    s = sum([len(v) for k, v in dictionary.items()])
    list_ = [None] * s
    i = 0
    for chord in dictionary:
        for melody in dictionary[chord]:
            list_[i] = (melody, chord)
            i += 1
    return list_


def dictitonary_to_deque(data_dictionary):
    data_length = dictionary_length(data_dictionary)
    data_list = dictionary_to_list(data_dictionary)
    random.shuffle(data_list)
    data_deque = deque(data_list)
    return data_deque


def split_dictionary(dictionary, p):
    list_ = dictionary_to_list(dictionary)
    random.shuffle(list_)
    slice_idx = int(p * len(list_))
    sub_list_1 = list_[:slice_idx]
    sub_list_2 = list_[slice_idx:]
    return sub_list_1, sub_list_2


def make_split(dataset, p):
    print('-' * 50)
    print('Creating Train')
    print('-' * 50)
    train_set, tmp_set = split_dictionary(dataset, p)
    train_set = list_to_dictionary(train_set)
    tmp_set = list_to_dictionary(tmp_set)
    print('-' * 50)
    print('Spliting val and test')
    print('-' * 50)
    eval_set, test_set = split_dictionary(tmp_set, 1/2)
    eval_set = list_to_dictionary(eval_set)
    test_set = list_to_dictionary(test_set)
    return train_set, eval_set, test_set


def list_to_dictionary(list_):
    dictionary = {}
    for elt in list_:
        if elt[1] in dictionary:
            dictionary[elt[1]].append(elt[0])
        else:
            dictionary[elt[1]] = [elt[0]]
    return dictionary


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def fraction_to_int(c):
    c = str(c)
    if len(c) == 1:
        return int(c)
    else:
        return int(c.split('/')[0]) / int(c.split('/')[1])


def lcm(a, b):
    return abs(a * b) / math.gcd(a, b) if a and b else 0


def get_denominator(c):
    if len(c.split('/')) == 1:
        return '1'
    else:
        return c.split('/')[-1]


def make_list_of_names(path_name):
    list_ = []
    for file_name in os.listdir(path_name):
        if os.path.isfile(path_name + file_name) and os.path.getsize(path_name + file_name):
            list_.append(file_name)
    print("Number of empty files encountered: {}".format(len(os.listdir(path_name)) - len(list_)))
    return list_


def extract_subsplit(split, split_length):
    list_ = dictionary_to_list(split)
    random.shuffle(list_)
    list_ = list_[:split_length]
    sub_split = list_to_dictionary(list_)
    return sub_split


def new_list(list_length):
    list_ = [None] * list_length
    return list_


def create_note_dictionary():
    all_notes = [i for i in range(12)]
    all_notes.append('rest')
    all_notes.append('hold')
    all_notes.append('PAD')
    note_dictionary = {}
    for note in all_notes:
        tensor = ByteTensor(len(all_notes)).fill_(0)
        tensor[all_notes.index(note)] = 1
        note_dictionary[note] = tensor
    return note_dictionary

def load_chord_dictionary():
    with open('data/dictionaries/chord_dictionary.pkl', 'rb') as f:
        chord_dictionary = pickle.load(f)
    return chord_dictionary



def get_max_length(data):
    m = -1
    if type(data) is list:
        for data_ in data:
            for chord in data_:
                for melody in data_[chord]:
                    if len(melody) > m:
                        m = len(melody)
    else:
        for chord in data:
            for melody in data[chord]:
                if len(melody) > m:
                    m = len(melody)
    return m


def get_sequence_scaling_factor(input_sequence):
    dur_list = [note[1] for note in input_sequence]
    den_list = [get_denominator(str(c)) for c in dur_list]
    a = lcm(int(den_list[0]), int(den_list[1]))
    for i in range(3, len(den_list)):
        if int(den_list[i]) > 1:
            a = lcm(int(a), int(den_list[i]))
    return a


def note_to_midi(note):
    if note == 'rest':
        return note
    elif note == 'hold':
        return note
    else:
        return pitch.Pitch(note).midi


def convert_sequence(input_melody):
    scaling_factor = get_sequence_scaling_factor(input_melody)
    melody = []
    # midi conversion and duration scaling
    for note in input_melody:
        if int(note[1]) > 0 or note[1] != '0':
            if note[0] != 'rest':
                melody.append((pitch.Pitch(note[0]).midi % 12, fraction_to_int(note[1]) * scaling_factor))
            else:
                melody.append((note[0], fraction_to_int(note[1]) * scaling_factor))
    output_melody = []
    # quantization
    for note, dur in melody:
        output_melody.append(note)
        output_melody.extend(['hold'] * (int(dur) - 1))
    output_melody.append('PAD')
    # sequence to tensor conversion
    tensor = Tensor(len(melody) + 1, 1, 15).fill_(0)
    note_dictionary = create_note_dictionary()
    tmp = 0
    for n, note in enumerate(output_melody):
        # if note in range(12) U 'rest'
        if note != 'hold' and note != 'PAD':
            tmp = note
            tensor[n][:][:] = note_dictionary[note]
        elif note == 'PAD':
            tensor[n][:][:] = note_dictionary[note]
        elif note == 'hold':
            tensor[n][:][:] = note_dictionary[note] + note_dictionary[tmp]
    return tensor


def parse_xml_to_str(path):
    try:
        s = converter.parse(path)
        if len([p for p in s.parts]) != 1:
            list_ = [len(p.iter.getElementsByClass('Measure')) for p in s.parts]
            part = list_.index(max(list_))
        else:
            part = [p for p in s.parts][0]
        measures = [m for m in part.iter.getElementsByClass('Measure')]
        melody_notes = [None] * len(measures)
        for i in range(len(measures)):
            notes = [note for note in measures[i].iter.getElementsByClass('Note')]
            pitch_list = [note.pitch.nameWithOctave for note in notes[i]]
            duration_list = [Fraction(note.quarterLength).__str__() for note in notes[i]]
            for j in range(len(pitch_list)):
                if pitch_list[j] != '0':
                    melody_notes.append((pitch_list[j], duration_list[j]))
        return melody_notes
    except converter.ConverterFileException:
        print(path.split('/')[-1], ' is not a correct file')


def wrap_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def dataset_from_name(dataset_name:str):
    with open('piano_roll/' + dataset_name, 'rb') as f:
            dataset = pickle.load(f)
    return dataset


def melody_transpose(melody, transposition):
    transposed_melody = []
    for note in melody:
        if note != 'rest' and note != 'hold' and note != 'PAD':
            transposed_melody.append((note + transposition) % 12)
        else:
            transposed_melody.append(note)
    return transposed_melody


@lru_cache(maxsize=10000)
def chord_transpose(chord, transposition):
    new_scs = chord.transposed_by(Interval(transposition))
    return new_scs


def data_transpose(melody, chord, transposition):
    return melody_transpose(melody, transposition), chord_transpose(chord, transposition)


def tensor_to_str(tensor):
    s = '['
    if len(tensor.size()) > 1:
        tensor = tensor.squeeze(dim=0)
    for i in range(len(tensor)):
        s += str(tensor[i].item())
        if i < len(tensor) - 1:
            s += ', '
    s += ']'
    return s


def one_hot_to_string(chord):
    if len(chord.size()) > 1:
        chord = chord.squeeze(dim=0)
    l_ = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    s = '['
    for i in range(len(l_)):
        if chord[i] == 1:
            s += l_[i]
            s += ' '
    s += ']'
    return s


def melody_to_batch(input_melody, n_features, note_dictionary):
    batch_size = len(input_melody)
    melody_length = len(input_melody[0])
    batch_tensor = torch.Tensor(batch_size, melody_length, n_features)
    for i, melody in enumerate(input_melody):
        for n, note in enumerate(melody):
            if note != 'hold' and note != 'PAD':
                tmp = note
                batch_tensor[i, n, :] = note_dictionary[note]
            elif note == 'hold':
                batch_tensor[i, n, :] = note_dictionary[note] + note_dictionary[tmp]
            else:
                batch_tensor[i, n, :] = note_dictionary['PAD']
    return batch_tensor


def pitch_list(chord):
    pitch_list = [p.midi for p in chord.chord.pitches]
    pitch_list.sort()
    short_pitch_list = [p % 12 for p in pitch_list]
    return short_pitch_list


@lru_cache(maxsize=10000)
def sc_to_tensor(chord):
    chord_pitch = pitch_list(chord)
    chord_tensor = torch.zeros(12)
    for p in chord_pitch:
        chord_tensor[p] = 1
    return chord_tensor


def to_one_hot(tensor, n_tokens):
    one_hot_tensor = Tensor(tensor.size()[0], n_tokens).fill(0)
    for i in range(len(tensor)):
        one_hot_tensor[i, tensor[i]] = 1
    return one_hot_tensor


def decode_chord_0(output):
    f = nn.LogSoftmax()
    decoded_chord = Tensor(output.size()[0], 1)
    log_probabilities = f(output)
    for i in range(len(output)):
        # TODO PROBABLY A BUG ON [I, 0]
        decoded_chord[i, 0] = torch.argmax(log_probabilities[i, :])
    return decode_chord_0


def get_chord_0(output, chord_look_up_table):
    decoded_chord = decode_chord_0(output)
    output_chord = [None] * decoded_chord.size()[0]
    for i in range(decoded_chord.size()[0]):
        output_chord[i] = search_in_table(chord_look_up_table, decoded_chord[i])
    return output_chord


def search_in_table(chord_look_up_table, idx):
    for chord in chord_look_up_table:
        if chord_look_up_table[chord] == idx:
            return chord
    return -1


def chord_to_batch_0(target, n_tokens):
    batch_tensor = wrap_cuda(torch.LongTensor(len(target), n_tokens))
    for i in range(len(target)):
        batch_tensor[i, :] = wrap_cuda(to_one_hot(target, n_tokens))


def chord_to_batch(target, chord_look_up_table):
    batch_tensor = wrap_cuda(torch.LongTensor(len(target), 12))
    for i in range(len(target)):
        batch_tensor[i, :] = wrap_cuda(chord_look_up(sc_to_tensor(target[i]), chord_look_up_table))
    return batch_tensor


def chord_look_up(tensor, chord_table):
    for k in chord_table:
        if (wrap_cuda(k) == wrap_cuda(tensor)).sum().item() == len(tensor):
            return wrap_cuda(k)
    else:
        print('Chord not found')
        raise KeyError


def table_look_up(tensor, chord_table):
    for k in chord_table:
        if (wrap_cuda(k) == wrap_cuda(tensor)).sum().item() == len(tensor):
            return chord_table[k]


def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)

#a method that reduces the pitch dimmension of the input: C0 -> C6
def cut(piano_roll): # piano_roll of dim seq_len x pitches
    return piano_roll[:, 24:96]

#a method that restores the original midi pitch of the output matrix
def expand(piano_roll):
    pr = np.zeros((48, 128))
    pr[:, 24:96] = piano_roll
    return pr

def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj


def output_tensor_to_png_mat(tensor):
    decoded = (tensor >= 0.5).type('torch.FloatTensor')
    decoded_mat = decoded.numpy()
    decoded_mat = expand(decoded_mat)
    formated_mat = np.ones(decoded_mat.shape)
    formated_mat = formated_mat - decoded_mat
    formated_mat = formated_mat * 255
    return formated_mat


def mat_to_png(mat):
    mat = mat.astype('int8')
    mat = expand(mat)
    mat = mat.tolist()
    img = png.from_array(mat[::-1], 'L')
    img.save('tmp.png')
    return img


def smart_track_to_png(st):
    # atom = 24
    # n_bars = 2
    mat = Converter.as_matrix(st, 24 * 2)
    img = png.from_array(mat, 24 * 2)
    img.save(os.path.join(os.getcwd(), 'tmp_png/tmp.png'))
    return img


def png_to_smart_track(png_file_name: str,
                       track_name: str,
                       track_duration: int,
                       threshold: int = 0) -> SmartTrack:
    res = SmartTrack(name=track_name, duration=track_duration)
    arr = Converter.png_to_array(png_file_name)
    d, m = divmod(track_duration, len(arr[0]))
    if m is not 0:
        raise ValueError(
            f'target track duration {track_duration} is not a multiple of png file width {len(arr[0])}')
    for pitch, row in enumerate(arr):
        seq = SmartSequence(duration=track_duration)
        crt_interval = None
        for i, v in enumerate(row):
            if v <= threshold:
                if crt_interval:
                    crt_interval.end += d
                else:
                    crt_interval = Interval(start=d * i, end=d * (i + 1))
            else:
                if crt_interval:
                    seq.append(crt_interval)
                    crt_interval = None
        if crt_interval:
            seq.append(crt_interval)
        res.put_sequence(pitch, seq)
    return res


def matrix_distance(m1, m2):
    assert m1.shape == m2.shape

    d = count_nonzero(logical_xor(m1, m2))
    return d / (m1.shape[0] * m1.shape[1])


def reconstruction(output, file_name='tmp_name'):
    mat = output_tensor_to_png_mat(output)
    img = png.from_array(np.transpose(mat)[::-1].astype('int16'), 'L')
    img.save('tmp.png')
    st = png_to_smart_track('tmp.png', track_name=file_name, track_duration=480)
    return st

