from os import listdir, path, rename
from typing import List

from music21.harmony import ChordStepModificationException
from music21.pitch import PitchException
from music21.repeat import ExpanderException
from tqdm import tqdm

from ls.lead_sheet import GapError, MultiplePartsError, NoChordSymbolError, NotMonophonicError, OverlapError, \
    SuperimposedChordsError, WikifoniaLeadSheet
from ls.spotify_chord import SpotifyChordSymbol

wikifonia = '../data/wikifonia/'
database = '../data/wikifonia-db/'

parsing_error_folder = path.join(wikifonia, 'parsing-error')
expander_error_folder = path.join(wikifonia, 'expander-error')
superimposed_chords_folder = path.join(wikifonia, 'superimposed-chords')
no_chord_symbols_folder = path.join(wikifonia, 'no-chord-symbols')
not_monophonic_folder = path.join(wikifonia, 'not-monophonic')
gaps_folder = path.join(wikifonia, 'gaps')
overlaps_folder = path.join(wikifonia, 'overlaps')
multiple_parts_folder = path.join(wikifonia, 'multiple-parts')


def clean_wikifonia_dir() -> None:
    """
    Try to parse every Wikifonia music XML file and put erroneous files in sub-folders

    :return:
    """
    src_dir = wikifonia
    for i, f in enumerate(listdir(src_dir)):
        if not f.endswith('.mxl'):
            continue
        file = path.join(src_dir, f)
        try:
            WikifoniaLeadSheet(file)
            print(f'{i+1} \t {f}')
        except (ChordStepModificationException,
                TypeError,
                ZeroDivisionError,
                PitchException,
                IndexError,
                AttributeError,
                ValueError) as err:
            print(f'\t=> SKIPS\t{i+1} \t {err}')
            rename(file, path.join(parsing_error_folder, f))
        except (ExpanderException, UnboundLocalError) as err:
            print(f'\t=> SKIPS\t{i+1} \t {err}')
            rename(file, path.join(expander_error_folder, f))
        except SuperimposedChordsError as err:
            print(f'\t=> SKIPS\t{i+1} \t {err}')
            rename(file, path.join(superimposed_chords_folder, f))
        except NoChordSymbolError as err:
            print(f'\t=> SKIPS\t{i+1} \t {err}')
            rename(file, path.join(no_chord_symbols_folder, f))
        except NotMonophonicError as err:
            print(f'\t=> SKIPS\t{i+1} \t {err}')
            rename(file, path.join(not_monophonic_folder, f))
        except GapError as err:
            print(f'\t=> SKIPS\t{i+1} \t {err}')
            rename(file, path.join(gaps_folder, f))
        except OverlapError as err:
            print(f'\t=> SKIPS\t{i+1} \t {err}')
            rename(file, path.join(overlaps_folder, f))
        except MultiplePartsError as err:
            print(f'\t=> SKIPS\t{i+1} \t {err}')
            rename(file, path.join(multiple_parts_folder, f))

        continue


def make_database():
    src_dir = wikifonia
    for i, f in enumerate(listdir(src_dir)):
        if not f.endswith('.xml'):
            continue

        if f not in ['Henry Mancini, Johnny Mercer - '
                     'Moon River.1.xml']:
            continue
        print(f'{i}\t{f}')
        file = path.join(src_dir, f)
        ls = WikifoniaLeadSheet(file)
        f_ = path.splitext(f)[0].encode('ascii', 'ignore').decode('ascii')
        with open(path.join(database, f_ + '.db'), 'w') as out:
            out.write(f'{f_}\n')
            for n_c in ls.get_note_and_chord_list(False):
                out.write(n_c.csv_str() + '\n')


def get_all_chord_name_seqs(db_file_path, n_seqs: int = 0) -> List[List[str]]:
    src_dir = db_file_path
    res = []
    all_files = listdir(src_dir)
    all_files.sort()
    for i, f in enumerate(all_files):
        if not f.endswith('.db'):
            continue
        if i >= n_seqs and n_seqs:
            break
        file = path.join(src_dir, f)
        with open(file, 'r') as input_file:
            prev_chord = 'NC'
            seq = []
            res.append(seq)
            for line_idx, line in enumerate(input_file.readlines()[1:]):
                pos, _, _, a_chord = line.split(',')
                a_chord = a_chord.strip()[1:-1]
                if a_chord == 'NC':
                    if a_chord != prev_chord:
                        seq = []
                        res.append(seq)
                elif a_chord != prev_chord:
                    seq.append(a_chord)
                elif pos == '0':
                    seq.append(a_chord)
                prev_chord = a_chord
    return res


def get_all_chord_seqs(db_file_path, n_seqs: int = 0) -> List[List[SpotifyChordSymbol]]:
    all_seqs = get_all_chord_name_seqs(db_file_path, n_seqs)
    res = []
    for i in tqdm(range(len(all_seqs))):
        seq = []
        res.append(seq)
        for c in all_seqs[i]:
            seq.append(SpotifyChordSymbol.from_figure(c))
    return res

# abp = SpotifyChordSymbol.from_figure('Bbpedal')
# print(abp)
# bbp = abp.transposed_by(Interval('m2'))
# print(bbp)
