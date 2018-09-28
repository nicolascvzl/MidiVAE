from os import listdir, path
from typing import List

from ls.spotify_chord import SpotifyChord, SpotifyChordSymbol

database = '../data/wikifonia-db/'


def get_all_chords() -> List[str]:
    src_dir = database
    all_chords = set()
    all_files = listdir(src_dir)
    all_files.sort()
    for i, f in enumerate(all_files):
        if not f.endswith('.db'):
            continue
        file = path.join(src_dir, f)
        with open(file, 'r') as input_file:
            for line in input_file.readlines()[1:]:
                a_chord = line.split(',')[-1].strip()[1:-1]
                all_chords.add(a_chord)
                if a_chord.startswith('Chord Symbol'):
                    print(f)
    res = [c for c in all_chords]
    res.sort()
    return res


# clean_wikifonia_dir()
all_chords = get_all_chords()
all_chords_in_c = set()
all_chords_symb_in_c = set()
for i, c in enumerate(all_chords):
    sch = SpotifyChord.from_figure(c)
    sch_symb = SpotifyChordSymbol.from_figure(c)
    sch_in_c, interval_chord = sch.transposed_to()
    sch_symb_in_c, interval_chord_symb = sch_symb.transposed_to()
    all_chords_in_c.add(sch_in_c)
    all_chords_symb_in_c.add(sch_symb_in_c)

print('All Spotify Unique Chords')
for i, sc in enumerate(all_chords_in_c):
    print(f'{i}\t{sc}\t{sc.note_string()}')

print('All Spotify Unique Chord Symbols')
for i, sc in enumerate(all_chords_symb_in_c):
    print(f'{i}\t{sc}')
