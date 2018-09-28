import os
from typing import List

from converter import Converter
from seq.smart_multi_track import SmartMultiTrack
from seq.track import SmartTrack

root = '/Users/proy/GoogleDrive/ProjectsProy/DancingMIDI/midis-for-vae'


def make_all_pngs():
    names, chunks = Converter.midi_chunks(root, chunk_len_in_beats=1)  # type: List[SmartTrack]
    for i in range(len(names)):
        name = names[i]
        chunk = chunks[i]
        if chunk.empty:
            continue
        for semitones in range(-6, 6):
            Converter.save_as_png(chunk.transpose(semitones),
                                  n_steps=24,
                                  png_file_name=os.path.join(root, f'{name}_{semitones}.png'))


def read_png():
    f = os.path.join(root, 'Piano Duke Clipping.mid_13_5.png')
    arr = Converter.png_to_smart_track(f, track_name='test', track_duration=480)
    print(arr)


# make_all_pngs()
# read_png()

# Example: For Nicolas, splits one file and saves one bar as png and creates the matrix
st = SmartMultiTrack.read(name='Duke',
                          file_name=os.path.join(root, "/Users/nicolasc/Desktop/test_1.mid"),
                          atomic_size_in_beats=0.125,
                          residual_ratio=0.25)
st.pad_to_next_bar()
bar_len = st.beats_to_ticks(4)
bars = st.split_every(step=bar_len, start=12 * bar_len, end=14 * bar_len)  # type: List[SmartMultiTrack]
matrix = Converter.as_matrix(bars[0].track(0), n_steps=96)
image = Converter.save_as_png(bars[0].track(0), n_steps=96, png_file_name=os.path.join(root, 'toto.png'))
print(matrix)
