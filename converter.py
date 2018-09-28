import os
from typing import List, Tuple

import png

from seq.interval import Interval
from seq.smart_multi_track import SmartMultiTrack
from seq.smart_seq import SmartSequence
from seq.track import SmartTrack


class Converter(object):
    @staticmethod
    def midi_files(root: str):
        return [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.mid')]

    @staticmethod
    def midi_chunks(root: str, chunk_len_in_beats: float) -> Tuple[List[str], List[SmartTrack]]:
        names = []  # type: List[str]
        chunks = []  # type: List[SmartTrack]
        for f in Converter.midi_files(root):
            smt = SmartMultiTrack.read(name=f, file_name=f, atomic_size_in_beats=.25, residual_ratio=.125)
            tr = smt.track(0)
            splits = tr.split_every(int(smt.beats_to_ticks(1) * chunk_len_in_beats))  # type: List[SmartTrack]
            for i in range(len(splits)):
                names.append(f + '_' + str(i))
            chunks.extend(splits)
        return names, chunks

    @staticmethod
    def as_matrix(track: SmartTrack, n_steps: int) -> List[List[int]]:
        time_step, m = divmod(track.duration, n_steps)
        if m is not 0:
            raise ValueError(f'duration {track.duration} is not a multiple of number of steps {n_steps}')
        n_steps, mod = divmod(track.duration, time_step)
        if mod > 0:
            n_steps += 1
            track.pad_to(n_steps * time_step)
        matrix = [[255 for x in range(n_steps)] for y in range(128)]
        for pitch, seq in track:
            if pitch == -1:
                continue
            for event in seq:
                s = event.start
                e = event.end
                d1, m1 = divmod(s, time_step)
                d2, m2 = divmod(e, time_step)
                for i in range(d1, min(n_steps, (d2 + (1 if m2 > 0 else 0)) + 1)):
                    matrix[pitch][i] = 0
        return matrix

    @staticmethod
    def save_as_png(track: SmartTrack, n_steps: int, png_file_name: str):
        m = Converter.as_matrix(track, n_steps)
        img = png.from_array(m, 'L')
        img.save(png_file_name)

    @staticmethod
    def png_to_array(png_file_name: str) -> List[List[int]]:
        res = []
        with open(png_file_name, 'rb') as f:
            r = png.Reader(f)
            _, _, l, _ = r.read()
            for x in l:
                res.append(list(x))
        return res

    @staticmethod
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
