from copy import deepcopy
from functools import lru_cache
from re import match, search
from typing import List, Optional, Tuple

from music21 import interval
from music21.chord import Chord
from music21.harmony import ChordSymbol
from music21.interval import Interval
from music21.pitch import Pitch
from music21.note import Note


def split(figure):
    _figure = figure.replace('bpedal', '-pedal')
    root = match(r'[A-Ga-g][#-]*', _figure).group()
    _figure = _figure.replace(root, '', 1)
    root_pitch = Pitch(root)
    root = str(root_pitch)
    bass = search(r'/[A-Ga-g][#-]*', _figure)
    structure = _figure.replace(bass.group(), '', 1) if bass else _figure
    if bass:
        bass_pitch = Pitch(bass.group()[1:])
        bass = '/' + str(bass_pitch)
    return root, structure.strip(), bass if bass else ''


@lru_cache(maxsize=1000000)
def _get_transposed_pitch(pitch, interv):
    return pitch.transpose(Interval(interv), inPlace=False)


class SpotifyChordSymbol:
    def __init__(self,
                 original_figure: str = None,
                 spotify_figure: str = None,
                 figure: str = None,
                 bass: str = None,
                 root: str = None,
                 kind: str = None):
        # This is the original figure, which is kept because it is needed to create a SpotifyChord equivalent to THIS
        #  SpotifyChordSymbol
        self.original_figure = original_figure
        if spotify_figure == 'NC':
            self.bass = None
            self.root = None
            self.structure = 'NC'
        elif figure:
            assert root
            self.bass = Pitch(bass[1:]) if bass else None
            if self.bass:
                self.bass.octave = None
            self.root = Pitch(root)
            self.root.octave = None
            self.structure = figure
        else:
            assert root and kind
            chord_symbol = ChordSymbol(bass=bass[1:], root=root, kind=kind)
            self.bass = Pitch(bass[1:]) if bass else None
            if self.bass:
                self.bass.octave = None
            self.root = Pitch(root)
            self.root.octave = None
            self.structure = chord_symbol.figure

    def as_spotify_chord(self):
        return SpotifyChord.from_figure(self.original_figure)

    def __str__(self):
        return self.structure

    def __repr__(self):
        return self.__str__()

    def __hash__(self) -> int:
        return self.structure.__hash__()

    def __eq__(self, o: object) -> bool:
        if type(o) is not SpotifyChordSymbol:
            return False
        return self.structure.__eq__(o.structure)

    @classmethod
    def from_bass_root_kind(cls, bass, root, kind):
        return SpotifyChord(bass=bass, root=root, kind=kind)

    @classmethod
    def from_figure(cls, figure: str) -> 'SpotifyChordSymbol':
        if figure == 'Am/A-':
            return SpotifyChordSymbol(bass='/A-', root='A', kind='minor')
        if figure == 'E7/E-':
            return SpotifyChordSymbol(bass='/E-', root='E', kind='dominant-seventh')
        if figure == 'NC':
            return SpotifyChordSymbol(spotify_figure='NC')
        root, structure, bass = split(figure)
        spotify_figure = None
        if structure in ['', '+', '7+', '7+ add 9', 'dim',
                         'add 2', 'add 4', '7 add 11', 'm add 2', 'm7 add 4', 'm add 4',
                         'maj7',
                         'M7', 'M9', 'M11', 'M13',
                         '/o7', 'o7',
                         'm', 'm6', 'm7', 'm9', 'm11', 'm13',
                         '6', '7', '9', '11', '13',
                         'sus', 'sus2', 'sus4', 'sus add 7',
                         'm add 9', 'm7 add 9', 'm7 add 11', '6 add 9', 'm6 add 9',
                         'pedal', 'power']:
            explicit_figure = root + structure + bass
        elif structure == '9 alter b5':
            explicit_figure = root + bass + '9b5'
        elif structure == '7 alter b5':
            explicit_figure = root + bass + '7b5'
        elif structure == '7 alter #5':
            explicit_figure = root + bass + '7#5'
        elif structure == 'm7 alter b5':
            explicit_figure = root + bass + '/o7'
        elif structure == '7 add 4 subtract 3':
            explicit_figure = root + bass + 'sus add 7'
        elif structure == 'sus add 7 add 9':
            # TODO Check if the 7th should be diminished
            explicit_figure = root + bass + ',45b79'
            spotify_figure = root + bass + '9sus4 '
        elif structure == 'sus add 7 add 13':
            # TODO Check if the 7th should be diminished
            # Pitches are not in the right order
            explicit_figure = root + bass + ',45b713'
            spotify_figure = root + bass + 'sus4 add7 add13'
        elif structure == 'sus add 7 add 9 add 13':
            # TODO Check if the 7th should be diminished
            # Pitches are not in the right order
            explicit_figure = root + bass + ',45b7913'
            spotify_figure = root + bass + 'sus4 add7 add9 add13'
        elif structure == 'sus add 7 add b9':
            # TODO Check if the 7th should be diminished
            explicit_figure = root + bass + ',45b7b9'
            spotify_figure = root + bass + 'b9sus4 '
        elif structure == 'sus add 7 add 9 add 11 add 13':
            explicit_figure = root + bass + ',45791113'
            spotify_figure = root + bass + '13sus4 '
        elif structure == 'pedal':
            explicit_figure = root + bass + 'pedal'
        elif structure == '7 add #11':
            explicit_figure = root + bass + '7 #11'
        elif structure == '7 alter #5 add #9':
            explicit_figure = root + bass + '+7 #9'
        elif structure == '9 alter #5':
            explicit_figure = root + bass + '9 #5'
        elif structure == 'm alter #5':
            explicit_figure = root + bass + 'm #5'
        elif structure == '7 alter b5 add b9':
            explicit_figure = root + bass + '7 b5b9'
        elif structure == 'm7 alter #5':
            explicit_figure = root + bass + 'm7 #5'
        elif structure == '7 alter b5 add #9':
            explicit_figure = root + bass + '7 b5#9'
        elif structure == '7 add b9':
            explicit_figure = root + bass + '7 b9'
        elif structure == '7 add #9':
            explicit_figure = root + bass + '7 #9'
        elif structure == '9 add #11':
            explicit_figure = root + bass + '9 #11'
        elif structure == '7 add b13':
            explicit_figure = root + bass + '7 b13'
        elif structure == '7 add #9 add #11':
            explicit_figure = root + bass + '7 #9#11'
        elif structure == '7 add b9 add b13':
            explicit_figure = root + bass + '7 b9b13'
        elif structure == 'add 9':
            explicit_figure = root + bass + 'add9'
        elif structure == 'maj7 add #11':
            explicit_figure = root + bass + 'maj7 #11'
        elif structure == 'sus add b9 add b9':
            explicit_figure = root + bass + 'sus b9'
        elif structure == 'm7 add b9':
            explicit_figure = root + bass + 'm7 b9'
        elif structure == 'add #11':
            explicit_figure = root + bass + ',#11'
        elif structure == '7 subtract 5 add b9 add #9 add #11 add b13':
            # TODO Pitches are not in the correct order
            explicit_figure = root + bass + ',3b7b9#9#11b13'
            spotify_figure = root + bass + '7 -5 b9 #9 #11 b13'
        elif structure == '13 alter b5':
            explicit_figure = root + bass + '13b5'
        elif structure == 'alter b5':
            explicit_figure = root + bass + ',b5'
        elif structure == 'maj7 alter #5':
            explicit_figure = root + bass + 'maj7 #5'
        elif structure == 'maj7 alter b5':
            explicit_figure = root + bass + 'maj7 b5'
        elif structure == '7 alter #5 add b9':
            explicit_figure = root + bass + '+7 b9'
        elif structure == '7 alter #5 add #9 add #11':
            explicit_figure = root + bass + '+7 #9#11'
        elif structure == '7+ add b9':
            explicit_figure = root + bass + '+7 b9'
        elif structure == '7 add #9 add b13':
            explicit_figure = root + bass + '7 #9b13'
        elif structure == 'M9 add #11':
            explicit_figure = root + bass + 'M9 #11'
        elif structure == 'M9 alter b5':
            explicit_figure = root + bass + 'M9 b5'
        elif structure == '7 add #11 add b9 add #5':
            explicit_figure = root + bass + '+7 b9#11'
        elif structure in ['7+ add #9', '7 add #9 alter #5']:
            explicit_figure = root + bass + '+7 #9'
        elif structure == 'm11 alter b5':
            explicit_figure = root + bass + 'm11 b5'
        elif structure == 'bpedal':
            # TODO This is a hack, replacing the 'b' by a '-'
            explicit_figure = root + bass + '-pedal'
        elif structure in ['13 add #11', '13 alter #11']:
            # TODO Check if a better solution exists
            # 1) This chord should be 13 alter #11
            # 2) The order of the pitches is wrong
            explicit_figure = root + bass + ',35b7913#11'
            spotify_figure = '13 alter #11'
        elif structure in ['13 add b9', '13 alter b9']:
            # TODO Check if a better solution exists
            # 1) This chord should be 13 alter #11
            # 2) The order of the pitches is wrong
            explicit_figure = root + bass + ',35b7b91113'
            spotify_figure = '13 alter b9'
        elif structure in ['13 add b9', '13 alter b9']:
            # TODO Check if a better solution exists
            # 1) This chord should be 13 alter #11
            # 2) The order of the pitches is wrong
            explicit_figure = root + bass + ',35b7b91113'
            spotify_figure = '13 alter b9'
        elif structure == '13 subtract 5 subtract 11':
            # TODO Check if a better solution exists
            # 1) This chord should be 13 alter #11
            # 2) The order of the pitches is wrong
            explicit_figure = root + bass + ',3b7913'
            spotify_figure = '13 sus4 omit 11'
        elif structure == 'm9 alter b5':
            explicit_figure = root + bass + 'm9 b5'
        elif structure == '7 subtract 5':
            explicit_figure = root + bass + '7 omit5'
        elif structure == 'pedal add 5':
            explicit_figure = root + bass + 'omit3'
        else:
            raise ValueError('unknown chord symbol ' + root + bass + structure)
        return SpotifyChordSymbol(original_figure=figure,
                                  spotify_figure=spotify_figure,
                                  figure=explicit_figure,
                                  bass=bass,
                                  root=root)

    @staticmethod
    def split(figure):
        _figure = figure.replace('bpedal', '-pedal')
        root = match(r'[A-Ga-g][#-]*', _figure).group()
        _figure = _figure.replace(root, '', 1)
        root_pitch = Pitch(root)
        # root_pitch.simplifyEnharmonic(inPlace=True)
        root = str(root_pitch)
        bass = search(r'/[A-Ga-g][#-]*', _figure)
        structure = _figure.replace(bass.group(), '', 1) if bass else _figure
        if bass:
            bass_pitch = Pitch(bass.group()[1:])
            # bass_pitch.simplifyEnharmonic(inPlace=True)
            bass = '/' + str(bass_pitch)
        return root, structure.strip(), bass if bass else ''

    def is_no_chord(self):
        return self.structure == 'NC'

    def transposed_to(self, pitch: Pitch = Pitch()) -> Tuple['SpotifyChordSymbol', Optional[Interval]]:
        if self.is_no_chord():
            return self, None
        tr_int = interval.notesToInterval(self.root, pitch)
        return self.transposed_by(tr_int), tr_int

    def transposed_by(self, tr_int: Interval) -> 'SpotifyChordSymbol':

        if self.is_no_chord():
            return self
        if tr_int.semitones == 0:
            return deepcopy(self)
        new_root = _get_transposed_pitch(self.root, tr_int.directedName)
        new_root.simplifyEnharmonic(inPlace=True)
        if self.bass:
            new_bass = _get_transposed_pitch(self.bass, tr_int.directedName)
            new_bass.simplifyEnharmonic(inPlace=True)
            new_structure = self.structure.replace('/' + str(self.bass), '/' + str(new_bass), 1)
            new_structure = new_structure.replace(str(self.root), str(new_root), 1)
            return SpotifyChordSymbol(root=str(new_root),
                                      bass='/' + str(new_bass) if new_bass else None,
                                      figure=new_structure)
        else:
            new_structure = self.structure.replace(str(self.root), str(new_root), 1)
            return SpotifyChordSymbol(root=str(new_root),
                                      figure=new_structure)

    @staticmethod
    def get_transposed_seq(seq: List['SpotifyChordSymbol'], interv) -> List['SpotifyChordSymbol']:
        tr_seq = [c.transposed_by(interv) for c in seq]
        return tr_seq


class SpotifyChord():
    def __init__(self,
                 spotify_figure: str = None,
                 figure: str = None,
                 bass: str = None,
                 root: str = None,
                 kind: str = None,
                 chord: Chord = None):
        if spotify_figure == 'NC':
            self.bass = None
            self.root = None
            self.chord = Chord()
            self.structure = 'NC'
        elif chord:
            assert root and figure
            self.bass = Pitch(bass[1:]) if bass else None
            self.root = Pitch(root)
            self.chord = chord
            self.structure = spotify_figure if spotify_figure else figure
        elif figure:
            assert root
            chord_symbol = ChordSymbol(figure=figure)
            self.chord = Chord(chord_symbol.pitches)
            self.bass = Pitch(bass[1:]) if bass else None
            if self.bass:
                self.bass.octave = None
            self.root = Pitch(root)
            self.root.octave = None
            self.structure = figure
        else:
            assert root and kind
            chord_symbol = ChordSymbol(bass=bass, root=root, kind=kind)
            self.chord = Chord(chord_symbol.pitches)
            self.bass = Pitch(bass) if bass else None
            if self.bass:
                self.bass.octave = None
            self.root = Pitch(root)
            self.root.octave = None
            self.structure = chord_symbol.figure

    def __str__(self):
        return self.structure

    def __repr__(self):
        return self.__str__()

    def __hash__(self) -> int:
        return self.structure.__hash__()

    def __eq__(self, o: object) -> bool:
        if type(o) is not SpotifyChord:
            return False
        return self.structure.__eq__(o.structure)

    def has_same_notes(self, a_chord: 'SpotifyChord') -> bool:
        if self.get_num_notes() != a_chord.get_num_notes():
            return False
        for i in range(self.get_num_notes()):
            if self.note_at(i) != a_chord.note_at(i):
                return False
        return True

    def get_num_notes(self) -> int:
        if self.is_no_chord():
            return 0
        return len(self.chord.pitches)

    @classmethod
    def from_bass_root_kind(cls, bass, root, kind):
        return SpotifyChord(bass=bass, root=root, kind=kind)

    @classmethod
    def from_figure(cls, figure: str) -> 'SpotifyChord':
        # if figure of type Xkind/X- : build it from an other enharmonic as bass
        if len(figure.split('/')) > 1 and list(figure)[0] == list(figure.split('/')[1])[0]:
            return SpotifyChord.from_figure(
                list(figure)[0] + '/' + Note(list(figure.split('/')[1])[0]).pitch.getAllCommonEnharmonics()[0].name)
        #if figure == 'Am/A-':
        #    return SpotifyChord(bass='A-', root='A', kind='minor')
        #if figure == 'E7/E-':
        #    return SpotifyChord(bass='E-', root='E', kind='dominant-seventh')
        if figure == 'NC':
            return SpotifyChord(spotify_figure='NC')
        root, structure, bass = split(figure)
        spotify_figure = None
        if structure in ['', '+', '7+', '7+ add 9', 'dim',
                         'add 2', 'add 4', '7 add 11', 'm add 2', 'm7 add 4', 'm add 4',
                         'maj7',
                         'M7', 'M9', 'M11', 'M13',
                         '/o7', 'o7',
                         'm', 'm6', 'm7', 'm9', 'm11', 'm13',
                         '6', '7', '9', '11', '13',
                         'sus', 'sus2', 'sus4', 'sus add 7',
                         'm add 9', 'm7 add 9', 'm7 add 11', '6 add 9', 'm6 add 9',
                         'pedal', 'power']:
            # explicit_figure = figure
            explicit_figure = root + structure + bass
        elif structure == '9 alter b5':
            explicit_figure = root + bass + '9b5'
        elif structure == '7 alter b5':
            explicit_figure = root + bass + '7b5'
        elif structure == '7 alter #5':
            explicit_figure = root + bass + '7#5'
        elif structure == 'm7 alter b5':
            explicit_figure = root + bass + '/o7'
        elif structure == '7 add 4 subtract 3':
            explicit_figure = root + bass + 'sus add 7'
        elif structure == 'sus add 7 add 9':
            # TODO Check if the 7th should be diminished
            explicit_figure = root + bass + ',45b79'
            spotify_figure = root + bass + '9sus4 '
        elif structure == 'sus add 7 add 13':
            # TODO Check if the 7th should be diminished
            # Pitches are not in the right order
            explicit_figure = root + bass + ',45b713'
            spotify_figure = root + bass + 'sus4 add7 add13'
        elif structure == 'sus add 7 add 9 add 13':
            # TODO Check if the 7th should be diminished
            # Pitches are not in the right order
            explicit_figure = root + bass + ',45b7913'
            spotify_figure = root + bass + 'sus4 add7 add9 add13'
        elif structure == 'sus add 7 add b9':
            # TODO Check if the 7th should be diminished
            explicit_figure = root + bass + ',45b7b9'
            spotify_figure = root + bass + 'b9sus4 '
        elif structure == 'sus add 7 add 9 add 11 add 13':
            explicit_figure = root + bass + ',45791113'
            spotify_figure = root + bass + '13sus4 '
        elif structure == 'pedal':
            explicit_figure = root + bass + 'pedal'
        elif structure == '7 add #11':
            explicit_figure = root + bass + '7 #11'
        elif structure == '7 alter #5 add #9':
            explicit_figure = root + bass + '+7 #9'
        elif structure == '9 alter #5':
            explicit_figure = root + bass + '9 #5'
        elif structure == 'm alter #5':
            explicit_figure = root + bass + 'm #5'
        elif structure == '7 alter b5 add b9':
            explicit_figure = root + bass + '7 b5b9'
        elif structure == 'm7 alter #5':
            explicit_figure = root + bass + 'm7 #5'
        elif structure == '7 alter b5 add #9':
            explicit_figure = root + bass + '7 b5#9'
        elif structure == '7 add b9':
            explicit_figure = root + bass + '7 b9'
        elif structure == '7 add #9':
            explicit_figure = root + bass + '7 #9'
        elif structure == '9 add #11':
            explicit_figure = root + bass + '9 #11'
        elif structure == '7 add b13':
            explicit_figure = root + bass + '7 b13'
        elif structure == '7 add #9 add #11':
            explicit_figure = root + bass + '7 #9#11'
        elif structure == '7 add b9 add b13':
            explicit_figure = root + bass + '7 b9b13'
        elif structure == 'add 9':
            explicit_figure = root + bass + 'add9'
        elif structure == 'maj7 add #11':
            explicit_figure = root + bass + 'maj7 #11'
        elif structure == 'sus add b9 add b9':
            explicit_figure = root + bass + 'sus b9'
        elif structure == 'm7 add b9':
            explicit_figure = root + bass + 'm7 b9'
        elif structure == 'add #11':
            explicit_figure = root + bass + ',#11'
        elif structure == '7 subtract 5 add b9 add #9 add #11 add b13':
            # TODO Pitches are not in the correct order
            explicit_figure = root + bass + ',3b7b9#9#11b13'
            spotify_figure = root + bass + '7 -5 b9 #9 #11 b13'
        elif structure == '13 alter b5':
            explicit_figure = root + bass + '13b5'
        elif structure == 'alter b5':
            explicit_figure = root + bass + ',b5'
        elif structure == 'maj7 alter #5':
            explicit_figure = root + bass + 'maj7 #5'
        elif structure == 'maj7 alter b5':
            explicit_figure = root + bass + 'maj7 b5'
        elif structure == '7 alter #5 add b9':
            explicit_figure = root + bass + '+7 b9'
        elif structure == '7 alter #5 add #9 add #11':
            explicit_figure = root + bass + '+7 #9#11'
        elif structure == '7+ add b9':
            explicit_figure = root + bass + '+7 b9'
        elif structure == '7 add #9 add b13':
            explicit_figure = root + bass + '7 #9b13'
        elif structure == 'M9 add #11':
            explicit_figure = root + bass + 'M9 #11'
        elif structure == 'M9 alter b5':
            explicit_figure = root + bass + 'M9 b5'
        elif structure == '7 add #11 add b9 add #5':
            explicit_figure = root + bass + '+7 b9#11'
        elif structure in ['7+ add #9', '7 add #9 alter #5']:
            explicit_figure = root + bass + '+7 #9'
        elif structure == 'm11 alter b5':
            explicit_figure = root + bass + 'm11 b5'
        elif structure == 'bpedal':
            # TODO This is a hack, replacing the 'b' by a '-'
            explicit_figure = root + bass + '-pedal'
        elif structure in ['13 add #11', '13 alter #11']:
            # TODO Check if a better solution exists
            # 1) This chord should be 13 alter #11
            # 2) The order of the pitches is wrong
            explicit_figure = root + bass + ',35b7913#11'
            spotify_figure = '13 alter #11'
        elif structure in ['13 add b9', '13 alter b9']:
            # TODO Check if a better solution exists
            # 1) This chord should be 13 alter #11
            # 2) The order of the pitches is wrong
            explicit_figure = root + bass + ',35b7b91113'
            spotify_figure = '13 alter b9'
        elif structure in ['13 add b9', '13 alter b9']:
            # TODO Check if a better solution exists
            # 1) This chord should be 13 alter #11
            # 2) The order of the pitches is wrong
            explicit_figure = root + bass + ',35b7b91113'
            spotify_figure = '13 alter b9'
        elif structure == '13 subtract 5 subtract 11':
            # TODO Check if a better solution exists
            # 1) This chord should be 13 alter #11
            # 2) The order of the pitches is wrong
            explicit_figure = root + bass + ',3b7913'
            spotify_figure = '13 sus4 omit 11'
        elif structure == 'm9 alter b5':
            explicit_figure = root + bass + 'm9 b5'
        elif structure == '7 subtract 5':
            explicit_figure = root + bass + '7 omit5'
        elif structure == 'pedal add 5':
            explicit_figure = root + bass + 'omit3'
        else:
            raise ValueError('unknown chord symbol ' + root + bass + structure)
        return SpotifyChord(spotify_figure=spotify_figure,
                            figure=explicit_figure,
                            bass=bass,
                            root=root)

    def note_string(self) -> str:
        if self.is_no_chord():
            return ''
        return str([str(p) for p in self.chord.pitches])

    def is_no_chord(self):
        return self.structure == 'NC'

    def transposed_to(self, pitch: Pitch = Pitch()) -> Tuple['SpotifyChord', Optional[Interval]]:
        if self.is_no_chord():
            return self, None
        tr_int = interval.notesToInterval(self.root, pitch)
        return self.transposed_by(tr_int), tr_int

    def transposed_by(self, tr_int: Interval) -> 'SpotifyChord':
        if self.is_no_chord():
            return self
        if tr_int.semitones == 0:
            return deepcopy(self)
        new_root = _get_transposed_pitch(self.root, tr_int.directedName)
        new_root.simplifyEnharmonic(inPlace=True)
        new_chord = self.chord.transpose(tr_int, inPlace=False)
        if self.bass:
            new_bass = _get_transposed_pitch(self.bass, tr_int.directedName)
            new_bass.simplifyEnharmonic(inPlace=True)
            new_structure = self.structure.replace('/' + str(self.bass), '/' + str(new_bass), 1)
            new_structure = new_structure.replace(str(self.root), str(new_root), 1)
            return SpotifyChord(root=str(new_root),
                                bass='/' + str(new_bass) if new_bass else None,
                                chord=new_chord,
                                figure=new_structure)
        else:
            new_structure = self.structure.replace(str(self.root), str(new_root), 1)
            return SpotifyChord(root=str(new_root),
                                chord=new_chord,
                                figure=new_structure)

    def note_at(self, i):
        if self.is_no_chord() or i >= self.get_num_notes():
            return None
        return self.chord.pitches[i]
