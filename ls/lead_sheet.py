import os
from collections import defaultdict
from copy import copy, deepcopy
from fractions import Fraction
from itertools import groupby
from math import floor
from typing import List, Optional, Union, Tuple

from music21 import converter, harmony
from music21.bar import Repeat
from music21.chord import Chord
from music21.clef import Clef, BassClef
from music21.duration import Duration
from music21.harmony import ChordSymbol
from music21.note import GeneralNote, Note, Rest
from music21.repeat import Coda, ExpanderException, RepeatExpression
from music21.spanner import RepeatBracket
from music21.stream import Measure, Part, Stream


class NoteAndChord:
    """
    Represent a note and a chord symbol that are associated to one another. The note may be a **rest**,
    a **pitch-class**, or a **note with an octave**. The chord symbol may be **none**, in which case, it is displayed
    'NC'. **Caution**: the **get_offset** and **quarter_length** get_duration are set according to the **note**'s
    get_offset and
    get_duration, that of the chord symbol are simply **ignored**.
    """

    def __init__(self,
                 note_or_rest: Union[Note, Rest],
                 chord_symbol: Optional[ChordSymbol],
                 offset: float):
        self.note_or_rest = deepcopy(note_or_rest)
        self.chord_symbol = deepcopy(chord_symbol)
        self.offset = Fraction(offset).limit_denominator(192)
        self.duration = self.note_or_rest.duration
        self.quarter_length = Fraction(self.note_or_rest.quarterLength).limit_denominator(192)

    def __str__(self):
        return 'beat = ' + str(self.offset) \
               + ', dur. = ' + str(self.quarter_length) \
               + ' ' + self.note_name() \
               + ' (' + self.chord_symbol_name() + ')' \
               + (' ...' if self.note_or_rest.tie else '')

    def csv_str(self):
        return f'{self.get_offset()},{self.get_quarter_length()},"{self.note_name()}","{self.chord_symbol_name()}"'

    def note_name(self) -> str:
        return self.note_or_rest.nameWithOctave if self.is_note() else self.note_or_rest.name

    def chord_symbol_name(self) -> str:
        return self.chord_symbol.figure if self.chord_symbol else 'NC'

    def has_chord_symbol(self) -> bool:
        return self.chord_symbol is not None

    def is_rest(self) -> bool:
        return self.note_or_rest.isRest

    def is_note(self) -> bool:
        return self.note_or_rest.isNote

    def get_offset(self) -> float:
        return self.offset

    def get_duration(self) -> float:
        return self.duration

    def get_quarter_length(self):
        return self.quarter_length


class NoChordSymbolError(Exception):
    pass


class NotMonophonicError(Exception):
    pass


class GapError(Exception):
    pass


class OverlapError(Exception):
    pass


class SuperimposedChordsError(Exception):
    pass


class MultiplePartsError(Exception):
    pass


class NoBarError(Exception):
    pass

class WrongBarDurationError(Exception):
    pass


class WikifoniaLeadSheet:

    def __init__(self, xml_fname: str):
        """
        The XML file must be a simple lead sheet file format, e.g., with a single melody (no bass line).

        :param xml_fname: is an XML file, e.g., Solar.xml
        """
        self.ls = converter.parse(xml_fname)  # type: Stream
        self.title = self.composer = None
        if self.ls.metadata:
            if self.ls.metadata.title:
                self.title = self.ls.metadata.title
            if self.ls.metadata.composer:
                self.composer = self.ls.metadata.composer
        if self.is_expandable(self.ls):
            try:
                self.ls = self.ls.expandRepeats()
            except ExpanderException as err:
                self.remove_da_capo_et_al()
                self.ls = self.ls.expandRepeats()
        self.name = os.path.splitext(os.path.basename(xml_fname))[0]
        self.sanity_check()
        # self.flat_ls = self.ls.flat  # type: Stream
        assert self.ls.highestTime == self.ls.flat.highestTime
        self.end_beat = self.ls.highestTime

    @staticmethod
    def is_expandable(ls: Stream) -> bool:
        """
        This method is used to fix a bug in music21. Lead sheets with no repeats are not expandable, but nio specific
        exception is raised, therefore it is impossible to distinguish lead sheets that are not expandable because of
        a **wrong** structure, or because of an **absence** of structure.

        :param ls:
        :return:
        """
        for _ in ls.flat.getElementsByClass([RepeatExpression, RepeatBracket, Repeat]):
            return True
        return False

    def remove_da_capo_et_al(self):
        """
        Remove all elements of type RepeatExpression, encompassing Codas, segnos, DaCapos etc.
        """
        for p in self.ls.getElementsByClass(Part):
            for m in p.getElementsByClass(Measure):
                m.removeByClass([Coda, RepeatExpression])

    def sanity_check(self):
        if len(self.ls.getElementsByClass(Part)) > 1:
            raise MultiplePartsError(f'MULTIPLE PARTS: {self.name}')
        self.check_has_chords()
        try:
            self.check_is_monophonic()
        except (NotMonophonicError, OverlapError, GapError):
            self.fix_not_monophonic()
            self.check_is_monophonic()
        try:
            self.check_has_no_superimposed_chords()
        except SuperimposedChordsError:
            self.fix_superimposed_chords()
            self.check_has_no_superimposed_chords()

    def check_has_chords(self):
        if len(self.ls.recurse().getElementsByClass(ChordSymbol)) == 0:
            raise NoChordSymbolError(f'NO CHORD SYMBOLS: {self.name}')

    def fix_not_monophonic(self):
        for elt in self.ls.recurse():
            if type(elt) == Chord:
                # replaces the chord by its soprano (highest) note
                elt.sortDiatonicAscending(inPlace=True)
                site = elt.activeSite
                offset = elt.getOffsetBySite(site)
                site.remove(elt)
                site.insert(offset, elt[-1])

        crt_pos, crt_site, previous = None, None, None
        to_remove = []
        for elt in self.ls.recurse():
            if type(elt) not in [Chord, Note, Rest]:
                continue
            if type(elt) == Chord:
                raise ValueError('should not happen, chords were removed')

            offset = Fraction(elt.getOffsetBySite(elt.activeSite))
            offset.limit_denominator(max_denominator=10)
            ql = Fraction(elt.quarterLength)
            ql.limit_denominator(max_denominator=10)
            if elt.activeSite == crt_site:
                assert crt_pos is not None
                if offset < crt_pos:
                    if previous.getOffsetBySite(crt_site) == elt.getOffsetBySite(crt_site) \
                            and previous.quarterLength == elt.quarterLength:
                        if previous.isRest:
                            to_remove.append(previous)
                        elif elt.isRest:
                            to_remove.append(elt)
                        else:
                            raise OverlapError(f'OVERLAPS: {self.name} (beat: {crt_pos} in {crt_site})')
                    else:
                        raise OverlapError(f'OVERLAPS: {self.name} (beat: {crt_pos} in {crt_site})')
                if offset > crt_pos:
                    raise GapError(f'GAP: {self.name} (beat: {crt_pos} in {crt_site})')
            crt_site = elt.activeSite
            previous = elt
            crt_pos = offset + ql
            for elt in to_remove:
                self.ls.remove(elt, recurse=True)

    def check_is_monophonic(self):
        crt_pos = None
        crt_site = None
        for elt in self.ls.recurse():
            if type(elt) not in [Chord, Note, Rest]:
                continue
            if type(elt) == Chord:
                raise NotMonophonicError(f'NOT MONOPHONIC: {self.name} (beat: {crt_pos})')

            offset = Fraction(elt.getOffsetBySite(elt.activeSite))
            offset.limit_denominator(max_denominator=10)
            ql = Fraction(elt.quarterLength)
            ql.limit_denominator(max_denominator=10)
            if elt.activeSite == crt_site:
                assert crt_pos is not None
                if offset < crt_pos:
                    raise OverlapError(f'OVERLAPS: {self.name} (beat: {crt_pos})')
                if offset > crt_pos:
                    raise GapError(f'GAP: {self.name} (beat: {crt_pos})')
            else:
                crt_site = elt.activeSite
            crt_pos = offset + ql

    def check_has_no_superimposed_chords(self):
        previous_pos = None
        previous_site = None
        previous = None
        for elt in self.ls.recurse():
            if type(elt) == ChordSymbol:
                if previous is not None:
                    if previous_site == elt.activeSite and previous_pos == elt.getOffsetBySite(elt.activeSite):
                        raise SuperimposedChordsError(
                            f'SUPERIMPOSED CHORDS SYMBOLS: {self.name} at {previous_pos} in {previous_site} ({elt}, '
                            f'{previous})')
                previous = elt
                previous_site = elt.activeSite
                previous_pos = elt.getOffsetBySite(elt.activeSite)

    def fix_superimposed_chords(self):
        part = self.ls.getElementsByClass(Part)[0]  # type: Part
        if not part.hasMeasures():
            raise NoBarError()

        for measure in part.getElementsByClass(Measure):
            all_chord_symbols = [cs for cs in measure.getElementsByClass(ChordSymbol)]

            # create the dictionary mapping temporal positions to superimposed chords
            offset_to_chord_symbol = defaultdict(list)
            for chord_symbol in all_chord_symbols:
                offset_to_chord_symbol[chord_symbol.offset].append(chord_symbol)

            # remove duplicate superimposed chords
            for offset, chord_symbols in offset_to_chord_symbol.items():
                if len(chord_symbols) == 1:
                    continue
                unique_chord_symbols = [x[0] for x in groupby(chord_symbols, key=lambda cs: cs.figure)]
                if len(unique_chord_symbols) < len(chord_symbols):
                    for i in range(len(chord_symbols) - 1):
                        if chord_symbols[i].figure == chord_symbols[i + 1].figure:
                            measure.remove(chord_symbols[i + 1])
                    offset_to_chord_symbol[offset] = unique_chord_symbols

            # try to guess correct chord positions
            for offset, chord_symbols in offset_to_chord_symbol.items():
                if len(chord_symbols) == 1:
                    continue
                mql = measure.quarterLength
                # compute the number of free positions for future chords, the current position inclusive
                n = 1
                for pos in range(int(offset) + 1, int(mql)):
                    if pos in offset_to_chord_symbol:
                        break
                    n += 1
                if len(chord_symbols) > n:
                    raise SuperimposedChordsError(f'SUPERIMPOSED CHORDS SYMBOLS: {self.name} at {offset}  in {measure} '
                                                  f'chords {offset_to_chord_symbol[offset]}')
                if len(chord_symbols) == n:
                    # every beat gets a chord (no extra room)
                    for idx in range(1, n):
                        chord_symbols[idx].setOffsetBySite(measure, idx + offset)
                else:
                    # there is more room than chords
                    if len(chord_symbols) == 2:
                        middle_available = True
                        for pos in range(int(offset) + 1, int(mql / 2) + 1):
                            if pos in offset_to_chord_symbol:
                                middle_available = False
                        if middle_available:
                            chord_symbols[1].setOffsetBySite(measure, int(mql / 2))
                        else:
                            chord_symbols[1].setOffsetBySite(measure, offset + 1)
                    else:
                        for pos in range(1, len(chord_symbols)):
                            chord_symbols[pos].setOffsetBySite(measure, offset + pos)
            measure.isSorted = False

    def getParts(self) -> Stream:
        return self.ls.getElementsByClass(Part)

    def get_note_and_chord_list(self, use_measures: bool = False) -> List[NoteAndChord]:

        part = self.ls.getElementsByClass(Part)[0]  # type: Part
        if not part.hasMeasures() and use_measures:
            print('warning, no measures in lead sheet')

        if part.hasMeasures() and use_measures:
            return self.get_note_and_chord_list_by_measure()

        res = []
        crt_chord, crt_note = None, None
        for elt in self.ls.recurse().getElementsByClass(GeneralNote):
            offset = elt.offset
            ql = elt.quarterLength
            if isinstance(elt, ChordSymbol):
                crt_chord = elt
                # check that no note crosses a chord change
                if crt_note:
                    assert crt_note.offset + crt_note.quarterLength \
                           <= crt_chord.offset
            else:
                crt_note = elt
            if crt_note:
                res.append(NoteAndChord(crt_note, crt_chord, offset))
                crt_note = None

        return res

    def get_note_and_chord_list_by_measure(self):
        res = []
        part = self.ls.getElementsByClass(Part)[0]  # type: Part
        crt_chord, crt_note = None, None
        for measure in part.getElementsByClass(Measure):
            for elt in measure.getElementsByClass(GeneralNote):
                offset = elt.getOffsetBySite(self.ls.flat)
                if isinstance(elt, ChordSymbol):
                    crt_chord = elt
                    # check that no note crosses a chord change
                    if crt_note:
                        assert crt_note.offset + crt_note.quarterLength \
                               <= crt_chord.offset
                else:
                    crt_note = elt
                if crt_note:
                    res.append(NoteAndChord(crt_note, crt_chord, offset))
                    crt_note = None

        return res

    def printChordSymbols(self):
        for part in self.getParts():  # type: Part
            for elt in part.recurse().getElementsByClass(ChordSymbol):  # type: ChordSymbol
                print(str(elt) + "\t" + str(elt.activeSite) + "\t" + str(elt.getOffsetInHierarchy(part)))

    def getChordSequence(self):
        s = Stream()
        for part in self.getParts():  # type: Part
            for elt in part.recurse().getElementsByClass(ChordSymbol):  # type: ChordSymbol
                s.insert(elt.getOffsetInHierarchy(part), copy(elt))
        return s

    def printChordSequenceForFile(self):
        chords = []
        for part in self.getParts():  # type: Part
            chords = part.recurse().getElementsByClass(ChordSymbol)
            for elt in chords:  # type: ChordSymbol
                start = elt.getOffsetInHierarchy(part)
                print(str(harmony.chordSymbolFigureFromChord(elt)) + "@" + str(start), end=', ')
        if len(chords) > 0:
            print()

    def writeChordSequenceToFile(self, outfile):
        chords = []
        for part in self.getParts():  # type: Part
            chords = part.recurse().getElementsByClass(ChordSymbol)
            outfile.write(', '.join(
                [str(harmony.chordSymbolFigureFromChord(elt)) + "@" + str(elt.getOffsetInHierarchy(part)) for elt in
                 chords]))

    def hasChords(self):
        for part in self.getParts():  # type: Part
            chords = part.recurse().getElementsByClass(ChordSymbol)
            for elt in chords:  # type: ChordSymbol
                return True
        return False

    def show(self, *args, **kwargs):
        self.ls.show(*args, **kwargs)

    def melody_and_chords_streams(self) -> Tuple[Stream, Stream]:
        """
        The chord stream contains realized chords and chord symbols and rests for NC

        :return:
        """
        melody = Stream()
        chord_dict = defaultdict(list)
        measure_duration = None
        for measure_idx, measure in enumerate(self.ls.recurse().getElementsByClass(Measure)):
            if measure_duration is None:
                measure_duration = measure.duration.quarterLength
            else:
               if measure_duration != measure.duration.quarterLength:
                   raise WrongBarDurationError()
            mel_measure = measure.cloneEmpty()
            if measure_idx == 0:
                anacrusis = measure.barDuration.quarterLength - measure.duration.quarterLength
                if anacrusis:
                    mel_measure.append(Rest(duration=Duration(anacrusis)))
            for elt in measure:
                if elt.isClassOrSubclass((ChordSymbol,)):
                    chord_dict[measure_idx].append(elt)
                else:
                    mel_measure.append(deepcopy(elt))
            melody.append(mel_measure)
        chords = deepcopy(melody)
        clef = None
        for _clef in chords.recurse().getElementsByClass(Clef):
            clef = _clef
            break
        if clef:
            clef.activeSite.insert(0, BassClef())
            clef.activeSite.remove(clef)
        last_chord_symbol = None
        for measure_idx, measure in enumerate(chords.getElementsByClass(Measure)):
            original_measure_duration = measure.duration.quarterLength
            measure.removeByClass([Rest, Note])
            if chord_dict[measure_idx]:
                beats = [floor(ch.beat) for ch in chord_dict[measure_idx]] \
                        + [1 + original_measure_duration]
                durations = [(beats[i + 1] - beats[i]) for i in range(len(beats) - 1)]
                if beats[0] > 1:
                    if last_chord_symbol is None:
                        measure.insert(0, Rest(duration=Duration(beats[0] - 1)))
                    else:
                        _cs = deepcopy(last_chord_symbol)
                        _cs.duration = Duration(beats[0] - 1)
                        measure.insert(0, _cs)
                for chord_symbol_idx, chord_symbol in enumerate(chord_dict[measure_idx]):
                    chord_symbol.duration = Duration(durations[chord_symbol_idx])
                    measure.insert(beats[chord_symbol_idx] - 1, chord_symbol)
                    last_chord_symbol = chord_symbol
            else:
                if last_chord_symbol is None:
                    measure.insert(0, Rest(duration=Duration(original_measure_duration)))
                else:
                    _cs = deepcopy(last_chord_symbol)
                    _cs.duration = Duration(original_measure_duration)
                    measure.insert(0, _cs)
        return melody, chords
