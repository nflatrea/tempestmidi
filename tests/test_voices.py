import numpy as np
import pytest

from tempestmidi.source import Note
from tempestmidi.voices import VoiceAllocator, midi_to_hz


def slot_pitches(va, t):
    hz, lvl = va.state([t])
    return {round(float(h), 3) for h, l in zip(hz[0], lvl[0]) if l > 0}


def test_a4():
    assert midi_to_hz(69) == 440.0
    va = VoiceAllocator([Note(0, 1, 0, 69, 127)], slots=4)
    hz, _ = va.state([0.5])
    assert hz[0, 0] == pytest.approx(440.0)


def test_chord_exceeds_slots_highest_win():
    pitches = [60, 62, 64, 65, 67, 69]
    va = VoiceAllocator([Note(0, 1, 0, p, 100) for p in pitches], slots=4)
    assert slot_pitches(va, 0.5) == {round(midi_to_hz(p), 3) for p in [64, 65, 67, 69]}
    assert va.dropped_notes == 2


def test_freed_slot_taken_by_held_note():
    notes = [Note(0, 2, 0, 60, 100), Note(0, 2, 0, 62, 100), Note(0, 1, 0, 64, 100)]
    va = VoiceAllocator(notes, slots=2, release=0.0)
    assert slot_pitches(va, 0.5) == {round(midi_to_hz(p), 3) for p in [62, 64]}
    assert slot_pitches(va, 1.5) == {round(midi_to_hz(p), 3) for p in [60, 62]}
    assert va.dropped_notes == 0


def test_stable_assignment():
    notes = [Note(0, 2, 0, 60, 100), Note(1, 2, 0, 72, 100)]
    va = VoiceAllocator(notes, slots=4)
    hz0, _ = va.state([0.5])
    hz1, _ = va.state([1.5])
    assert hz0[0, 0] == pytest.approx(midi_to_hz(60))
    assert hz1[0, 0] == pytest.approx(midi_to_hz(60))  # first note kept slot 0
    assert hz1[0, 1] == pytest.approx(midi_to_hz(72))


def test_drums_ignored():
    va = VoiceAllocator([Note(0, 1, 9, 36, 127)], slots=4)
    _, lvl = va.state(np.linspace(0, 1, 50))
    assert np.all(lvl == 0)
    assert va.drum_notes == 1


def test_attack_ramp_and_release():
    va = VoiceAllocator([Note(0, 1, 0, 69, 127)], slots=1, attack=0.01, release=0.02)
    _, lvl = va.state([0.0, 0.005, 0.01, 0.5, 1.0, 1.01, 1.02, 1.5])
    assert lvl[:, 0] == pytest.approx([0, 0.5, 1, 1, 1, 0.5, 0, 0])


def test_default_level_is_full():
    va = VoiceAllocator([Note(0, 1, 0, 69, 64)], slots=1)
    _, lvl = va.state([0.5])
    assert lvl[0, 0] == 1.0


def test_velocity_scales_level():
    va = VoiceAllocator([Note(0, 1, 0, 69, 64)], slots=1, velocity=True)
    _, lvl = va.state([0.5])
    assert lvl[0, 0] == pytest.approx(64 / 127)


def test_idle_zero():
    va = VoiceAllocator([Note(1, 2, 0, 69, 100)], slots=3)
    _, lvl = va.state([0.0, 0.5, 3.0])
    assert np.all(lvl == 0)
    assert np.all(va.state([1.5])[1][0, 1:] == 0)


def test_note_ids_unique_and_idle():
    notes = [Note(0, 1, 0, 60, 100), Note(1, 2, 0, 62, 100), Note(0, 2, 0, 50, 100)]
    va = VoiceAllocator(notes, slots=2, release=0)
    ids = va.note_ids([0.5, 1.5, 3.0])
    assert ids[0, 0] != ids[1, 0]  # new note in same slot -> new id
    assert ids[0, 1] == ids[1, 1]  # sustained note keeps its id
    assert np.all(ids[2] == -1)
