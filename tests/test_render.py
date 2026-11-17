from fractions import Fraction

import numpy as np
import pytest

from tempestmidi.render import FrameRenderer, VideoMode, sigma_delta


class ConstVoices:
    """Fixed per-slot (hz, level)."""

    def __init__(self, hz, lvl):
        self.hz = np.asarray(hz, float)
        self.lvl = np.asarray(lvl, float)
        self.slots = len(hz)

    def state(self, t):
        t = np.atleast_1d(t)
        return (np.broadcast_to(self.hz, (t.size, self.slots)).copy(),
                np.broadcast_to(self.lvl, (t.size, self.slots)).copy())


def line_signal(r, frames, ch=None):
    """Concatenate per-line emission (visible rows; 0 in vblank)."""
    m = r.mode
    x = np.zeros(frames * m.v_total)
    for n in range(frames):
        rows = r.rows(n)
        v = rows.mean(axis=1) if ch is None else rows[:, ch]
        x[n * m.v_total: n * m.v_total + m.height] = v
    return x


def spectrum(x, rate, pad=8):
    x = x - x.mean()
    n = pad * x.size
    return np.fft.rfftfreq(n, 1 / rate), np.abs(np.fft.rfft(x, n))


def peak(freqs, spec, lo=100, hi=1000):
    band = (freqs > lo) & (freqs < hi)
    return freqs[band][np.argmax(spec[band])]


def amp_at(freqs, spec, f):
    return spec[np.argmin(np.abs(freqs - f))]


def test_mode_default_and_parse():
    m = VideoMode()
    assert (m.width, m.height, m.v_total) == (1920, 1080, 1125)
    assert m.line_rate == 67500.0
    m2 = VideoMode.parse("1920x1080x1125@60000/1001")
    assert m2.line_rate == pytest.approx(1125 * 60000 / 1001)
    assert VideoMode.parse("1280x720x750@59.94").refresh == Fraction("59.94")
    with pytest.raises(ValueError):
        VideoMode.parse("1080p")


def test_unknown_mix():
    with pytest.raises(ValueError):
        FrameRenderer(ConstVoices([440], [1]), mix="strips")


def test_sigma_delta_mean_and_bits():
    x = np.full(1000, 0.3)
    b = sigma_delta(x)
    assert set(np.unique(b)) <= {0, 1}
    assert b.mean() == pytest.approx(0.3, abs=0.002)
    assert np.array_equal(sigma_delta(np.array([1.0, 0, 1, 1, 0])), [1, 0, 1, 1, 0])


@pytest.mark.parametrize("mix", ["sd", "arp", "rgb"])
def test_rows_uniform_full_width_0_255(mix):
    r = FrameRenderer(ConstVoices([300, 450, 700, 1000], [1, 0.3, 0.77, 0.5]), mix=mix)
    for n in (0, 1, 37):
        f = r.render(n)
        assert f.shape[:2] == (1080, 1920) and f.dtype == np.uint8
        assert f.ndim == (3 if mix == "rgb" else 2)
        assert set(np.unique(f)) <= {0, 255}
        assert np.all(f == f[:, :1])  # every row uniform across width


def test_single_voice_equals_eliza_square():
    f = 523.25
    r = FrameRenderer(ConstVoices([f], [1.0]))
    m = r.mode
    for n in (0, 1, 2):
        g = n * m.v_total + np.arange(m.height)
        expected = (np.mod(f * g / m.line_rate, 1.0) < 0.5) * 255
        assert np.array_equal(r.render(n)[:, 0], expected)


@pytest.mark.parametrize("mix", ["sd", "arp", "rgb"])
def test_silence_black(mix):
    r = FrameRenderer(ConstVoices([440, 523], [0, 0]), mix=mix)
    assert not r.render(5).any()


@pytest.mark.parametrize("mode", [VideoMode(), VideoMode.parse("1920x1080x1125@60000/1001")])
def test_pitch_440_within_half_percent(mode):
    r = FrameRenderer(ConstVoices([440.0], [1.0]), mode)
    x = line_signal(r, int(round(mode.fps)))
    assert peak(*spectrum(x, mode.line_rate)) == pytest.approx(440.0, rel=0.005)


def test_sd_chord_has_both_pitches():
    r = FrameRenderer(ConstVoices([220.0, 330.0], [1.0, 1.0]))
    freqs, spec = spectrum(line_signal(r, 60), r.mode.line_rate)
    floor = np.median(spec[(freqs > 100) & (freqs < 1000)])
    assert amp_at(freqs, spec, 220) > 20 * floor
    assert amp_at(freqs, spec, 330) > 20 * floor


def test_arp_alternates_slots():
    hz = [220.0, 330.0]
    r = FrameRenderer(ConstVoices(hz, [1.0, 1.0]), mix="arp")
    m = r.mode
    for n in range(4):
        g = n * m.v_total + np.arange(m.height)
        expected = np.mod(hz[n % 2] * g / m.line_rate, 1.0) < 0.5
        assert np.array_equal(r.rows(n)[:, 0], expected)


def test_rgb_channels_equal_slots():
    hz = [220.0, 330.0, 495.0]
    r = FrameRenderer(ConstVoices(hz, [1, 1, 1]), mix="rgb")
    m = r.mode
    g = 3 * m.v_total + np.arange(m.height)
    f = r.render(3)
    for ch in range(3):
        expected = (np.mod(hz[ch] * g / m.line_rate, 1.0) < 0.5) * 255
        assert np.array_equal(f[:, 0, ch], expected)


def test_emission_white_row_is_one():
    r = FrameRenderer(ConstVoices([0.01, 0.01, 0.01], [1, 1, 1]), mix="rgb")
    assert r.emission(0)[0] == 1.0


def test_phase_continuity_across_frames():
    r = FrameRenderer(ConstVoices([523.25], [1.0]))
    assert r.line_times(1)[0] == pytest.approx(1125 / 67500)  # vblank counted, no reset


# -- stress: more than 2 voices ---------------------------------------------------

from tempestmidi.source import Note  # noqa: E402
from tempestmidi.voices import VoiceAllocator  # noqa: E402


def dense_song(seconds=4.0):
    """Sustained 3-note chord (bars of 1 s) + fast 80 ms melody run on top."""
    notes = []
    for bar in range(int(seconds)):
        for p in (48, 52, 55):
            notes.append(Note(bar, bar + 1, 0, p, 100))
    t, k = 0.0, 0
    while t < seconds:
        notes.append(Note(t, t + 0.08, 1, 72 + (k * 5) % 12, 100))
        t += 0.08
        k += 1
    return notes


def test_arp_new_note_drawn_next_frame():
    va = VoiceAllocator(dense_song(), slots=4, attack=0, release=0)
    r = FrameRenderer(va, mix="arp")
    m = r.mode
    nf = int(4 * m.fps)
    starts = np.arange(nf) * m.v_total / m.line_rate
    ids = va.note_ids(starts)
    for n in range(1, nf):
        new = set(ids[n][ids[n] >= 0]) - set(ids[n - 1][ids[n - 1] >= 0])
        if len(new) == 1:  # exactly one onset: it must be drawn now
            s = r.arp_slot(n)
            assert ids[n][s] in new


def test_arp_no_undrawn_notes_dense():
    va = VoiceAllocator(dense_song(), slots=4, attack=0, release=0)
    r = FrameRenderer(va, mix="arp")
    m = r.mode
    nf = int(4 * m.fps)
    starts = np.arange(nf) * m.v_total / m.line_rate
    ids = va.note_ids(starts)
    visible = set(ids[ids >= 0].tolist())
    drawn = {int(ids[n][r.arp_slot(n)]) for n in range(nf) if r.arp_slot(n) >= 0}
    assert visible <= drawn


def test_arp_seekable_matches_sequential():
    va = VoiceAllocator(dense_song(), slots=4)
    seq = FrameRenderer(va, mix="arp")
    order = [seq.arp_slot(n) for n in range(200)]
    rnd = FrameRenderer(va, mix="arp")
    assert rnd.arp_slot(150) == order[150] and rnd.arp_slot(3) == order[3]


def test_rgb_balanced_lanes():
    hz = [220.0, 0, 0, 330.0, 495.0, 0]
    r = FrameRenderer(ConstVoices(hz, [1, 0, 0, 1, 1, 0]), mix="rgb")
    m = r.mode
    g = np.arange(m.height)
    rows = r.rows(0)
    for ch, f in enumerate([220.0, 330.0, 495.0]):
        assert np.array_equal(rows[:, ch], np.mod(f * g / m.line_rate, 1.0) < 0.5)


def test_rgb_six_voices_two_per_lane():
    hz = [200.0, 250.0, 300.0, 350.0, 400.0, 450.0]
    r = FrameRenderer(ConstVoices(hz, [1] * 6), mix="rgb")
    x = line_signal(r, 60, ch=0)  # red lane carries slots 0 and 3
    freqs, spec = spectrum(x, r.mode.line_rate)
    floor = np.median(spec[(freqs > 100) & (freqs < 1000)])
    assert amp_at(freqs, spec, 200) > 20 * floor and amp_at(freqs, spec, 350) > 20 * floor
    # exact lane content: red = sigma-delta mix of slots 0 and 3 only
    m = r.mode
    t = np.arange(m.height) / m.line_rate
    hi = (np.mod(np.array(hz)[None, :] * t[:, None], 1.0) < 0.5).astype(float)
    assert np.array_equal(r.rows(0)[:, 0], sigma_delta(hi[:, [0, 3]].sum(1) / 2))
