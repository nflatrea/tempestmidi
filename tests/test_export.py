import re
import subprocess

import numpy as np
import pytest

from tempestmidi.export import _dc_block, export_video, ffmpeg_exe, preview_mix
from tempestmidi.render import FrameRenderer, VideoMode
from tempestmidi.source import Note
from tempestmidi.voices import VoiceAllocator, midi_to_hz

MODE = VideoMode()


def renderer(pitches, duration, mix="sd"):
    notes = [Note(0, duration, 0, p, 127) for p in pitches]
    va = VoiceAllocator(notes, slots=max(1, len(pitches)), attack=0, release=0)
    return FrameRenderer(va, MODE, mix=mix)


def decode(path, color=False, w=MODE.width, h=MODE.height):
    fmt, ch = ("rgb24", 3) if color else ("gray", 1)
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", fmt, "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8)
    return a.reshape(-1, h, w, ch) if color else a.reshape(-1, h, w)


def probe(path):
    return subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          capture_output=True).stderr.decode()


def decoded_audio_seconds(path, rate=48000):
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:a",
                          "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"],
                         capture_output=True, check=True).stdout
    return len(raw) / 2 / rate


def within16(v):
    v = v.astype(int)
    return np.all(np.minimum(v, 255 - v) <= 16)


@pytest.fixture(scope="module")
def mono_mp4(tmp_path_factory):
    out = tmp_path_factory.mktemp("v") / "mono.mp4"
    r = renderer([57], 2.0)  # A3 220 Hz
    return out, export_video(r, 2.0, str(out)), r


def test_cfr_60fps_420_and_summary(mono_mp4):
    out, summ, _ = mono_mp4
    info = probe(out)
    assert "1920x1080" in info and re.search(r"\b60 fps", info) and "yuv420p" in info
    assert summ.frames == 120 and summ.voices == 1 and summ.line_rate == 67500.0
    assert "frames=120" in summ.line() and "mix=sd" in summ.line()


def test_round_trip_mono_pitch_and_rows(mono_mp4):
    out, _, r = mono_mp4
    frames = decode(out)
    assert len(frames) == 120 and within16(frames)
    x = np.zeros(len(frames) * MODE.v_total)
    for n, f in enumerate(frames):
        x[n * MODE.v_total: n * MODE.v_total + MODE.height] = f[:, 960] > 127
    x -= x.mean()
    pad = 8 * x.size
    spec = np.abs(np.fft.rfft(x, pad))
    freqs = np.fft.rfftfreq(pad, 1 / MODE.line_rate)
    band = (freqs > 100) & (freqs < 1000)
    assert freqs[band][np.argmax(spec[band])] == pytest.approx(midi_to_hz(57), rel=0.01)
    for n in (0, 7, 119):
        assert np.array_equal(frames[n] > 127, r.render(n) > 127)


def test_round_trip_sd_chord(tmp_path):
    r = renderer([60, 64, 67, 72], 1.0)
    out = tmp_path / "c.mp4"
    export_video(r, 1.0, str(out))
    frames = decode(out)
    assert within16(frames)
    for n in (0, 30, 59):
        assert np.array_equal(frames[n] > 127, r.render(n) > 127)


def test_round_trip_rgb(tmp_path):
    r = renderer([57, 64, 69], 1.0, mix="rgb")
    out = tmp_path / "rgb.mp4"
    export_video(r, 1.0, str(out))
    assert "yuv444p" in probe(out)
    frames = decode(out, color=True)
    assert within16(frames)
    for n in (0, 30, 59):
        assert np.array_equal(frames[n] > 127, r.render(n) > 127)


def test_no_audio_by_default(mono_mp4):
    assert "Audio:" not in probe(mono_mp4[0])


def test_preview_audio_track(tmp_path):
    out = tmp_path / "a.mp4"
    summ = export_video(renderer([57, 64], 2.0), 2.0, str(out), preview_audio=True)
    assert "Audio:" in probe(out)
    assert decoded_audio_seconds(out) == pytest.approx(summ.frames / 60, abs=1 / 60)


@pytest.mark.parametrize("mix", ["sd", "rgb"])
def test_preview_mix_spectrum(mix):
    y = preview_mix(renderer([57, 64], 1.0, mix=mix), 1.0)
    assert y.size == 48000 and np.max(np.abs(y)) == pytest.approx(0.9)
    seg = y[4800:]  # skip DC-block settling
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(seg.size, 1 / 48000)
    floor = np.median(spec[(freqs > 100) & (freqs < 1000)])
    for f in (midi_to_hz(57), midi_to_hz(64)):
        assert spec[np.argmin(np.abs(freqs - f))] > 20 * floor


def test_dc_block_removes_offset():
    y = _dc_block(np.full(48000, 0.7), 48000)
    assert abs(y[-1]) < 1e-3
