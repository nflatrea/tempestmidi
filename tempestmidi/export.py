"""Video export: rendered frames -> mp4 via the bundled ffmpeg binary."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .render import FrameRenderer

AUDIO_RATE = 48000


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class ExportSummary:
    duration: float
    frames: int
    voices: int
    mix: str
    line_rate: float
    dropped_notes: int
    output: str

    def line(self) -> str:
        return (f"done: {self.output}  duration={self.duration:.2f}s frames={self.frames} "
                f"voices={self.voices} mix={self.mix} line_rate={self.line_rate:.2f}Hz "
                f"dropped_notes={self.dropped_notes}")


def preview_mix(renderer: FrameRenderer, duration: float, rate: int = AUDIO_RATE) -> np.ndarray:
    """Expected demodulated audio: per-line emission of the rendered rows (0 during
    vertical blanking), lowpassed, DC-removed, normalized, resampled to `rate`."""
    mode = renderer.mode
    lr = mode.line_rate
    frames = renderer.frame_count(duration)
    out = np.zeros(frames * mode.v_total)
    for n in range(frames):
        a = n * mode.v_total
        out[a:a + mode.height] = renderer.emission(n)
    n_lines = min(out.size, int(np.ceil(duration * lr)))
    out = out[:n_lines]
    # moving-average lowpass (~6 kHz first null at 67.5 kHz) before linear resampling
    k = max(1, int(round(lr / 6000)))
    if out.size >= k > 1:
        out = np.convolve(out, np.ones(k) / k, mode="same")
    t_out = np.arange(int(round(duration * rate))) / rate
    y = np.interp(t_out, np.arange(n_lines) / lr, out)
    y = _dc_block(y, rate)
    peak = np.max(np.abs(y)) if y.size else 0.0
    return (y / peak * 0.9) if peak > 0 else y


def _dc_block(x: np.ndarray, rate: int, cutoff: float = 20.0) -> np.ndarray:
    if x.size == 0:
        return x
    a = np.exp(-2 * np.pi * cutoff / rate)
    # y[n] = x[n] - x[n-1] + a*y[n-1], vectorized in blocks via lfilter-free loop on chunks
    y = np.empty_like(x)
    prev_x = x[0]
    prev_y = 0.0
    for i in range(0, x.size, 4096):
        blk = x[i:i + 4096]
        dx = np.diff(np.concatenate([[prev_x], blk]))
        # IIR y = dx + a*y_prev: closed form with powers of a
        k = np.arange(blk.size)
        powers = a ** k
        acc = np.cumsum(dx / (a ** k)) * powers  # sum_j dx_j a^(k-j)
        yb = acc + prev_y * a ** (k + 1)
        y[i:i + blk.size] = yb
        prev_x, prev_y = blk[-1], yb[-1]
    return y


def write_wav(path: str, samples: np.ndarray, rate: int = AUDIO_RATE) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def export_video(renderer: FrameRenderer, duration: float, output: str,
                 preview_audio: bool = False, crf: int | None = None, dropped_notes: int = 0,
                 progress: Callable[[int, int], None] | None = None) -> ExportSummary:
    """Render `duration` seconds of frames and encode them to `output` (mp4)."""
    mode = renderer.mode
    if crf is None:
        # rgb: yuv444p already requires High 4:4:4 Predictive, so lossless (crf 0)
        # costs no extra compatibility; lossy chroma left lanes up to 18 off
        crf = 0 if renderer.color else 4
    frames = renderer.frame_count(duration)
    rate = f"{mode.refresh.numerator}/{mode.refresh.denominator}"
    video_seconds = frames / mode.fps

    tmp_wav = None
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24" if renderer.color else "gray",
           "-s", f"{mode.width}x{mode.height}",
           "-r", rate, "-i", "-"]
    if preview_audio:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        write_wav(tmp_wav, preview_mix(renderer, video_seconds))
        cmd += ["-i", tmp_wav]
    cmd += ["-map", "0:v"]
    if preview_audio:
        cmd += ["-map", "1:a", "-c:a", "aac", "-b:a", "128k", "-t", f"{video_seconds:.6f}"]
    # deblocking smears horizontal band edges -> disabled; flat content keeps files small
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-x264-params", "no-deblock=1",
            # rgb lanes: 4:4:4 so chroma is not shared between adjacent rows
            "-pix_fmt", "yuv444p" if renderer.color else "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-r", rate, "-movflags", "+faststart", output]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for n in range(frames):
            proc.stdin.write(renderer.render(n).tobytes())
            if progress:
                progress(n + 1, frames)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")
    except BrokenPipeError as e:
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
        raise RuntimeError(f"ffmpeg failed: {err.strip()}") from e
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    return ExportSummary(duration=video_seconds, frames=frames, voices=renderer.slots, mix=renderer.mix,
                         line_rate=mode.line_rate, dropped_notes=dropped_notes, output=output)


def stderr_progress(min_interval: float = 1.0) -> Callable[[int, int], None]:
    last = [0.0]

    def cb(i: int, total: int) -> None:
        now = time.monotonic()
        if i == total or now - last[0] >= min_interval:
            last[0] = now
            sys.stderr.write(f"\rrendering frame {i}/{total} ({100 * i / total:5.1f}%)")
            if i == total:
                sys.stderr.write("\n")
            sys.stderr.flush()

    return cb
