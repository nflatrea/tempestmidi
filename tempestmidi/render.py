"""Frame renderer: voice slots -> full-width scanline band frames.

Every row is uniform across the full width to maximize emitted power.
White vs black rows encode to different TMDS symbol streams, so the 
row on/off sequence AM-keys the HDMI emission. Voices are mixed per row:

- sd  : normalized sum of voices -> 1 bit per row (first-order sigma-delta)
- arp : one active voice per frame, longest-waiting first (chiptune arpeggio)
- rgb : active voices spread over color channels R, G, B (separate TMDS lanes)

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

MIX_MODES = ("sd", "arp", "rgb")


@dataclass(frozen=True)
class VideoMode:
    width: int = 1920
    height: int = 1080
    v_total: int = 1125
    refresh: Fraction = Fraction(60)

    def __post_init__(self):
        if not (0 < self.height <= self.v_total) or self.width <= 0 or self.refresh <= 0:
            raise ValueError(f"invalid video mode {self}")

    @property
    def line_rate(self) -> float:
        """Lines per second (horizontal frequency) = v_total * refresh."""
        return float(self.v_total * self.refresh)

    @property
    def fps(self) -> float:
        return float(self.refresh)

    @classmethod
    def parse(cls, text: str) -> "VideoMode":
        """Parse 'WIDTHxHEIGHTxVTOTAL@REFRESH', refresh as int, decimal or fraction."""
        m = re.fullmatch(r"\s*(\d+)x(\d+)x(\d+)@([\d./]+)\s*", text)
        if not m:
            raise ValueError(f"bad mode {text!r}, expected e.g. 1920x1080x1125@60")
        w, h, vt, r = m.groups()
        return cls(int(w), int(h), int(vt), Fraction(r))

    def __str__(self) -> str:
        return f"{self.width}x{self.height}x{self.v_total}@{self.refresh}"


def sigma_delta(x: np.ndarray, acc0: float = 0.5) -> np.ndarray:
    """First-order sigma-delta along axis 0 for values in [0, 1] -> {0, 1}."""
    s = acc0 + np.cumsum(x, axis=0)
    fl = np.floor(s)
    prev = np.concatenate([np.full((1,) + x.shape[1:], np.floor(acc0)), fl[:-1]], axis=0)
    return (fl - prev).astype(np.uint8)


def _normalized_mix(level: np.ndarray, high: np.ndarray) -> np.ndarray:
    """sum(level*high) / max(1, sum(level)) per row."""
    return (level * high).sum(axis=1) / np.maximum(1.0, level.sum(axis=1))


class FrameRenderer:
    """Renders frame n of the TEMPEST video for a voice source.

    `voices` must provide `.slots` and `.state(t) -> (hz, level)` arrays of
    shape (len(t), slots), like `VoiceAllocator`.
    """

    def __init__(self, voices, mode: VideoMode = VideoMode(), mix: str = "sd"):
        if mix not in MIX_MODES:
            raise ValueError(f"unknown mix mode {mix!r}, expected one of {MIX_MODES}")
        self.voices = voices
        self.mode = mode
        self.mix = mix
        self.slots = voices.slots
        self._rows = np.arange(mode.height)
        self._arp_sel: list[int] = []  # selected slot per frame (-1 = none), computed in order
        self._arp_last: dict[int, int] = {}  # note id -> last frame drawn

    @property
    def color(self) -> bool:
        return self.mix == "rgb"

    def line_times(self, n: int) -> np.ndarray:
        """Absolute times (s) of the visible rows of frame n (vblank lines counted)."""
        g = n * self.mode.v_total + self._rows
        return g / self.mode.line_rate

    def slot_state(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(high 0/1, level) per visible row and slot for frame n."""
        t = self.line_times(n)
        hz, lvl = self.voices.state(t)
        high = (np.mod(hz * t[:, None], 1.0) < 0.5).astype(float)
        return high, lvl

    def rows(self, n: int) -> np.ndarray:
        """Row on/off states (height, 3) as uint8 0/1 for channels R, G, B."""
        high, lvl = self.slot_state(n)
        if self.mix == "sd":
            bit = sigma_delta(_normalized_mix(lvl, high))
            return np.repeat(bit[:, None], 3, axis=1)
        if self.mix == "arp":
            s = self.arp_slot(n)
            if s < 0:
                return np.zeros((self.mode.height, 3), np.uint8)
            bit = sigma_delta(lvl[:, s] * high[:, s])
            return np.repeat(bit[:, None], 3, axis=1)
        # rgb: slots active in this frame -> R, G, B in rotation (balanced lanes)
        out = np.zeros((self.mode.height, 3), np.uint8)
        active = np.flatnonzero(lvl.max(axis=0) > 0)
        for ch in range(3):
            idx = active[ch::3]
            if idx.size:
                out[:, ch] = sigma_delta(_normalized_mix(lvl[:, idx], high[:, idx]))
        return out

    # arpeggio scheduling
    def _note_ids(self, t: np.ndarray) -> np.ndarray:
        if hasattr(self.voices, "note_ids"):
            return self.voices.note_ids(t)
        _, lvl = self.voices.state(t)  # fallback: one "note" per slot
        return np.where(lvl > 0, np.arange(self.slots)[None, :], -1)

    def arp_slot(self, n: int) -> int:
        """Slot drawn in frame n (`arp` mode): the active note that has waited
        longest since it was last drawn (never-drawn first, ties -> lowest slot)."""
        lr, vt, h = self.mode.line_rate, self.mode.v_total, self.mode.height
        while len(self._arp_sel) <= n:
            a = len(self._arp_sel)
            b = max(n + 1, a + 600)
            frames = np.arange(a, b)
            start = self._note_ids(frames * vt / lr)  # active at frame start
            end = self._note_ids((frames * vt + h - 1) / lr)  # fallback: began mid-frame
            for i, f in enumerate(frames):
                ids = start[i] if (start[i] >= 0).any() else end[i]
                best, best_key = -1, None
                for s in np.flatnonzero(ids >= 0):
                    key = self._arp_last.get(int(ids[s]), -1)
                    if best_key is None or key < best_key:
                        best, best_key = int(s), key
                if best >= 0:
                    self._arp_last[int(ids[best])] = int(f)
                self._arp_sel.append(best)
        return self._arp_sel[n]

    def emission(self, n: int) -> np.ndarray:
        """Relative emission per visible row (mean of channel on/off states)."""
        return self.rows(n).mean(axis=1)

    def render(self, n: int) -> np.ndarray:
        """Frame n: uint8 (height, width) gray, or (height, width, 3) RGB in rgb mode.
        Every row is uniform; values are 0 or 255 only."""
        r = self.rows(n) * np.uint8(255)
        w = self.mode.width
        if self.color:
            return np.ascontiguousarray(np.broadcast_to(r[:, None, :], (r.shape[0], w, 3)))
        return np.ascontiguousarray(np.broadcast_to(r[:, :1], (r.shape[0], w)))

    def frame_count(self, duration: float) -> int:
        return max(1, int(np.ceil(duration * self.mode.fps)))
