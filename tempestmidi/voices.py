"""Voice allocation: notes -> fixed voice slots with level envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .source import Note

PERCUSSION_CHANNEL = 9  # GM channel 10, 0-based


def midi_to_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


@dataclass
class _SlotTrack:
    on: list[float] = field(default_factory=list)
    off: list[float] = field(default_factory=list)
    hz: list[float] = field(default_factory=list)
    vel: list[float] = field(default_factory=list)


class VoiceAllocator:
    """Assigns notes to `slots` voice slots (highest pitches win) and answers
    per-slot (frequency, level) queries at arbitrary times."""

    def __init__(self, notes: list[Note], slots: int = 4,
                 attack: float = 0.005, release: float = 0.030, velocity: bool = False):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        self.slots = slots
        self.attack = attack
        self.release = release
        self.velocity = velocity
        melodic = [n for n in notes if n.channel != PERCUSSION_CHANNEL and n.end > n.start]
        self.drum_notes = len(notes) - len(melodic)
        self.notes = melodic
        self.duration = max((n.end for n in notes), default=0.0)
        self._tracks = [_SlotTrack() for _ in range(slots)]
        self.dropped_notes = self._allocate()
        self._arrays = [
            tuple(np.asarray(getattr(tr, k), dtype=float) for k in ("on", "off", "hz", "vel"))
            for tr in self._tracks
        ]

    # allocation sweep
    def _allocate(self) -> int:
        notes = self.notes
        times = sorted({n.start for n in notes} | {n.end for n in notes})
        by_start: dict[float, list[int]] = {}
        by_end: dict[float, list[int]] = {}
        for i, n in enumerate(notes):
            by_start.setdefault(n.start, []).append(i)
            by_end.setdefault(n.end, []).append(i)

        active: set[int] = set()
        slot_of: dict[int, int] = {}  # note index -> slot
        since: dict[int, float] = {}  # note index -> time slot acquired
        ever: set[int] = set()

        def close(i: int, t: float):
            s = slot_of.pop(i)
            tr = self._tracks[s]
            n = notes[i]
            tr.on.append(since.pop(i))
            tr.off.append(t)
            tr.hz.append(midi_to_hz(n.pitch))
            tr.vel.append(n.velocity / 127.0 if self.velocity else 1.0)

        for t in times:
            for i in by_end.get(t, ()):
                active.discard(i)
            for i in by_start.get(t, ()):
                active.add(i)
            ranked = sorted(active, key=lambda i: (-notes[i].pitch, notes[i].start, i))
            chosen = set(ranked[: self.slots])
            for i in [i for i in slot_of if i not in chosen]:  # ended or displaced
                close(i, t)
            used = set(slot_of.values())
            free = [s for s in range(self.slots) if s not in used]
            for i in ranked[: self.slots]:
                if i not in slot_of:
                    slot_of[i] = free.pop(0)
                    since[i] = t
                    ever.add(i)
        for i in list(slot_of):  # should be empty: every note ends at some event time
            close(i, times[-1])
        return len(notes) - len(ever)

    # queries
    def state(self, t) -> tuple[np.ndarray, np.ndarray]:
        """Return (hz, level) arrays of shape (len(t), slots) for times `t` in seconds.
        Idle slots report level 0 (and hz 0 when nothing has played yet).
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        hz = np.zeros((t.size, self.slots))
        lvl = np.zeros((t.size, self.slots))
        for s, (on, off, f, vel) in enumerate(self._arrays):
            if on.size == 0:
                continue
            k = np.searchsorted(on, t, side="right") - 1
            valid = k >= 0
            kk = np.where(valid, k, 0)
            t_on, t_off, v = on[kk], off[kk], vel[kk]
            att = self._attack_gain
            held = t < t_off
            l_held = v * att(t - t_on)
            l_rel = v * att(t_off - t_on) * np.clip(1.0 - (t - t_off) / self.release, 0.0, 1.0) \
                if self.release > 0 else np.zeros_like(t)
            level = np.where(held, l_held, l_rel)
            lvl[:, s] = np.where(valid, level, 0.0)
            hz[:, s] = np.where(valid, f[kk], 0.0)
        return hz, lvl

    def note_ids(self, t) -> np.ndarray:
        """Per-slot id of the note sounding (level > 0) at times `t`, shape
        (len(t), slots); -1 when the slot is silent. Ids are unique per note-on
        in a slot (a re-acquired note gets a new id)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        _, lvl = self.state(t)
        ids = np.full((t.size, self.slots), -1, dtype=np.int64)
        offset = 0
        for s, (on, *_rest) in enumerate(self._arrays):
            if on.size:
                k = np.searchsorted(on, t, side="right") - 1
                ids[:, s] = np.where((k >= 0) & (lvl[:, s] > 0), offset + k, -1)
            offset += on.size
        return ids

    def _attack_gain(self, dt):
        if self.attack <= 0:
            return np.ones_like(dt)
        return np.clip(dt / self.attack, 0.0, 1.0)
