"""MIDI sources: bitmidi.com client with local cache, local files, note normalization."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

BITMIDI_BASE = "https://bitmidi.com"
USER_AGENT = "tempestmidi/v1 (+https://github.com/; MIDI to TEMPEST video converter)"

class SourceError(Exception):
    """MIDI source failure (network, API format, parse)."""

@dataclass(frozen=True)
class Note:
    start: float  # seconds
    end: float    # seconds
    channel: int  # 0-based (GM percussion = 9)
    pitch: int    # MIDI note number
    velocity: int # 1..127

@dataclass(frozen=True)
class SearchResult:
    id: int
    name: str


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "tempestmidi"


class BitmidiClient:
    """Minimal client for undocumented bitmidi.com's JSON API."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        session: Any = None,
        min_interval: float = 1.0,
        timeout: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.min_interval = min_interval
        self.timeout = timeout
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    # networking
    def _get(self, path: str, params: dict | None = None):
        if self._last is not None:
            wait = self.min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        url = BITMIDI_BASE + path
        try:
            resp = self.session.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=self.timeout
            )
            resp.raise_for_status()
        except Exception as e:  # requests.RequestException and friends
            raise SourceError(f"bitmidi request failed: {url}: {e}") from e
        finally:
            self._last = self._clock()
        return resp

    # API
    def search(self, query: str, page: int = 0) -> list[SearchResult]:
        resp = self._get("/api/midi/search", {"q": query, "page": page})
        try:
            data = resp.json()
            results = data["result"]["results"]
            if not isinstance(results, list):
                raise TypeError("results is not a list")
            return [SearchResult(id=int(r["id"]), name=str(r["name"])) for r in results]
        except (ValueError, KeyError, TypeError) as e:
            raise SourceError(f"unexpected bitmidi API response format: {e}") from e

    def cache_path(self, midi_id: int) -> Path:
        return self.cache_dir / f"{int(midi_id)}.mid"

    def fetch(self, midi_id: int) -> Path:
        """Return local path of MIDI id, downloading into the cache if needed."""
        path = self.cache_path(midi_id)
        if path.exists():
            return path
        resp = self._get(f"/uploads/{int(midi_id)}.mid")
        content = resp.content
        if not content.startswith(b"MThd"):
            raise SourceError(f"bitmidi id {midi_id}: response is not a MIDI file")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return path


def resolve_source(spec: str, client: BitmidiClient | None = None) -> Path:
    """`bitmidi:<id>` -> cached path, otherwise a local path."""
    if spec.startswith("bitmidi:"):
        try:
            midi_id = int(spec.split(":", 1)[1])
        except ValueError as e:
            raise SourceError(f"invalid bitmidi id in {spec!r}") from e
        return (client or BitmidiClient()).fetch(midi_id)
    path = Path(spec)
    if not path.is_file():
        raise SourceError(f"file not found: {spec}")
    return path


def load_notes(path: str | os.PathLike) -> list[Note]:
    """Parse a MIDI type 0/1 file into time-ordered notes (seconds, tempo applied)."""
    import mido

    try:
        mid = mido.MidiFile(str(path))
    except Exception as e:  # mido raises OSError/EOFError/ValueError/KeyError
        raise SourceError(f"cannot parse MIDI file {path}: {e}") from e
    if mid.type not in (0, 1):
        raise SourceError(f"unsupported MIDI file type {mid.type} (only 0 and 1)")

    notes: list[Note] = []
    active: dict[tuple[int, int], list[tuple[float, int]]] = {}
    t = 0.0
    try:
        for msg in mid:  # merged tracks, delta times in seconds with tempo applied
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append((t, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                stack = active.get((msg.channel, msg.note))
                if stack:
                    start, vel = stack.pop(0)
                    if t > start:
                        notes.append(Note(start, t, msg.channel, msg.note, vel))
    except Exception as e:
        raise SourceError(f"cannot read MIDI events from {path}: {e}") from e
    for (ch, pitch), stack in active.items():  # close dangling notes at end of file
        for start, vel in stack:
            if t > start:
                notes.append(Note(start, t, ch, pitch, vel))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
