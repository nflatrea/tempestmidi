import json

import mido
import pytest

from tempestmidi.source import BitmidiClient, SourceError, load_notes, resolve_source


class FakeResp:
    def __init__(self, status=200, body=b"", json_data=None):
        self.status_code = status
        self.content = body
        self._json = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._json is None:
            return json.loads(self.content.decode())
        return self._json


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(tmp_path, responses, **kw):
    s = FakeSession(responses)
    c = BitmidiClient(cache_dir=tmp_path / "cache", session=s, sleep=lambda x: None, **kw)
    return c, s


def midi_bytes(tmp_path):
    mid = mido.MidiFile(type=0)
    tr = mido.MidiTrack()
    tr.append(mido.Message("note_on", note=60, velocity=100, time=0))
    tr.append(mido.Message("note_off", note=60, velocity=0, time=480))
    mid.tracks.append(tr)
    p = tmp_path / "x.mid"
    mid.save(p)
    return p.read_bytes()


# -- bitmidi ---------------------------------------------------------------

def test_search_results(tmp_path):
    data = {"result": {"results": [{"id": 1, "name": "a.mid"}, {"id": 2, "name": "b.mid"}]}}
    c, s = client(tmp_path, [FakeResp(json_data=data)])
    res = c.search("mario", page=1)
    assert [(r.id, r.name) for r in res] == [(1, "a.mid"), (2, "b.mid")]
    url, params, headers = s.calls[0]
    assert url.endswith("/api/midi/search")
    assert params == {"q": "mario", "page": 1}
    assert "tempestmidi" in headers["User-Agent"]


def test_search_empty(tmp_path):
    c, _ = client(tmp_path, [FakeResp(json_data={"result": {"results": []}})])
    assert c.search("zzz") == []


def test_search_bad_json(tmp_path):
    c, _ = client(tmp_path, [FakeResp(body=b"<html>nope</html>")])
    with pytest.raises(SourceError, match="format"):
        c.search("x")
    c, _ = client(tmp_path, [FakeResp(json_data={"oops": 1})])
    with pytest.raises(SourceError, match="format"):
        c.search("x")


def test_fetch_network_error_cache_unchanged(tmp_path):
    c, _ = client(tmp_path, [ConnectionError("down")])
    with pytest.raises(SourceError):
        c.fetch(72257)
    cache = tmp_path / "cache"
    assert not cache.exists() or list(cache.iterdir()) == []


def test_fetch_non_midi_not_cached(tmp_path):
    c, _ = client(tmp_path, [FakeResp(body=b"<html>")])
    with pytest.raises(SourceError):
        c.fetch(5)
    cache = tmp_path / "cache"
    assert not cache.exists() or list(cache.iterdir()) == []


def test_fetch_then_cache_hit(tmp_path):
    body = midi_bytes(tmp_path)
    c, s = client(tmp_path, [FakeResp(body=body)])
    p = c.fetch(72257)
    assert p.read_bytes() == body
    assert len(s.calls) == 1
    p2 = c.fetch(72257)  # no response queued: any request would fail
    assert p2 == p and len(s.calls) == 1
    assert [x.name for x in (tmp_path / "cache").iterdir()] == ["72257.mid"]


def test_rate_limit(tmp_path):
    now = [0.0]
    slept = []
    s = FakeSession([FakeResp(json_data={"result": {"results": []}})] * 2)
    c = BitmidiClient(cache_dir=tmp_path, session=s, min_interval=1.0,
                      clock=lambda: now[0], sleep=slept.append)
    c.search("a")
    now[0] = 0.25
    c.search("b")
    assert slept == [pytest.approx(0.75)]


def test_resolve_source(tmp_path):
    body = midi_bytes(tmp_path)
    c, _ = client(tmp_path, [FakeResp(body=body)])
    assert resolve_source("bitmidi:9", c).name == "9.mid"
    assert resolve_source(str(tmp_path / "x.mid")) == tmp_path / "x.mid"
    with pytest.raises(SourceError):
        resolve_source(str(tmp_path / "missing.mid"))
    with pytest.raises(SourceError):
        resolve_source("bitmidi:abc", c)


# -- local load / normalization ---------------------------------------------

def test_tempo_change(tmp_path):
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))  # 120 bpm
    tr.append(mido.Message("note_on", note=60, velocity=90, time=0))
    tr.append(mido.Message("note_off", note=60, time=480))  # 0.5 s
    tr.append(mido.MetaMessage("set_tempo", tempo=1000000, time=0))  # 60 bpm
    tr.append(mido.Message("note_on", note=62, velocity=80, time=0))
    tr.append(mido.Message("note_off", note=62, time=480))  # 1.0 s
    mid.tracks.append(tr)
    p = tmp_path / "t.mid"
    mid.save(p)
    n = load_notes(p)
    assert [(x.pitch, round(x.start, 6), round(x.end, 6)) for x in n] == [(60, 0, 0.5), (62, 0.5, 1.5)]
    assert n[0].velocity == 90


def test_multitrack_merge(tmp_path):
    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    t0 = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000, time=0)])
    t1 = mido.MidiTrack([
        mido.Message("note_on", channel=0, note=64, velocity=100, time=480),
        mido.Message("note_on", channel=0, note=64, velocity=0, time=480),
    ])
    t2 = mido.MidiTrack([
        mido.Message("note_on", channel=1, note=48, velocity=100, time=0),
        mido.Message("note_off", channel=1, note=48, time=960),
    ])
    mid.tracks += [t0, t1, t2]
    p = tmp_path / "m.mid"
    mid.save(p)
    n = load_notes(p)
    assert [(x.channel, x.pitch, round(x.start, 6), round(x.end, 6)) for x in n] == [
        (1, 48, 0, 1.0), (0, 64, 0.5, 1.0)]


def test_invalid_file(tmp_path):
    p = tmp_path / "bad.mid"
    p.write_bytes(b"not a midi file at all")
    with pytest.raises(SourceError):
        load_notes(p)
