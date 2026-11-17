"""Command line interface."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tempestmidi",
        description="Convert MIDI files into polyphonic TEMPEST videos (HDMI emission music).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search bitmidi.com")
    s.add_argument("query")
    s.add_argument("--page", type=int, default=0)

    f = sub.add_parser("fetch", help="download a bitmidi.com MIDI file into the cache")
    f.add_argument("id", type=int)

    c = sub.add_parser("convert", help="convert a MIDI file into a TEMPEST mp4")
    c.add_argument("source", help="bitmidi:<id> or path to a .mid file")
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--voices", type=int, default=4, help="max simultaneous voices (default 4)")
    c.add_argument("--mix", choices=["sd", "arp", "rgb"], default="sd",
                   help="voice mixing, all full-width rows: sd = 1-bit sigma-delta sum (default), "
                        "arp = one voice per frame (arpeggio), rgb = voice per color/TMDS lane")
    c.add_argument("--velocity", action="store_true",
                   help="scale voice level by note velocity (default: full level, max power)")
    c.add_argument("--mode", default="1920x1080x1125@60",
                   help="WIDTHxHEIGHTxVTOTAL@REFRESH (refresh may be a fraction, e.g. 60000/1001)")
    c.add_argument("--preview-audio", action="store_true",
                   help="add an audio track with the expected demodulated mix")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "search":
            return _cmd_search(args)
        if args.cmd == "fetch":
            return _cmd_fetch(args)
        if args.cmd == "convert":
            return _cmd_convert(args)
    except Exception as e:  # noqa: BLE001 - CLI boundary
        from .source import SourceError
        if isinstance(e, SourceError):
            print(f"error: {e}", file=sys.stderr)
            return 1
        raise
    return 2


def _cmd_search(args) -> int:
    from .source import BitmidiClient

    results = BitmidiClient().search(args.query, page=args.page)
    if not results:
        print("no results")
        return 0
    for r in results:
        print(f"{r.id:>8}  {r.name}")
    print(f"(page {args.page}; convert with: tempestmidi convert bitmidi:<id> -o out.mp4)")
    return 0


def _cmd_fetch(args) -> int:
    from .source import BitmidiClient

    print(BitmidiClient().fetch(args.id))
    return 0


def _cmd_convert(args) -> int:
    from .export import export_video, stderr_progress
    from .render import FrameRenderer, VideoMode
    from .source import load_notes, resolve_source
    from .voices import VoiceAllocator

    try:
        mode = VideoMode.parse(args.mode)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.voices < 1:
        print("error: --voices must be >= 1", file=sys.stderr)
        return 2
    path = resolve_source(args.source)
    notes = load_notes(path)
    voices = VoiceAllocator(notes, slots=args.voices, velocity=args.velocity)
    if voices.duration <= 0:
        print(f"error: no notes in {path}", file=sys.stderr)
        return 1
    duration = voices.duration + voices.release
    renderer = FrameRenderer(voices, mode, mix=args.mix)
    print(f"{path}: {len(voices.notes)} notes ({voices.drum_notes} drum notes ignored), "
          f"{duration:.1f}s, mode {mode}, mix {args.mix}, line rate {mode.line_rate:.2f} Hz", file=sys.stderr)
    summary = export_video(renderer, duration, args.output, preview_audio=args.preview_audio,
                           dropped_notes=voices.dropped_notes, progress=stderr_progress())
    print(summary.line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
