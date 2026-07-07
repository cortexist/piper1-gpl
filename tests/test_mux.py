"""Tests for the PCM + phoneme-timing mux framing (piper.mux)."""

import io
from dataclasses import dataclass

from piper.mux import (
    CONFIG,
    PCM,
    SCHEDULE,
    parse_config,
    parse_schedule,
    read_frame,
    schedule_payload,
    write_frame,
)


@dataclass
class _Alignment:
    phoneme: str
    num_samples: int


def test_frame_round_trip() -> None:
    """Frames survive the wire, including empty and binary payloads."""
    buf = io.BytesIO()
    write_frame(buf, CONFIG, b"rate=22050\nwidth=2\nchannels=1\n")
    write_frame(buf, SCHEDULE, b"0.0\t80.0\th\n")
    write_frame(buf, PCM, bytes(range(256)))
    write_frame(buf, PCM, b"")

    buf.seek(0)
    kind, payload = read_frame(buf)
    assert kind == CONFIG
    assert parse_config(payload) == {"rate": 22050, "width": 2, "channels": 1}

    kind, payload = read_frame(buf)
    assert (kind, payload) == (SCHEDULE, b"0.0\t80.0\th\n")

    kind, payload = read_frame(buf)
    assert (kind, payload) == (PCM, bytes(range(256)))

    kind, payload = read_frame(buf)
    assert (kind, payload) == (PCM, b"")

    assert read_frame(buf) is None  # clean end of stream


def test_truncated_frame() -> None:
    """A frame cut mid-payload reads as end of stream, not garbage."""
    buf = io.BytesIO()
    write_frame(buf, PCM, b"full payload")
    data = buf.getvalue()
    assert read_frame(io.BytesIO(data[:-3])) is None


def test_schedule_round_trip() -> None:
    """Alignments serialize to cumulative start times and parse back."""
    alignments = [
        _Alignment("^", 2205),   # 100 ms at 22050 Hz
        _Alignment("h", 4410),   # 200 ms
        _Alignment("@", 0),      # zero-length phonemes keep their slot
        _Alignment("$", 1102),
    ]
    events = parse_schedule(schedule_payload(alignments, 22050))
    assert [e[2] for e in events] == ["^", "h", "@", "$"]
    starts = [e[0] for e in events]
    durs = [e[1] for e in events]
    assert starts == [0.0, 100.0, 300.0, 300.0]
    assert durs[0] == 100.0 and durs[1] == 200.0 and durs[2] == 0.0


def test_parse_control() -> None:
    """Control lines classify; marker tokens inside are dialect-agnostic."""
    from piper.control import parse_control

    line = '<|tool_call>call:set_voice{speaker_id:<|"|>happy<|"|>}<tool_call|>'
    assert parse_control(line) == ("set_voice", "happy")
    assert parse_control("  <|tool_call>set_voice{3}<tool_call|>  ") == ("set_voice", "3")
    assert parse_control("<|tool_call>call:do_thing{x:1}<tool_call|>") == ("ignore", None)
    assert parse_control("Plain speech, not control.") is None
    assert parse_control("<|tool_call>unterminated") is None


def test_resolve_speaker() -> None:
    """Names resolve through the voice config; numbers bound-check."""
    from types import SimpleNamespace
    from piper.control import resolve_speaker

    cfg = SimpleNamespace(num_speakers=3, speaker_id_map={"happy": 2, "sad": 1})
    assert resolve_speaker(cfg, "happy") == 2
    assert resolve_speaker(cfg, "1") == 1
    assert resolve_speaker(cfg, "7") is None
    assert resolve_speaker(cfg, "angry") is None
    assert resolve_speaker(cfg, None) is None
