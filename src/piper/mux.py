"""Framing for piper's mixed PCM + phoneme-timing stream (--output-mux).

One byte stream carries both the audio and the timing a lip-sync frontend or
a mechanical actuator needs, so a single pipe can feed both. Frames are
[kind: 1 byte][length: uint32 little-endian][payload]:

- ``C``: stream config, ``key=value`` lines (``rate``, ``width``, ``channels``).
  Written once, first.
- ``A``: one sentence's phoneme schedule, written BEFORE that sentence's
  audio: tab-separated ``start_ms<TAB>duration_ms<TAB>phoneme`` lines, times
  relative to the start of the sentence. With ``--stream`` the schedule comes
  from the encoder half's duration output, so it precedes the first PCM chunk
  by the whole sentence.
- ``P``: signed 16-bit mono PCM samples.
- ``M``: metadata from an in-band control line (see piper.control) — e.g.
  the value of a ``set_voice`` call, written when the call is seen, BEFORE
  the audio it colors — whether or not this voice has a matching speaker
  (a face can change even when the voice cannot). The pipe does not
  interpret the payload; meaning belongs to the two ends.

``python3 -m piper.demux`` separates the two again: PCM to an audio sink,
schedule lines to stdout, each a little ahead of the audio clock.
"""

import struct
from typing import BinaryIO, Iterable, List, Optional, Tuple

CONFIG = b"C"
SCHEDULE = b"A"
PCM = b"P"
META = b"M"


def write_frame(out: BinaryIO, kind: bytes, payload: bytes) -> None:
    """Write one frame."""
    out.write(kind + struct.pack("<I", len(payload)) + payload)
    out.flush()


def read_frame(inp: BinaryIO) -> Optional[Tuple[bytes, bytes]]:
    """Read one frame; None at a clean end of stream."""
    head = inp.read(5)
    if len(head) < 5:
        return None

    kind, length = head[:1], struct.unpack("<I", head[1:])[0]
    payload = b""
    while len(payload) < length:
        more = inp.read(length - len(payload))
        if not more:
            return None

        payload += more

    return kind, payload


def schedule_payload(alignments: Iterable, sample_rate: int) -> bytes:
    """Serialize PhonemeAlignment objects into an ``A`` frame payload."""
    lines = []
    start = 0.0
    for alignment in alignments:
        dur_ms = alignment.num_samples * 1000.0 / sample_rate
        lines.append("%.1f\t%.1f\t%s" % (start, dur_ms, alignment.phoneme))
        start += dur_ms

    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_schedule(payload: bytes) -> List[Tuple[float, float, str]]:
    """Parse an ``A`` frame payload into (start_ms, duration_ms, phoneme)."""
    events = []
    for line in payload.decode("utf-8").splitlines():
        if not line:
            continue

        start, dur, phoneme = line.split("\t", 2)
        events.append((float(start), float(dur), phoneme))

    return events


def parse_config(payload: bytes) -> dict:
    """Parse a ``C`` frame payload into a dict of ints."""
    config = {}
    for line in payload.decode("utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = int(value)

    return config
