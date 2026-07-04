"""Separate piper's --output-mux stream: PCM to an audio sink, phonemes to stdout.

The counterpart of ``--output-mux`` (see piper.mux for the framing): reads the
mixed stream on stdin, forwards PCM to the sink command's stdin, and prints
each phoneme event to stdout a little AHEAD of the audio clock — including the
following phoneme — so an animation or a mechanical actuator can prepare the
transition before the sound arrives:

    start_ms<TAB>duration_ms<TAB>phoneme<TAB>next_phoneme

Times are milliseconds on the stream clock (cumulative across sentences). The
audio clock is the bytes forwarded to the sink: a real-time sink (aplay, a
sound server) applies backpressure, so bytes-forwarded tracks playback. With
no ``--sink`` the PCM is discarded and events pace against the wall clock —
useful for actuators that want only the visemes, and for tests.

    piper -m voice.onnx --output-raw --stream --output-mux \\
      | python3 -m piper.demux --sink "aplay -r 22050 -f S16_LE -t raw -c 1 -"
"""

import argparse
import shlex
import subprocess
import sys
import time

from .mux import CONFIG, PCM, SCHEDULE, parse_config, parse_schedule, read_frame

# One pacing slice of audio; also the jitter bound on event emission.
_SLICE_MS = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sink",
        help="Command to receive raw PCM on stdin (e.g. \"aplay -r 22050 "
        "-f S16_LE -t raw -c 1 -\"); omit to discard PCM and pace by wall clock",
    )
    parser.add_argument(
        "--lead-ms",
        "--lead_ms",
        type=float,
        default=50.0,
        help="How far ahead of the audio clock events are emitted (default: 50)",
    )
    args = parser.parse_args()

    sink = None
    if args.sink:
        sink = subprocess.Popen(shlex.split(args.sink), stdin=subprocess.PIPE)

    inp = sys.stdin.buffer
    rate, width = 22050, 2
    events = []                 # (start_ms, dur_ms, phoneme), stream clock
    sent_ms = 0.0               # stream time of completed sentences
    pos_bytes = 0               # PCM bytes forwarded
    wall0 = None                # wall-clock anchor for the sink-less mode

    def emit_due(pos_ms):
        while events and events[0][0] <= pos_ms + args.lead_ms:
            start, dur, phoneme = events.pop(0)
            nxt = events[0][2] if events else "-"
            sys.stdout.write("%.0f\t%.0f\t%s\t%s\n" % (start, dur, phoneme, nxt))
            sys.stdout.flush()

    while True:
        frame = read_frame(inp)
        if frame is None:
            break

        kind, payload = frame
        if kind == CONFIG:
            config = parse_config(payload)
            rate = config.get("rate", rate)
            width = config.get("width", width)
        elif kind == SCHEDULE:
            # The schedule precedes its sentence's audio; its times are
            # sentence-relative, and the sentence starts where the previous
            # audio ended.
            base = sent_ms
            for start, dur, phoneme in parse_schedule(payload):
                events.append((base + start, dur, phoneme))
                sent_ms = max(sent_ms, base + start + dur)
        elif kind == PCM:
            slice_bytes = max(width, int(rate * width * _SLICE_MS / 1000))
            for off in range(0, len(payload), slice_bytes):
                piece = payload[off: off + slice_bytes]
                pos_ms = pos_bytes * 1000.0 / (rate * width)
                emit_due(pos_ms)
                if sink is not None:
                    sink.stdin.write(piece)
                    sink.stdin.flush()
                else:
                    if wall0 is None:
                        wall0 = time.monotonic() - pos_ms / 1000.0
                    ahead = wall0 + pos_ms / 1000.0 - time.monotonic()
                    if ahead > 0:
                        time.sleep(ahead)
                pos_bytes += len(piece)

    emit_due(float("inf"))      # end of stream: flush whatever remains
    if sink is not None:
        sink.stdin.close()
        sink.wait()


if __name__ == "__main__":
    main()
