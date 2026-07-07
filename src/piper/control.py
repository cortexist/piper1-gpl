"""In-band control lines for the streaming CLI.

An LLM upstream of piper can steer the voice by emitting a tool-call span
that the clause layer (little-gemma's clausecat with --allow-control-token)
passes through verbatim as its own line:

    <|tool_call>call:set_voice{speaker_id:<|"|>happy<|"|>}<tool_call|>

A line that is entirely one ``<|tool_call>…<tool_call|>`` span is a control
line, never speech. ``set_voice`` selects the speaker of a multi-speaker
voice for every following utterance — by name (the voice config's
``speaker_id_map``) or by number. Unrecognized calls are consumed silently
(better than speaking their payload). The ``<|…|>`` marker tokens inside
(Gemma's string quotes, for instance) are stripped before parsing, so the
exact token dialect does not matter.
"""

import logging
import re
from typing import Optional, Tuple

_LOGGER = logging.getLogger(__name__)

_CONTROL = re.compile(r"^\s*<\|tool_call>(.*)<tool_call\|>\s*$")
_SET_VOICE = re.compile(r"set_voice\s*\{(.*)\}")
_TOKEN = re.compile(r"<[^<>]{1,24}>")


def parse_control(line: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Classify a line. None: ordinary text, speak it. ("set_voice", value):
    switch speaker. ("ignore", None): a control line piper does not know —
    consume without speaking.
    """
    m = _CONTROL.match(line)
    if m is None:
        return None

    inner = _TOKEN.sub("", m.group(1))
    call = _SET_VOICE.search(inner)
    if call is None:
        return ("ignore", None)

    args = call.group(1)
    value = args.split(":", 1)[1] if ":" in args else args
    return ("set_voice", value.strip().strip("\"'") or None)


def resolve_speaker(config, value: Optional[str]) -> Optional[int]:
    """A speaker id (number) or name (config.speaker_id_map) -> id."""
    if value is None:
        return None

    if value.lstrip("-").isdigit():
        sid = int(value)
        return sid if 0 <= sid < max(1, config.num_speakers) else None

    return config.speaker_id_map.get(value)
