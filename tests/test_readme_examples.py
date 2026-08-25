"""The README's runnable examples must actually run.

CORELIB_PLAN §9.5 shows a Generator example and, since #110, a ``feed`` and a
``Binding`` example. These tests execute them: the object the README defines has
to behave like generated code — the same bytes from the streaming path as from
the one-shot path, and a decode that survives being fed one byte at a time. An
example that drifts from the API stops running here rather than in a reader's
editor.

Nothing below asserts what the README *says*; only that the code in it works.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sofab import Encoder

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "README.md"
_INDEX_RST = _ROOT / "docs" / "index.rst"

# §6.1.1's explicit list of spellings a port must not invent, plus the two the
# generated layer would collide with. Matched case-insensitively on word
# boundaries so ``_marshal`` / ``Unmarshal`` are caught too.
_BANNED_NAMES = (
    "marshal",
    "unmarshal",
    "serialize_to",
    "to_bytes",
    "from_bytes",
    "decode_from",
    "decode_into",
)
_BANNED_RE = re.compile(r"(?<![\w.])_?(" + "|".join(_BANNED_NAMES) + r")\b", re.IGNORECASE)


def _sections(text: str) -> dict[str, str]:
    """Split a Markdown document into ``heading -> body`` (headings verbatim)."""
    out: dict[str, str] = {}
    heading = ""
    buf: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if line.startswith("#") and not fenced:
            out[heading] = "\n".join(buf)
            heading, buf = line.strip(), []
        else:
            buf.append(line)
    out[heading] = "\n".join(buf)
    return out


def _python_blocks(body: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", body, re.DOTALL | re.MULTILINE)


@pytest.fixture(scope="module")
def readme() -> str:
    return _README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def generator_example(readme: str) -> list[str]:
    """The ``### Code generator`` section's Python blocks, in order."""
    body = _sections(readme).get("### Code generator")
    assert body is not None, "README lost its `### Code generator` section (CORELIB_PLAN §9.5)"
    blocks = _python_blocks(body)
    assert blocks, "the Code generator section has no runnable example"
    return blocks


def _run_readme_block(readme: str, heading: str, ns: dict) -> dict:
    body = _sections(readme).get(heading)
    assert body is not None, f"README lost its `{heading}` section"
    blocks = _python_blocks(body)
    assert blocks, f"`{heading}` has no runnable example"
    for i, block in enumerate(blocks):
        exec(compile(block, f"README.md#{heading}[{i}]", "exec"), ns)
    return ns

def test_generator_example_runs_and_round_trips(generator_example: list[str]) -> None:
    """Execute the README example; it must behave like the code sofabgen emits."""
    ns: dict[str, object] = {}
    for i, block in enumerate(generator_example):
        exec(compile(block, f"README.md#code-generator[{i}]", "exec"), ns)

    point_cls = ns["Point"]
    assert isinstance(point_cls, type)
    methods = {n for n, v in vars(point_cls).items() if callable(v) or isinstance(v, classmethod)}
    assert {"serialize", "deserialize", "encode", "decode"} <= methods
    assert not [n for n in methods if _BANNED_RE.match(n)]

    # The one-shot bytes are the bytes the primitives produce for the same fields.
    reference = Encoder()
    reference.write_signed(1, 3)
    reference.write_signed(2, 4)
    assert ns["wire"] == reference.getvalue()
    assert ns["got"] == point_cls(x=3, y=4)

    # The streaming half: same bytes out of a buffer far smaller than the message,
    # and a decode fed in tiny chunks reaches the same object.
    assert ns["streamed"] == ns["wire"], "serialize() through a sink must match encode()"
    assert ns["got_streamed"] == point_cls(x=3, y=4)


def test_binding_example_runs_and_decodes(readme: str) -> None:
    """The ``Binding`` example must decode the message it claims to."""
    enc = Encoder()
    enc.write_unsigned(1, 300)
    enc.write_string(3, "grüß dich")
    enc.write_unsigned_array(4, [7, 8, 9])
    enc.flush()

    ns: dict = {"payload": enc.getvalue()}
    _run_readme_block(readme, "### Decode into your own storage (`Binding`)", ns)

    assert ns["u"][0] == 300
    assert ns["objs"][0] == "grüß dich"
    words, u = ns["words"], ns["u"]
    assert list(u[8 : 8 + u[3]]) == [7, 8, 9]
    assert len(words) == ns["b"].tree_words_required * 8


def test_feed_example_matches_the_api(readme: str) -> None:
    """The ``Deserialize`` example is written against a socket, so it is checked
    for shape rather than executed: the names it uses must exist and mean what
    the surrounding table says."""
    from sofab import Decoder, Status

    body = _sections(readme).get("### Deserialize")
    assert body is not None, "README lost its `### Deserialize` section"
    src = "\n".join(_python_blocks(body))
    assert "Decoder(visitor=" in src and "dec.feed(" in src
    assert "dec.error" in src
    for name in ("COMPLETE", "INCOMPLETE", "INVALID"):
        assert f"`Status.{name}`" in body, f"the outcome table lost {name}"
        assert hasattr(Status, name)
    assert hasattr(Decoder, "feed") and hasattr(Decoder, "reset")
    # §5.2: no finalize step may exist, and the README has to say so — the
    # absence is part of the contract, not an omission.
    assert "no** `finish()`/`end()`" in body
    for banned in ("finish", "finalize", "end", "close"):
        assert not hasattr(Decoder, banned), banned


def test_array_begin_example_fills_the_destination(readme: str) -> None:
    """The ``on_array_begin`` example must do what the section claims: fill the
    handler's own storage, leave the typed hook alone, and apply the width."""
    from sofab import Decoder, Status, Visitor

    ns: dict = {"Visitor": Visitor}
    _run_readme_block(readme, "#### Integer arrays: `on_array_begin`", ns)
    handler_cls = ns["Handler"]

    enc = Encoder()
    enc.write_unsigned_array(7, [1, 2, 0xFFFF])
    enc.write_unsigned_array(8, [1 << 20])  # a different id: the list route
    enc.flush()

    got: list = []
    handler = handler_cls()
    handler.on_unsigned_array = lambda fid, vals: got.append((fid, list(vals)))
    assert Decoder(visitor=handler).feed(enc.getvalue()) is Status.COMPLETE

    assert list(handler.ports[:3]) == [1, 2, 0xFFFF]
    assert got == [(8, [1 << 20])], "id 7 went to the destination, not the hook"


def test_string_begin_example_fills_the_destination(readme: str) -> None:
    """The ``on_string_begin`` example: the handler's own bytearray takes the
    payload's UTF-8, and ``on_string`` is not called for that field."""
    from sofab import Decoder, Status, Visitor

    ns: dict = {"Visitor": Visitor}
    _run_readme_block(readme, "#### Strings: `on_string_begin`", ns)
    handler = ns["Handler"]()

    got: list = []
    handler.on_string = lambda fid, value: got.append((fid, value))

    enc = Encoder()
    enc.write_string(3, "grüß dich")
    enc.write_string(4, "elsewhere")
    enc.flush()
    assert Decoder(visitor=handler).feed(enc.getvalue()) is Status.COMPLETE

    utf8 = "grüß dich".encode()
    assert bytes(handler.name[: len(utf8)]) == utf8
    assert got == [(4, "elsewhere")], "id 3 went to the destination, not the hook"


def test_float_array_begin_example_fills_the_destination(readme: str) -> None:
    from sofab import Decoder, Status, Visitor

    ns: dict = {"Visitor": Visitor}
    _run_readme_block(readme, "#### Float arrays: `on_float_array_begin`", ns)
    handler = ns["Handler"]()

    got: list = []
    handler.on_float64_array = lambda fid, vals: got.append((fid, list(vals)))

    enc = Encoder()
    enc.write_float64_array(5, [0.5, 1.5, 2.5])
    enc.write_float64_array(6, [9.0])
    enc.flush()
    assert Decoder(visitor=handler).feed(enc.getvalue()) is Status.COMPLETE

    assert list(handler.samples[:3]) == [0.5, 1.5, 2.5]
    assert got == [(6, [9.0])], "id 5 went to the destination, not the hook"


def test_bit_exact_float_example_round_trips_a_signaling_nan(readme: str) -> None:
    """The transcoder in the README must reproduce the wire bytes exactly —
    which is the whole reason §6.5 requires the channel it uses."""
    from sofab import Decoder, Status, Visitor

    out = Encoder()
    ns: dict = {"Visitor": Visitor, "out": out}
    _run_readme_block(
        readme, "#### Bit-exact floats: `on_float32_bits` and `write_float32_bits`", ns
    )

    # A signaling NaN, which a widening conversion would quiet.
    snan = bytes.fromhex("0100807f")
    one = Encoder()
    one.write_float32(7, 0.0)
    one.flush()
    other = Encoder()
    other.write_float32_array(8, [0.0])
    other.flush()
    wire = one.getvalue()[:-4] + snan + other.getvalue()[:-4] + snan

    assert Decoder(visitor=ns["Transcoder"]()).feed(wire) is Status.COMPLETE
    out.flush()
    assert out.getvalue() == wire
