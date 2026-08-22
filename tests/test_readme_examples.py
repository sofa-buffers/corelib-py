"""The documented generated-object layer must be the one the generator emits.

CORELIB_PLAN §6.1.1 closes the name set of the generated layer: ``encode`` /
``decode`` (one-shot) and ``serialize`` / ``deserialize`` (streaming). Every
other spelling a port might invent — ``marshal``, ``unmarshal``, ``to_bytes``,
``from_bytes``, ``serialize_to``, ``decode_from``, ``decode_into`` — is banned,
and §9 requires every fact in the README to match the code as it stands today,
so documenting a name the ``sofabgen`` Python backend does not emit is a defect
in itself. §9.5 further requires the Usage *Generator* example to show **both**
halves: the one-shot ``encode()`` / ``decode()`` pair *and* the streaming
``serialize`` / chunk-fed decode path.

These tests therefore do not merely grep: they execute the README's Generator
example and check that the object it defines behaves like generated code —
same bytes from the streaming path as from the one-shot path, and a decode that
survives being fed one byte at a time. A doc example that drifts from the API
stops running, and a banned name reintroduced anywhere in the published docs
fails the name check.
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


def test_no_banned_generated_layer_names_in_docs(readme: str) -> None:
    """No ``marshal``/``unmarshal``/… anywhere in the published documentation."""
    for name, text in (("README.md", readme), ("docs/index.rst", _INDEX_RST.read_text("utf-8"))):
        found = sorted({m.group(0) for m in _BANNED_RE.finditer(text)})
        assert not found, f"{name} uses generated-layer names banned by §6.1.1: {found}"


def test_generator_example_shows_both_halves(generator_example: list[str]) -> None:
    """§9.5: the one-shot pair *and* the streaming ``serialize`` / chunk-fed path."""
    src = "\n".join(generator_example)
    for needed in ("def serialize(", "def deserialize(", "def encode(", "def decode("):
        assert needed in src, f"the generated-code stand-in does not define `{needed}…`"
    assert "over_buffer(" in src, "the example never streams out through a caller-supplied buffer"


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


def test_docs_landing_page_matches_the_shipped_engines() -> None:
    """docs/index.rst is the published landing page (§12.2) and §9's facts bind it."""
    text = _INDEX_RST.read_text(encoding="utf-8")
    assert "Pure-Python runtime" not in text, (
        "the landing page calls the library pure-Python, but the package ships a "
        "compiled accelerator that is active by default (sofab.IMPL == 'native')"
    )
    lowered = text.lower()
    assert "accelerator" in lowered and "fallback" in lowered


# --- the push-decode sections (CORELIB_PLAN §5.2 / §5.3) ---------------------
#
# Same rule as the Generator example: the README states facts about the API, so
# its examples are executed rather than eyeballed. Both sections here document
# calls that did not exist before push mode, which is exactly the kind of doc a
# refactor drifts away from silently.


def _run_readme_block(readme: str, heading: str, ns: dict) -> dict:
    body = _sections(readme).get(heading)
    assert body is not None, f"README lost its `{heading}` section"
    blocks = _python_blocks(body)
    assert blocks, f"`{heading}` has no runnable example"
    for i, block in enumerate(blocks):
        exec(compile(block, f"README.md#{heading}[{i}]", "exec"), ns)
    return ns


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
    """The ``feed`` example is written against a socket, so it is checked for
    shape rather than executed: the names it uses must exist and mean what the
    surrounding table says."""
    from sofab import Decoder, Status

    body = _sections(readme).get("### Deserialize push (`feed`)")
    assert body is not None, "README lost its `### Deserialize push (`feed`)` section"
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
