"""The README's *shape* is a contract, not an editorial choice (CORELIB_PLAN §9).

§9 fixes one structure for the whole family of ``corelib-*`` READMEs — "do not
change the section ordering and do not invent new top-level sections" — so a
reader who knows one port's README can navigate this one by position. These
tests pin that structure, plus the handful of facts §9 names explicitly, so a
restructuring or a trim cannot quietly drop one:

* §9    the ``## `` chapters, membership and order
* §9.1  the centered logo, the ``# SofaBuffers`` title, the tagline, the org link
* §9.2  the badge block: CI, coverage, Docs — in that order
* §9.4  no API-documentation chapter at any heading level
* §9.5  the Usage chapter still shows every example the plan lists
* §9.6  ``MIN_OUTPUT_BUFFER`` is stated in the Memory handling chapter, with the
        value the package actually exposes
* every in-document ``](#anchor)`` link resolves to a heading

Two neighbouring checks live elsewhere on purpose, so no fact is asserted twice:

* §6.1.1's closed generated-object name set (no ``marshal`` / ``unmarshal`` /
  ``serialize_to`` / ``to_bytes`` / ``from_bytes`` / ``decode_from`` /
  ``decode_into``) is owned by ``tests/test_readme_examples.py``, which checks
  README.md *and* ``docs/index.rst`` and additionally executes the Usage
  generator example.
* §6.4's strict-UTF-8 knob: this port has none, legitimately. Python ``str`` is a
  Unicode string type, and §6.4 rules those "always strict — the option is a
  no-op for them and they MAY omit it entirely"; only byte-container targets
  (C ``char[]``, Go ``string``, Zig ``[]const u8``) MUST expose it. So the
  README is not required to document a knob, and this file does not demand one.
  What it does demand is that the premise stays true:
  ``test_strict_utf8_option_is_legitimately_absent`` fails the moment the
  package grows a strict-UTF-8 switch, at which point §6.4 obliges the README to
  document it and this test says so.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import sofab
from sofab import Encoder, SofaRangeError

_ROOT = Path(__file__).resolve().parent.parent
_README = _ROOT / "README.md"

# §9's sanctioned chapters, in order: §9.2, §9.3, §9.5, §9.6, §9.7, §9.8. (§9.1
# is the header block, whose `# SofaBuffers` is the document title; §9.4 forbids
# an API-documentation chapter.) Anything else at `## ` level is an invented
# section — demote it to a `###` subsection of the chapter it belongs to rather
# than adding a row here.
_TOP_LEVEL_SECTIONS = (
    "SofaBuffers Python library",
    "Why this design",
    "Usage",
    "Memory handling",
    "Build & test",
    "Benchmarks",
)

# §9.5's example list, mapped onto this port's names. Python has no separate
# OStream/IStream classes: `Encoder(writer)` *is* the output-stream wrapper and
# `Decoder(reader)` *is* the push-fed input stream, so those two bullets are
# covered by the Serialize / Deserialize stream sections rather than by chapters
# of their own.
_USAGE_SUBSECTIONS = (
    "Serialize",
    "Serialize stream",
    "Deserialize",
    "Deserialize stream",
    "Code generator",
)

# Each §9.5 bullet, as something that must appear in the Usage chapter's
# runnable code. Matched against the concatenated ```python blocks, so prose
# alone never satisfies one.
_USAGE_EXAMPLES = (
    ("simple encode", (r"Encoder\(\)", r"\.getvalue\(\)")),
    ("simple decode", (r"Decoder\(", r"\.next\(\)")),
    ("streaming a message larger than the buffer", (r"over_buffer\(", r"flush=")),
    ("OStream — the encoder over a writer sink", (r"Encoder\(\w+\)",)),
    ("IStream — the decoder over a read\\(n\\) source", (r"Decoder\(\w+\)",)),
    ("Generator — one-shot encode/decode", (r"def encode\(", r"def decode\(")),
    ("Generator — streaming serialize/deserialize", (r"def serialize\(", r"def deserialize\(")),
)

_HEADING_RE = re.compile(r"^(#{1,6}) +(.*?)\s*$")
_FORBIDDEN_CHAPTERS = ("api reference", "api documentation", "source documentation", "api")


@pytest.fixture(scope="module")
def readme() -> str:
    return _README.read_text(encoding="utf-8")


def _headings(text: str) -> list[tuple[int, str]]:
    """``(level, title)`` for every ATX heading, skipping fenced code blocks.

    The fence skip is what keeps a ``# comment`` inside a shell example from
    being read as a chapter.
    """
    out: list[tuple[int, str]] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = _HEADING_RE.match(line)
        if m:
            out.append((len(m.group(1)), m.group(2)))
    assert out, "README.md: no headings found — the parser is broken, not the document"
    return out


def _chapter(text: str, title: str) -> str:
    """The body of the ``## <title>`` chapter, up to the next ``## ``."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) == 2 and m.group(2) == title:
            start = i + 1
            break
    assert start is not None, f"README.md: no `## {title}` chapter (CORELIB_PLAN §9)"
    body: list[str] = []
    fenced = False
    for line in lines[start:]:
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            m = _HEADING_RE.match(line)
            if m and len(m.group(1)) == 2:
                break
        body.append(line)
    return "\n".join(body)


def _python_blocks(body: str) -> str:
    return "\n".join(re.findall(r"^```python\n(.*?)^```", body, re.DOTALL | re.MULTILINE))


def test_top_level_sections_match_the_plan(readme: str) -> None:
    """§9: the chapter list is closed and its order is fixed."""
    got = [title for level, title in _headings(readme) if level == 2]

    sanctioned = set(_TOP_LEVEL_SECTIONS)
    invented = [t for t in got if t not in sanctioned]
    assert not invented, (
        f"README.md: invented top-level section(s) {invented} — §9: \"do not invent new "
        "top-level sections\"; demote each to a `###` subsection of the chapter it belongs to"
    )
    missing = [t for t in _TOP_LEVEL_SECTIONS if t not in got]
    assert not missing, f"README.md: missing top-level section(s) {missing} (CORELIB_PLAN §9)"
    assert tuple(got) == _TOP_LEVEL_SECTIONS, (
        f"README.md: chapters are {got}, want {list(_TOP_LEVEL_SECTIONS)} — §9 fixes the order"
    )


def test_header_block(readme: str) -> None:
    """§9.1: centered logo, title, tagline, link back to the organization."""
    head = readme.split("\n## ", 1)[0]
    assert re.search(
        r'<p align="center"><img src="assets/sofabuffers_logo\.png"[^>]*></p>', head
    ), "README.md: §9.1 wants the centered logo as the first line"
    assert re.search(r"^# SofaBuffers\s*$", head, re.MULTILINE), "README.md: no `# SofaBuffers`"
    assert "<b>Structured Objects For Anyone</b>" in head, "README.md: §9.1 tagline is gone"
    assert "<i>... so optimized, feels amazing.</i>" in head, "README.md: §9.1 tagline is gone"
    assert "https://github.com/sofa-buffers" in head, "README.md: no link to the organization"


def test_badge_block(readme: str) -> None:
    """§9.2: CI, coverage and Docs badges, in that order, opening the chapter."""
    chapter = _chapter(readme, "SofaBuffers Python library")
    badges = re.findall(r"^\[!\[([^\]]+)\]\((.*?)\)\]\((.*?)\)\s*$", chapter, re.MULTILINE)
    labels = [label.lower() for label, _, _ in badges]
    assert labels[:3] == ["ci", "coverage", "docs"], (
        f"README.md: badge block is {labels}, want CI then coverage then Docs (§9.2)"
    )
    docs_target = badges[2][2]
    assert "sofa-buffers.github.io/corelib-py" in docs_target, (
        f"README.md: the Docs badge must point at the published API reference, not {docs_target!r} "
        "— §9.2/§9.4 make it the only pointer to API documentation"
    )


def test_no_api_documentation_chapter(readme: str) -> None:
    """§9.4: the Docs badge is the single entry point — at every heading level."""
    offenders = [
        (level, title)
        for level, title in _headings(readme)
        if title.strip().lower().rstrip(":") in _FORBIDDEN_CHAPTERS
    ]
    assert not offenders, (
        f"README.md: {offenders} — §9.4 forbids an API-documentation section; demoting it to "
        "a subsection does not make it allowed"
    )


def test_usage_shows_every_example_the_plan_lists(readme: str) -> None:
    """§9.5: the six examples the plan enumerates are all still runnable code."""
    usage = _chapter(readme, "Usage")
    subsections = [
        m.group(2) for m in map(_HEADING_RE.match, usage.splitlines()) if m and len(m.group(1)) == 3
    ]
    missing = [s for s in _USAGE_SUBSECTIONS if s not in subsections]
    assert not missing, f"README.md: Usage lost subsection(s) {missing} (CORELIB_PLAN §9.5)"
    positions = [subsections.index(s) for s in _USAGE_SUBSECTIONS]
    assert positions == sorted(positions), (
        f"README.md: Usage subsections are out of order: {subsections}"
    )

    code = _python_blocks(usage)
    assert code, "README.md: the Usage chapter carries no runnable Python at all"
    for name, patterns in _USAGE_EXAMPLES:
        for pattern in patterns:
            assert re.search(pattern, code), (
                f"README.md: Usage no longer shows the {name!r} example "
                f"(nothing matching /{pattern}/ in its code blocks) — CORELIB_PLAN §9.5"
            )
    for name in _USAGE_SUBSECTIONS:
        section_code = _python_blocks(usage.split(f"### {name}\n", 1)[1].split("\n### ", 1)[0])
        assert section_code.strip(), f"README.md: `### {name}` has no runnable example (§9.5)"


def test_memory_handling_states_min_output_buffer(readme: str) -> None:
    """§9.6: the constant lives in the Memory handling chapter, and it is correct."""
    memory = _chapter(readme, "Memory handling")
    assert "MIN_OUTPUT_BUFFER" in memory, (
        "README.md: §9.6 requires MIN_OUTPUT_BUFFER to be stated in the `## Memory handling` "
        "chapter — it is the number a caller needs before it can size a streaming buffer"
    )
    stated = re.search(r"`MIN_OUTPUT_BUFFER` is `(\d+)`", memory)
    assert stated, "README.md: Memory handling never states MIN_OUTPUT_BUFFER's value (§5.1/§9.6)"
    assert int(stated.group(1)) == sofab.MIN_OUTPUT_BUFFER, (
        f"README.md says MIN_OUTPUT_BUFFER is {stated.group(1)}, "
        f"sofab.MIN_OUTPUT_BUFFER is {sofab.MIN_OUTPUT_BUFFER}"
    )
    assert sofab.MIN_OUTPUT_BUFFER <= 20, "CORELIB_PLAN §5.1 caps the declaration at 20"


def test_strict_utf8_option_is_legitimately_absent() -> None:
    """§6.4: a Unicode-string port MAY omit the knob — but only while it has none.

    Python ``str`` cannot hold non-UTF-8 bytes, so this port is always strict and
    ``SOFAB_STRICT_UTF8`` is a no-op it omits entirely. That is why the README
    checks above do not require the knob to be documented. If the package ever
    grows one, this fails and §6.4's documentation duty starts applying.
    """
    switches = [n for n in dir(sofab) if "STRICT" in n.upper() or "UTF8" in n.upper()]
    assert not switches, (
        f"sofab now exposes {switches}: this port is no longer knob-free, so CORELIB_PLAN §6.4 "
        "requires the README to document the option, its default, and its effect — add that "
        "check to this file"
    )
    enc = Encoder()
    with pytest.raises(SofaRangeError):
        enc.write_string(1, "\ud800")


def test_internal_links_resolve(readme: str) -> None:
    """A heading that moves takes its anchor with it; every `](#…)` must still land."""
    anchors = {_github_anchor(title) for _, title in _headings(readme)}
    links = re.findall(r"\]\(#([^)]+)\)", readme)
    assert links, "README.md: no in-document links found — the scan is broken"
    broken = sorted({link for link in links if link not in anchors})
    assert not broken, f"README.md: link(s) to {broken} match no heading"


def _github_anchor(title: str) -> str:
    """Slugify a heading the way GitHub does: lowercase, punctuation dropped."""
    out = []
    for ch in title.lower():
        if ch.isalnum() or ch == "-":
            out.append(ch)
        elif ch == " ":
            out.append("-")
    return "".join(out)
