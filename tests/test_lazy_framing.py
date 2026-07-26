"""Lazy sequence framing (MESSAGE_SPEC §2, CORELIB_PLAN §6).

A sequence-typed *field* whose value equals its declared default is **omitted**
rather than emitted as an empty ``begin``/``end`` frame — but a wrapper-array
*element* keeps its frame even when all-default, because element presence is what
carries a dynamic array's length (*highest present id + 1*, §5.1).

Whether a sequence is emitted depends on what its children turn out to be, while
its header must precede them, and buffering the sub-message is not an option for a
streaming format. So the header is held back:
``write_sequence_begin_lazy`` pushes the id onto a pending run and writes nothing;
any field write commits the whole run, outermost header first; ``write_sequence_end``
drops a sequence whose header is still pending (header *and* end marker);
``write_sequence_end_keep`` behaves like a write and forces the frame out.

Because the message layer already omits every child equal to its default,
"not one child was written" *is* "the object equals its declared default",
evaluated per child field, recursively, for free — no byte image is ever compared,
so in-memory padding cannot influence the decision.

These are the framing tests from ``corelib-rs/tests/ostream_tests.rs``, run against
**both** engines: the pure-Python encoder and the compiled accelerator carry
separate copies of the hold-back logic, so both have to be proven.
"""

from __future__ import annotations

import pytest

from sofab.encoder import Encoder as PyEncoder

_ENGINES = [pytest.param(PyEncoder, id="python")]
try:  # pragma: no cover - depends on whether the extension was built
    from sofab._speedups import Encoder as NativeEncoder

    _ENGINES.append(pytest.param(NativeEncoder, id="native"))
except ImportError:  # pragma: no cover
    pass

engine = pytest.mark.parametrize("Encoder", _ENGINES)


def _encode(Encoder, fn) -> bytes:
    enc = Encoder()
    fn(enc)
    return enc.getvalue()


# --- the seven framing tests -------------------------------------------------


@engine
def test_lazy_sequence_without_content_emits_nothing(Encoder):
    """An all-default sequence carries no information, so the field is omitted —
    where the eager API would have written the two-byte empty frame ``0E 07``."""

    def build(e):
        e.write_sequence_begin_lazy(1)
        e.write_sequence_end()

    assert _encode(Encoder, build) == b""


@engine
def test_end_keep_frames_a_contentless_sequence(Encoder):
    """``end_keep`` forces a contentless frame onto the wire — the array-element
    and explicit-empty cases of §2 / §5.1."""

    def build(e):
        e.write_sequence_begin_lazy(1)
        e.write_sequence_end_keep()

    assert _encode(Encoder, build) == bytes([0x0E, 0x07])


@engine
def test_end_keep_commits_the_enclosing_run(Encoder):
    """Forcing a frame forces its ancestors too: the outer sequence got content
    (the inner frame), so it is framed as well."""

    def build(e):
        e.write_sequence_begin_lazy(1)
        e.write_sequence_begin_lazy(2)
        e.write_sequence_end_keep()
        e.write_sequence_end()

    assert _encode(Encoder, build) == bytes([0x0E, 0x16, 0x07, 0x07])


@engine
def test_end_keep_matches_end_once_content_exists(Encoder):
    """With content the two closers make no difference — the headers are already
    out."""

    def with_keep(e):
        e.write_sequence_begin_lazy(1)
        e.write_unsigned(0, 42)
        e.write_sequence_end_keep()

    def with_end(e):
        e.write_sequence_begin_lazy(1)
        e.write_unsigned(0, 42)
        e.write_sequence_end()

    keep = _encode(Encoder, with_keep)
    assert keep == bytes([0x0E, 0x00, 0x2A, 0x07])
    assert keep == _encode(Encoder, with_end)


@engine
def test_lazy_sequence_commits_the_whole_run_on_first_content(Encoder):
    """One child field commits the whole held-back run, outermost header first, so
    a non-default leaf deep inside brings every enclosing frame back in wire
    order."""

    def build(e):
        e.write_sequence_begin_lazy(1)
        e.write_sequence_begin_lazy(2)
        e.write_unsigned(0, 42)
        e.write_sequence_end()
        e.write_sequence_end()

    assert _encode(Encoder, build) == bytes([0x0E, 0x16, 0x00, 0x2A, 0x07, 0x07])


@engine
def test_lazy_sequence_drops_only_the_empty_inner_one(Encoder):
    """Only the empty inner sequence drops; the outer one has content (the leaf)
    and is framed. This is the interleaving a naive "drop the whole run" gets
    wrong."""

    def build(e):
        e.write_sequence_begin_lazy(1)
        e.write_sequence_begin_lazy(2)
        e.write_sequence_end()
        e.write_unsigned(0, 42)
        e.write_sequence_end()

    assert _encode(Encoder, build) == bytes([0x0E, 0x00, 0x2A, 0x07])


@engine
def test_lazy_sequence_after_content_is_independent(Encoder):
    """A lazily framed sequence *after* content in the same scope, and the sibling
    order, stay intact."""

    def build(e):
        e.write_unsigned(0, 1)
        e.write_sequence_begin_lazy(1)
        e.write_sequence_end()
        e.write_unsigned(2, 3)

    assert _encode(Encoder, build) == bytes([0x00, 0x01, 0x10, 0x03])


@engine
def test_lazy_framing_is_buffer_size_independent(Encoder):
    """Held-back headers are encoder state, not buffer content, so a flush can
    never split a pending run: a 3-byte output buffer sees exactly the bytes a
    one-shot encode produces."""
    chunks: list[bytes] = []
    enc = Encoder.over_buffer(bytearray(3), 0, chunks.append)
    enc.write_sequence_begin_lazy(1)
    enc.write_sequence_begin_lazy(2)
    enc.write_sequence_end()
    enc.write_unsigned(0, 42)
    enc.write_sequence_end()
    enc.flush()
    assert b"".join(chunks) == bytes([0x0E, 0x00, 0x2A, 0x07])


# --- supporting invariants ---------------------------------------------------


@engine
def test_writer_choke_point_is_complete(Encoder):
    """Invariant 1: *every* writer must commit the pending run before its first
    byte. Each case opens a sequence lazily, writes exactly one field through a
    different writer, and closes with the dropping ``end``: if that writer bypassed
    the choke point, the ``0x0E`` header would be missing (and ``end`` would then
    have dropped a frame that already had content)."""
    writers = [
        ("unsigned", lambda e: e.write_unsigned(0, 1)),
        ("signed", lambda e: e.write_signed(0, -1)),
        ("bool", lambda e: e.write_bool(0, True)),
        ("fp32", lambda e: e.write_float32(0, 1.0)),
        ("fp64", lambda e: e.write_float64(0, 1.0)),
        ("string", lambda e: e.write_string(0, "x")),
        ("blob", lambda e: e.write_bytes(0, b"x")),
        ("unsigned_array", lambda e: e.write_unsigned_array(0, [1])),
        ("signed_array", lambda e: e.write_signed_array(0, [-1])),
        ("fp32_array", lambda e: e.write_float32_array(0, [1.0])),
        ("fp64_array", lambda e: e.write_float64_array(0, [1.0])),
        ("nested_kept_frame", lambda e: (e.write_sequence_begin_lazy(0), e.write_sequence_end_keep())),
    ]
    for name, write in writers:
        def build(e, write=write):
            e.write_sequence_begin_lazy(1)
            write(e)
            e.write_sequence_end()

        out = _encode(Encoder, build)
        assert out[:1] == b"\x0e", f"{name} bypassed the pending-run commit: {out.hex()}"
        assert out[-1:] == b"\x07", f"{name} lost its sequence end: {out.hex()}"


@engine
def test_pending_run_is_a_suffix_so_end_pops_the_innermost(Encoder):
    """Invariant 2: the pending ids are a contiguous suffix of the open sequences,
    so ``end`` popping the last entry always closes the *innermost* sequence — even
    when an eagerly-framed ancestor sits below it."""

    def build(e):
        e.write_sequence_begin_lazy(1)  # gets content below -> framed
        e.write_unsigned(0, 7)
        e.write_sequence_begin_lazy(2)  # pending, dropped
        e.write_sequence_end()
        e.write_sequence_begin_lazy(3)  # pending, kept
        e.write_sequence_end_keep()
        e.write_sequence_end()  # closes id 1, which is NOT pending -> end marker

    assert _encode(Encoder, build) == bytes([0x0E, 0x00, 0x07, 0x1E, 0x07, 0x07])


@engine
def test_all_default_message_encodes_to_zero_bytes(Encoder):
    """MESSAGE_SPEC §2: with every sequence omitted, a message whose every field
    equals its default is the empty byte string."""

    def build(e):
        for depth in range(8):
            e.write_sequence_begin_lazy(depth)
        for _ in range(8):
            e.write_sequence_end()

    assert _encode(Encoder, build) == b""


@engine
def test_depth_bookkeeping_is_unchanged(Encoder):
    """Invariant 5: ``begin_lazy`` still increments depth and still rejects past
    MAX_DEPTH; ``end`` and ``end_keep`` both decrement."""
    from sofab.types import MAX_DEPTH, SofaRangeError, SofaStateError

    enc = Encoder()
    for _ in range(MAX_DEPTH):
        enc.write_sequence_begin_lazy(0)
    with pytest.raises(SofaRangeError):
        enc.write_sequence_begin_lazy(0)
    enc.write_sequence_end()  # room again
    enc.write_sequence_begin_lazy(0)
    with pytest.raises(SofaRangeError):
        enc.write_sequence_begin_lazy(0)

    # end_keep decrements too: close everything, then over-close.
    for _ in range(MAX_DEPTH):
        enc.write_sequence_end_keep()
    with pytest.raises(SofaStateError):
        enc.write_sequence_end_keep()
    with pytest.raises(SofaStateError):
        enc.write_sequence_end()


@engine
def test_begin_lazy_rejects_an_out_of_range_id(Encoder):
    """The id check moved out of the header path (a lazy begin writes no header),
    so it has to be done up front — an invalid id must not sit in the pending run
    waiting to blow up at commit time."""
    from sofab.types import ID_MAX, SofaRangeError

    enc = Encoder()
    with pytest.raises(SofaRangeError):
        enc.write_sequence_begin_lazy(ID_MAX + 1)
    with pytest.raises(SofaRangeError):
        enc.write_sequence_begin_lazy(-1)
    # ...and nothing was left open or pending.
    enc.write_unsigned(0, 1)
    assert enc.getvalue() == bytes([0x00, 0x01])
