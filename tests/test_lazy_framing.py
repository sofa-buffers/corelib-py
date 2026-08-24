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
from vectors import ENCODER_ENGINES
from vectors import uvarint as _varint

from sofab.types import MAX_DEPTH

engine = pytest.mark.parametrize("Encoder", ENCODER_ENGINES)


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
def test_a_run_committed_across_flushes_is_byte_identical(Encoder):
    """A committed run split across flush boundaries yields exactly the one-shot
    bytes: encoding through a 3-byte output buffer that drains repeatedly produces
    the same wire image as the in-memory encoder.

    Note what this does *not* — and cannot — show: a flush landing while a header
    is still held back. That is unreachable by construction, not merely untested.
    A held-back header occupies no buffer space (the pending ids are encoder
    state, never buffer content), and the buffer only fills through a write, which
    commits the whole run *before* its first byte goes out. So there is no state in
    which a pending run straddles a flush. What is left to prove is this: that the
    bytes of an already-committed run survive being chopped up by the sink.
    """
    chunks: list[bytes] = []
    enc = Encoder.over_buffer(bytearray(3), 0, lambda c: chunks.append(bytes(c)))
    enc.write_sequence_begin_lazy(1)
    enc.write_sequence_begin_lazy(2)
    enc.write_sequence_end()
    enc.write_unsigned(0, 42)
    enc.write_string(3, "a longer payload, so the 3-byte buffer drains repeatedly")
    enc.write_sequence_end()
    enc.flush()

    def build(e):
        e.write_sequence_begin_lazy(1)
        e.write_sequence_begin_lazy(2)
        e.write_sequence_end()
        e.write_unsigned(0, 42)
        e.write_string(3, "a longer payload, so the 3-byte buffer drains repeatedly")
        e.write_sequence_end()

    assert len(chunks) > 1  # the sink really did drain mid-message
    assert b"".join(chunks) == _encode(Encoder, build)
    assert b"".join(chunks).startswith(bytes([0x0E, 0x00, 0x2A]))


# --- supporting invariants ---------------------------------------------------


#: Every public writer, one entry each — parametrized so each is an independent
#: test case (a plain loop would report as one case and stop at the first
#: offender, hiding any writer after it).
_WRITERS = [
    pytest.param(lambda e: e.write_unsigned(0, 1), id="unsigned"),
    pytest.param(lambda e: e.write_signed(0, -1), id="signed"),
    pytest.param(lambda e: e.write_bool(0, True), id="bool"),
    pytest.param(lambda e: e.write_float32(0, 1.0), id="fp32"),
    pytest.param(lambda e: e.write_float64(0, 1.0), id="fp64"),
    pytest.param(lambda e: e.write_string(0, "x"), id="string"),
    pytest.param(lambda e: e.write_bytes(0, b"x"), id="blob"),
    pytest.param(lambda e: e.write_unsigned_array(0, [1]), id="unsigned_array"),
    pytest.param(lambda e: e.write_signed_array(0, [-1]), id="signed_array"),
    pytest.param(lambda e: e.write_float32_array(0, [1.0]), id="fp32_array"),
    pytest.param(lambda e: e.write_float64_array(0, [1.0]), id="fp64_array"),
    pytest.param(
        lambda e: (e.write_sequence_begin_lazy(0), e.write_sequence_end_keep()),
        id="nested_kept_frame",
    ),
]


@engine
@pytest.mark.parametrize("write", _WRITERS)
def test_writer_choke_point_is_complete(Encoder, write):
    """Invariant 1: *every* writer must commit the pending run before its first
    byte. Each case opens a sequence lazily, writes exactly one field through a
    different writer, and closes with the dropping ``end``: if that writer bypassed
    the choke point, the ``0x0E`` header would be missing (and ``end`` would then
    have dropped a frame that already had content)."""

    def build(e):
        e.write_sequence_begin_lazy(1)
        write(e)
        e.write_sequence_end()

    out = _encode(Encoder, build)
    assert out[:1] == b"\x0e", f"bypassed the pending-run commit: {out.hex()}"
    assert out[-1:] == b"\x07", f"lost its sequence end: {out.hex()}"


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
@pytest.mark.parametrize("depth", [1, 8, 40, MAX_DEPTH])
def test_all_default_message_encodes_to_zero_bytes(Encoder, depth):
    """MESSAGE_SPEC §2: with every sequence omitted, a message whose every field
    equals its default is the empty byte string — at *every* depth.

    CORELIB_PLAN §6 ("how deep the hold-back reaches"): both engines can allocate,
    so both hold back to the full ``MAX_DEPTH`` and are canonical everywhere. The
    pending run grows on demand — a Python list, a doubling heap block in the
    accelerator — so there is no window to overflow and no eager-framing fallback
    that would emit the empty frames §2 omits. 40 levels is deliberately past any
    plausible fixed window (and ``MAX_DEPTH`` is past every one).
    """

    def build(e):
        for i in range(depth):
            e.write_sequence_begin_lazy(i)
        for _ in range(depth):
            e.write_sequence_end()

    assert _encode(Encoder, build) == b""


@engine
@pytest.mark.parametrize("depth", [40, MAX_DEPTH])
def test_deep_nesting_commits_the_whole_run_in_order(Encoder, depth):
    """The other half of the same claim: nested that deep, one leaf write must
    still bring *every* enclosing header back, outermost first, and each ``end``
    must then emit its marker. Nothing may be framed eagerly and nothing dropped."""

    def build(e):
        for i in range(depth):
            e.write_sequence_begin_lazy(i)
        e.write_unsigned(0, 42)
        for _ in range(depth):
            e.write_sequence_end()

    expected = bytearray()
    for i in range(depth):
        expected += _varint((i << 3) | 0x06)  # header: id=i, wire type 6 = SEQUENCE_START
    expected += bytes([0x00, 0x2A])  # the leaf: id 0, unsigned, value 42
    expected += b"\x07" * depth
    assert _encode(Encoder, build) == bytes(expected)


@engine
def test_depth_bookkeeping_is_unchanged(Encoder):
    """Invariant 5: ``begin_lazy`` still increments depth and still rejects past
    MAX_DEPTH; ``end`` and ``end_keep`` both decrement."""
    from sofab.types import MAX_DEPTH, SofaArgumentError

    enc = Encoder()
    for _ in range(MAX_DEPTH):
        enc.write_sequence_begin_lazy(0)
    with pytest.raises(SofaArgumentError):
        enc.write_sequence_begin_lazy(0)
    enc.write_sequence_end()  # room again
    enc.write_sequence_begin_lazy(0)
    with pytest.raises(SofaArgumentError):
        enc.write_sequence_begin_lazy(0)

    # end_keep decrements too: close everything, then over-close.
    for _ in range(MAX_DEPTH):
        enc.write_sequence_end_keep()
    with pytest.raises(SofaArgumentError):
        enc.write_sequence_end_keep()
    with pytest.raises(SofaArgumentError):
        enc.write_sequence_end()


@engine
def test_begin_lazy_rejects_an_out_of_range_id(Encoder):
    """The id check moved out of the header path (a lazy begin writes no header),
    so it has to be done up front — an invalid id must not sit in the pending run
    waiting to blow up at commit time."""
    from sofab.types import ID_MAX, SofaArgumentError

    enc = Encoder()
    with pytest.raises(SofaArgumentError):
        enc.write_sequence_begin_lazy(ID_MAX + 1)
    with pytest.raises(SofaArgumentError):
        enc.write_sequence_begin_lazy(-1)
    # ...and nothing was left open or pending.
    enc.write_unsigned(0, 1)
    assert enc.getvalue() == bytes([0x00, 0x01])
