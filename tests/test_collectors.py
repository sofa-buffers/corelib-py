"""The static helper layer (CORELIB_PLAN §6.6.1).

    the reassembly buffers, **sequence collectors and array builders** a port
    holds so the generator need not emit them into every generated package

It ships beside the codec and is not part of it: the generated layer calls a
collector, the collector calls the codec, and *this* layer is the one allowed to
allocate (§6.6). The shared ``sequence_growth`` cases drive it too — see
``test_sequence_growth.py``; this file is the contract underneath them.
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES

from sofab import (
    BytesSeq,
    Encoder,
    Float32Seq,
    Float64Seq,
    NestedSeq,
    SignedSeq,
    SofaDecodeError,
    SofaLimitError,
    Status,
    StringSeq,
    UnsignedSeq,
    Visitor,
)

WRAPPER = 4


class Row(Visitor):
    def __init__(self) -> None:
        self.fields: dict[int, int] = {}

    def on_unsigned(self, field_id, value):
        self.fields[field_id] = value


def root(make):
    """The object holding the array field, handing its scope to a collector."""

    class Root(Visitor):
        def on_sequence_begin(self, field_id):
            return make() if field_id == WRAPPER else None

    return Root()


def _wrap(write) -> bytes:
    enc = Encoder()
    enc.write_sequence_begin_lazy(WRAPPER)
    write(enc)
    enc.write_sequence_end_keep()
    enc.flush()
    return enc.getvalue()


# --- placement ---------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_elements_are_placed_at_their_id_not_appended(engine):
    """MESSAGE_SPEC §2 omits an interior element equal to its default, so the
    gap has to be filled: appending would shorten the array by every gap."""
    out: list = []
    wire = _wrap(lambda e: [e.write_string(0, "a"), e.write_string(3, "d")])
    assert engine(visitor=root(lambda: StringSeq(out))).feed(wire) is Status.COMPLETE
    assert out == ["a", "", "", "d"]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_reopened_id_overwrites_rather_than_appends(engine):
    out: list = []
    wire = _wrap(lambda e: [e.write_string(0, "a"), e.write_string(0, "b")])
    assert engine(visitor=root(lambda: StringSeq(out))).feed(wire) is Status.COMPLETE
    assert out == ["b"]


@pytest.mark.parametrize("engine", ENGINES)
def test_an_empty_wrapper_collects_nothing(engine):
    out: list = []
    assert engine(visitor=root(lambda: StringSeq(out))).feed(_wrap(lambda e: None)) is (
        Status.COMPLETE
    )
    assert out == []


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "cls,write,want",
    [
        (BytesSeq, lambda e: e.write_bytes(1, b"z"), [b"", b"z"]),
        (UnsignedSeq, lambda e: e.write_unsigned(1, 9), [0, 9]),
        (SignedSeq, lambda e: e.write_signed(1, -9), [0, -9]),
        (Float32Seq, lambda e: e.write_float32(1, 1.5), [0.0, 1.5]),
        (Float64Seq, lambda e: e.write_float64(1, 1.5), [0.0, 1.5]),
    ],
    ids=["bytes", "unsigned", "signed", "float32", "float64"],
)
def test_each_element_type_fills_its_gap_with_its_own_default(engine, cls, write, want):
    out: list = []
    assert engine(visitor=root(lambda: cls(out))).feed(_wrap(write)) is Status.COMPLETE
    assert out == want


# --- which bound applies (§6.2.1) -------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_schema_cap_makes_an_over_index_invalid(engine):
    """The schema bounded the array, so an id past it is a statement about
    validity (§7.1) -- not the receiver's capacity."""
    out: list = []
    wire = _wrap(lambda e: e.write_string(8, "x"))
    dec = engine(visitor=root(lambda: StringSeq(out, cap=4)))
    # A schema-bound violation is the INVALID outcome, not an exception: the
    # decoder answers §7.1 in the status, and reserves the error channel for the
    # policy rejection a receiver limit is (§6.3).
    assert dec.feed(wire) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert not isinstance(dec.error, SofaLimitError)
    assert out == []


@pytest.mark.parametrize("engine", ENGINES)
def test_without_a_schema_cap_the_receiver_limit_applies(engine):
    out: list = []
    wire = _wrap(lambda e: e.write_string(8, "x"))
    dec = engine(visitor=root(lambda: StringSeq(out, max_dyn_array_count=4)))
    with pytest.raises(SofaLimitError):
        dec.feed(wire)
    assert out == []


@pytest.mark.parametrize("engine", ENGINES)
def test_a_schema_cap_takes_the_receiver_limit_off_the_field(engine):
    """§6.2.1: a receiver limit "MUST NOT be applied to a field the schema
    already bounds"."""
    out: list = []
    wire = _wrap(lambda e: e.write_string(6, "x"))
    dec = engine(visitor=root(lambda: StringSeq(out, cap=8, max_dyn_array_count=2)))
    assert dec.feed(wire) is Status.COMPLETE
    assert len(out) == 7


@pytest.mark.parametrize("engine", ENGINES)
def test_the_index_is_judged_before_the_list_grows(engine):
    """An index near 2**31 must cost a comparison, not an allocation."""
    out: list = []
    wire = _wrap(lambda e: e.write_string(1 << 30, "x"))
    dec = engine(visitor=root(lambda: StringSeq(out, cap=4)))
    assert dec.feed(wire) is Status.INVALID
    assert out == []


@pytest.mark.parametrize("engine", ENGINES)
def test_the_last_legal_index_is_accepted(engine):
    out: list = []
    wire = _wrap(lambda e: e.write_string(3, "d"))
    assert engine(visitor=root(lambda: StringSeq(out, cap=4))).feed(wire) is (
        Status.COMPLETE
    )
    assert out == ["", "", "", "d"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("cls,write", [(StringSeq, "write_string"), (BytesSeq, "write_bytes")])
def test_an_element_over_the_declared_maxlen_is_invalid(engine, cls, write):
    out: list = []
    payload = "y" * 40 if cls is StringSeq else b"y" * 40
    wire = _wrap(lambda e: getattr(e, write)(0, payload))
    dec = engine(visitor=root(lambda: cls(out, elem_max=8)))
    assert dec.feed(wire) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert not isinstance(dec.error, SofaLimitError)


# --- framed elements ---------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_framed_elements_each_get_their_own_handler(engine):
    out: list = []
    enc = Encoder()
    enc.write_sequence_begin_lazy(WRAPPER)
    for index, value in ((0, 11), (2, 33)):
        enc.write_sequence_begin_lazy(index)
        enc.write_unsigned(0, value)
        enc.write_sequence_end_keep()
    enc.write_sequence_end_keep()
    enc.flush()

    dec = engine(visitor=root(lambda: NestedSeq(out, factory=Row)))
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    assert len(out) == 3
    assert out[0].fields == {0: 11}
    assert out[1] is None  # the gap keeps the element default
    assert out[2].fields == {0: 33}


@pytest.mark.parametrize("engine", ENGINES)
def test_a_framed_element_is_placed_before_it_is_filled(engine):
    """The list's shape is settled at the index, so a handler that keeps a
    reference sees the object the list holds."""
    out: list = []
    enc = Encoder()
    enc.write_sequence_begin_lazy(WRAPPER)
    enc.write_sequence_begin_lazy(0)
    enc.write_unsigned(0, 7)
    enc.write_sequence_end_keep()
    enc.write_sequence_end_keep()
    enc.flush()
    dec = engine(visitor=root(lambda: NestedSeq(out, factory=Row)))
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    assert out[0].fields == {0: 7}


@pytest.mark.parametrize("engine", ENGINES)
def test_a_framed_element_over_the_cap_is_refused_before_the_factory_runs(engine):
    made = []

    def factory():
        made.append(1)
        return Row()

    out: list = []
    enc = Encoder()
    enc.write_sequence_begin_lazy(WRAPPER)
    enc.write_sequence_begin_lazy(9)
    enc.write_unsigned(0, 1)
    enc.write_sequence_end_keep()
    enc.write_sequence_end_keep()
    enc.flush()
    dec = engine(visitor=root(lambda: NestedSeq(out, factory=factory, cap=4)))
    assert dec.feed(enc.getvalue()) is Status.INVALID
    assert out == []
    assert made == [1], "the factory runs, but its element is never placed"


# --- the descent itself ------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("chunk", [None, 1, 3])
def test_the_descent_survives_a_chunk_boundary(engine, chunk):
    out: list = []
    wire = _wrap(lambda e: [e.write_string(0, "a"), e.write_string(1, "bbbbbbbb")])
    dec = engine(visitor=root(lambda: StringSeq(out)))
    status = Status.COMPLETE
    if chunk is None:
        status = dec.feed(wire)
    else:
        for i in range(0, len(wire), chunk):
            status = dec.feed(wire[i : i + chunk])
    assert status is Status.COMPLETE
    assert out == ["a", "bbbbbbbb"]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_parent_resumes_after_the_scope_closes(engine):
    out: list = []
    seen: list = []

    class Root(Visitor):
        def on_sequence_begin(self, field_id):
            return StringSeq(out) if field_id == WRAPPER else None

        def on_unsigned(self, field_id, value):
            seen.append((field_id, value))

    enc = Encoder()
    enc.write_unsigned(1, 10)
    enc.write_sequence_begin_lazy(WRAPPER)
    enc.write_string(0, "a")
    enc.write_sequence_end_keep()
    enc.write_unsigned(2, 20)
    enc.flush()

    assert engine(visitor=Root()).feed(enc.getvalue()) is Status.COMPLETE
    assert out == ["a"]
    assert seen == [(1, 10), (2, 20)], "the parent must get the fields after the scope"


@pytest.mark.parametrize("engine", ENGINES)
def test_two_wrappers_get_two_collectors(engine):
    a: list = []
    b: list = []

    class Root(Visitor):
        def on_sequence_begin(self, field_id):
            return StringSeq(a if field_id == 4 else b)

    enc = Encoder()
    enc.write_sequence_begin_lazy(4)
    enc.write_string(0, "x")
    enc.write_sequence_end_keep()
    enc.write_sequence_begin_lazy(5)
    enc.write_string(1, "y")
    enc.write_sequence_end_keep()
    enc.flush()

    assert engine(visitor=Root()).feed(enc.getvalue()) is Status.COMPLETE
    assert a == ["x"]
    assert b == ["", "y"]


@pytest.mark.parametrize("engine", ENGINES)
def test_reset_puts_the_callers_handler_back(engine):
    """A message abandoned mid-descent must not leave the child in charge."""
    out: list = []
    seen: list = []

    class Root(Visitor):
        def on_sequence_begin(self, field_id):
            return StringSeq(out)

        def on_unsigned(self, field_id, value):
            seen.append(value)

    enc = Encoder()
    enc.write_sequence_begin_lazy(WRAPPER)
    enc.write_string(0, "a")
    enc.flush()  # no end marker: the descent is still open

    dec = engine(visitor=Root())
    assert dec.feed(enc.getvalue()) is Status.INCOMPLETE
    dec.reset()

    tail = Encoder()
    tail.write_unsigned(1, 42)
    tail.flush()
    assert dec.feed(tail.getvalue()) is Status.COMPLETE
    assert seen == [42], "the caller's handler must be back in charge"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_string_elements_maxlen_is_a_byte_length_not_a_character_count(engine):
    """MESSAGE_SPEC §1 makes ``maxlen`` a bound on the payload's **wire byte
    length**, and §7 makes a longer payload INVALID.

    ``'hél'`` is three code points and **four** UTF-8 bytes, so a collector that
    measured the decoded ``str`` accepted it against ``elem_max=3`` — a bound it
    violates. The sibling ``BytesSeq`` never had the bug, because ``len()`` on
    ``bytes`` is already the wire length; this pins the two to the same rule.
    """
    out: list = []
    wire = _wrap(lambda e: e.write_string(0, "hél"))
    dec = engine(visitor=root(lambda: StringSeq(out, elem_max=3)))
    assert dec.feed(wire) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert not isinstance(dec.error, SofaLimitError)
    assert out == []

    # Four is the byte length, so four is what the bound has to accept.
    ok: list = []
    assert engine(visitor=root(lambda: StringSeq(ok, elem_max=4))).feed(wire) is (
        Status.COMPLETE
    )
    assert ok == ["hél"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("cls,write", [(StringSeq, "write_string"), (BytesSeq, "write_bytes")])
def test_an_over_long_element_is_refused_at_the_fixlen_word(engine, cls, write):
    """The verdict comes from the length header, not from the built element, so
    a payload the message truncates behind is still INVALID (§7.1, §5.2.3)."""
    out: list = []
    payload = "y" * 40 if cls is StringSeq else b"y" * 40
    wire = _wrap(lambda e: getattr(e, write)(0, payload))
    # Cut the message inside the payload: the length word has arrived, the
    # bytes it announces have not.
    dec = engine(visitor=root(lambda: cls(out, elem_max=8)))
    assert dec.feed(wire[:6]) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert out == []


@pytest.mark.parametrize("engine", ENGINES)
def test_an_elements_bound_does_not_judge_the_other_subtype(engine):
    """A ``blob`` reaching a ``StringSeq`` is a MESSAGE_SPEC §7.3 type mismatch:
    it is skipped, so it is never judged against a bound that is not its."""
    out: list = []
    wire = _wrap(lambda e: e.write_bytes(0, b"y" * 40))
    assert engine(visitor=root(lambda: StringSeq(out, elem_max=8))).feed(wire) is (
        Status.COMPLETE
    )
    assert out == []


@pytest.mark.parametrize("engine", ENGINES)
def test_a_child_handler_gets_its_own_on_field(engine):
    """The hook flags belong to the **handler**, not to the decode.

    A root that does not override ``on_field`` handing a scope to a child that
    does used to leave the pure engine's per-loop flag stale — the child was
    never asked, and the walk asserted on the ``Field`` nobody had built. The
    collectors' own ``elem_max`` rides on exactly this, so it is pinned here as
    well as through them.
    """

    seen: list = []

    class Child(Visitor):
        def on_field(self, field):
            seen.append((field.id, field.type, field.size))
            return None

    class Root(Visitor):
        def on_sequence_begin(self, field_id):
            return Child() if field_id == WRAPPER else None

    wire = _wrap(lambda e: (e.write_string(0, "ab"), e.write_unsigned(1, 7)))
    dec = engine(visitor=Root())
    assert dec.feed(wire) is Status.COMPLETE
    assert [(i, s) for i, _t, s in seen] == [(0, 2), (1, 0)]


# --- growth geometry (§7.2 item 8) -------------------------------------------


def test_the_container_extends_to_the_index_in_one_pass():
    """§7.2 item 8: "Test it where the language offers [an allocation-counting
    facility]; where it does not, say so in the port's README rather than
    reporting the case as passed." Python offers ``tracemalloc``, so it is
    tested.

    The property is that a **sparse** wrapper array does not cost O(n²): placing
    at a far index extends the container to at least ``index + 1`` in one pass,
    rather than re-copying the whole list per element. CPython's ``list`` gives
    that for free — appending is amortised O(1) — and the point of the case is
    to notice if the collector ever stops using it.
    """
    import tracemalloc

    span = 1 << 14

    def place(step):
        out: list = []
        coll = UnsignedSeq(out, cap=span)
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            for index in range(0, span, step):
                coll.on_unsigned(index, index)
            return tracemalloc.get_traced_memory()[1] - base, out
        finally:
            tracemalloc.stop()

    dense, out_dense = place(1)
    sparse, out_sparse = place(1 << 8)

    # Both reach the same length: the gap is filled, not skipped (MESSAGE_SPEC §2).
    assert len(out_dense) == span
    assert len(out_sparse) == span - (1 << 8) + 1
    assert out_sparse[0] == 0 and out_sparse[1] == 0 and out_sparse[1 << 8] == 1 << 8

    # And the sparse walk costs no more than the dense one: if the container were
    # rebuilt per element rather than extended, 64 far placements over 16,384
    # slots would peak at many times a single list of that size.
    assert sparse <= dense * 2, (
        f"a sparse array peaked at {sparse} bytes against {dense} for the dense "
        "one; the container is being rebuilt rather than extended"
    )
    # One list of `span` slots is ~8 bytes per slot; anything near a multiple of
    # that is a copy per element.
    assert sparse < span * 8 * 3, f"{sparse} bytes for {span} slots"
