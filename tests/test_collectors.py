"""The static helper layer (CORELIB_PLAN §6.6.1): ``sofab.collectors``.

    the reassembly buffers, **sequence collectors and array builders** a port
    holds so the generator need not emit them into every generated package

The first half drives :func:`reserve_leaf`, :func:`reserve_elem` and
:func:`reserve_row` **directly**, on a plain list: a test that brings its own
container tests its own container, so the helpers' contract is pinned here with
nothing of the codec in between.

The second half drives them the way generated code does — from inside a **flat**
visitor that routes every scope itself — so that the verdicts they raise reach
the caller as the decoder's categories (INVALID in the status, a limit or an
argument refusal as an exception), on both engines and across chunk boundaries.
The shared ``sequence_growth`` cases drive them too (``test_sequence_growth.py``).
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import NO_CAPS

try:
    from sofab import _speedups as _native
except ImportError:  # pragma: no cover - no compiled extension
    _native = None  # type: ignore[assignment]

import sofab
from sofab import (
    UNBOUNDED,
    Encoder,
    FixlenSubtype,
    SofaArgumentError,
    SofaDecodeError,
    SofaError,
    SofaLimitError,
    Status,
    Visitor,
    WireType,
    collectors,
    reserve_elem,
    reserve_leaf,
    reserve_row,
)

#: Both implementations of the helpers: the pure ``sofab.collectors`` functions
#: and, where the extension is built, their compiled twins in ``sofab._speedups``
#: (which ``sofab`` re-exports when the native engine is active). Every direct
#: test below runs against each, so the two cannot drift apart.
HELPERS = [pytest.param(collectors, id="python")]
if _native is not None:  # pragma: no cover - native-only branch
    HELPERS.append(pytest.param(_native, id="native"))


@pytest.fixture(params=HELPERS)
def h(request):
    return request.param


class Row:
    """A framed element: a stand-in for a generated struct class."""

    def __init__(self) -> None:
        self.fields: dict[int, int] = {}


# =============================================================================
# The helpers themselves
# =============================================================================

# --- placement ---------------------------------------------------------------


def test_a_leaf_is_reserved_at_its_id(h):
    out: list[str] = []
    h.reserve_leaf(out, 0, "", 4, 8)
    out[0] = "a"
    assert out == ["a"]


def test_a_gap_left_by_omitted_interior_elements_keeps_the_default(h):
    """MESSAGE_SPEC §2 omits an interior element equal to its default, so the
    gap has to be filled: appending would shorten the array by every gap."""
    out: list[str] = []
    h.reserve_leaf(out, 0, "", 8, 8)
    out[0] = "a"
    h.reserve_leaf(out, 3, "", 8, 8)
    out[3] = "d"
    assert out == ["a", "", "", "d"]


def test_the_length_is_the_highest_id_plus_one(h):
    out: list[bytes] = []
    h.reserve_leaf(out, 5, b"", UNBOUNDED, 8)
    assert len(out) == 6
    assert out == [b""] * 6


def test_a_reserve_never_overwrites_a_slot_already_present(h):
    """The helper grows; the value store replaces (§7.4). Reserving again for a
    repeated id -- or for a header a resumed decode asks about twice -- must not
    wipe the value already there."""
    out: list[str] = []
    h.reserve_leaf(out, 1, "", 4, 8)
    out[1] = "b"
    h.reserve_leaf(out, 1, "", 4, 8)
    assert out == ["", "b"]
    h.reserve_leaf(out, 0, "", 4, 8)  # a lower id: nothing grows, nothing moves
    assert out == ["", "b"]


def test_a_framed_element_gets_its_own_object_per_slot(h):
    """A shared mutable default would alias every element onto one object."""
    out: list[Row] = []
    h.reserve_elem(out, 2, Row, 4, 8)
    assert len(out) == 3
    assert len({id(r) for r in out}) == 3
    out[0].fields[0] = 1
    assert out[1].fields == {} and out[2].fields == {}


def test_a_reopened_framed_element_merges_rather_than_restarts(h):
    """§7.4: a framed element's fields arrive one at a time, so a re-opened id
    routes into what its earlier fields built."""
    made: list[int] = []

    def make() -> Row:
        made.append(1)
        return Row()

    out: list[Row] = []
    h.reserve_elem(out, 0, make, 4, 8)
    out[0].fields[0] = 7
    first = out[0]
    h.reserve_elem(out, 0, make, 4, 8)
    assert out[0] is first and first.fields == {0: 7}
    assert made == [1]


def test_a_native_matrix_row_is_reserved_with_list_as_the_factory(h):
    rows: list[list[int]] = []
    h.reserve_elem(rows, 1, list, 2, 8)
    assert rows == [[], []]
    assert rows[0] is not rows[1]


def test_a_row_is_replaced_whole_and_its_gaps_are_fresh_rows(h):
    """An array wrapper *is* the array's value (§7.4): a repeated row id replaces
    the row, and a caller's reference to the earlier list is left intact."""
    rows: list[list[str]] = []
    h.reserve_row(rows, 2, UNBOUNDED, 8)
    assert rows == [[], [], []]
    assert len({id(r) for r in rows}) == 3
    rows[2].append("x")
    kept = rows[2]
    h.reserve_row(rows, 2, UNBOUNDED, 8)
    assert rows[2] == [] and rows[2] is not kept
    assert kept == ["x"]
    assert len(rows) == 3


# --- the schema bound (§7.1) -------------------------------------------------


@pytest.mark.parametrize(
    "reserve",
    [
        lambda h, out, i: h.reserve_leaf(out, i, "", 4, 8),
        lambda h, out, i: h.reserve_elem(out, i, Row, 4, 8),
        lambda h, out, i: h.reserve_row(out, i, 4, 8),
    ],
    ids=["leaf", "elem", "row"],
)
def test_the_last_index_under_the_schema_count_is_accepted(h, reserve):
    out: list = []
    reserve(h, out, 3)
    assert len(out) == 4


@pytest.mark.parametrize(
    "reserve",
    [
        lambda h, out, i: h.reserve_leaf(out, i, "", 4, 8),
        lambda h, out, i: h.reserve_elem(out, i, Row, 4, 8),
        lambda h, out, i: h.reserve_row(out, i, 4, 8),
    ],
    ids=["leaf", "elem", "row"],
)
def test_an_index_at_the_schema_count_is_invalid_and_extends_nothing(h, reserve):
    """A ``count`` is a capacity: an id at it contradicts the schema (§7.1).
    The check runs before any growth (§7.2 item 8), so the list is not left
    partially extended and a lower id delivered afterwards still lands."""
    out: list = []
    reserve(h, out, 1)
    with pytest.raises(SofaDecodeError) as exc:
        reserve(h, out, 4)
    assert not isinstance(exc.value, SofaLimitError)
    assert len(out) == 2, "the refused id extended the list"
    reserve(h, out, 2)
    assert len(out) == 3


def test_a_far_index_costs_a_comparison_not_an_allocation(h):
    out: list = []
    with pytest.raises(SofaDecodeError):
        h.reserve_leaf(out, 1 << 30, "", 4, 8)
    assert out == []


def test_the_factory_never_runs_for_a_refused_index(h):
    made: list[int] = []

    def make() -> Row:
        made.append(1)
        return Row()

    out: list[Row] = []
    with pytest.raises(SofaDecodeError):
        h.reserve_elem(out, 9, make, 4, 8)
    assert out == [] and made == []


def test_a_schema_count_takes_the_receiver_cap_off_the_field(h):
    """§6.2.1: a receiver limit "MUST NOT be applied to a field the schema
    already bounds" -- a declared count above the receiver cap wins."""
    out: list[str] = []
    h.reserve_leaf(out, 6, "", 8, 2)
    assert len(out) == 7


# --- the receiver cap (§6.2.1, §6.3) ----------------------------------------


@pytest.mark.parametrize(
    "reserve",
    [
        lambda h, out, i, r: h.reserve_leaf(out, i, "", UNBOUNDED, r),
        lambda h, out, i, r: h.reserve_elem(out, i, Row, UNBOUNDED, r),
        lambda h, out, i, r: h.reserve_row(out, i, UNBOUNDED, r),
    ],
    ids=["leaf", "elem", "row"],
)
def test_past_the_receiver_cap_on_an_unbounded_array_is_limit_exceeded(h, reserve):
    """Well-formed input this receiver declines: the policy category, not
    INVALID -- and, like the schema bound, judged before any growth."""
    out: list = []
    reserve(h, out, 3, 4)
    with pytest.raises(SofaLimitError) as exc:
        reserve(h, out, 4, 4)
    assert not isinstance(exc.value, SofaDecodeError)
    assert "limit" in str(exc.value).lower()
    assert len(out) == 4
    reserve(h, out, 1, 4)  # a lower id still lands
    assert len(out) == 4


@pytest.mark.parametrize("rcap", [-1, None, float("inf"), 8.0])
@pytest.mark.parametrize(
    "reserve",
    [
        lambda h, out, r: h.reserve_leaf(out, 0, "", UNBOUNDED, r),
        lambda h, out, r: h.reserve_elem(out, 0, Row, UNBOUNDED, r),
        lambda h, out, r: h.reserve_row(out, 0, UNBOUNDED, r),
    ],
    ids=["leaf", "elem", "row"],
)
def test_an_unstated_receiver_cap_is_an_argument_error(h, reserve, rcap):
    """§6.2.1: the helper "MUST NOT read an omitted argument as *unlimited*".
    A cap that states no number admits no element, and the refusal is the
    ``InvalidArgument`` tier -- not a limit nobody configured (§6.3)."""
    out: list = []
    with pytest.raises(SofaError) as exc:
        reserve(h, out, rcap)
    assert isinstance(exc.value, SofaArgumentError)
    assert not isinstance(exc.value, SofaLimitError)
    assert out == []


def test_the_receiver_cap_is_a_required_argument(h):
    """It cannot be left out; there is no default to fall back on."""
    with pytest.raises(TypeError):
        h.reserve_leaf([], 0, "", UNBOUNDED)  # type: ignore[call-arg]


def test_unbounded_is_the_exported_sentinel(h):
    assert UNBOUNDED == -1
    for name in ("UNBOUNDED", "reserve_leaf", "reserve_elem", "reserve_row"):
        assert name in sofab.__all__


def test_a_receiver_cap_wider_than_a_machine_word_is_compared_not_clamped(h):
    out: list = []
    h.reserve_leaf(out, 3, "", UNBOUNDED, 1 << 70)
    assert len(out) == 4
    with pytest.raises(SofaArgumentError):
        h.reserve_leaf(out, 3, "", UNBOUNDED, -(1 << 70))


def test_a_list_subclass_is_grown_like_a_list(h):
    class Tracked(list):
        pass

    out = Tracked()
    h.reserve_leaf(out, 1, "", 4, 8)
    h.reserve_elem(out, 2, Row, 4, 8)
    assert len(out) == 3 and out[:2] == ["", ""] and isinstance(out[2], Row)
    rows = Tracked()
    h.reserve_row(rows, 1, UNBOUNDED, 8)
    assert rows == [[], []]


def test_the_refusal_text_is_the_same_in_both_implementations(h):
    """The native twins build their refusal with the pure module's own
    ``_refusal``, so a message classified on its text reads the same."""
    for args, cls in (((4, 4, 8), SofaDecodeError), ((4, UNBOUNDED, 4), SofaLimitError)):
        with pytest.raises(cls) as exc:
            h.reserve_leaf([], args[0], "", args[1], args[2])
        assert str(exc.value) == str(collectors._refusal(*args))


def test_sofab_exports_the_twins_of_the_active_engine():
    """``sofab.reserve_*`` follow ``sofab.IMPL`` exactly as ``Decoder`` does."""
    src = _native if sofab.IMPL == "native" else collectors
    assert src is not None
    assert sofab.reserve_leaf is src.reserve_leaf
    assert sofab.reserve_elem is src.reserve_elem
    assert sofab.reserve_row is src.reserve_row


# --- growth geometry (§7.2 item 8) -------------------------------------------


def test_the_container_extends_to_the_index_in_one_pass(h):
    """§7.2 item 8: "Test it where the language offers [an allocation-counting
    facility]; where it does not, say so in the port's README rather than
    reporting the case as passed." Python offers ``tracemalloc``, so it is
    tested.

    The property is that a **sparse** wrapper array does not cost O(n²): placing
    at a far index extends the container to at least ``index + 1`` in one pass,
    rather than re-copying the whole list per element. CPython's ``list`` gives
    that for free — appending is amortised O(1) — and the point of the case is
    to notice if the helper ever stops using it.
    """
    import tracemalloc

    span = 1 << 14

    def place(step):
        out: list = []
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            for index in range(0, span, step):
                h.reserve_leaf(out, index, 0, span, 8)
                out[index] = index
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


# =============================================================================
# Through a decode: the helpers called from a flat visitor
# =============================================================================

TAGS = 3  # array<string, count 4, maxlen 8>
ROWS = 4  # array<Row, count 4>
MATRIX = 5  # array<array<string>>, no count
DYN = 6  # array<string>, no count
BLOBS = 7  # array<blob, count 4, maxlen 8>

TAGS_CAP = 4
ROWS_CAP = 4
ELEM_MAXLEN = 8


class Doc(Visitor):
    """The shape a generated visitor takes: flat, routing each scope itself,
    growing each list through the helpers and storing values by index."""

    def __init__(self, rcap: int) -> None:
        self.rcap = rcap
        self.tags: list[str] = []
        self.rows: list[Row] = []
        self.matrix: list[list[str]] = []
        self.dyn: list[str] = []
        self.blobs: list[bytes] = []
        self._scope: list[str] = ["root"]
        self._ix = 0
        self._rix = 0
        self.after: list[tuple[int, int]] = []

    # A leaf scope: its list, its schema count, its element subtype.
    def _leaf(self, scope: str):
        if scope == "tags":
            return self.tags, TAGS_CAP, FixlenSubtype.STRING, ""
        if scope == "dyn":
            return self.dyn, UNBOUNDED, FixlenSubtype.STRING, ""
        if scope == "blobs":
            return self.blobs, TAGS_CAP, FixlenSubtype.BLOB, b""
        if scope == "mrow":
            return self.matrix[self._rix], UNBOUNDED, FixlenSubtype.STRING, ""
        return None

    def on_field(self, field):
        leaf = self._leaf(self._scope[-1])
        if leaf is None:
            return None
        out, cap, subtype, default = leaf
        # §7.3: a mistyped element is skipped, before the bound is asked.
        if field.type is not WireType.FIXLEN or field.subtype is not subtype:
            return False
        reserve_leaf(out, field.id, default, cap, self.rcap)
        return None

    def on_schema_bound(self, field_id, n, wtype, subtype):
        scope = self._scope[-1]
        if scope in ("tags", "blobs") and wtype is WireType.FIXLEN:
            return ELEM_MAXLEN
        return -1

    def on_sequence_begin(self, field_id):
        scope = self._scope[-1]
        if scope == "root":
            name = {TAGS: "tags", ROWS: "rows", MATRIX: "matrix", DYN: "dyn", BLOBS: "blobs"}.get(field_id)
            if name is None:
                return False
            setattr(self, name, [])  # §7.4: the array field is replaced
            self._scope.append(name)
        elif scope == "rows":
            reserve_elem(self.rows, field_id, Row, ROWS_CAP, self.rcap)
            self._ix = field_id
            self._scope.append("row")
        elif scope == "matrix":
            reserve_row(self.matrix, field_id, UNBOUNDED, self.rcap)
            self._rix = field_id
            self._scope.append("mrow")
        else:
            return False
        return None

    def on_sequence_end(self):
        self._scope.pop()

    def on_string(self, field_id, value):
        self._leaf(self._scope[-1])[0][field_id] = value

    def on_bytes(self, field_id, value):
        self._leaf(self._scope[-1])[0][field_id] = value

    def on_unsigned(self, field_id, value):
        if self._scope[-1] == "row":
            self.rows[self._ix].fields[field_id] = value
        else:
            self.after.append((field_id, value))


def _wrap(field: int, write) -> bytes:
    enc = Encoder()
    enc.write_sequence_begin_lazy(field)
    write(enc)
    enc.write_sequence_end_keep()
    enc.flush()
    return enc.getvalue()


def _feed(dec, wire: bytes, chunk: int | None = None) -> Status:
    if chunk is None:
        return dec.feed(wire)
    status = Status.INCOMPLETE
    for i in range(0, len(wire), chunk):
        status = dec.feed(wire[i : i + chunk])
    return status


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("chunk", [None, 1, 3])
def test_leaves_are_placed_by_id_through_a_decode(engine, chunk):
    doc = Doc(rcap=8)
    wire = _wrap(TAGS, lambda e: [e.write_string(0, "a"), e.write_string(3, "dddddddd")])
    assert _feed(engine(**NO_CAPS, visitor=doc), wire, chunk) is Status.COMPLETE
    assert doc.tags == ["a", "", "", "dddddddd"]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_repeated_leaf_id_replaces(engine):
    doc = Doc(rcap=8)
    wire = _wrap(TAGS, lambda e: [e.write_string(1, "a"), e.write_string(1, "b")])
    assert engine(**NO_CAPS, visitor=doc).feed(wire) is Status.COMPLETE
    assert doc.tags == ["", "b"]


@pytest.mark.parametrize("engine", ENGINES)
def test_framed_elements_are_reserved_and_routed(engine):
    enc = Encoder()
    enc.write_sequence_begin_lazy(ROWS)
    for index, value in ((0, 11), (2, 33)):
        enc.write_sequence_begin_lazy(index)
        enc.write_unsigned(0, value)
        enc.write_sequence_end_keep()
    enc.write_sequence_end_keep()
    enc.write_unsigned(9, 99)
    enc.flush()
    doc = Doc(rcap=8)
    assert engine(**NO_CAPS, visitor=doc).feed(enc.getvalue()) is Status.COMPLETE
    assert [r.fields for r in doc.rows] == [{0: 11}, {}, {0: 33}]
    assert doc.after == [(9, 99)], "the parent resumes after the scope closes"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_wrapper_row_matrix_through_a_decode(engine):
    enc = Encoder()
    enc.write_sequence_begin_lazy(MATRIX)
    enc.write_sequence_begin_lazy(0)
    enc.write_string(1, "x")
    enc.write_sequence_end_keep()
    enc.write_sequence_begin_lazy(2)
    enc.write_string(0, "y")
    enc.write_sequence_end_keep()
    enc.write_sequence_begin_lazy(0)  # §7.4: a repeated row id replaces the row
    enc.write_string(0, "z")
    enc.write_sequence_end_keep()
    enc.write_sequence_end_keep()
    enc.flush()
    doc = Doc(rcap=8)
    assert engine(**NO_CAPS, visitor=doc).feed(enc.getvalue()) is Status.COMPLETE
    assert doc.matrix == [["z"], [], ["y"]]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("field", [TAGS, ROWS])
def test_past_the_schema_count_a_decode_is_invalid(engine, field):
    if field == TAGS:
        wire = _wrap(TAGS, lambda e: [e.write_string(0, "a"), e.write_string(TAGS_CAP, "x")])
    else:

        def write(e):
            e.write_sequence_begin_lazy(0)
            e.write_sequence_end_keep()
            e.write_sequence_begin_lazy(ROWS_CAP)
            e.write_unsigned(0, 1)
            e.write_sequence_end_keep()

        wire = _wrap(ROWS, write)
    doc = Doc(rcap=64)  # a receiver cap well above the count: it must not apply
    dec = engine(**NO_CAPS, visitor=doc)
    # A schema-bound violation is the INVALID outcome, not an exception: the
    # decoder answers §7.1 in the status, and reserves the error channel for the
    # policy rejection a receiver limit is (§6.3).
    assert dec.feed(wire) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert not isinstance(dec.error, SofaLimitError)
    assert len(doc.tags if field == TAGS else doc.rows) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_an_over_count_leaf_is_refused_at_its_header(engine):
    """The index is judged from ``on_field``, so a message that ends right after
    the over-count element's header is INVALID, not INCOMPLETE (§5.2)."""
    wire = _wrap(TAGS, lambda e: e.write_string(TAGS_CAP, "xxxxxxxx"))
    dec = engine(**NO_CAPS, visitor=Doc(rcap=8))
    assert dec.feed(wire[:5]) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("field", [DYN, MATRIX])
def test_past_the_receiver_cap_a_decode_is_limit_exceeded(engine, field):
    if field == DYN:
        wire = _wrap(DYN, lambda e: [e.write_string(0, "a"), e.write_string(4, "x")])
    else:

        def write(e):
            e.write_sequence_begin_lazy(4)
            e.write_sequence_end_keep()

        wire = _wrap(MATRIX, write)
    doc = Doc(rcap=4)
    with pytest.raises(SofaLimitError):
        engine(**NO_CAPS, visitor=doc).feed(wire)
    assert len(doc.dyn) <= 1 and doc.matrix == []


@pytest.mark.parametrize("engine", ENGINES)
def test_an_unstated_receiver_cap_fails_the_decode_as_an_argument_error(engine):
    wire = _wrap(DYN, lambda e: e.write_string(0, "a"))
    with pytest.raises(SofaArgumentError):
        engine(**NO_CAPS, visitor=Doc(rcap=-1)).feed(wire)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "field,write,payload",
    [
        (TAGS, "write_string", "y" * (ELEM_MAXLEN + 1)),
        (BLOBS, "write_bytes", b"y" * (ELEM_MAXLEN + 1)),
    ],
    ids=["string", "blob"],
)
def test_an_element_payload_over_maxlen_is_invalid(engine, field, write, payload):
    """An element's ``maxlen`` is not the helper's argument: the visitor declares
    it from ``on_schema_bound`` and the codec judges it at the length word."""
    wire = _wrap(field, lambda e: getattr(e, write)(0, payload))
    dec = engine(**NO_CAPS, visitor=Doc(rcap=8))
    assert dec.feed(wire) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert not isinstance(dec.error, SofaLimitError)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_string_elements_maxlen_is_a_byte_length(engine):
    """``maxlen`` bounds the wire byte length (MESSAGE_SPEC §1). Eight code
    points of which one is two UTF-8 bytes is nine bytes: over the bound."""
    wire = _wrap(TAGS, lambda e: e.write_string(0, "é" + "y" * 7))
    dec = engine(**NO_CAPS, visitor=Doc(rcap=8))
    assert dec.feed(wire) is Status.INVALID
    ok = _wrap(TAGS, lambda e: e.write_string(0, "é" + "y" * 6))
    doc = Doc(rcap=8)
    assert engine(**NO_CAPS, visitor=doc).feed(ok) is Status.COMPLETE
    assert doc.tags == ["é" + "y" * 6]


@pytest.mark.parametrize("engine", ENGINES)
def test_an_over_maxlen_element_is_refused_at_the_length_word(engine):
    """A payload the message truncates behind is still INVALID (§7.1, §5.2)."""
    wire = _wrap(TAGS, lambda e: e.write_string(0, "y" * 40))
    dec = engine(**NO_CAPS, visitor=Doc(rcap=8))
    assert dec.feed(wire[:6]) is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_mistyped_element_past_the_count_is_skipped_not_refused(engine):
    """§7.3 runs before the bound: a ``blob`` in a string array is skipped like
    an unknown id, so its over-count index is never judged."""
    wire = _wrap(TAGS, lambda e: [e.write_string(0, "a"), e.write_bytes(TAGS_CAP + 5, b"z")])
    doc = Doc(rcap=8)
    assert engine(**NO_CAPS, visitor=doc).feed(wire) is Status.COMPLETE
    assert doc.tags == ["a"]
