"""One decode surface, and the proof that it stays one (CORELIB_PLAN §5.3.1).

§5.3.1: "A corelib exposes exactly one decode surface: the visitor … A port
**MUST NOT** offer any second decode surface: no pull-parser, no iterator or
``next()``-style API, no cursor, no convenience wrapper that decodes by another
route", because "**every additional surface is a second implementation of every
rule in this document**".

This port had two. A :class:`sofab.Binding` used to be a decode path of its own
inside each engine, and the two had already drifted apart exactly as the clause
predicts: with ``max_dyn_string_len=4``, a 10-byte string at a field the schema
bounds at ``maxlen=10`` decoded COMPLETE through the binding and raised
:class:`sofab.SofaLimitError` through the visitor, because §6.2.1's
schema-bound exemption had been implemented on one route only.

A table is now reached **through** the one surface, never beside it: a handler
declares its slots once from :meth:`sofab.Visitor.destinations`, and
``Decoder(binding=...)`` is the constructor shorthand for a handler that
declares exactly that. It is the same bargain :meth:`sofab.Visitor.on_string_begin`
and :meth:`sofab.Visitor.on_array_begin` already strike per field -- name a
destination and the codec writes there instead of calling back -- made once for
the whole message.

What makes that one surface is not where the value lands but where the *rules*
live. There is exactly one implementation of each: the schema bound is settled
in ``Decoder._settle_bound``, reached from a table entry and from
:meth:`sofab.Visitor.on_schema_bound` alike, and the receiver cap, the §7.3 tag
test, the UTF-8 check, the declared element width and the resume transaction are
not duplicated at all -- a mapped field and an unmapped one run the same code
until the assignment itself. So the two cannot disagree: there is one of them.

These tests hold that down from both ends -- structurally (no engine has a
second implementation of the rules, and no adapter is left to grow one) and
behaviourally (a table and a hand-written visitor that declare the same thing
reach the same outcome, over a matrix of messages and receiver limits, on both
engines).
"""

from __future__ import annotations

import inspect

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import NO_CAPS, ROOMY_REASSEMBLY, Recorder, Slots, Status, capped, walk

from sofab import (
    ARRAY_MAX,
    FIXLEN_MAX,
    Binding,
    Encoder,
    Field,
    FixlenSubtype,
    SofaDecodeError,
    SofaLimitError,
    Visitor,
    WireType,
)
from sofab import binding as binding_mod
from sofab import decoder as pure_decoder
from sofab.decoder import Decoder as PyDecoder

try:
    from sofab._speedups import Decoder as NativeDecoder
except ImportError:  # pragma: no cover - the pure-only build
    NativeDecoder = None

# --- structural: there is only one path left ---------------------------------


def test_a_table_is_reached_through_the_visitor():
    """The whole of what a table is now: something a handler *declares*, on the
    one surface -- not a second thing a caller can hand the decoder instead."""
    b = Binding().unsigned(1, at=0)
    words = bytearray(8)

    class Declaring(Visitor):
        def destinations(self):
            return (b, words, None)

    # The base class declares none, so an ordinary visitor is unaffected.
    assert Visitor().destinations() is None

    enc = Encoder()
    enc.write_unsigned(1, 42)
    msg = bytes(enc.getvalue())

    for engine in (PyDecoder, NativeDecoder):
        if engine is None:
            continue
        words[:] = bytes(8)
        assert engine(**NO_CAPS, visitor=Declaring()).feed(msg) is Status.COMPLETE
        assert memoryview(words).cast("Q")[0] == 42

        # ``binding=`` is the same declaration, written at the constructor.
        w2 = bytearray(8)
        assert engine(**NO_CAPS, binding=b, words=w2).feed(msg) is Status.COMPLETE
        assert memoryview(w2).cast("Q")[0] == 42


def test_no_engine_has_a_second_implementation_of_the_rules():
    """The rule §5.3.1 is really about: not two surfaces, so not two
    implementations.

    The old second *value path* (``_take_bound``) is gone, and so is the adapter
    that replaced it (``sofab.binding.handler`` / ``_BoundVisitor``) -- either
    one is a place a rule could be written down twice. What is left is a single
    schema-bound site both routes reach.
    """
    assert not hasattr(pure_decoder.Decoder, "_take_bound")
    assert not hasattr(binding_mod, "handler")
    assert not hasattr(binding_mod, "_BoundVisitor")
    # The one site, and the hook half that feeds it rather than re-deciding.
    assert hasattr(pure_decoder.Decoder, "_settle_bound")
    src = inspect.getsource(pure_decoder.Decoder._schema_bound)
    assert "_settle_bound" in src
    assert "SofaDecodeError" not in src, "the hook half must not re-decide the bound"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_binding_decodes_through_the_visitor_it_was_given(engine):
    """A field the table does not name reaches the fallback visitor, which is
    only possible because both travel the same path."""
    enc = Encoder()
    enc.write_unsigned(1, 7)
    enc.write_unsigned(2, 9)
    b = Binding().unsigned(1, at=0)
    rec = Recorder()
    words = bytearray(8)
    dec = engine(**NO_CAPS, binding=b, words=words, visitor=rec)
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    assert Slots(words, []).u[0] == 7
    assert rec.events == [("u", 2, 9)]


# --- the divergence the clause exists to prevent ------------------------------

BOUNDED = "x" * 10


def _bounded_string_message() -> bytes:
    enc = Encoder()
    enc.write_string(1, BOUNDED)
    return enc.getvalue()


class _DeclaringVisitor(Recorder):
    """A hand-written visitor that declares the same schema bound a
    ``Binding().string(1, at=0, maxlen=…)`` entry does."""

    def __init__(self, declared: int) -> None:
        super().__init__()
        self.declared = declared
        self.asked: list[tuple] = []

    def on_schema_bound(self, field_id: int, n: int, wtype, subtype) -> int:
        self.asked.append((field_id, n))
        return self.declared if field_id == 1 else -1


@pytest.mark.parametrize("engine", ENGINES)
def test_the_audited_divergence_is_gone(engine):
    """The exact case AUDIT_v2 measured: same message, same limits, and the two
    surfaces now answer alike when they declare alike."""
    data = _bounded_string_message()

    # Declared by the table.
    b = Binding().string(1, at=0, maxlen=10)
    words = bytearray(8)
    objs: list = [None]
    dec = engine(reassembly=ROOMY_REASSEMBLY, max_dyn_array_count=ARRAY_MAX, max_dyn_blob_len=FIXLEN_MAX, binding=b, words=words, objects=objs, max_dyn_string_len=4)
    assert dec.feed(data) is Status.COMPLETE
    assert objs[0] == BOUNDED

    # Declared by a hand-written visitor. Same answer.
    rec = _DeclaringVisitor(10)
    status, _rec, _dec = walk(engine, data, recorder=rec, max_dyn_string_len=4)
    assert status is Status.COMPLETE
    assert rec.events == [("str", 1, BOUNDED)]
    assert rec.asked == [(1, 10)]     # the id, and the length the WIRE announced


@pytest.mark.parametrize("engine", ENGINES)
def test_declaring_nothing_is_capped_on_either_surface(engine):
    """And the other way round: a table that declares no bound is capped
    exactly as a visitor that declares none is."""
    data = _bounded_string_message()

    b = Binding().string(1, at=0)  # no maxlen: the schema leaves it open
    words = bytearray(8)
    with pytest.raises(SofaLimitError) as bound_exc:
        engine(reassembly=ROOMY_REASSEMBLY, max_dyn_array_count=ARRAY_MAX, max_dyn_blob_len=FIXLEN_MAX, binding=b, words=words, objects=[None], max_dyn_string_len=4).feed(data)

    with pytest.raises(SofaLimitError) as visitor_exc:
        walk(engine, data, max_dyn_string_len=4)

    assert str(bound_exc.value) == str(visitor_exc.value)


# --- the equivalence, over a matrix ------------------------------------------


class _Mirror(Visitor):
    """A hand-written visitor that mirrors a ``Binding`` field for field: the
    same destinations, the same declared bounds, written out by hand.

    Any rule implemented for one and not the other shows up as a disagreement
    between this and the table it mirrors.
    """

    def __init__(self, table: dict[int, tuple], words: bytearray, objects: list) -> None:
        self.table = table
        self.slots = Slots(words, objects)
        self.objects = objects
        self._e: tuple | None = None

    def on_field(self, field: Field) -> bool | None:
        e = self.table.get(field.id)
        if e is None or e[0] != field.type or (e[1] is not None and e[1] != field.subtype):
            self._e = None
            return False
        self._e = e
        return None

    def on_schema_bound(self, field_id: int, n: int, wtype, subtype) -> int:
        return -1 if self._e is None else self._e[3]

    def on_array_begin(self, field_id, wtype, count):
        e = self._e
        assert e is not None
        view = self.slots.q if wtype is WireType.ARRAY_SIGNED else self.slots.u
        return (view[e[2] : e[2] + e[3]], None, None)

    def on_float_array_begin(self, field_id, subtype, count):
        e = self._e
        assert e is not None
        return self.slots.d[e[2] : e[2] + e[3]]

    def on_unsigned(self, field_id, value):
        self.slots.u[self._e[2]] = value          # type: ignore[index]

    def on_signed(self, field_id, value):
        self.slots.q[self._e[2]] = value          # type: ignore[index]

    def on_float32(self, field_id, value):
        self.slots.d[self._e[2]] = value          # type: ignore[index]

    def on_float64(self, field_id, value):
        self.slots.d[self._e[2]] = value          # type: ignore[index]

    def on_string(self, field_id, value):
        self.objects[self._e[2]] = value          # type: ignore[index]

    def on_bytes(self, field_id, value):
        self.objects[self._e[2]] = value          # type: ignore[index]


def _table(binding: Binding) -> dict[int, tuple]:
    """``{id: (wire type, subtype, at, declared)}`` — what a hand-written
    visitor needs to mirror ``binding``."""
    return {
        e.field_id: (e.wt, e.st, e.at, e.declared) for e in binding.entries
    }


def _outcome(fn):
    """``(status, exception type, message)`` for one decode, so two decodes can
    be compared whatever they did."""
    try:
        return (fn(), None, None)
    except Exception as exc:  # noqa: BLE001 - comparing the verdict is the point
        return (None, type(exc), str(exc))


BIG_STR = "s" * 40
BIG_BLOB = b"b" * 40


def _messages() -> list[tuple[str, bytes, Binding]]:
    out = []

    enc = Encoder()
    enc.write_string(1, BIG_STR)
    out.append(("string, declared", enc.getvalue(), Binding().string(1, at=0, maxlen=64)))

    enc = Encoder()
    enc.write_string(1, BIG_STR)
    out.append(("string, undeclared", enc.getvalue(), Binding().string(1, at=0)))

    enc = Encoder()
    enc.write_string(1, BIG_STR)
    out.append(("string, over its bound", enc.getvalue(), Binding().string(1, at=0, maxlen=8)))

    enc = Encoder()
    enc.write_bytes(1, BIG_BLOB)
    out.append(("blob, declared", enc.getvalue(), Binding().bytes(1, at=0, maxlen=64)))

    enc = Encoder()
    enc.write_bytes(1, BIG_BLOB)
    out.append(("blob, undeclared", enc.getvalue(), Binding().bytes(1, at=0)))

    enc = Encoder()
    enc.write_unsigned_array(1, list(range(20)))
    out.append(("u array", enc.getvalue(), Binding().unsigned_array(1, at=0, cap=32)))

    enc = Encoder()
    enc.write_unsigned_array(1, list(range(20)))
    out.append(("u array over cap", enc.getvalue(), Binding().unsigned_array(1, at=0, cap=8)))

    enc = Encoder()
    enc.write_signed_array(1, [-3] * 20)
    out.append(("i array", enc.getvalue(), Binding().signed_array(1, at=0, cap=32)))

    enc = Encoder()
    enc.write_float32_array(1, [0.5] * 20)
    out.append(("fp32 array", enc.getvalue(), Binding().float32_array(1, at=0, cap=32)))

    enc = Encoder()
    enc.write_float64_array(1, [0.5] * 20)
    out.append(("fp64 array", enc.getvalue(), Binding().float64_array(1, at=0, cap=32)))

    enc = Encoder()
    enc.write_unsigned(1, 12345)
    enc.write_signed(2, -9)
    enc.write_float32(3, 0.5)
    enc.write_float64(4, 0.25)
    b = (Binding().unsigned(1, at=0).signed(2, at=1)
         .float32(3, at=2).float64(4, at=3))
    out.append(("scalars", enc.getvalue(), b))

    # §7.3: the table declares a type the wire contradicts.
    enc = Encoder()
    enc.write_string(1, BIG_STR)
    out.append(("mistyped", enc.getvalue(), Binding().unsigned(1, at=0)))

    return out


LIMITS = [
    {},
    {"max_dyn_string_len": 8},
    {"max_dyn_blob_len": 8},
    {"max_dyn_array_count": 8},
    {"max_dyn_string_len": 8, "max_dyn_blob_len": 8, "max_dyn_array_count": 8},
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("limits", LIMITS, ids=lambda d: ",".join(d) or "no limits")
@pytest.mark.parametrize("name,data,binding", _messages(), ids=lambda v: v if isinstance(v, str) else "")
@pytest.mark.parametrize("chunk", [None, 3])
def test_a_table_and_a_hand_written_visitor_agree(engine, limits, name, data, binding, chunk):
    """The property §5.3.1 buys: one surface, so one outcome. Decode the same
    bytes under the same receiver limits through a ``Binding`` and through a
    visitor declaring exactly what the table declares, and compare the verdict
    *and* every slot."""
    nwords = max(binding.tree_words_required, 1)
    nobjs = max(binding.tree_objects_required, 1)

    def run(handler_kw, words, objects):
        dec = engine(words=words, objects=objects, reassembly=len(data) + 16,
                     **handler_kw, **limits)
        status = Status.COMPLETE
        if chunk is None:
            status = dec.feed(data)
        else:
            for off in range(0, len(data), chunk):
                status = dec.feed(data[off : off + chunk])
        return status if status is not Status.INVALID else dec.error

    bw, bo = bytearray(nwords * 8), [None] * nobjs
    bound_outcome = _outcome(lambda: run({"binding": binding}, bw, bo))

    vw, vo = bytearray(nwords * 8), [None] * nobjs
    mirror = _Mirror(_table(binding), vw, vo)
    visitor_outcome = _outcome(lambda: run({"visitor": mirror}, vw, vo))

    assert repr(bound_outcome) == repr(visitor_outcome), name
    assert bytes(bw) == bytes(vw), name
    assert bo == vo, name


# --- on_schema_bound itself ---------------------------------------------------


class _Bound(Recorder):
    def __init__(self, declared, ids=(1,)):
        super().__init__()
        self.declared = declared
        self.ids = ids

    def on_schema_bound(self, field_id, n, wtype, subtype):
        return self.declared if field_id in self.ids else -1


@pytest.mark.parametrize("engine", ENGINES)
def test_over_the_declared_bound_is_invalid_not_a_limit(engine):
    """§6.2.1: a schema bound is "a statement about *validity*", so exceeding it
    is INVALID — never the policy rejection a receiver cap gives."""
    enc = Encoder()
    enc.write_string(1, BIG_STR)
    status, _rec, dec = walk(engine, enc.getvalue(), recorder=_Bound(8))
    assert status is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert "exceeds the 8 the schema declares" in str(dec.error)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "write,what",
    [
        (lambda e: e.write_unsigned_array(1, list(range(20))), "array count"),
        (lambda e: e.write_signed_array(1, [1] * 20), "array count"),
        (lambda e: e.write_float32_array(1, [1.0] * 20), "array count"),
        (lambda e: e.write_float64_array(1, [1.0] * 20), "array count"),
        (lambda e: e.write_bytes(1, BIG_BLOB), "fixlen length"),
    ],
)
def test_every_bounded_kind_reaches_the_verdict(engine, write, what):
    enc = Encoder()
    write(enc)
    status, _rec, dec = walk(engine, enc.getvalue(), recorder=_Bound(8))
    assert status is Status.INVALID
    assert str(dec.error).startswith(f"{what} 20" if what == "array count" else what)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_declared_bound_takes_the_receiver_cap_off(engine):
    """§6.2.1: the caps "**MUST NOT** be applied to a field the schema already
    bounds"."""
    enc = Encoder()
    enc.write_bytes(1, BIG_BLOB)
    status, rec, _dec = walk(
        engine, enc.getvalue(), recorder=_Bound(64), max_dyn_blob_len=8
    )
    assert status is Status.COMPLETE
    assert rec.events == [("blob", 1, BIG_BLOB)]


@pytest.mark.parametrize("engine", ENGINES)
def test_declaring_nothing_leaves_the_cap_in_force(engine):
    enc = Encoder()
    enc.write_bytes(1, BIG_BLOB)
    with pytest.raises(SofaLimitError):
        walk(engine, enc.getvalue(), recorder=_Bound(-1), max_dyn_blob_len=8)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_scalar_is_never_asked(engine):
    """A scalar carries neither a count nor a length, so there is nothing to
    bound and the hook is not asked for one."""

    class Asked(Recorder):
        def __init__(self):
            super().__init__()
            self.asked = []

        def on_schema_bound(self, field_id, n, wtype, subtype):
            self.asked.append((field_id, n))
            return -1

    enc = Encoder()
    enc.write_unsigned(1, 5)
    enc.write_signed(2, -5)
    enc.write_float32(3, 1.0)
    enc.write_float64(4, 1.0)
    enc.write_string(5, "x")
    rec = Asked()
    status, _rec, _dec = walk(engine, enc.getvalue(), recorder=rec)
    assert status is Status.COMPLETE
    assert rec.asked == [(5, 1)]      # only the string, with its byte length


@pytest.mark.parametrize("engine", ENGINES)
def test_a_skipped_field_is_never_asked(engine):
    """§6.7.2: a skipped field is walked, not read — so nothing is bound-checked
    for it and nothing is capped for it."""

    class Decline(Recorder):
        def __init__(self):
            super().__init__()
            self.asked = []

        def on_field(self, field):
            super().on_field(field)
            return False

        def on_schema_bound(self, field_id, n, wtype, subtype):
            self.asked.append(field_id)
            return 1

    enc = Encoder()
    enc.write_string(1, BIG_STR)
    rec = Decline()
    status, _rec, _dec = walk(engine, enc.getvalue(), recorder=rec, max_dyn_string_len=2)
    assert status is Status.COMPLETE
    assert rec.asked == []


# --- the one combination a single surface cannot serve -----------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_the_fp32_channel_is_per_field_not_per_decoder(engine):
    """§6.5's raw-bits channel is chosen by overriding the hook, and a table
    entry is a declaration about *its own* field.

    So the combination is well defined rather than ambiguous: a field the table
    names gets the widened double in its slot -- that is what the table asked
    for -- and a field it does not name reaches the fallback, which asked for the
    bits. Different fields, different declarations, and no field is ever offered
    both.
    """
    enc = Encoder()
    enc.write_float32(1, 1.5)     # bound: widened into the slot
    enc.write_float32(2, 2.5)     # unbound: raw bits to the fallback
    msg = bytes(enc.getvalue())

    class Bits(Visitor):
        def __init__(self):
            self.bits = []

        def on_float32_bits(self, field_id, bits):
            self.bits.append((field_id, bits))

    sink = Bits()
    words = bytearray(8)
    b = Binding().float32(1, at=0)
    assert engine(**NO_CAPS, binding=b, words=words, visitor=sink).feed(msg) is Status.COMPLETE
    assert memoryview(words).cast("d")[0] == 1.5
    assert sink.bits == [(2, 0x40200000)]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_fp32_array_channel_is_per_field_too(engine):
    enc = Encoder()
    enc.write_float32_array(1, [1.5, 2.5])
    enc.write_float32_array(2, [3.5])
    msg = bytes(enc.getvalue())

    class ArrayBits(Visitor):
        def __init__(self):
            self.seen = []

        def on_float32_array_bits(self, field_id, count, payload):
            self.seen.append((field_id, count, bytes(payload)))

    sink = ArrayBits()
    words = bytearray(16)
    b = Binding().float32_array(1, at=0, cap=2)
    assert engine(**NO_CAPS, binding=b, words=words, visitor=sink).feed(msg) is Status.COMPLETE
    assert list(memoryview(words).cast("d")) == [1.5, 2.5]
    assert [(fid, n) for fid, n, _ in sink.seen] == [(2, 1)]


def test_both_engines_settle_a_declared_bound_the_same_way():
    """A table cannot mean one thing in one engine and something else in the
    other: the same declaration, the same message and the same receiver limit
    reach the same verdict and the same slots in both."""
    native = pytest.importorskip("sofab._speedups")
    enc = Encoder()
    enc.write_string(1, "x" * 10)
    msg = bytes(enc.getvalue())

    out = []
    for engine in (pure_decoder.Decoder, native.Decoder):
        b = Binding().string(1, at=0, maxlen=10)
        objects: list = [None]
        dec = engine(reassembly=ROOMY_REASSEMBLY, max_dyn_array_count=ARRAY_MAX, max_dyn_blob_len=FIXLEN_MAX, 
            binding=b, words=bytearray(8), objects=objects, max_dyn_string_len=4
        )
        out.append((dec.feed(msg), objects[0]))
    assert out[0] == out[1] == (Status.COMPLETE, "x" * 10)


# --- a table and a visitor, together -----------------------------------------
#
# The combination is where "one surface" has to be shown twice over: the table
# takes the fields it names, and every hook the fallback would have been offered
# alone is still offered to it, through the same one path.


class _Every(Recorder):
    """A fallback that overrides every hook a decode can reach, and records
    which ones fired."""

    def __init__(self) -> None:
        super().__init__()
        self.begins: list[tuple] = []
        self.bounds: list[int] = []
        self.seq_answer: object = None

    def on_schema_bound(self, field_id, n, wtype, subtype):
        self.bounds.append(field_id)
        return -1

    def on_sequence_begin(self, field_id):
        super().on_sequence_begin(field_id)
        return self.seq_answer

    def on_array_begin(self, field_id, wtype, count):
        self.begins.append(("array", field_id, count))
        return None

    def on_float_array_begin(self, field_id, subtype, count):
        self.begins.append(("farray", field_id, count))
        return None

    def on_blob_begin(self, field_id, size):
        self.begins.append(("blob", field_id, size))
        return None

    def on_string_begin(self, field_id, size):
        self.begins.append(("string", field_id, size))
        return None


@pytest.mark.parametrize("engine", ENGINES)
def test_the_fallback_still_gets_every_hook(engine):
    enc = Encoder()
    enc.write_unsigned(1, 1)          # bound
    enc.write_unsigned(2, 2)
    enc.write_signed(3, -3)
    enc.write_float32(4, 0.5)
    enc.write_float64(5, 0.25)
    enc.write_string(6, "six")
    enc.write_bytes(7, b"seven")
    enc.write_unsigned_array(8, [1, 2])
    enc.write_signed_array(9, [-1, -2])
    enc.write_float32_array(10, [1.5])
    enc.write_float64_array(11, [2.5])

    b = Binding().unsigned(1, at=0)
    rec = _Every()
    words = bytearray(8)
    dec = engine(**NO_CAPS, binding=b, words=words, visitor=rec)
    assert dec.feed(enc.getvalue()) is Status.COMPLETE
    assert Slots(words, []).u[0] == 1
    assert rec.events == [
        ("u", 2, 2), ("s", 3, -3), ("f32", 4, 0.5), ("f64", 5, 0.25),
        ("str", 6, "six"), ("blob", 7, b"seven"),
        ("ua", 8, (1, 2)), ("sa", 9, (-1, -2)),
        ("f32a", 10, (1.5,)), ("f64a", 11, (2.5,)),
    ]
    assert rec.begins == [
        ("string", 6, 3), ("blob", 7, 5), ("array", 8, 2), ("array", 9, 2),
        ("farray", 10, 1), ("farray", 11, 1),
    ]
    # Asked for every aggregate the fallback took, and for none of the scalars
    # or the bound field.
    assert rec.bounds == [6, 7, 8, 9, 10, 11]


def _nested() -> bytes:
    enc = Encoder()
    enc.write_sequence_begin_lazy(1)
    enc.write_unsigned(1, 11)
    enc.write_sequence_end()
    enc.write_sequence_begin_lazy(2)
    enc.write_unsigned(1, 22)
    enc.write_sequence_end()
    return enc.getvalue()


@pytest.mark.parametrize("engine", ENGINES)
def test_an_unbound_sequence_with_no_fallback_is_skipped(engine):
    inner = Binding().unsigned(1, at=0, count_at=1)
    b = Binding().sequence(1, inner, count_at=2)
    words = bytearray(3 * 8)
    dec = engine(**NO_CAPS, binding=b, words=words)
    assert dec.feed(_nested()) is Status.COMPLETE
    s = Slots(words, [])
    assert (s.u[0], s.u[1], s.u[2]) == (11, 1, 1)   # id 2's sub-tree skipped whole


@pytest.mark.parametrize("engine", ENGINES)
def test_an_unbound_sequence_goes_to_the_fallback_flat(engine):
    inner = Binding().unsigned(1, at=0, count_at=1)
    b = Binding().sequence(1, inner, count_at=2)
    rec = _Every()
    words = bytearray(3 * 8)
    dec = engine(**NO_CAPS, binding=b, words=words, visitor=rec)
    assert dec.feed(_nested()) is Status.COMPLETE
    assert Slots(words, []).u[0] == 11
    # The fallback heard its own scope open and close, and every close: a bound
    # sequence is descended into without asking it, but it is still told.
    assert rec.events == [("seq}",), ("seq{", 2), ("u", 1, 22), ("seq}",)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_fallback_can_decline_an_unbound_sequence(engine):
    b = Binding().unsigned(9, at=0)
    rec = _Every()
    rec.seq_answer = False
    words = bytearray(8)
    dec = engine(**NO_CAPS, binding=b, words=words, visitor=rec)
    assert dec.feed(_nested()) is Status.COMPLETE
    # Both sub-trees were offered and both declined, so nothing inside either
    # was decoded and no on_sequence_end fired for them.
    assert rec.events == [("seq{", 1), ("seq{", 2)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_fallback_can_name_a_child_for_an_unbound_sequence(engine):
    child = Recorder()
    b = Binding().unsigned(9, at=0)
    rec = _Every()
    rec.seq_answer = child
    words = bytearray(8)
    dec = engine(**NO_CAPS, binding=b, words=words, visitor=rec)
    assert dec.feed(_nested()) is Status.COMPLETE
    assert child.events == [("u", 1, 11), ("seq}",), ("u", 1, 22), ("seq}",)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_mistyped_sequence_id_is_the_fallbacks(engine):
    """§7.3 again, at a sequence: the table declares a scalar where the wire
    opened a sub-tree, so the row does not match and the fallback is asked."""
    b = Binding().unsigned(1, at=0)
    rec = _Every()
    words = bytearray(8)
    dec = engine(**NO_CAPS, binding=b, words=words, visitor=rec)
    assert dec.feed(_nested()) is Status.COMPLETE
    assert rec.events[0] == ("seq{", 1)


def test_the_base_visitor_declares_no_bound():
    """The default: a handler that says nothing leaves every field to the
    receiver caps."""
    assert Visitor().on_schema_bound(1, 0, WireType.FIXLEN, FixlenSubtype.STRING) == -1


# --- the map's own state, across a reset and a scope it does not name ---------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_reset_inside_a_mapped_sequence_rewinds_the_map(engine):
    """A decode abandoned inside a nested scope leaves the map pointing at the
    child. ``reset`` rewinds it to the one the caller declared -- the slots are
    construction-time state (§6.6), so the index moves, not the storage."""
    child = Binding().unsigned(1, at=1)
    b = Binding().unsigned(1, at=0).sequence(2, child, count_at=2)

    enc = Encoder()
    enc.write_unsigned(1, 7)
    enc.write_sequence_begin_lazy(2)
    enc.write_unsigned(1, 9)
    enc.write_sequence_end()
    msg = bytes(enc.getvalue())

    words = bytearray(8 * 8)
    dec = engine(**NO_CAPS, binding=b, words=words)
    # Stop inside the sequence: the child map is live and unbalanced.
    assert dec.feed(msg[:-1]) is Status.INCOMPLETE
    u = memoryview(words).cast("Q")
    assert (u[0], u[1], u[2]) == (7, 9, 1)

    dec.reset()
    words[:] = bytes(len(words))
    # The next message decodes against the ROOT map again, not the child's.
    assert dec.feed(msg) is Status.COMPLETE
    assert (u[0], u[1], u[2]) == (7, 9, 1)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_map_with_no_fallback_skips_a_sequence_it_does_not_name(engine):
    """§4.9's fresh id scope with nobody to hand it to: the whole sub-tree is
    consumed and discarded, and the fields after it still land."""
    b = Binding().unsigned(1, at=0).unsigned(9, at=1)

    enc = Encoder()
    enc.write_unsigned(1, 7)
    enc.write_sequence_begin_lazy(5)      # unmapped, no fallback
    enc.write_unsigned(1, 1234)           # must NOT reach slot 0
    enc.write_sequence_end()
    enc.write_unsigned(9, 11)
    msg = bytes(enc.getvalue())

    words = bytearray(4 * 8)
    dec = engine(**NO_CAPS, binding=b, words=words)
    assert dec.feed(msg) is Status.COMPLETE
    u = memoryview(words).cast("Q")
    assert (u[0], u[1]) == (7, 11)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_skipped_unmapped_sequence_resumes_across_a_chunk(engine):
    """The skip is a transaction like every other read (§5.2): running out of
    bytes inside a sub-tree nobody wants is INCOMPLETE, and the retry finishes
    *that* skip rather than restarting the field walk behind it."""
    b = Binding().unsigned(1, at=0).unsigned(9, at=1)

    enc = Encoder()
    enc.write_unsigned(1, 7)
    enc.write_sequence_begin_lazy(5)      # unmapped, no fallback
    enc.write_unsigned(1, 1234)
    enc.write_unsigned(2, 5678)
    enc.write_sequence_end()
    enc.write_unsigned(9, 11)
    msg = bytes(enc.getvalue())

    words = bytearray(4 * 8)
    dec = engine(**capped(reassembly=64), binding=b, words=words)
    u = memoryview(words).cast("Q")
    # One byte at a time, so the skip is suspended at every boundary inside the
    # sub-tree and has to be resumed rather than restarted.
    status = None
    for i in range(len(msg)):
        status = dec.feed(msg[i : i + 1])
    assert status is Status.COMPLETE
    assert (u[0], u[1]) == (7, 11)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_hook_is_told_what_the_wire_announced(engine):
    """The second argument is the count or length the SENDER stated, which is
    what the handler needs to answer without the decoder building an object for
    it: a byte length for a string or blob, an element count for an array."""

    class Seen(Recorder):
        def __init__(self):
            super().__init__()
            self.seen = []

        def on_schema_bound(self, field_id, n, wtype, subtype):
            self.seen.append((field_id, n))
            return -1

    enc = Encoder()
    enc.write_string(1, "abcde")            # 5 bytes
    enc.write_bytes(2, b"xyz")              # 3 bytes
    enc.write_unsigned_array(3, [1, 2, 3, 4])   # 4 elements
    enc.write_float64_array(4, [1.0, 2.0])      # 2 elements
    enc.write_unsigned(5, 7)                # a scalar: never asked
    rec = Seen()
    status, _rec, _dec = walk(engine, enc.getvalue(), recorder=rec)
    assert status is Status.COMPLETE
    assert rec.seen == [(1, 5), (2, 3), (3, 4), (4, 2)]


# --- the hook is told the tag, so §7.3 reaches it too (#133) ------------------


class _TaggedBound(Recorder):
    """A hand-written handler whose schema declares exactly one field: id 1, a
    ``string`` with ``maxlen=32``. It answers the bound only for that tag --
    which is what a table entry has the decoder do on its behalf."""

    def on_schema_bound(self, field_id, n, wtype, subtype):
        if (
            field_id == 1
            and wtype is WireType.FIXLEN
            and subtype is FixlenSubtype.STRING
        ):
            return 32
        return -1


@pytest.mark.parametrize("engine", ENGINES)
def test_a_bounded_id_under_another_tag_is_skipped_not_invalid(engine):
    """§7.3: "a decoder MUST treat the field as it treats an unknown field id"
    -- and "against a schema bound, this clause wins".

    The bug this closes: the hook was told the id and the announced length and
    nothing else, so a handler could not tell its own 32-byte-bounded string
    from a 40-byte blob that happens to reuse id 1. It answered 32 for the blob
    and the decode came back INVALID, where the same declaration written as a
    ``Binding`` entry -- which gets the §7.3 tag test run ahead of it -- came
    back COMPLETE with the slot untouched.
    """
    enc = Encoder()
    enc.write_bytes(1, b"x" * 40)      # a BLOB at the id the schema calls a string
    data = enc.getvalue()

    # Declared by the table: the tag does not match, so the field is skipped.
    b = Binding().string(1, at=0, maxlen=32)
    words, objs = bytearray(b.tree_words_required * 8), [None]
    assert engine(**NO_CAPS, binding=b, words=words, objects=objs).feed(data) is Status.COMPLETE
    assert objs[0] is None

    # Declared by the hook. Same declaration, same answer.
    status, rec, _dec = walk(engine, data, recorder=_TaggedBound())
    assert status is Status.COMPLETE
    assert rec.events == [("blob", 1, b"x" * 40)]


@pytest.mark.parametrize("engine", ENGINES)
def test_the_bound_still_binds_the_field_it_was_declared_for(engine):
    """The other half: answering -1 for a foreign tag must not cost the bound
    on the field the schema really declared."""
    enc = Encoder()
    enc.write_string(1, "y" * 40)     # the declared tag, over the declared bound
    status, _rec, dec = walk(engine, enc.getvalue(), recorder=_TaggedBound())
    assert status is Status.INVALID
    assert isinstance(dec.error, SofaDecodeError)
    assert "exceeds the 32" in str(dec.error)

    enc = Encoder()
    enc.write_string(1, "y" * 10)     # and inside it, unchanged
    status, rec, _dec = walk(engine, enc.getvalue(), recorder=_TaggedBound())
    assert status is Status.COMPLETE
    assert rec.events == [("str", 1, "y" * 10)]


@pytest.mark.parametrize(
    "write, wtype, subtype",
    [
        (lambda e: e.write_string(1, "abc"), WireType.FIXLEN, FixlenSubtype.STRING),
        (lambda e: e.write_bytes(1, b"abc"), WireType.FIXLEN, FixlenSubtype.BLOB),
        (lambda e: e.write_unsigned_array(1, [1, 2]), WireType.ARRAY_UNSIGNED, None),
        (lambda e: e.write_signed_array(1, [-1, 2]), WireType.ARRAY_SIGNED, None),
        (
            lambda e: e.write_float32_array(1, [1.0]),
            WireType.ARRAY_FIXLEN,
            FixlenSubtype.FP32,
        ),
        (
            lambda e: e.write_float64_array(1, [1.0]),
            WireType.ARRAY_FIXLEN,
            FixlenSubtype.FP64,
        ),
    ],
    ids=["string", "blob", "u-array", "s-array", "fp32-array", "fp64-array"],
)
@pytest.mark.parametrize("engine", ENGINES)
def test_every_bounded_kind_reports_its_own_tag(engine, write, wtype, subtype):
    """One hook spans four wire types and three subtypes, so each must arrive
    labelled with the one it actually is -- ``None`` for an integer array, which
    carries no subtype word on the wire (§4.8)."""

    class Seen(Recorder):
        def __init__(self):
            super().__init__()
            self.tags = []

        def on_schema_bound(self, field_id, n, wt, st):
            self.tags.append((wt, st))
            return -1

    enc = Encoder()
    write(enc)
    status, rec, _dec = walk(engine, enc.getvalue(), recorder=Seen())
    assert status is Status.COMPLETE
    assert rec.tags == [(wtype, subtype)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_bounded_kind_that_carries_no_bound_is_still_not_asked(engine):
    """An fp32/fp64 *scalar* is a fixlen field with a fixed width, so there is
    nothing for a schema to bound and the hook is not reached -- unchanged by
    the tag, and the reason ``subtype`` never arrives as FP32/FP64 outside an
    array."""

    class Seen(Recorder):
        def __init__(self):
            super().__init__()
            self.tags = []

        def on_schema_bound(self, field_id, n, wt, st):
            self.tags.append((wt, st))
            return -1

    enc = Encoder()
    enc.write_float32(1, 1.5)
    enc.write_float64(2, 2.5)
    enc.write_unsigned(3, 7)
    status, rec, _dec = walk(engine, enc.getvalue(), recorder=Seen())
    assert status is Status.COMPLETE
    assert rec.tags == []


@pytest.mark.parametrize("engine", ENGINES)
def test_declaring_a_bound_costs_no_object(engine):
    """The point of the signature: the hook is handed the id and the announced
    count/length as plain ``int``s and the tag as the enum members the decoder
    already holds, so a handler that declares schema bounds and overrides
    nothing else never causes a ``Field`` -- or anything else -- to be built."""

    class Bounds(Visitor):
        def __init__(self):
            self.seen = []

        def on_schema_bound(self, field_id, n, wtype, subtype):
            self.seen.append((type(field_id), type(n), wtype, subtype))
            return -1

    enc = Encoder()
    enc.write_string(1, "abc")
    enc.write_unsigned_array(2, [1, 2])
    sink = Bounds()
    assert engine(**NO_CAPS, visitor=sink).feed(bytes(enc.getvalue())) is Status.COMPLETE
    assert sink.seen == [
        (int, int, WireType.FIXLEN, FixlenSubtype.STRING),
        (int, int, WireType.ARRAY_UNSIGNED, None),
    ]
    # Interned members, not values coerced per field: identity, not equality.
    assert sink.seen[0][2] is WireType.FIXLEN
    assert sink.seen[0][3] is FixlenSubtype.STRING


def test_only_on_field_makes_the_decoder_build_one():
    """The mechanism behind the test above, read off the pure engine's own flag:
    ``on_schema_bound`` no longer forces a ``Field`` per field, ``on_field``
    still does."""

    class Bounds(Visitor):
        def on_schema_bound(self, field_id, n, wtype, subtype):
            return -1

    class Fields(Visitor):
        def on_field(self, field):
            return None

    assert pure_decoder.Decoder(**NO_CAPS, visitor=Bounds())._make_field is False
    assert pure_decoder.Decoder(**NO_CAPS, visitor=Fields())._make_field is True
