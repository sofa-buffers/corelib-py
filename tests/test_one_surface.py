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

A ``Binding`` is now a :class:`sofab.Visitor` built from the table
(``sofab.binding.handler``) and nothing else, and the declaration the exemption
turns on -- what the *schema* bounds a field at -- is a hook on that one
surface, :meth:`sofab.Visitor.on_schema_bound`. So the two cannot disagree:
there is one of them. These tests hold that down from both ends -- structurally
(no engine has a second value path left) and behaviourally (a table and a
hand-written visitor that declare the same thing reach the same outcome, over a
matrix of messages and receiver limits, on both engines).
"""

from __future__ import annotations

import pytest
from vectors import DECODER_ENGINES as ENGINES
from vectors import Recorder, Slots, Status, walk

from sofab import (
    Binding,
    Encoder,
    Field,
    FixlenSubtype,
    SofaArgumentError,
    SofaDecodeError,
    SofaLimitError,
    Visitor,
    WireType,
)
from sofab import binding as binding_mod
from sofab import decoder as pure_decoder

# --- structural: there is only one path left ---------------------------------


def test_a_binding_is_a_visitor():
    """The whole of what a table is now: a handler on the one surface."""
    b = Binding().unsigned(1, at=0)
    h = binding_mod.handler(b, bytearray(8), None, None)
    assert isinstance(h, Visitor)


def test_no_engine_has_a_second_value_path():
    """The rule §5.3.1 is really about: not two surfaces, so not two
    implementations. The bound-decode methods are gone from the pure decoder,
    and the native decoder no longer takes a compiled table."""
    assert not hasattr(pure_decoder.Decoder, "_take_bound")
    assert not hasattr(Binding, "_compiled")
    b = Binding().unsigned(1, at=0)
    assert not hasattr(b, "_compiled")


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
    dec = engine(binding=b, words=words, visitor=rec)
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

    def on_schema_bound(self, field: Field) -> int:
        self.asked.append((field.id, field.type, field.subtype))
        return self.declared if field.id == 1 else -1


@pytest.mark.parametrize("engine", ENGINES)
def test_the_audited_divergence_is_gone(engine):
    """The exact case AUDIT_v2 measured: same message, same limits, and the two
    surfaces now answer alike when they declare alike."""
    data = _bounded_string_message()

    # Declared by the table.
    b = Binding().string(1, at=0, maxlen=10)
    words = bytearray(8)
    objs: list = [None]
    dec = engine(binding=b, words=words, objects=objs, max_dyn_string_len=4)
    assert dec.feed(data) is Status.COMPLETE
    assert objs[0] == BOUNDED

    # Declared by a hand-written visitor. Same answer.
    rec = _DeclaringVisitor(10)
    status, _rec, _dec = walk(engine, data, recorder=rec, max_dyn_string_len=4)
    assert status is Status.COMPLETE
    assert rec.events == [("str", 1, BOUNDED)]
    assert rec.asked == [(1, WireType.FIXLEN, FixlenSubtype.STRING)]


@pytest.mark.parametrize("engine", ENGINES)
def test_declaring_nothing_is_capped_on_either_surface(engine):
    """And the other way round: a table that declares no bound is capped
    exactly as a visitor that declares none is."""
    data = _bounded_string_message()

    b = Binding().string(1, at=0)  # no maxlen: the schema leaves it open
    words = bytearray(8)
    with pytest.raises(SofaLimitError) as bound_exc:
        engine(binding=b, words=words, objects=[None], max_dyn_string_len=4).feed(data)

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

    def on_schema_bound(self, field: Field) -> int:
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

    def on_schema_bound(self, field):
        return self.declared if field.id in self.ids else -1


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

        def on_schema_bound(self, field):
            self.asked.append(field.id)
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
    assert rec.asked == [5]


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

        def on_schema_bound(self, field):
            self.asked.append(field.id)
            return 1

    enc = Encoder()
    enc.write_string(1, BIG_STR)
    rec = Decline()
    status, _rec, _dec = walk(engine, enc.getvalue(), recorder=rec, max_dyn_string_len=2)
    assert status is Status.COMPLETE
    assert rec.asked == []


# --- the one combination a single surface cannot serve -----------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_a_binding_refuses_a_raw_fp32_fallback(engine):
    """Which fp32 route a handler takes is a property of *the handler* (§6.5's
    channel is chosen by overriding the hook), and one surface means one
    handler. A table delivers the widened double, so it cannot also carry a
    fallback that wanted the bits — refused, rather than silently delivering
    the wrong one."""

    class Bits(Visitor):
        def on_float32_bits(self, field_id, bits):
            pass

    b = Binding().unsigned(1, at=0)
    with pytest.raises(SofaArgumentError, match="on_float32_bits"):
        engine(binding=b, words=bytearray(8), visitor=Bits())


@pytest.mark.parametrize("engine", ENGINES)
def test_a_binding_refuses_a_raw_fp32_array_fallback(engine):
    class ArrayBits(Visitor):
        def on_float32_array_bits(self, field_id, count, payload):
            pass

    b = Binding().unsigned(1, at=0)
    with pytest.raises(SofaArgumentError, match="on_float32_array_bits"):
        engine(binding=b, words=bytearray(8), visitor=ArrayBits())


def test_both_engines_share_the_one_table_compiler():
    """Both engines turn a table into a handler with the same function, so a
    table cannot mean one thing in one engine and something else in the other."""
    native = pytest.importorskip("sofab._speedups")
    assert pure_decoder._binding_handler is binding_mod.handler
    assert native._binding_handler is binding_mod.handler


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

    def on_schema_bound(self, field):
        self.bounds.append(field.id)
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
    dec = engine(binding=b, words=words, visitor=rec)
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
    dec = engine(binding=b, words=words)
    assert dec.feed(_nested()) is Status.COMPLETE
    s = Slots(words, [])
    assert (s.u[0], s.u[1], s.u[2]) == (11, 1, 1)   # id 2's sub-tree skipped whole


@pytest.mark.parametrize("engine", ENGINES)
def test_an_unbound_sequence_goes_to_the_fallback_flat(engine):
    inner = Binding().unsigned(1, at=0, count_at=1)
    b = Binding().sequence(1, inner, count_at=2)
    rec = _Every()
    words = bytearray(3 * 8)
    dec = engine(binding=b, words=words, visitor=rec)
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
    dec = engine(binding=b, words=words, visitor=rec)
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
    dec = engine(binding=b, words=words, visitor=rec)
    assert dec.feed(_nested()) is Status.COMPLETE
    assert child.events == [("u", 1, 11), ("seq}",), ("u", 1, 22), ("seq}",)]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_mistyped_sequence_id_is_the_fallbacks(engine):
    """§7.3 again, at a sequence: the table declares a scalar where the wire
    opened a sub-tree, so the row does not match and the fallback is asked."""
    b = Binding().unsigned(1, at=0)
    rec = _Every()
    words = bytearray(8)
    dec = engine(binding=b, words=words, visitor=rec)
    assert dec.feed(_nested()) is Status.COMPLETE
    assert rec.events[0] == ("seq{", 1)


def test_the_base_visitor_declares_no_bound():
    """The default: a handler that says nothing leaves every field to the
    receiver caps."""
    assert Visitor().on_schema_bound(Field(1, WireType.FIXLEN)) == -1
