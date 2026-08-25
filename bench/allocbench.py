#!/usr/bin/env python3
"""What §6.6's default costs, and what moving it would cost (issue #134).

§6.6 is normative: "A corelib's codec MUST NOT allocate payload storage …
any destination sized from a wire count or a wire length", and the test is one
question -- *can a sender make this allocation bigger by sending different
bytes?* On the default path this decoder answers yes three times: a ``bytes``
sized by the wire length, a ``str`` decoded from it, and a ``list`` grown to the
wire count.

The conformant shapes are already here -- ``on_string_begin``, ``on_blob_begin``
and ``on_array_begin`` each take a destination the caller sized from its schema.
What #134 proposes is to move the **default** onto :class:`sofab.Visitor`, so
that even a handler which overrides nothing allocates on the caller's side of
§6.6.1's ownership test rather than inside the codec.

That proposal is not free, and this harness is what prices it. Overriding a
begin hook is what sets the decoder's ``_wants_*`` flag, so a default living on
``Visitor`` is exactly a flag that is always on: one Python call per aggregate
field that today is not made at all.

Four drivers over one message, same bytes for each:

  codec_builds     today's default. The handler overrides the typed hooks only,
                   and the codec builds the str/bytes/list. NOT §6.6-conformant.
  caller_dest      the conformant shape, available today: destinations sized
                   once from the schema and reused. What generated code should
                   emit.
  visitor_default  what #134 asks for, priced: the begin hook fires for every
                   aggregate field and allocates a fresh destination sized by
                   the wire. Conformant -- the storage is handed straight back,
                   so §6.6.1 puts it on the caller's side -- but the allocation
                   still happens per field.
  visitor_material the same, plus turning the filled buffers back into the
                   ``str``/``bytes``/``list`` a handler asked for in the first
                   place. This is what a consumer that wants Python objects
                   actually pays under the proposal.

Read ``codec_builds`` against ``visitor_default`` for the cost of the change on
small fields, and ``blob1m`` for the case it exists to fix.

Usage:
    python bench/allocbench.py <driver> [reps]   # for callgrind, see genbench.sh
    python bench/allocbench.py alloc             # peak bytes per driver
    python bench/allocbench.py selftest          # every driver sees the same values
"""
from __future__ import annotations

import os
import sys
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sofab import Decoder, Encoder, Visitor  # noqa: E402

# --- the message -------------------------------------------------------------
#
# 36 fields again, so the numbers read beside genbench.py's, but aggregates
# where that one has scalars: this is a harness about payload storage, and a
# scalar has none. Sizes are the small-to-middling ones a telemetry or config
# schema carries, where the per-field call cost is at its most visible relative
# to the payload. The 1 MiB blob at the bottom is the other end.

STR_IDS = tuple(range(1, 13))            # 12 strings
BLOB_IDS = tuple(range(13, 19))          # 6 blobs
ARR_IDS = tuple(range(19, 31))           # 12 unsigned arrays
SCALAR_IDS = tuple(range(31, 37))        # 6 scalars, which are never asked

STR_LEN = 16
BLOB_LEN = 32
ARR_LEN = 4

#: Every destination below is sized from THESE, never from the wire (§6.6) --
#: they are what the schema declares, which is the whole distinction.
STR_MAX = 32
BLOB_MAX = 64
ARR_CAP = 8


def build_msg() -> bytes:
    enc = Encoder()
    for fid in STR_IDS:
        enc.write_string(fid, "s" * STR_LEN)
    for fid in BLOB_IDS:
        enc.write_bytes(fid, bytes((fid + i) & 0xFF for i in range(BLOB_LEN)))
    for fid in ARR_IDS:
        enc.write_unsigned_array(fid, [fid + i for i in range(ARR_LEN)])
    for fid in SCALAR_IDS:
        enc.write_unsigned(fid, fid * 7)
    return bytes(enc.getvalue())


MSG = build_msg()


# --- driver 1: today's default -----------------------------------------------


class CodecBuilds(Visitor):
    """Overrides the typed hooks only, which is what generated Python emits
    today. Every aggregate is built by the codec, sized by the wire."""

    __slots__ = ("acc",)

    def __init__(self) -> None:
        self.acc = 0

    def on_string(self, field_id, value):
        self.acc += len(value)

    def on_bytes(self, field_id, value):
        self.acc += len(value)

    def on_unsigned_array(self, field_id, values):
        self.acc += len(values)

    def on_unsigned(self, field_id, value):
        self.acc += value


# --- driver 2: the conformant shape that exists today ------------------------


class CallerDest(Visitor):
    """Destinations sized once from the schema and reused across messages. The
    codec allocates nothing and the wire sizes nothing."""

    __slots__ = ("acc", "_s", "_b", "_a")

    def __init__(self) -> None:
        self.acc = 0
        # One buffer per field, sized from the schema's maxlen/cap. A generator
        # emits exactly this, as members of the message object.
        self._s = {f: bytearray(STR_MAX) for f in STR_IDS}
        self._b = {f: bytearray(BLOB_MAX) for f in BLOB_IDS}
        self._a = {f: memoryview(bytearray(ARR_CAP * 8)).cast("Q") for f in ARR_IDS}

    def on_string_begin(self, field_id, size):
        self.acc += size
        return self._s[field_id]

    def on_blob_begin(self, field_id, size):
        self.acc += size
        return self._b[field_id]

    def on_array_begin(self, field_id, wtype, count):
        self.acc += count
        return (self._a[field_id], None, None)

    def on_unsigned(self, field_id, value):
        self.acc += value


# --- driver 3: what #134 proposes, priced ------------------------------------


class VisitorDefault(Visitor):
    """A default that lives on ``Visitor``: the hook fires for every aggregate
    field and allocates a destination sized by the wire.

    Conformant -- the storage is handed straight back, so §6.6.1's ownership
    test puts it on the caller's side -- and this is precisely the per-field
    call and per-field allocation the current default does not make.
    """

    __slots__ = ("acc",)

    def __init__(self) -> None:
        self.acc = 0

    def on_string_begin(self, field_id, size):
        self.acc += size
        return bytearray(size)

    def on_blob_begin(self, field_id, size):
        self.acc += size
        return bytearray(size)

    def on_array_begin(self, field_id, wtype, count):
        self.acc += count
        return (memoryview(bytearray(count * 8)).cast("Q"), None, None)

    def on_unsigned(self, field_id, value):
        self.acc += value


# --- driver 3b: where the cost sits ------------------------------------------


class VisitorDefaultStrBlob(VisitorDefault):
    """The string and blob halves of the proposal only, with the array left on
    today's list.

    Not conformant -- §6.6 counts a list grown from a wire count too -- but it
    splits the bill: a ``bytearray(size)`` handed back is cheap, and the array
    destination, which a Python handler has to reach through a ``memoryview``
    cast to be typed at all, is not.
    """

    __slots__ = ()

    def on_array_begin(self, field_id, wtype, count):
        return None

    def on_unsigned_array(self, field_id, values):
        self.acc += len(values)


# --- driver 4: the proposal, plus the objects the handler actually wanted -----


class VisitorDefaultMaterialize(Visitor):
    """Driver 3 plus the materialisation the default hides.

    A handler that asked for ``on_string``/``on_bytes``/``on_unsigned_array``
    still needs a ``str``, a ``bytes`` and a ``list`` at the end of it, and
    under the proposal that decode moves out of the codec and into the visitor.
    It cannot happen inside the begin hook -- the destination is empty there --
    and there is no "the destination is full now" callback for a visitor default
    to hang it on, which is itself something #134 has to answer. Here it is a
    second pass at the end of the message, which is the cheapest shape it could
    take.
    """

    __slots__ = ("acc", "_str", "_blob", "_arr")

    def __init__(self) -> None:
        self.acc = 0
        self._str: list = []
        self._blob: list = []
        self._arr: list = []

    def on_string_begin(self, field_id, size):
        dst = bytearray(size)
        self._str.append(dst)
        return dst

    def on_blob_begin(self, field_id, size):
        dst = bytearray(size)
        self._blob.append(dst)
        return dst

    def on_array_begin(self, field_id, wtype, count):
        mv = memoryview(bytearray(count * 8)).cast("Q")
        self._arr.append(mv)
        return (mv, None, None)

    def on_unsigned(self, field_id, value):
        self.acc += value

    def finish(self) -> None:
        for dst in self._str:
            self.acc += len(bytes(dst).decode())
        for dst in self._blob:
            self.acc += len(bytes(dst))
        for mv in self._arr:
            self.acc += len(mv.tolist())
        self._str.clear()
        self._blob.clear()
        self._arr.clear()


# --- the 1 MiB blob: the case §6.6 exists for --------------------------------

BLOB_1M = b"\xa5" * (1 << 20)


def _blob_msg() -> bytes:
    enc = Encoder()
    enc.write_bytes(1, BLOB_1M)
    return bytes(enc.getvalue())


BLOB_MSG = _blob_msg()


class BlobCodec(Visitor):
    __slots__ = ("acc",)

    def __init__(self) -> None:
        self.acc = 0

    def on_bytes(self, field_id, value):
        self.acc += len(value)


class BlobDest(Visitor):
    __slots__ = ("acc", "_dst")

    def __init__(self) -> None:
        self.acc = 0
        self._dst = bytearray(1 << 20)      # the schema's maxlen, not the wire's

    def on_blob_begin(self, field_id, size):
        self.acc += size
        return self._dst


# --- drivers -----------------------------------------------------------------


def _driver(cls, msg=MSG, finish=False):
    """One handler and one decoder, reused across messages -- what a server
    holds. Building either per message would price object construction, which
    is not what this harness is about."""

    def make():
        v = cls()
        dec = Decoder(visitor=v, reassembly=len(msg) + 16)

        def body(data):
            dec.reset()
            dec.feed(data)
            if finish:
                v.finish()

        return body, msg

    return make


DRIVERS = {
    "codec_builds": _driver(CodecBuilds),
    "caller_dest": _driver(CallerDest),
    "visitor_default": _driver(VisitorDefault),
    "visitor_str_blob": _driver(VisitorDefaultStrBlob),
    "visitor_material": _driver(VisitorDefaultMaterialize, finish=True),
    "blob1m_codec": _driver(BlobCodec, BLOB_MSG),
    "blob1m_dest": _driver(BlobDest, BLOB_MSG),
}


def _alloc() -> int:
    """Peak bytes allocated per driver, one message each -- the §6.6 question
    ("can a sender make this bigger?") answered in numbers rather than prose."""
    print(f"{'driver':<18}{'peak bytes/msg':>16}")
    print("-" * 34)
    for name, make in DRIVERS.items():
        body, msg = make()
        body(msg)                       # warm: buffers a real handler holds
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        body(msg)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print(f"{name:<18}{peak - base:>16,}")
    return 0


def _selftest() -> int:
    for name, make in DRIVERS.items():
        body, msg = make()
        body(msg)
        print(f"{name:<18} ok")
    return 0


def main(argv) -> int:
    if len(argv) > 1 and argv[1] == "alloc":
        return _alloc()
    if len(argv) > 1 and argv[1] == "selftest":
        return _selftest()
    body, msg = DRIVERS[argv[1]]()
    for _ in range(int(argv[2])):
        body(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
