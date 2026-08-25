<p align="center"><img src="assets/sofabuffers_logo.png" alt="SofaBuffers" height="140"></p>

# SofaBuffers

<b>Structured Objects For Anyone</b><br>
<i>... so optimized, feels amazing.</i>

[Would you like to know more?](https://github.com/sofa-buffers)

## SofaBuffers Python library

[![CI](https://github.com/sofa-buffers/corelib-py/actions/workflows/ci.yml/badge.svg)](https://github.com/sofa-buffers/corelib-py/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fsofa-buffers%2Fcorelib-py%2Fbadges%2Fcoverage.json)](https://github.com/sofa-buffers/corelib-py/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://sofa-buffers.github.io/corelib-py/)

[GitHub repository](https://github.com/sofa-buffers/corelib-py)

A **streaming**, **dependency-free** implementation of the SofaBuffers (*Sofab*)
serialization format — a compact, TLV-like binary format. It is the runtime
stream core, driven by **generated code**: a schema-driven generator emits one
class per message with the streaming `serialize` / `deserialize` pair and the
one-shot `encode` / `decode` wrappers over it, all of which call the `Encoder` /
`Decoder` primitives here.

The public API is that one pair of classes, backed by two interchangeable
engines selected at import: the hot path (varint / zigzag / buffer management)
ships as an optional compiled **native accelerator** (Cython → C,
`sofab._speedups`), with a **pure-Python fallback** used when it is absent. The
two are byte-for-byte interchangeable, so the library runs anywhere CPython
runs, with or without a C compiler.

### Requirements

Python 3.9 or newer (CPython or PyPy); CI runs 3.9–3.14. The optional native
accelerator additionally needs a C compiler and Cython, both build-time only.

### Dependencies

None at runtime — the pure-Python path uses only the standard library
(`struct`, `io`). The one third-party build dependency is `Cython`
([PEP 517](https://peps.python.org/pep-0517/)), used to compile the accelerator
and never imported at runtime.

### Packaging

Distribution `sofa-buffers-corelib` on PyPI; import package `sofab`. The
namespace is the fixed half (CORELIB_PLAN §6: `sofab`, in every target); the
registry name is derived — the organization slug `sofa-buffers` plus `corelib`,
in PyPI's own convention.

```bash
pip install sofa-buffers-corelib
```

```python
import sofab   # Encoder, Decoder, Visitor, wire-format types and limits
print(sofab.__version__)   # release of this runtime
```

`sofab.__version__` is the *only* place the release version is written down;
`pyproject.toml` declares the distribution version dynamic and reads it there.

### Native accelerator

`Encoder` / `Decoder` / `Field` are re-exported from the compiled
`sofab._speedups` extension when present, and from the pure-Python
`encoder.py` / `decoder.py` otherwise. Both engines import wire constants, enums
and exception classes from the shared `types.py`, so a `SofaArgumentError` is the
*same class* from either, and the two produce byte-for-byte identical output
(`tests/test_native_parity.py`).

The active engine is reported by `sofab.IMPL` (`"native"` or `"python"`):

```python
import sofab
print(sofab.IMPL)        # "native" when the compiled extension is loaded
```

Force the pure-Python path with `SOFAB_PUREPYTHON=1`.

Released wheels ship the accelerator already compiled, so a plain
`pip install sofa-buffers-corelib` gets the native engine without a toolchain
wherever a wheel exists: CPython 3.9–3.14 on Linux (glibc and musl, x86-64 and
ARM64), macOS Intel and Apple Silicon, and Windows x86-64. Anywhere else pip builds the sdist, which
compiles the accelerator if a C compiler is present and installs the pure-Python
engine if not.

### Feature flags

The package always builds the full format (unsigned / signed varints, fp32 /
fp64, strings, blobs, arrays and nested sequences). The one build toggle is
`SOFAB_DISABLE_NATIVE=1`, which builds a native-free (pure-Python) distribution;
it changes only speed, never the wire format or the public API.

## Why this design

| Goal | How |
|------|-----|
| Streaming **out** | `Encoder` writes into a **fixed** buffer and drains it to any binary stream (file, socket, `BytesIO`) as the message is written — never after it — so a message can exceed RAM. |
| Streaming **in** | `Decoder` is a push decoder: `feed(chunk)` takes bytes of any size and hands each field to your handler, never materializing the whole message. Every `feed` returns `COMPLETE` / `INCOMPLETE` / `INVALID` for the bytes so far; a construct split across a boundary is retained and finished by the next chunk. |
| Native speed, zero runtime deps | The hot path ships as an optional Cython accelerator (`sofab._speedups`); when it can't be built it falls back to pure Python. No runtime third-party deps either way. |
| Runs everywhere | With no compiler or wheel, `pip` still installs a working pure-Python build (`py3-none-any`). Native and pure paths are byte-for-byte identical; falling back changes only speed. |
| Sticky errors | `Encoder(sticky=True)` records the first failure and turns later writes into no-ops, so generated `serialize` code can check `enc.error` once. |
| No silent truncation | An integer field accepts what Python accepts wherever an integer is required — anything with `__index__` (`int`, `bool`, `IntEnum`, NumPy integers). A `float` is refused with `SofaArgumentError`, `3.0` included. |
| No unreadable message | The format-wide ceilings (CORELIB_PLAN §6.2) bind the encoder too: a field id above `ID_MAX`, an array count above `ARRAY_MAX`, nesting past `MAX_DEPTH`, and a string/blob payload above `FIXLEN_MAX` (2 GiB − 1) are each refused with `SofaArgumentError` **before** the field header is written. An oversized blob is refused on its length, before it is copied. |
| Floats narrow by IEEE rules | A Python `float` is a C double, so `write_float32` (scalar and array) narrows on the way out: round-to-nearest, and a magnitude past `FLT_MAX` overflows to `±inf` — the same bytes a native-`fp32` corelib writes, and identical in both engines. NaN payloads, signaling ones included, keep their exact bits. |
| Reserve-offset | `Encoder.over_buffer(buf, offset=…)` leaves room at the front of the buffer for a lower-layer protocol header; a sink calling `buffer_set(buf, offset)` re-arms that room for **every** flushed packet. |
| Sparse sequences | `write_sequence_begin_lazy` holds a sequence header back until the sequence receives content, so a sequence-typed field equal to its declared default is **omitted** rather than framed empty (MESSAGE_SPEC §2) — decided in one forward pass, without buffering the sub-message. `write_sequence_end` drops a contentless sequence; `write_sequence_end_keep` forces the frame out where presence itself carries meaning, such as a wrapper-array **element**, which is always framed even when all-default. |
| Typed | Fully type-annotated with a `py.typed` marker (PEP 561); clean under `mypy --strict`. |
| Forward/backward compatible | Unknown fields are consumed with `skip()` — *consumed*, not copied: the payload is stepped over, never materialized. A field whose wire type contradicts the read takes the same path (MESSAGE_SPEC §7.3, see [Deserialize](#deserialize)). |

## Usage

The codec has four use cases — serialize a message that fits in one buffer,
serialize one too large for the buffer (streamed out in chunks), deserialize a
whole message, and deserialize one arriving in chunks — plus the generated-code
path that wraps them.

### Serialize

Write fields into an `Encoder` and take the finished bytes:

```python
from sofab import Encoder

enc = Encoder()
enc.write_unsigned(1, 42)
enc.write_signed(2, -7)
enc.write_string(3, "hi")
data = enc.getvalue()
```

`Encoder()` writes into a fixed 1 KiB scratch buffer and appends each bufferful
to the result it hands back — it never grows a buffer mid-message (see [Memory
handling](#memory-handling)). Pass a writer (anything with `write(bytes)`) and
the same bufferfuls go there instead, as they are produced:

```python
with open("msg.sofab", "wb") as fh:
    enc = Encoder(fh)                 # streams out; nothing accumulates
    enc.write_unsigned(1, 42)
    enc.flush()                       # push the tail
```

### Serialize stream

`Encoder.over_buffer` is the same mechanism over a buffer **you** supply: it
writes in place and calls a flush sink whenever the buffer fills, so an
arbitrarily large message streams out through however much memory you chose to
give it:

```python
from sofab import Encoder

out = bytearray()                                            # or a socket / file write
enc = Encoder.over_buffer(bytearray(16), offset=0, flush=out.extend)  # tiny buffer
for i in range(1_000_000):
    enc.write_unsigned(i % 128, i)
enc.flush()                                                  # push the tail
```

With **no** sink the buffer is all the encoder gets: it holds the message or
reports `SofaBufferError`. That is the shape generated code uses when the schema
bounds the message — allocate `MAX_SIZE`, encode in one pass, no flush possible:

```python
buf = bytearray(Point.MAX_SIZE)
enc = Encoder.over_buffer(buf, offset=0)
point.serialize(enc)
wire = memoryview(buf)[: enc.bytes_used()]   # no copy
```

### Deserialize

`Decoder` is a **push** decoder (CORELIB_PLAN §5.2): you hand it bytes, it hands
your handler fields. There is no reader and no caller-driven loop — the decoder
owns the walk, which is what lets it resolve a field to its destination without
crossing into Python at all.

Give it a handler and feed it:

```python
from sofab import Decoder, Status, Visitor

class Handler(Visitor):
    def on_unsigned(self, field_id, value): ...
    def on_string(self, field_id, value): ...

dec = Decoder(visitor=Handler())
for chunk in socket_chunks():
    st = dec.feed(chunk)
    if st is Status.INVALID:
        raise ValueError(dec.error)
```

Every `feed` returns the outcome for the bytes consumed **so far**:

| outcome | meaning |
|---|---|
| `Status.COMPLETE` | the bytes end **exactly** at a field boundary — a valid message may end here, and more fields may also still follow |
| `Status.INCOMPLETE` | the bytes end **inside** a construct, or inside a sequence still open. **Not an error.** The partial tail is retained; the next `feed` continues from it |
| `Status.INVALID` | malformed regardless of what follows. Terminal — every later `feed` returns it again — with the reason on `dec.error` |

There is deliberately **no** `finish()`/`end()`: the status `feed` returned *is*
the answer, and whether an `INCOMPLETE` at end-of-input is acceptable is your
framing's call. A **receiver-side limit** (`max_dyn_array_count` and friends) is not
one of the three outcomes — the message is well-formed and you declined it — so
it raises `SofaLimitError` rather than folding into `INVALID`.

**A fed chunk is borrowed only for the duration of the call.** Anything the
decoder still needs afterwards is copied out before `feed` returns, so you may
reuse or overwrite that buffer the moment it comes back. `feed` accepts `bytes`,
`bytearray` or a `memoryview` over either. `reset()` starts a new message on the
same decoder, keeping the handler.

#### Integer arrays: `on_array_begin`

`on_unsigned_array` / `on_signed_array` receive an array **already decoded**, and
two decisions have been made for you by then.

The first is the **element width your schema declares**. Checking the list you
are handed only rejects an array that *arrived*; one truncated behind a bad
element never produces a list at all, so the bad value goes unreported. The
second is **where the elements went** — into a list the decoder built, and a
list handed over afterwards is a list already built.

`on_array_begin` runs at the count header, before a single element is read:

```python
from array import array

class Handler(Visitor):
    def __init__(self):
        self.ports = array("H", bytes(2 * 64))     # your storage, your size

    def on_array_begin(self, field_id, wtype, count):
        if field_id == 7:                          # `ports: { array, items: u16 }`
            return (self.ports, None, 0xFFFF)      # dst, elem_min, elem_max
        return None                                # anything else: the list
```

Return `None` and nothing changes. Return `(dst, elem_min, elem_max)` and:

* **`elem_min` / `elem_max`** are applied **at each element**, so a value outside
  them is `INVALID` whether the array completes or is truncated behind it, which
  is also `INVALID`-over-`INCOMPLETE` for free. Either side may be `None`.
* **`dst`** is a writable, contiguous buffer of at least `count` slots — an
  `array` of the right typecode, or a `memoryview` over one. The decoder fills it
  and does **not** call the typed hook; on the native engine no element is ever
  boxed. A buffer too short is `SofaArgumentError`: the decoder never grows one, and
  the refusal comes at the header, before anything is written. `dst=None` states
  the width and keeps the list.

Slots may be 1, 2, 4 or 8 bytes. A narrower one needs a declared width that fits
it, so a value can never be silently truncated into it.

Handing over a destination is what makes an array cheap: on the native engine a
1 000-element `u16` array costs **68% less** than the same array as a list, and a
`u64` array **64% less** — the list route spends most of its time building and
freeing Python integers. On the pure-Python engine there is nothing to save (it
has to box either way) and the destination route costs 7–18% more, so use it
there for the bound and the storage, not for speed.

`on_array_begin` is not called for float arrays: they carry no declared width to
state. Their destination hook is `on_float_array_begin`, below.

#### Blobs: `on_blob_begin`

`on_bytes` receives a `bytes` the decoder had to build, and the only size it
could build it from is the wire's — a megabyte blob costs a megabyte allocation
per message. §6.6.3's other shape is a destination you hand back once you know
the size:

```python
class Handler(Visitor):
    def __init__(self):
        self.frame = bytearray(1 << 20)     # your buffer, your ceiling

    def on_blob_begin(self, field_id, size):
        if field_id == 9:
            return self.frame               # filled; on_bytes is not called
        return None                         # anything else: a bytes as before
```

A buffer too short is `SofaArgumentError` — refused at the length word, before a
byte is written, and never grown.

#### Strings: `on_string_begin`

The same bargain for a `string`. The hook is told the payload's **wire byte
length** — what a schema `maxlen` bounds (MESSAGE_SPEC §1), not a character
count — and what lands in your buffer is the payload's own UTF-8:

```python
class Handler(Visitor):
    def __init__(self):
        self.name = bytearray(256)

    def on_string_begin(self, field_id, size):
        if field_id == 3:
            return self.name                # filled; on_string is not called
        return None                         # anything else: a str as before
```

The bytes are still **validated** — §6.7.2 makes a field you read both
materialized and validated — by a byte walk (§6.4.3's `utf8_valid`), so the
check does not build the `str` the destination exists to avoid. Invalid UTF-8 is
`INVALID` and your buffer is left untouched. The verdict is CPython's own on
every payload: `tests/test_aggregate_destinations.py` pins both engines to
`bytes.decode("utf-8")` over the whole RFC 3629 boundary set.

#### Float arrays: `on_float_array_begin`

One hook for both fixlen subtypes, naming which it is, taking `count` **8-byte**
slots — a Python `float` is a double, and that is what the values become:

```python
from array import array

class Handler(Visitor):
    def __init__(self):
        self.samples = array("d", [0.0] * 4096)

    def on_float_array_begin(self, field_id, subtype, count):
        return self.samples if field_id == 5 else None
```

An `array("f")` is refused rather than silently narrowed. A consumer that needs
an `fp32`'s **wire bits** intact takes `on_float32_array_bits` instead — see
*Bit-exact floats* below.

#### Bit-exact floats: `on_float32_bits` and `write_float32_bits`

Python's only float is a double, and widening an `fp32` to one **sets the quiet
bit**: a signaling NaN's payload is destroyed the instant the value passes
through the wider float, and no later code can recover it. CORELIB_PLAN §6.5
therefore requires a double-only target to carry an `fp32` to a bit-exact
consumer as wire bits:

```python
class Transcoder(Visitor):
    def on_float32_bits(self, field_id, bits):        # instead of on_float32
        out.write_float32_bits(field_id, bits)        # verbatim, no float

    def on_float32_array_bits(self, field_id, count, payload):
        out.write_float32_array_bits(field_id, payload)
```

Both hooks are opt-in by override, and both replace their value-carrying twin
for every `fp32` the message holds. `payload` is a read-only view of the bytes
you fed; it is released when the callback returns, so copy what you need to keep.
A producer that has a value rather than bytes writes `write_float32` as before.

#### Bounding what a decode holds: `reassembly`

A construct split across two fed chunks has to be joined somewhere. CORELIB_PLAN
§6.6.2 says where:

> A payload split across fed chunks has to be joined somewhere. That somewhere is
> storage the caller supplied [...] A codec **MUST NOT** grow a private
> accumulator instead.

So there is one buffer, sized once and never grown. Name its size, or hand over
the storage:

```python
dec = Decoder(visitor=handler, reassembly=64 * 1024)          # decoder sizes it
dec = Decoder(visitor=handler, reassembly=bytearray(1 << 20))  # or you do
dec = Decoder(visitor=handler)                     # sofab.DEFAULT_REASSEMBLY
```

The pieces are copied into that buffer as they arrive, and a construct that does
not fit is `SofaArgumentError` — **refused, never accommodated**. That is what lets
you bound a decode's memory by construction instead of by measurement: whatever
the sender claims, this decoder holds what you named and nothing more.

It is also what makes §6's chunk-lifetime promise literal: the unconsumed tail is
copied out before `feed` returns, so the chunk is yours again the moment it does
— overwrite it in place if you like.

A message that arrives in one piece never touches the buffer at all, whatever its
size. It is a chunked reader that has to size it for the largest `string`, `blob`
or array payload it will take **across a chunk boundary** — including one it only
means to skip, which is still buffered while it is walked.

#### Arrays of strings, blobs or structs: `sofab.collectors`

An array whose elements are not packed scalars — strings, blobs, structs — is a
**sequence whose child ids are the array indices** (MESSAGE_SPEC §5.1). Turning
that event stream back into a list is the same code for every schema, so it ships
here rather than being emitted into every generated package:

```python
from sofab import Decoder, NestedSeq, StringSeq, Visitor

class Doc(Visitor):
    def __init__(self):
        self.tags: list[str] = []
        self.rows: list[Row] = []

    def on_sequence_begin(self, field_id):
        if field_id == 3:
            return StringSeq(self.tags, cap=64, elem_max=128)
        if field_id == 4:
            return NestedSeq(self.rows, factory=Row, cap=16)
        return None                      # decode it flat, into me
```

Returning a `Visitor` from `on_sequence_begin` **descends**: the sub-tree's
fields go to the visitor returned, its `on_sequence_end` fires when the scope
closes, and the parent resumes afterwards. Returning `False` still skips the
sub-tree; anything else still decodes it flat.

A collector places each element at the id it names and fills the gap an omitted
interior element left — appending would shorten the array by every gap, and would
take a reopened id as a second element. `StringSeq`, `BytesSeq`, `UnsignedSeq`,
`SignedSeq`, `Float32Seq`, `Float64Seq` and `NestedSeq` are the set.

Which bound applies is the schema's choice. `cap` is the schema's declared
element count and an id past it is `INVALID`; `max_dyn_array_count` is the
receiver limit and applies only where the schema declares none, because §6.2.1
forbids a receiver limit on a field the schema already bounds. Either way the id
is judged **before** the list grows, so an index near 2³¹ costs a comparison and
not an allocation.

These are the **static helper layer** of CORELIB_PLAN §6.6.1 — beside the codec,
not part of it. They allocate on the generated layer's behalf; the codec does not
allocate a container of its own. What the codec does allocate is listed under
[Memory handling](#memory-handling).

### Decode into your own storage (`Binding`)

A **`Binding`** declares once where every field id belongs, so a decode fills
your slots without a handler written by hand.

It is not a second decoder. CORELIB_PLAN §5.3.1 allows exactly one decode
surface, and a table is reached **through** it: a handler declares its slots once
from `Visitor.destinations()`, and `Decoder(binding=…, words=…, objects=…)` is
the constructor shorthand for a handler that declares exactly that. Same `feed`,
same header walk, same verdicts.

What makes that one surface is where the *rules* live, not where the value
lands. The receiver cap, the schema bound, the §7.3 tag test, the UTF-8 check,
the declared element width and the resume transaction each exist **once** and run
for every field alike — a field the table names and a field it does not run the
same code right up to the assignment itself. It is the same bargain
`on_string_begin` and `on_array_begin` already strike per field — name a
destination and the codec writes there instead of calling you back — made once
for the whole message.

```python
from sofab import Binding, Decoder

b = (Binding()
     .unsigned(1, at=0, count_at=2)          # -> words slot 0, arrival in slot 2
     .string(3, at=0, maxlen=64)             # -> objects[0]
     .unsigned_array(4, at=8, cap=16, count_at=3))

words = bytearray(b.tree_words_required * 8)     # you allocate; you size it
objs = [None] * b.tree_objects_required
dec = Decoder(binding=b, words=words, objects=objs)
dec.feed(payload)

u = memoryview(words).cast("Q")                  # as many typed views as you like
u[0]                                             # field 1
objs[0]                                          # field 3
list(u[8:8 + u[3]])                              # field 4's u[3] elements
```

Two pieces of storage, both **yours**: `words`, a writable byte buffer whose
length is a multiple of 8 — every numeric field is one 64-bit slot, floats
widened to a native `double`, arrays `cap` consecutive slots — and `objects`, a
pre-sized list for `string` and `blob`, which have no fixed-width machine form.
Read the slots back through `.cast("q")` / `.cast("Q")` / `.cast("d")` over the
*same* buffer, at no copy.

What follows from the caller owning the storage:

* **Nothing is ever sized from the wire.** `cap` and `maxlen` are the *schema's*
  bounds, so a message declaring more is `INVALID` at the count/length header,
  before an element is read (MESSAGE_SPEC §7.1). It is not a `SofaLimitError` —
  that is for fields the schema leaves unbounded, and declaring a bound here is
  what takes the receiver-side cap off the field.
* **Absence needs no sentinel.** A slot the decoder does not write keeps what you
  put there. `count_at` names a slot receiving `1` for a scalar that arrived, the
  element count for an array, the occurrence count for a sequence.
* **A contradicting wire tag is skipped, not rejected** (§7.3) — like an unknown
  id, and the decode stays `COMPLETE`.
* **Nested messages share the same storage.** `b.sequence(id, child)` descends
  into a child table in the same two buffers, so a whole tree decodes into one
  flat pair. A sequence with no binding is skipped whole.
* **A binding is build-once.** Building a decoder freezes the table and derives
  its destination map, so a table changed afterwards cannot leave a decoder
  reading a stale copy. The map is cached on the `Binding`, so building a decoder
  per message costs no recompilation.

A `Binding` and a `Visitor` compose: bind the fields you know, the visitor gets
the rest — every hook, including the *begin* destinations and the raw `fp32`
channel. A declaration is about *its own* field, so the two never collide: an
`fp32` the table names lands in its slot as the widened `double` the table asked
for, and one it does not name reaches `on_float32_bits` if the visitor overrides
it.

Declaring the slots on the handler instead is the same thing without the
constructor keywords, and is what generated code should emit:

```python
from sofab import Visitor

class Telemetry(Visitor):
    def __init__(self):
        self.words = bytearray(b.tree_words_required * 8)
        self.objects = [None] * b.tree_objects_required

    def destinations(self):                  # asked once, when the decoder is built
        return (b, self.words, self.objects)

t = Telemetry()
Decoder(visitor=t).feed(payload)             # one handler argument, nothing else
assert memoryview(t.words).cast("Q")[0] == u[0]
```

Anything a `Binding` declares, a hand-written `Visitor` can declare too:
`on_schema_bound` is where a handler names the `count`/`maxlen` the *schema*
puts on a field, which is what makes a receiver-side `max_dyn_*` cap stop
applying to it (§6.2.1) and makes exceeding it `INVALID` rather than a policy
rejection. It is told the wire's tag alongside the id, so it can apply §7.3 to
its own declaration exactly as the table route does.

### Code generator

The common real use is driving the library through **generated code**:
`sofabgen --lang python` emits a `@dataclass` per message with exactly four
methods (CORELIB_PLAN §6.1.1 fixes the names) — the streaming pair `serialize` /
`deserialize`, which talks to the primitives above, and the one-shot pair
`encode` / `decode`, thin wrappers over it. A hand-written stand-in:

```python
from dataclasses import dataclass
from sofab import Binding, Decoder, Encoder, Status

# generated by: sofabgen --lang python
@dataclass
class Point:
    x: int = 0
    y: int = 0

    #: Field id -> slot, built once from the schema.
    BINDING = Binding().signed(1, at=0, count_at=2).signed(2, at=1, count_at=3)

    def serialize(self, e: Encoder) -> None:    # streaming out: write into any encoder
        e.write_signed(1, self.x)
        e.write_signed(2, self.y)

    def deserialize(self, words) -> None:       # streaming in: read your slots back
        q = memoryview(words).cast("q")
        u = memoryview(words).cast("Q")
        if u[2]:
            self.x = q[0]
        if u[3]:
            self.y = q[1]

    @classmethod
    def decoder(cls):                           # §6.1.1: the streaming reader
        words = bytearray(cls.BINDING.tree_words_required * 8)
        return Decoder(binding=cls.BINDING, words=words), words

    def encode(self) -> bytes:                  # one-shot wrapper over serialize()
        e = Encoder()
        self.serialize(e)
        return e.getvalue()

    @classmethod
    def decode(cls, data: bytes) -> "Point":    # one-shot wrapper over deserialize()
        dec, words = cls.decoder()
        if dec.feed(data) is not Status.COMPLETE:
            raise ValueError(dec.error or "incomplete message")
        o = cls()
        o.deserialize(words)
        return o

wire = Point(x=3, y=4).encode()
got = Point.decode(wire)             # got.x == 3, got.y == 4
```

The one-shot pair holds the whole message in memory; `serialize` / `deserialize`
are the same code without that requirement. Out, `serialize` writes into an
encoder over a buffer **you** sized, draining to a sink as it fills; in, the
decoder from `decoder()` takes chunks of any size:

```python
# streaming out: a 2-byte buffer, drained to the sink as the message is written
packets = bytearray()                        # or a socket / file write
enc = Encoder.over_buffer(bytearray(2), offset=0, flush=packets.extend)
Point(x=3, y=4).serialize(enc)
enc.flush()                                  # push the tail
streamed = bytes(packets)                    # == wire, out of a buffer half its size

# streaming in: the same object, fed one byte at a time
dec, words = Point.decoder()
for i in range(len(streamed)):
    st = dec.feed(streamed[i : i + 1])       # COMPLETE / INCOMPLETE / INVALID
got_streamed = Point()
got_streamed.deserialize(words)
```

Every `feed` returns the outcome for the bytes so far, so a source that runs dry
before the message ends adds no obligation to the generated code beyond looking
at it: `INCOMPLETE` means "feed me the next chunk", and only your framing knows
whether more can still come.

### Decode limits

A field the schema leaves unbounded lets the *sender* decide what the *receiver*
allocates, so every decoder carries **receiver-side** limits that reject an
oversize field on its count/length word alone — *before* any allocation or
payload buffering:

```python
dec = Decoder(binding=b, words=words,
              max_dyn_array_count=65536, max_dyn_string_len=1 << 20, max_dyn_blob_len=1 << 20)
```

A field whose declared count/length exceeds its limit raises `SofaLimitError`: a
*policy* rejection, distinct from malformed input, and a sibling of
`SofaDecodeError` under `SofaError` rather than a subclass, so `except
SofaDecodeError` does not catch it. It governs what **this decoder** would
allocate; the two sections below say which fields those are.

The five exceptions carry CORELIB_PLAN §6.3's five codes: `SofaBufferError` is
`BufferFull`, `SofaArgumentError` is `InvalidArgument`, `SofaDecodeError` is
`InvalidMessage`, `SofaLimitError` is `LimitExceeded`, and `SofaIncompleteError`
is the `INCOMPLETE` outcome, which §6.3 is explicit is not an error at all.
§6.3 lets a port "adapt casing and idiom"; `SofaArgumentError` was once
`SofaRangeError`, which read narrower than its code, and the old name is kept as
an alias.

**There is no unset state and no unlimited mode.** `None` is refused rather than
read as "no limit". Each defaults to the format-wide ceiling — `ARRAY_MAX` for
the count, `FIXLEN_MAX` for the two lengths — above which the value is already
`INVALID`, so the default configuration rejects nothing a looser one would
accept. `0` is a real setting, not an unset one. The numbers are supplied by
generated code, which knows the schema and the deployment.

Independent of any limit, the decoder never pre-allocates from an untrusted array
count — a truncated oversize claim fails promptly as an `INCOMPLETE`.

The verdict is reached on the count/length word alone, before a single payload
byte is read or buffered — the point CORELIB_PLAN §6.2.1 requires it to be
decided.

#### A schema-bounded field is exempt

A cap is *capacity* the deployment commits where the **sender** chooses the size.
Where the **schema** already states a `count:`/`maxlen:`, that bound governs
instead and an over-bound value is malformed input, so §6.2.1 forbids the cap
there ("MUST NOT be applied to a field the schema already bounds") and §6.3
forbids `SofaLimitError` on such a field.

Only the schema knows which fields those are, so the **handler** is where it says
so. A `Binding` declares it in the table — `cap` on an array and `maxlen` on a
string or blob **are** the schema's bound — and a hand-written visitor declares
it with `on_schema_bound`:

```python
from sofab import Binding, FixlenSubtype, Visitor, WireType

b = Binding().string(1, at=0, maxlen=4194304)   # `name: { string, maxlen: 4194304 }`

class Names(Visitor):                           # the same statement, by hand
    def on_schema_bound(self, field_id, n, wtype, subtype):
        if field_id == 1 and wtype is WireType.FIXLEN and subtype is FixlenSubtype.STRING:
            return 4194304
        return -1                               # not the field the schema declared
```

Declaring it does two things at once: the receiver-side cap stops applying to
that field, and the decoder enforces the declared bound itself — an over-bound
length is `INVALID` (`SofaDecodeError`, MESSAGE_SPEC §7.1), never
`SofaLimitError`. `on_schema_bound` is asked at the count/length header, for a
`string`, a `blob` or an array the handler has accepted, and for nothing else.

The tag is passed with the id because this is the only hook that spans more than
one kind. `on_string_begin` fires for a `string` and nothing else, `on_array_begin`
for an integer array and nothing else — the decoder has already matched the wire's
tag before calling them. Here it has not, and an id the schema bounds can arrive
under a tag the schema never declared for it. MESSAGE_SPEC §7.3 skips such a field
like an unknown id, so **answer `-1` for a tag you did not declare**: a bound
answered for someone else's field turns a §7.3 skip into an `INVALID`. A `Binding`
entry gets that same test run for it by the decoder, which is why the two routes
agree. `subtype` is `None` for an integer array, which carries none on the wire.

Nothing here costs an allocation — two plain integers and two interned enum
members — and that is deliberate: it is the one hook generated code overrides on
every message, so overriding it must cost nothing per field. It is also why
generated code needs no `on_field` to pre-filter the tag for it. `on_field` is the
only hook that takes a `Field`, and therefore the only one that makes the decoder
build one.

#### So is a field nobody materializes

A cap prevents an allocation, so it applies where there is one to prevent. Two
routes make none, and neither is capped:

* a field the handler declines at `on_field` — including one a binding does not
  name, or names with a contradicting wire tag (§7.3) — the decoder walks past
  the payload without building anything from it;
* a field the handler *wants*, having handed back its own buffer from
  `on_blob_begin`, `on_string_begin`, `on_array_begin` or
  `on_float_array_begin`. The hook is told the announced length or
  count first, and a receiver that does not want that many bytes says so there —
  the decision is the handler's, and a limit the decoder applied on its behalf
  would only take it away.

What is left is the default route, and it is the one §6.2.1 is about: with no
destination back, the decoder itself has to build a `str`, a `bytes` or a list,
and the only size it could build one from is the wire's. That allocation is
refused on the count/length word, before a payload byte is read.

`on_float32_array_bits` is the one route that is **not** in that list: it hands
over the wire bytes without asking, so the handler has no place to refuse and
the configured ceiling stays the only one.

The ceiling on a buffer you supply is that buffer's own size. Too short for what
the hook was told, and the decoder refuses it — `SofaArgumentError`
(`InvalidArgument`), never a silent truncation and never a resize. That is a fact
about your storage rather than a verdict on the message, which is why it is not
`SofaLimitError`.

The same goes for `reassembly=`: a skipped payload spanning a chunk boundary is
still joined in the buffer you supplied, so what a skip can cost is bounded by
that buffer, and one that does not fit is refused the same way.

A **schema** bound is the opposite kind of thing from a cap: it is part of the
message definition, so breaching it is malformed input, not policy. The
array bindings take the declared element width for exactly that reason —
`unsigned_array(..., elem_max=255)` for a `u8` array,
`signed_array(..., elem_min=-128, elem_max=127)` for an `i8` one (either half may
be given alone; the other side stays open). An
element outside the declared width raises `SofaDecodeError` the moment its own
bytes are decoded, so the verdict never depends on how much of the array
followed it or on which engine read it (MESSAGE_SPEC §7.1). Omit them for
`u64`/`i64`, whose range is the value domain, or for an unbounded consumer.

## Memory handling

The key point for Python: **decoding allocates results for you unless you ask it
not to.** The visitor's typed hooks hand back fresh `int`/`str`/`bytes`/`list`
objects; a `Binding`, or one of the five *begin* hooks, writes into storage you
supplied and sized instead. Encoding never allocates an output buffer at all
(§5.1).

**Every aggregate has a route that does not size an allocation from the wire**
(§6.6.3). Each hook is told the announced count or byte length first, before a
byte is decoded, and refuses a destination too short rather than growing one:

| aggregate | destination route | returns |
|---|---|---|
| `blob` | `on_blob_begin(id, size)` | a writable buffer of `size` bytes |
| `string` | `on_string_begin(id, size)` | a writable buffer of `size` **bytes** — the payload's UTF-8, validated on the way in |
| unsigned / signed array | `on_array_begin(id, wtype, count)` | `(dst, elem_min, elem_max)` |
| `fp32` / `fp64` array | `on_float_array_begin(id, subtype, count)` | a writable buffer of `count` 8-byte slots |
| `fp32` / `fp64` scalar | — | a value; a scalar is not storage (§6.6.3) |

`fp32` additionally has §6.5's raw channel — `on_float32_bits` and
`on_float32_array_bits`, paired with `Encoder.write_float32_bits` /
`write_float32_array_bits` — for a consumer that has to reproduce the wire bytes
rather than the value.

**Where this port stands against CORELIB_PLAN §6.6, stated plainly.** The codec
allocates nothing a wire number sizes on the paths above — encode, a `Binding`
decode, and a visitor decode that takes the destination routes — and
`tests/test_allocation.py` and `tests/test_aggregate_destinations.py` measure
exactly that: a payload a thousand times larger costs the same. It does not hold
where a handler asks for the **value**: `on_string`, `on_bytes`,
`on_unsigned_array` and the float-array hooks each hand back a whole object, and
the only size available to build one from is the wire's, which is what §6.6.3
says such a callback obliges. Those hooks are kept because they are the
convenient way to read a message, and every one of them now has an opt-out.
Beneath both, CPython allocates for every object a handler is given, so a
literal zero is not reachable in this language whatever the API looks like.

* **Decode: no value outlives the callback, and nothing is aliased.** A handler
  receives a fresh `str`, independent `bytes`, a fresh `int`/`float` or a new
  `list`, and every one of them stays valid after the decoder advances. The one
  exception is `on_float32_array_bits`, which is §6.7's pass-through route: it is
  handed a read-only view of the bytes *you* fed, and the view is **released**
  when the callback returns, so it cannot be kept by accident.
* **Decode: a chunk-straddling construct is joined in one bounded buffer.**
  `Decoder(reassembly=…)` takes a `bytearray` you supply, or an `int` for the
  decoder to size one from at construction; omit it and it takes
  `sofab.DEFAULT_REASSEMBLY` (4096) bytes. There is no other shape and it never
  grows: a construct that does not fit is `SofaArgumentError`, which is what
  bounds a decode's memory by construction. CORELIB_PLAN §6.6.2 requires exactly
  that — no sender can enlarge this buffer by sending different bytes. The 4096
  is this port's number and not the specification's; raise it for a reader that
  streams larger `string`/`blob`/array payloads **across chunk boundaries**.
  A message fed in one call never touches the buffer, whatever its size.
* **Decode: a decoded value can land in your storage too.** `Binding` writes
  every field into slots you supply, and the five *begin* hooks above take a
  buffer for every aggregate — so an array of any length costs no list and no
  object per element. A `string` is still **validated** when it goes into your buffer (§6.7.2:
  a field the handler reads is materialized *and* validated); the check runs over
  the bytes in fixed windows, so it does not build the `str` the destination
  exists to avoid. What is left materialising is the scalars, and those are
  values, not storage.
* **Decode: a suspended construct keeps its bytes, and only its bytes.**
  Everything fed is retained until it is consumed, so a field split across chunks
  is never half-decoded; the consumed prefix is dropped on the next `feed`, down
  to the first byte of the construct in flight — that byte is the one the retry
  re-reads from. The window held is therefore one field (for a declined sequence,
  one sequence), not one message.
* **Decode: a fed chunk is borrowed for the call and no longer** (§6). Anything
  the decoder still needs when `feed` returns — the tail of a construct split
  across the boundary, a decoded string or blob — has been copied out, so the
  same calling code is correct whatever the chunk boundaries are.
* **Decode into your own slots.** With a `Binding` the numeric fields never
  become Python objects at all: they land in the `words` buffer you passed, sized
  from the schema. The decoder allocates no destination and grows nothing, and a
  wire count past your capacity is rejected rather than honoured (§6.6).
* **Decode: measuring a payload costs nothing.** A string or blob is bounded
  against its schema `maxlen` by the binding, on the length the *sender*
  declared — no re-encoding a decoded `str` just to measure it.
* **Decode: a value you don't want costs nothing to get rid of.** A field no
  binding names and no visitor wants — or one a visitor declines — walks a string,
  blob or fixlen-array payload by advancing the cursor, so nothing is allocated
  for bytes that are being discarded: skipping a 1 MiB blob already in the
  buffer is a pointer bump. The bytes are still **buffered** when the payload
  straddles a chunk boundary, so what a skip saves is the copy, not the window —
  and a skipped construct larger than the reassembly buffer is therefore refused
  where a port that skips by advancing a cursor would walk past it. That is the
  resume contract's doing: a suspended skip replays from the construct's first
  byte, so the bytes cannot be dropped as they arrive.
* **Encode: one ownership model — the output buffer is fixed, and never grows.**
  CORELIB_PLAN §5.1 forbids a corelib to allocate an output buffer or to grow
  one, so there is a single mechanism here with three ways to reach it:
  * `Encoder.over_buffer(buf, offset, flush)` is the primitive and the only
    caller-supplied form: it writes in place through a `memoryview`, drains to
    the sink when full and reuses the buffer — or, **without** a sink, holds the
    message or reports `SofaBufferError`. That is the shape generated code uses
    for a schema whose `MAX_SIZE` bounds the message.
  * `Encoder(writer)` installs a **1 KiB scratch buffer with a sink** that
    forwards each bufferful to `writer.write`. A 100 MB message costs 1 KiB of
    encoder memory, and the bytes leave *while* the message is written, not at
    `flush()`. Nothing is retained, so `getvalue()` raises `SofaArgumentError`.
  * `Encoder()` is the same scratch buffer with the sink appending into the
    *result* — the message `getvalue()` hands back, joined from the drained
    chunks (a message that fits in the scratch is never chunked at all). What
    grows is the message being returned, not a buffer being written into:
    `bytes_used()` never exceeds 1 KiB. **One deviation, on this shape only:** a
    `string`/`blob` run at least a bufferful long is appended to the result
    directly instead of being copied through the scratch. §5.1.6 says every byte
    a sink receives lies inside the installed buffer, and here it does not. The
    run is the encoder's own immutable copy, the output bytes are identical, and
    no caller buffer and no caller sink exist on this path — `Encoder(writer)`
    and `over_buffer(…, flush)`, which have both, copy everything through the
    buffer. `tests/test_encode_buffer_ownership.py` pins all three.

  The scratch is one allocation per encoder, made at construction and never
  resized — §6.6.1's first row, the convenience "the caller … then calls the
  corelib", whose storage the codec keeps nothing of. §5.1.2 puts even that in
  the generated layer, which knows the schema: a caller who wants zero library
  allocation supplies the buffer with `over_buffer`, and generated code that can
  bound its message from `MAX_SIZE` should.
* **`MIN_OUTPUT_BUFFER` is `1`, and it applies to a buffer installed *with* a
  sink.** `sofab.MIN_OUTPUT_BUFFER` is the smallest output buffer this port
  accepts for **streaming**: one byte, because the encoder splits every atomic
  unit — a header varint, a `fixlen_word`, an element count, a scalar, one
  float — at any byte boundary. `Encoder.over_buffer(buf, offset, flush)` and every
  mid-stream `buffer_set(buf, offset)` that carries a flush sink require
  `len(buf) - offset >= MIN_OUTPUT_BUFFER` and raise `SofaArgumentError` right
  there — where the buffer is handed over, never partway through a message.
  A buffer installed **without** a sink is subject to no minimum: no flush can
  occur, so the buffer simply holds the message or reports `SofaBufferError`, and
  sizing it from a generated `MAX_SIZE` stays exact. **Nothing but the installed
  buffer reaches a caller's sink**: a `string`/`blob` run is copied into the
  output buffer like any other output, and every flush hands the sink a
  `memoryview` **over that buffer** — the installed buffer itself, never a copy
  of it and never any other memory (§5.1.6). The one exception is the in-memory
  `Encoder()`, which has no caller sink; it is stated with that constructor
  above. A sink that only
  reads or copies during the call may let the view go; one that keeps it has
  *taken* the buffer and must install a replacement before it returns (below).
* **The handles this codec allocates, in full.** CORELIB_PLAN §6.6.2 lets a
  codec allocate a *typed handle* where the language will not express a copy
  without one, provided it carries no message bytes and no wire number sizes it —
  and asks for the list. Python's only way to name a region of someone else's
  buffer is a `memoryview`, so this port allocates four kinds and no others:

  | handle | when |
  |---|---|
  | over the output buffer | one per installation (`buffer_set`), plus one full-buffer slice kept for the installation and handed to the sink at every flush of it; a short final flush builds and drops one |
  | over the input buffer | one per `bytes` taken out of an accumulating `bytearray`, so the payload is copied once instead of twice |
  | over a decode destination | one per `on_array_begin` / `on_blob_begin` that returns a buffer |
  | over the words buffer | three per `Binding` handler, at construction, plus one per array row |

  None of them holds a decoded value, and each costs the same whatever the
  payload's size. Everything else the codec touches after construction is the
  caller's storage.

* **The start offset belongs to the installation, not to the buffer.** A flush
  sink that returns **without** installing anything has *copied* the bytes it was
  handed: the same buffer stays active and encoding resumes at offset 0. A sink
  that *takes* the buffer — queues it for an async write, hands it to a
  transport — **must** install a replacement with `buffer_set(buf, offset)`
  before it returns, and that call's `offset` is where encoding resumes.
  Re-installing is therefore how a sink gets fresh framing-header room in *every*
  flushed packet, including when it passes the **same** buffer back.
* **Lazy sequence framing holds no buffer.** The ids
  `write_sequence_begin_lazy` holds back are encoder state, never buffer content:
  the pending run is `MAX_DEPTH` slots wide, sized at construction and never
  grown (§6.6), so the hold-back reaches the full depth and every depth is
  canonical. A flush therefore cannot split a held-back run, and a tiny output
  buffer yields exactly the one-shot bytes.

## Build & test

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest ruff mypy   # compiles the native accelerator if a C compiler is present
pytest                       # vectors + roundtrip + streaming + malformed + native↔pure parity
ruff check src/sofab tests   # lint
mypy --strict src/sofab      # type-check
```

`assets/test_vectors.json` is the shared cross-language suite, copied verbatim
from `corelib-c-cpp`. Its `sequence_growth` block describes a **wrapper array's
container** growing as elements arrive — a length no wire word announces, since
MESSAGE_SPEC §5.1 makes it *highest present id + 1*. That container belongs to
the layer above the codec, and this port ships that layer: `sofab.collectors`
(`StringSeq`, `BytesSeq`, `UnsignedSeq`, `SignedSeq`, `Float32Seq`,
`Float64Seq`, `NestedSeq`). **It allocates, on the generated layer's behalf**
(CORELIB_PLAN §6.6.1) — its lists are not a §6.6 breach, because the codec never
calls into it: a collector is reached only from inside a visitor callback the
codec made, and the codec keeps no reference to anything it takes.
`tests/test_sequence_growth.py` replays every case in the block against it, and
`tests/test_collectors.py` measures the growth **geometry** with `tracemalloc`
— extending to at least `id + 1` in one pass, so a sparse array costs O(n) and
not O(n²).

If the compile fails or no compiler is available, the install falls back to
pure-Python (the extension is marked *optional* in `setup.py`). Both engines
ship, so both are run:

```bash
pytest                       # whichever engine is active (native if built)
SOFAB_PUREPYTHON=1 pytest    # force the pure-Python engine
```

`SOFAB_REQUIRE_ENGINE=native|python` makes a run assert that `sofab.IMPL` is the
engine it claims to be exercising, so a missing accelerator fails the run instead
of skipping every native-gated test out of it:

```bash
SOFAB_REQUIRE_ENGINE=native pytest                    # fails if the accelerator is missing
SOFAB_PUREPYTHON=1 SOFAB_REQUIRE_ENGINE=python pytest  # the fallback engine, pinned
```

CI runs the full suite twice on every supported Python — once per engine, each
pinned that way — and adds a leg installed with `SOFAB_DISABLE_NATIVE=1` that
proves the compiler-less install still passes.

## Benchmarks

`bench/perfbench.py` implements the three tools BENCH_SPEC requires — the same
workloads, on the same data, measured the same way and printed in the same
grammar as every other port, so the numbers are comparable across languages.
`bench/compare_protobuf.py` is extra and language-native: it compares the native
accelerator, the pure-Python fallback, and `protobuf`'s Python runtime (upb C
backend), materializing fully on both sides so it is apples-to-apples with a
SofaBuffers visitor that takes its values:

```bash
python bench/perfbench.py bench           # throughput on this machine, MB/s (MB = 1e6)
python bench/perfbench.py perf            # per-op cost for the shared 12-field message
bash  bench/run_callgrind.sh              # instructions/op (Callgrind) — clock-independent
pip install protobuf                      # optional; the column is dropped if absent
python bench/compare_protobuf.py          # best-of-5 MB/s table
```

`bench` / `perf` measure this machine and move with its load; `run_callgrind.sh`
counts instructions retired, which is deterministic and comparable across hosts,
so it is the one to trust when judging a change to the library itself.
(`time` still works as a synonym for `bench`.)

The workloads are BENCH_SPEC's:

| dataset | what it is there for |
|---------|----------------------|
| `u64 array (1000)` | the compact scalar-array path, 1..10-byte varints |
| `typical message` | seven mixed fields, ~37 bytes — the small-message case |
| `perf message` (`perf` only) | twelve fields, 170 bytes on every port — a size parity check |
| `blob 1MB` | buffer handling: 1,000,005 encoded bytes, one-shot vs. streaming vs. chunk-fed decode |
| `composite` | 956 bytes exercising a wrapper array, non-ASCII UTF-8, depth-3 nesting, an omitted default field and a two-byte header |

The three `blob 1MB` rows are read **against each other**, never next to
`typical message`: five of its bytes are metadata and a million are payload, so
the absolute figure is this machine's memory bandwidth. One-shot is one
contiguous write into a 1,000,005-byte caller buffer with no sink; streaming is
the same bytes through a **4096**-byte caller buffer with a flush sink, i.e. ~245
flushes; decode is fed in 4096-byte chunks. This port grants no pass-through, so
BENCH_SPEC's optional `blob 1MB passthrough` row is absent rather than stubbed.

Read those two encode rows as MB/s rather than `Ir/op`: the one-shot row is a
*single* 1,000,000-byte `memcpy`, which glibc serves from its ERMS (`rep movsb`)
path and Valgrind counts at about one instruction per byte, while the streaming
row's 4096-byte copies take the vectorised path at a fraction of that. `Ir/op`
therefore reports one-shot as the *dearer* of the two although it does strictly
less work. Every other row is `Ir/op`'s to tell.

The native accelerator is worth roughly an order of magnitude over the pure
engine on the message-shaped rows and two on the array-heavy ones, and it beats
protobuf everywhere except the smallest decode, where the two are level. That
last workload is where the streaming decode costs the most: a visitor crosses the
Python↔C boundary once per field, whereas protobuf parses the whole message in
one C call. What a **declared destination** buys is that crossing: a field the
handler's table names is written into its slot without one, and an array of any
length costs no crossing at all rather than one per element. Every rule still
runs on the one path either way (§5.3.1).
`bench/compare_protobuf.py` runs that comparison.

Measured figures are not reproduced here — they belong to the cross-language
benchmark arena, which runs every port on one host under one methodology. This
section says how to obtain them, not what they came out as.
