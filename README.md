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

Distribution `sofa-buffers-corelib` on PyPI; import package `sofab`.

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
and exception classes from the shared `types.py`, so a `SofaRangeError` is the
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
| No silent truncation | An integer field accepts what Python accepts wherever an integer is required — anything with `__index__` (`int`, `bool`, `IntEnum`, NumPy integers). A `float` is refused with `SofaRangeError`, `3.0` included. |
| No unreadable message | The format-wide ceilings (CORELIB_PLAN §6.2) bind the encoder too: a field id above `ID_MAX`, an array count above `ARRAY_MAX`, nesting past `MAX_DEPTH`, and a string/blob payload above `FIXLEN_MAX` (2 GiB − 1) are each refused with `SofaRangeError` **before** the field header is written. An oversized blob is refused on its length, before it is copied. |
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
  boxed. A buffer too short is `SofaRangeError`: the decoder never grows one, and
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

`on_array_begin` is not called for float arrays: they carry no declared width and
are already moved in one piece.

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

A buffer too short is `SofaRangeError` — refused at the length word, before a
byte is written, and never grown. Strings are not offered: a string you read must
be validated (§6.7.2), and this port validates by decoding, which builds the
`str` a destination would exist to avoid.

#### Bounding what a decode holds: `reassembly`

A construct split across two fed chunks has to be joined somewhere. By default
the decoder joins it in a `bytearray` of its own, which grows to whatever the
message says — convenient, and not what CORELIB_PLAN §6.6.2 asks for:

> A payload split across fed chunks has to be joined somewhere. That somewhere is
> storage the caller supplied [...] A codec **MUST NOT** grow a private
> accumulator instead.

Pass one and it does not:

```python
dec = Decoder(visitor=handler, reassembly=bytearray(64 * 1024))
```

The pieces are copied into your buffer as they arrive, and a construct that does
not fit is `SofaRangeError` — **refused, never accommodated**. That is what lets
you bound a decode's memory by construction instead of by measurement: whatever
the sender claims, this decoder holds your 64 KiB and nothing more.

It also makes §6's chunk-lifetime promise literal. Without a reassembly buffer a
`bytes` chunk is read where it lies and kept until the next `feed`; with one, the
unconsumed tail is copied out before `feed` returns, so the chunk is yours again
the moment it does — overwrite it in place if you like.

Nothing else changes: same verdicts, same values, same chunking-independence. A
message that arrives in one piece never touches the buffer at all.

### Decode into your own storage (`Binding`)

A `Visitor` costs one Python call per field. A **`Binding`** costs none: declare
once where every field id belongs, and the decoder writes there itself.

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
* **A binding is build-once.** Building a decoder freezes the table; the decoder
  caches what it derives, including a compiled lookup table shared by every
  decoder over that binding in the native engine.

A `Binding` and a `Visitor` compose: bind the fields you know, the visitor gets
the rest.

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
SofaDecodeError` does not catch it.

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

Only the schema knows which fields those are, and the binding is where it says
so — `cap` on an array and `maxlen` on a string or blob **are** the schema's
bound:

```python
b = Binding().string(1, at=0, maxlen=4194304)   # `name: { string, maxlen: 4194304 }`
```

Declaring it does two things at once: the receiver-side cap stops applying to
that field, and the decoder enforces the declared bound itself — an over-bound
length is `INVALID` (`SofaDecodeError`, MESSAGE_SPEC §7.1), never
`SofaLimitError`. A field with no declared bound — one the table names without a
`maxlen`, or does not name at all — stays under the cap, including when it is
only walked past.

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
not to.** The pull and visitor paths hand back fresh `int`/`str`/`bytes`/`list`
objects; a `Binding` writes into storage you supplied and sized instead, and
allocates only for `string` and `blob`, which have no fixed-width machine form.
Encoding never allocates an output buffer at all (§5.1).

* **Decode: the library owns the input buffer.** `Decoder` keeps a single
  internal buffer, extended by `feed` and never handed out, so there is **no
  zero-copy aliasing**: a handler receives a fresh `str`, independent `bytes`, a
  fresh `int`/`float` or a new `list`, and every one of them stays valid after
  the decoder advances.
* **Decode: where a chunk-straddling construct is joined is yours to decide.**
  By default the decoder joins it in a `bytearray` of its own, extended to
  whatever the message declares. Pass `reassembly=bytearray(n)` and it joins in
  **your** buffer instead, refusing a construct that does not fit rather than
  growing one — which is what bounds a decode's memory by construction, and what
  CORELIB_PLAN §6.6.2 asks of a codec. The default is the convenience; the
  parameter is the conformant path, and generated code should take it.
* **Decode: a decoded value can land in your storage too.** `Binding` writes the
  numeric fields straight into slots you supply; `on_array_begin` takes a buffer
  for an integer array's elements and `on_blob_begin` one for a blob's payload,
  and neither builds a Python object on the way. What is left materialising is a
  `str` (which must be validated, and this port validates by decoding), a float
  array, and the scalars — those are values, not storage.
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
  straddles a chunk boundary, so what a skip saves is the copy, not the window.
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
    `flush()`. Nothing is retained, so `getvalue()` raises `SofaRangeError`.
  * `Encoder()` is the same scratch buffer with the sink appending into the
    *result* — the message `getvalue()` hands back, joined from the drained
    chunks (a message that fits in the scratch is never chunked at all, and a
    `string`/`blob` run longer than the buffer becomes one chunk rather than
    being copied through it). What grows is the message being returned, not a
    buffer being written into: `bytes_used()` never exceeds 1 KiB.

  The scratch is one allocation per encoder, made at construction and never
  resized. A caller who wants zero library allocation supplies the buffer with
  `over_buffer`.
* **`MIN_OUTPUT_BUFFER` is `1`, and it applies to a buffer installed *with* a
  sink.** `sofab.MIN_OUTPUT_BUFFER` is the smallest output buffer this port
  accepts for **streaming**: one byte, because the encoder splits every atomic
  unit — a header varint, a `fixlen_word`, an element count, a scalar, one
  float — at any byte boundary. `Encoder.over_buffer(buf, offset, flush)` and every
  mid-stream `buffer_set(buf, offset)` that carries a flush sink require
  `len(buf) - offset >= MIN_OUTPUT_BUFFER` and raise `SofaRangeError` right
  there — where the buffer is handed over, never partway through a message.
  A buffer installed **without** a sink is subject to no minimum: no flush can
  occur, so the buffer simply holds the message or reports `SofaBufferError`, and
  sizing it from a generated `MAX_SIZE` stays exact. There is no pass-through: a
  `string`/`blob` run is copied into the output buffer like any other output, and
  every flush hands the sink a `bytes` snapshot of that buffer's prefix — a sink
  may retain what it receives without pinning caller memory.
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
  the pending run is allocated on the first hold-back — an encoder that never
  opens a sequence never pays for it — and grows on demand to the full
  `MAX_DEPTH`. A flush therefore cannot split a held-back run, and a tiny output
  buffer yields exactly the one-shot bytes.

## Build & test

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest ruff mypy   # compiles the native accelerator if a C compiler is present
pytest                       # vectors + roundtrip + streaming + malformed + native↔pure parity
ruff check src/sofab tests   # lint
mypy --strict src/sofab      # type-check
```

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
backend), materializing fully on both sides so it is apples-to-apples with the
SofaBuffers pull API:

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
last workload is where the streaming **pull** API costs the most: it crosses the
Python↔C boundary once per field when a visitor handles it — and not at all when
a `Binding` does — whereas protobuf parses the whole message in one C call. `bench/compare_protobuf.py` runs that
comparison.

Measured figures are not reproduced here — they belong to the cross-language
benchmark arena, which runs every port on one host under one methodology. This
section says how to obtain them, not what they came out as.
