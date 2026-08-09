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
stream core, meant to be driven by **generated code**: a schema-driven generator
emits one class per message plus `marshal` / `unmarshal` methods that call the
`Encoder` / `Decoder` primitives here — the same way protobuf's generated code
calls its runtime.

The public API is one pair of classes with two interchangeable engines selected
at import: the hot path (varint / zigzag / buffer management) ships as an
optional compiled **native accelerator** (Cython → C, `sofab._speedups`) loaded
automatically when present, with a **pure-Python fallback** used when it is not.
The two are byte-for-byte interchangeable, so the library runs anywhere CPython
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
```

## Why this design

| Goal | How |
|------|-----|
| Streaming **out** | `Encoder` writes to any binary stream (file, socket, `BytesIO`), so a message can exceed RAM and stream straight to the wire. |
| Streaming **in** | `Decoder` is a pull parser over any `read(n)` reader; `next()` returns one field header at a time, never materializing the whole message. A call that runs out of bytes reports `SofaIncompleteError` **without consuming anything**, so it can simply be re-issued when more arrive. |
| Native speed, zero runtime deps | The hot path ships as an optional Cython accelerator (`sofab._speedups`); when it can't be built it falls back to pure Python. No runtime third-party deps either way. |
| Runs everywhere | With no compiler or wheel, `pip` still installs a working pure-Python build (`py3-none-any`). Native and pure paths are byte-for-byte identical — falling back changes only speed. |
| Sticky errors | `Encoder(sticky=True)` records the first failure and turns later writes into no-ops, so generated `marshal` code can check `enc.error` once. |
| No silent truncation | An integer field accepts what Python accepts wherever an integer is required — anything with `__index__` (`int`, `bool`, `IntEnum`, NumPy integers). A `float` is refused with `SofaRangeError`, `3.0` included: writing `3` for a caller's `3.7` would change the value in a way the receiver could never detect. |
| Floats narrow by IEEE rules | A Python `float` is a C double, so `write_float32` (scalar and array) narrows on the way out: round-to-nearest, and a magnitude past `FLT_MAX` overflows to `±inf` — the same bytes a native-`fp32` corelib writes, and identical in both engines. NaN payloads, signaling ones included, keep their exact bits (§4.6/§6.5). |
| Reserve-offset | `Encoder.over_buffer(buf, offset=…)` leaves room at the front of the buffer for a lower-layer protocol header; a sink calling `buffer_set(buf, offset)` re-arms that room for **every** flushed packet. |
| Sparse sequences | `write_sequence_begin_lazy` holds a sequence header back until the sequence receives content, so a sequence-typed field equal to its declared default is **omitted** rather than framed empty (MESSAGE_SPEC §2) — decided in one forward pass, without buffering the sub-message. `write_sequence_end` drops a contentless sequence; `write_sequence_end_keep` forces the frame out where presence itself carries meaning — a wrapper-array **element** is still always framed, even when all-default, because element presence is what carries a dynamic array's length (§5.1). |
| Typed | Fully type-annotated with a `py.typed` marker (PEP 561); clean under `mypy --strict`. |
| Forward/backward compatible | Unknown fields are consumed with `skip()`. |

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

### Serialize stream

`Encoder.over_buffer` writes into a small fixed scratch buffer and calls a flush
sink whenever it fills, so an arbitrarily large message streams out through bounded
memory:

```python
from sofab import Encoder

out = bytearray()                                            # or a socket / file write
enc = Encoder.over_buffer(bytearray(16), offset=0, flush=out.extend)  # tiny buffer
for i in range(1_000_000):
    enc.write_unsigned(i % 128, i)
enc.flush()                                                  # push the tail
```

### Deserialize

`Decoder` is a pull parser: `next()` returns one field header at a time; read the
value with a typed accessor, or `skip()` an unknown field:

```python
import io
from sofab import Decoder

dec = Decoder(io.BytesIO(data))
while (field := dec.next()) is not None:   # None == clean EOF
    if   field.id == 1: v = dec.unsigned()
    elif field.id == 2: v = dec.signed()
    elif field.id == 3: s = dec.string()
    else:               dec.skip()         # unknown field
```

### Deserialize stream

Hand `Decoder` any object with `read(n)` (a socket, `sys.stdin.buffer`,
`gzip.GzipFile`, …) and pull fields with `next()` as they arrive. It refills on
demand, so the same loop decodes correctly even when fed one byte at a time,
wherever the bytes come from:

```python
from sofab import Decoder

dec = Decoder(reader)                # any read(n) source: file, socket, pipe
while (field := dec.next()) is not None:
    ...                              # pull each field, or dec.skip()
```

**When the bytes have not all arrived yet.** A reader that can return `b""`
before end-of-message — a non-blocking socket, a queue fed by another task — puts
the decoder in the position CORELIB_PLAN §5.2 calls `INCOMPLETE`: the bytes stop
*inside* a field. That is not an error and not the end of the message; it means
"feed me more". Two shapes signal it, and both are **resumable — the suspended
call consumed nothing**, so the answer to either is to obtain more bytes and
issue the *same* call again:

* `next()` returns `None` — the bytes stopped exactly *between* fields (§5.2
  `COMPLETE`: a message may end here, and more fields may also still follow);
* `SofaIncompleteError` is raised — the bytes stopped *inside* a field header or
  payload, or inside a sequence that is still open.

```python
while True:
    try:
        field = dec.next()
    except SofaIncompleteError:
        feed_more(); continue        # partial field retained; re-issue next()
    if field is None:
        if stream_ended: break       # your framing decides; the decoder never does
        feed_more(); continue
    value = read_the_value(dec, field)   # same retry rule for the typed reads
```

Whether an incomplete message is acceptable is the **caller's** decision, not the
decoder's: only your framing (a length prefix, a datagram boundary, EOF) knows
whether more bytes can still come.

### Code generator

The most common real use is driving the library through **generated code**:
`sofabgen` emits a `@dataclass` per message with `encode` / `decode` methods that
call the primitives above. A hand-written stand-in, encoded then decoded:

```python
import io
from dataclasses import dataclass
from sofab import Encoder, Decoder

# generated by: sofabgen --lang python
@dataclass
class Point:
    x: int = 0
    y: int = 0

    def _marshal(self, e: Encoder) -> None:
        e.write_signed(1, self.x)
        e.write_signed(2, self.y)

    def encode(self) -> bytes:
        e = Encoder()
        self._marshal(e)
        return e.getvalue()

    @classmethod
    def decode(cls, data: bytes) -> "Point":
        o = cls()
        dec = Decoder(io.BytesIO(data))
        while (f := dec.next()) is not None:
            if   f.id == 1: o.x = dec.signed()
            elif f.id == 2: o.y = dec.signed()
            else:           dec.skip()          # tolerate unknown fields
        return o

wire = Point(x=3, y=4).encode()
got = Point.decode(wire)             # got.x == 3, got.y == 4
```

## Decode limits

Array counts and string/blob lengths are optional on the wire, so by default the
decoder allocates whatever a message declares. Untrusted input can abuse that, so
`Decoder` takes optional **receiver-side** caps that reject an oversize field at
header-decode time — *before* any allocation or payload buffering:

```python
dec = Decoder(reader, max_array_count=65536, max_string_len=1 << 20, max_blob_len=1 << 20)
```

A field whose declared count/length exceeds its cap raises `SofaLimitError`. That
is a *policy* rejection, distinct from malformed input: it is a sibling of
`SofaDecodeError` under `SofaError`, **not** a subclass, so `except
SofaDecodeError` does not catch it. Each limit defaults to `None` (no cap —
today's behaviour); the values are meant to be supplied by generated code, not
guessed by the runtime. Independent of any limit, the decoder never pre-allocates
from an untrusted array count — a truncated oversize claim fails promptly as
`SofaIncompleteError` rather than attempting a huge allocation.

A **schema** bound is the opposite kind of thing: it is part of the message
definition, so breaching it is malformed input, not policy. The integer-array
reads take the declared element width for exactly that reason —
`read_unsigned_array(255)` for a `u8` array, `read_signed_array(-128, 127)` for
an `i8` one (either half may be given alone; the other side stays open). An
element outside the declared width raises `SofaDecodeError` the moment its own
bytes are decoded, so the verdict never depends on how much of the array
followed it or on which engine read it (MESSAGE_SPEC §7.1). Omit the argument
for `u64`/`i64`, whose range is the value domain, or for an unbounded consumer.

## Memory handling

The key point for Python: **the library allocates results for you — the caller
never provides a value buffer.**

* **Decode: a suspended call keeps its bytes, and only its bytes.** Everything
  the reader hands over is retained, so a field split across chunks is never
  half-consumed; the buffer's consumed prefix is dropped on the next refill,
  down to the first byte of the call in flight — that byte is the one a resumed
  call re-reads from. The window held is therefore one field (for a `skip()`
  over a sequence, one sequence), not one message.
* **Decode.** `Decoder` keeps a single internal buffer, refilled from the
  `read(n)` source and never handed out, so there is **no zero-copy aliasing**:
  `string()` returns a fresh `str`, `bytes()` independent `bytes`, scalars a
  fresh `int`/`float`, and arrays a new `list` — every result stays valid after
  the decoder advances. `fixlen_len()` peeks the current string/blob field's
  exact wire byte length **without** consuming it, so a caller can bound the
  field against a schema `maxlen` before reading — no re-encoding a decoded
  `str` just to measure it.
* **Encode.** Two ownership models. The default `Encoder()` / `Encoder(writer)`
  owns a growable `bytearray` — `getvalue()` hands back a copy, or `flush()`
  drains to the writer. `Encoder.over_buffer` is caller-owned and bounded: you
  provide a fixed `bytearray`, it writes in place via a `memoryview` and flushes
  to the sink + reuses the buffer when full.
* **The start offset belongs to the installation, not to the buffer.** A flush
  sink states what it did by what it does before returning. Returning **without**
  installing anything means it *copied* the bytes it was handed: the same buffer
  stays active and encoding resumes at offset 0. A sink that *takes* the buffer —
  queues it for an async write, hands it to a transport — **must** install a
  replacement with `buffer_set(buf, offset)` before it returns, and that call's
  `offset` is where encoding resumes. Re-installing is therefore how a sink gets
  fresh framing-header room in *every* flushed packet (one header per packet),
  including when it passes the **same** buffer back: a bare return would reserve
  nothing, since the offset is consumed by the installation that carried it.
* **Sequence framing is lazy.** `write_sequence_begin_lazy(id)` pushes the id onto
  a pending run and writes nothing; the first field write inside commits the whole
  run, outermost header first. `write_sequence_end()` then drops a sequence that
  never got content — header *and* end marker — which is exactly MESSAGE_SPEC §2's
  "omit a sequence-typed field equal to its declared default", since generated
  code already omits every child equal to *its* default. Close with
  `write_sequence_end_keep()` wherever the frame carries information regardless of
  its contents: a wrapper-array **element** (element presence is what carries a
  dynamic array's length, §5.1) or an array field that must encode as explicitly
  empty against a non-empty declared default. The two mistakes are not
  symmetric — `end_keep` where `end` would do costs one non-canonical empty frame
  every decoder normalizes away, while the reverse changes a decoded array's
  length — so `end_keep` is the safe choice when a call site is ambiguous. The
  pending run grows on demand, so the hold-back reaches the full `MAX_DEPTH` and
  every nesting depth is canonical, and it is allocated on the first hold-back —
  an encoder that never opens a sequence never pays for it. The pending ids are
  encoder state, never buffer content, so a flush cannot split a run *by
  construction*: a held-back header takes no buffer space, and the buffer only
  fills through a write, which commits the run before its first byte. A tiny
  output buffer therefore yields exactly the one-shot bytes.

## Native accelerator

`Encoder` / `Decoder` / `Field` are re-exported from the compiled
`sofab._speedups` extension when present, and from the pure-Python
`encoder.py` / `decoder.py` otherwise. The native core is a small Cython
implementation of the same algorithm — one contiguous buffer, an advancing
cursor, bulk `memcpy`, and varint/zigzag compiled to C. Both engines import wire
constants, enums and exception classes from the shared `types.py`, so a
`SofaRangeError` is the *same class* from either, and the two produce
byte-for-byte identical output (enforced by `tests/test_native_parity.py`).

The active engine is reported by `sofab.IMPL` (`"native"` or `"python"`):

```python
import sofab
print(sofab.IMPL)        # "native" when the compiled extension is loaded
```

Force the pure-Python path with `SOFAB_PUREPYTHON=1`.

## Feature flags

The package always builds the full format (unsigned / signed varints, fp32 /
fp64, strings, blobs, arrays and nested sequences). The one build toggle is
`SOFAB_DISABLE_NATIVE=1`, which builds a native-free (pure-Python) distribution;
it changes only speed, never the wire format or the public API.

## Build & test

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest ruff mypy   # compiles the native accelerator if a C compiler is present
pytest                       # vectors + roundtrip + streaming + malformed + native↔pure parity
ruff check src/sofab tests   # lint
mypy --strict src/sofab      # type-check
```

If the compile fails or no compiler is available, the install falls back to
pure-Python (the extension is marked *optional* in `setup.py`). To exercise both
engines:

```bash
pytest                       # whichever engine is active (native if built)
SOFAB_PUREPYTHON=1 pytest    # force the pure-Python engine
```

## Benchmarks

`bench/perfbench.py` runs the standard workloads; `bench/compare_protobuf.py`
compares the native accelerator, the pure-Python fallback, and (for a yardstick)
`protobuf`'s Python runtime (upb C backend), with full materialization on both
sides so it is apples-to-apples with the SofaBuffers pull API:

```bash
python bench/perfbench.py time            # throughput on this machine, MB/s (MB = 1e6)
python bench/perfbench.py perf            # per-op cost for the shared 12-field message
bash  bench/run_callgrind.sh              # instructions/op (Callgrind) — clock-independent
pip install protobuf                      # optional; the column is dropped if absent
python bench/compare_protobuf.py          # best-of-5 MB/s table
```

`time` / `perf` measure this machine and move with its load; `run_callgrind.sh`
counts instructions retired, which is deterministic and comparable across hosts,
so it is the one to trust when judging a change to the library itself.

Representative result (throughput MB/s, higher is better; one x86-64 host,
CPython 3.12 — the *ratios* are the point):

| Workload | sofab **native** | sofab pure | protobuf (upb) | native vs protobuf |
|----------|-----------------:|-----------:|---------------:|:------------------:|
| encode: u64 array (1000) | **≈840** | ≈11 | ≈160 | **≈5× faster** |
| encode: typical message  | **≈18**  | ≈4.4 | ≈10 | **≈1.7× faster** |
| decode: u64 array (1000) | **≈460** | ≈7.8 | ≈195 | **≈2.4× faster** |
| decode: typical message  | ≈9.2     | ≈2.1 | ≈9.0 | ≈1.0× (see note) |

The native accelerator is **~4× faster than the pure-Python fallback on a small
mixed message and ~60–75× on array-heavy ones**, and beats protobuf everywhere
except the smallest decode, where the two are level. That last workload is where
the streaming **pull** API costs the most: it crosses the Python↔C boundary twice
per field (`next()` then a typed read), whereas protobuf parses the whole message
in one C call — an inherent pull-vs-parse-tree trade-off that only shows on very
small messages, and the price of never having to hold one in memory.
