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
serialization format — a compact, TLV-like binary format. It is the **runtime
stream core** (the Python equivalent of the C `corelib`'s `istream` / `ostream`),
meant to be driven by **generated code**: a schema-driven code generator emits
one class per message plus `marshal` / `unmarshal` methods that call the
`Encoder` / `Decoder` primitives here — the same way protobuf's generated Python
code calls its runtime.

**Fast where it can be, portable everywhere.** The public API is a single pair of
classes with two interchangeable engines behind it, selected once at import: the
hot path (varint / zigzag / buffer management) ships as an optional compiled
**native accelerator** (Cython → C, `sofab._speedups`) that is loaded
automatically when present, with a **pure-Python fallback** used verbatim when it
is not. The two are byte-for-byte interchangeable — same public API, validated by
the same conformance vectors — so the library runs *anywhere CPython runs*, with
or without a C compiler, and generated code never has to care which engine is
active. See [API summary](#api-summary) for the accelerator details.

The wire format is specified, language-neutrally, in the
[SofaBuffers documentation](https://github.com/sofa-buffers/documentation). The
unit tests here validate against the shared, language-agnostic conformance suite
(`assets/test_vectors.json`, copied verbatim from that repo) to guarantee
byte-for-byte interoperability with the C, C++, Rust, Go, Java and C#
implementations.

**Requirements.** Python 3.9 or newer (CPython or PyPy); CI runs 3.9–3.13.
Building the optional native accelerator additionally needs a C compiler and
Cython — both are **build-time only** and entirely optional; without them `pip`
installs a working pure-Python build.

**Dependencies.** None at runtime — the pure-Python path uses only the standard
library (`struct`, `io`). The only third-party build dependency is `Cython`
(declared as a [PEP 517](https://peps.python.org/pep-0517/) build requirement),
pulled in to compile the accelerator and never imported at runtime.

**Package name.** Distribution `sofa-buffers-corelib` on PyPI; import package
`sofab`.

```bash
pip install sofa-buffers-corelib
```

```python
import sofab   # Encoder, Decoder, Visitor, wire-format types and limits
```

## Why this design

| Goal | How |
|------|-----|
| Streaming **out** | `Encoder` writes to any binary stream (file, socket, `BytesIO`), so a message can exceed RAM and stream straight to the wire. `Encoder.over_buffer` drains a small caller buffer through a flush sink when it fills. |
| Streaming **in** | `Decoder` is a pull parser over any `read(n)` reader; `next()` returns one field header at a time, never materializing the whole message. Large string / blob / array payloads are read in bulk. |
| Native speed, zero runtime deps | The varint / buffer hot path ships as an optional compiled Cython accelerator (`sofab._speedups`); when it can't be built it falls back to pure standard-library Python. No *runtime* third-party dependencies either way. |
| Runs everywhere | If there is no C compiler and no prebuilt wheel, `pip` still installs a working pure-Python-only build (`py3-none-any`). The native and pure paths are byte-for-byte identical, so falling back never changes behaviour — only speed. |
| Sticky errors | The encoder can record the first failure and turn later writes into no-ops (`Encoder(sticky=True)`), so generated `marshal` code can issue a run of writes and check `enc.error` once. |
| Reserve-offset | `Encoder.over_buffer(buf, offset=…)` leaves room at the front of the buffer for a lower-layer protocol header (saves a copy). |
| Typed | Fully type-annotated and ships a `py.typed` marker (PEP 561); clean under `mypy --strict`. |
| Forward/backward compatible | Unknown fields are consumed with `skip()` — old readers tolerate new fields, new readers tolerate missing ones. |
| 64-bit value type | Matches the C default configuration, so varint lengths and bytes are identical across languages. |

## Usage

### Simple encode / decode

```python
import io
from sofab import Encoder, Decoder

# ---- encode ----
enc = Encoder()
enc.write_unsigned(1, 42)
enc.write_signed(2, -7)
enc.write_string(3, "hi")
enc.write_unsigned_array(4, [10, 20, 30])
data = enc.getvalue()

# ---- decode (pull parser) ----
dec = Decoder(io.BytesIO(data))
while (field := dec.next()) is not None:   # None == clean EOF
    if   field.id == 1: v = dec.unsigned()
    elif field.id == 2: v = dec.signed()
    elif field.id == 3: s = dec.string()
    elif field.id == 4: a = dec.read_unsigned_array()
    else:               dec.skip()         # unknown field
```

### OStream — streaming encoder to a stream sink

An `Encoder` constructed with a *writer* (any object with `.write(bytes)`)
accumulates in an internal buffer and drains to the writer on `flush()`, so a
message streams straight to a file, socket or pipe:

```python
from sofab import Encoder

with open("out.sofab", "wb") as f:
    enc = Encoder(f)                 # writer = the file object
    enc.write_string(1, "hello")
    enc.write_float64(2, 3.14)
    enc.flush()                      # drains the buffer to f.write()
```

### Streaming a message larger than the buffer

`Encoder.over_buffer` writes into a small fixed scratch buffer and calls a flush
sink whenever it fills, so the "buffer" can be a socket, a pipe or a file —
nothing is held whole in memory (an arbitrarily large message streams out
through bounded memory):

```python
from sofab import Encoder

out = bytearray()                                            # or a socket / file write
enc = Encoder.over_buffer(bytearray(16), offset=0, flush=out.extend)  # tiny buffer
for i in range(1_000_000):
    enc.write_unsigned(i % 128, i)
enc.flush()                                                  # push the tail
```

### IStream — streaming / pull decoder

The decoder is symmetric: hand `Decoder` any object with `read(n)` (a socket,
`sys.stdin.buffer`, `gzip.GzipFile`, ...) and pull fields with `next()` as they
arrive. It refills from the reader on demand, so it decodes correctly even when
fed one byte at a time:

```python
from sofab import Decoder

with open("out.sofab", "rb") as f:
    dec = Decoder(f)                 # any read(n) source: file, socket, pipe
    while (field := dec.next()) is not None:
        ...                          # pull each field, or dec.skip()
```

For callback-style decoding, subclass `sofab.Visitor` and hand it to
`Decoder.drive(visitor)` (see [API summary](#api-summary)).

### Generated object code

The most common real use is driving the library through **generated code**: a
schema compiled by the generator emits a class per message whose `marshal` /
`unmarshal` methods call the primitives above. The pattern a generated class
follows:

```python
from sofab import Encoder, Decoder

class Point:                         # generated from a schema
    __slots__ = ("x", "y", "label")

    def marshal(self, enc: Encoder) -> None:
        enc.write_signed(1, self.x)
        enc.write_signed(2, self.y)
        enc.write_string(3, self.label)

    @classmethod
    def unmarshal(cls, dec: Decoder) -> "Point":
        obj = cls()
        while (f := dec.next()) is not None:
            if   f.id == 1: obj.x = dec.signed()
            elif f.id == 2: obj.y = dec.signed()
            elif f.id == 3: obj.label = dec.string()
            else:           dec.skip()          # tolerate unknown fields
        return obj
```

Generated `marshal` code typically constructs the encoder with `sticky=True` and
checks `enc.error` once after a run of writes, rather than wrapping each call.

## API summary

### Encoding

`Encoder` has two construction models, both first-class:

* `Encoder(writer=None, *, sticky=False)` — the in-memory / Go-style model. Bytes
  accumulate in an internal growable `bytearray`; `getvalue()` returns them, or
  `flush()` drains them to `writer.write()` when a writer is supplied. No size
  need be known up front.
* `Encoder.over_buffer(buf, offset=0, flush=None, *, sticky=False)` — the
  bounded caller-buffer / Rust-C-Java-style model. Writes in place into a fixed
  `bytearray`, reserving `offset` bytes at the front for a lower-layer header;
  when the buffer fills it is passed to the `flush` sink and reused.
  `buffer_set(buf, offset)` swaps in a fresh buffer mid-stream (typically from
  inside the sink), and `bytes_used()` reports how much of the current buffer is
  filled.

Values are written with **typed `write_*` methods** — one per logical type
(scalars, floats, strings, blobs and the four packed arrays; see below) — plus
`write_sequence_begin` / `write_sequence_end` to open and close nested
sub-messages. Each writer validates ranges and raises on error, unless the
encoder is in **sticky mode** (`sticky=True`), which latches the first failure in
`enc.error` and turns subsequent writes into no-ops — so generated code can issue
a run of writes and check once. `flush()` returns the number of bytes drained.

### Decoding

Decoding is a **pull loop**: `next()` returns the next `Field` header (or `None`
at clean EOF, and `SEQUENCE_START` / `SEQUENCE_END` markers for nested
sub-messages), then exactly one matching typed accessor consumes the value —
`unsigned()`, `signed()`, `bool()`, `float32()`, `float64()`, `string()`,
`bytes()`, or one of the `read_*_array()` methods. Each accessor checks the
pending field's wire type and raises `SofaStateError` on a mismatch; the current
header is also available via the `field` property. `skip()` discards the current
value — or, positioned on a `SEQUENCE_START`, the entire nested sub-tree in one
call — which is how unknown or unwanted fields are tolerated. **Sequences** are
descended by continuing the pull loop until the matching `SEQUENCE_END` marker
(depth is tracked for you), or dropped whole with `skip()`.

An **object / visitor** alternative is layered on the same hot path:
`Decoder.drive(visitor)` pulls the whole stream and dispatches each field to the
typed hooks of a `sofab.Visitor` subclass (`on_unsigned`, `on_signed`,
`on_float32`, `on_float64`, `on_string`, `on_bytes`, the four `on_*_array` hooks,
plus `on_sequence_begin` / `on_sequence_end`). All value hooks default to a
no-op that still consumes the field, so unhandled fields are skipped safely. The
control hooks `on_field` and `on_sequence_begin` may return `False` to decline a
field or a whole sub-tree *before* it is decoded, so skipping a large array or a
deep sequence costs nothing.

### Supported types

Python has no generics, so each logical type is a distinct method rather than a
template parameter:

| Logical type | Encode | Decode |
|--------------|--------|--------|
| Unsigned integer (uint64) | `write_unsigned` | `unsigned` |
| Signed integer (int64, zigzag) | `write_signed` | `signed` |
| Boolean (unsigned `0`/`1`) | `write_bool` | `bool` |
| 32-bit float | `write_float32` | `float32` |
| 64-bit float | `write_float64` | `float64` |
| String (UTF-8) | `write_string` | `string` |
| Blob (raw bytes) | `write_bytes` | `bytes` |
| Unsigned array | `write_unsigned_array` | `read_unsigned_array` |
| Signed array | `write_signed_array` | `read_signed_array` |
| fp32 array | `write_float32_array` | `read_float32_array` |
| fp64 array | `write_float64_array` | `read_float64_array` |

The four `*_array` writers accept any `Iterable`, and `write_bytes` accepts
`bytes`, `bytearray` or `memoryview`. **Fixlen arrays carry only fixed-width
element subtypes (fp32 / fp64)** — there is no `write_string_array` /
`write_bytes_array`, and a fixlen-array header whose subtype is not FP32/FP64 is
rejected with `SofaDecodeError`; model a list of strings or blobs as a `SEQUENCE`
of individual fields instead. Integer values out of the uint64 / int64 range, ids
above `ID_MAX`, and array counts outside `0 .. ARRAY_MAX` raise `SofaRangeError`
on encode (a zero-count array is a valid, fully-specified empty array on the
wire).

The module also exports the wire-format enums `WireType` and `FixlenSubtype`, the
`Field` descriptor, the error classes `SofaError` (base), `SofaDecodeError`,
`SofaRangeError`, `SofaStateError` and `SofaBufferError`, the `zigzag_encode` /
`zigzag_decode` helpers, and the limits `API_VERSION` (currently `1`), `ID_MAX`,
`ARRAY_MAX`, `FIXLEN_MAX`, `MAX_DEPTH` (255), `UNSIGNED_MAX`, `SIGNED_MIN` and
`SIGNED_MAX`.

> **Note on value width:** like the C default configuration, the scalar value
> type is 64-bit, so varint encodings match byte-for-byte across the C, C++,
> Rust, Go, Java, C# and Python implementations.

### Memory handling

This is where the Python port differs most from the C / embedded ports, where the
caller hands the library fixed destination buffers. **In Python the library
allocates the result objects for you** — the caller never pre-allocates a value
buffer.

* **Input buffer (decode).** `Decoder` keeps a single contiguous internal buffer
  that it refills from the `read(n)` source and advances a cursor over; that
  buffer is an implementation detail and is never handed out. There is **no
  zero-copy aliasing** into it: `string()` returns a freshly decoded `str`,
  `bytes()` an independent immutable `bytes`, scalars a fresh `int`/`float`, and
  arrays a brand-new `list` — every result stays valid after the decoder advances
  or the input is reused. You decode by *receiving* values, not by providing
  storage for them.
* **Output buffer (encode).** Two ownership models. The default `Encoder()` /
  `Encoder(writer)` owns a growable `bytearray`; `getvalue()` hands back one copy
  of the accumulated bytes, or `flush()` drains to the writer. `Encoder.over_buffer`
  is caller-owned and bounded: you provide a fixed `bytearray`, the encoder wraps
  it in a `memoryview` and writes in place, flushing to the sink and reusing the
  buffer when full — the only `memoryview` in the API is this internal write
  target, never a value handed back to you.
* **Message object.** Generated message classes are plain Python objects owning
  their own fields; `marshal` writes their values into an encoder and `unmarshal`
  reads fresh values out of a decoder, so a decoded message shares no storage with
  the input bytes and outlives the decoder.

### Native accelerator

`Encoder` / `Decoder` / `Field` are re-exported from the compiled
`sofab._speedups` extension when it is present, and from the pure-Python
`encoder.py` / `decoder.py` otherwise. The native core is a small Cython
translation of the *same* pointer-advance algorithm the
[`corelib-cpp`](https://github.com/sofa-buffers/corelib-cpp) implementation uses
(one contiguous buffer, an advancing cursor, bulk `memcpy`, varint/zigzag in C) —
**not** a binding to the footprint-optimized bare-metal C core. Both engines
import the wire constants, enums and exception classes from the shared
`types.py`, so a `SofaRangeError` from the native encoder is the *same class* the
pure one raises, and the two produce byte-for-byte identical output (enforced by
`tests/test_native_parity.py`).

Which engine is active is reported by `sofab.IMPL` (`"native"` or `"python"`):

```python
import sofab
print(sofab.IMPL)        # "native" when the compiled extension is loaded
```

Force the pure-Python path (for debugging or A/B checks) with the environment
variable `SOFAB_PUREPYTHON=1`.

## Feature flags

The C library exposes compile-time `SOFAB_DISABLE_*` switches (fixlen, fp64,
array, sequence, overflow checks) to strip whole code paths for tiny
microcontrollers. **Python has no equivalent** — the package always builds the
full format (unsigned / signed varints, fp32 / fp64, strings, blobs, arrays and
nested sequences), since the desktop, server and cloud targets it runs on are not
code-size constrained.

The one build toggle is `SOFAB_DISABLE_NATIVE=1`, which builds a native-free
(pure-Python) distribution; it changes only *speed*, never the wire format or the
public API.

## Build & test

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest ruff mypy   # compiles the native accelerator if a C compiler is present
pytest                       # vectors + roundtrip + streaming + malformed + native↔pure parity
ruff check src/sofab tests   # lint
mypy --strict src/sofab      # type-check
```

Building the native accelerator needs only a C compiler; **Cython** is pulled in
automatically as a build-time dependency and is never imported at runtime. If the
compile fails or no compiler is available, the install silently falls back to
pure-Python (the extension is marked *optional* in `setup.py`). To exercise both
engines locally:

```bash
pytest                       # runs against whichever engine is active (native if built)
SOFAB_PUREPYTHON=1 pytest    # force the pure-Python engine
```

`tests/test_native_parity.py` asserts the native and pure engines produce
byte-identical output and cross-decode each other; it is skipped automatically
when the extension is not built. The tests are split by concern, with the
conformance suite validating against the shared, language-agnostic vectors copied
verbatim from the `documentation` repo:

- `test_conformance_vectors.py` — every vector in `assets/test_vectors.json`, encode (byte-exact) **and** decode
- `test_vectors_ostream.py` / `test_vectors_istream.py` — extra encoder/decoder coverage incl. the full-scale example
- `test_roundtrip.py` — encode → decode value preservation (scalars, arrays, strings/blobs, sequences, boundary values)
- `test_streaming.py` — 1-byte-granularity decode + tiny-scratch-buffer encode match the one-shot path
- `test_malformed.py` — malformed-input decode errors + encoder range / state errors + sticky mode
- `test_visitor.py` — the `Decoder.drive(Visitor)` path against the pull loop
- `test_varint.py` — varint / zigzag codec

CI (`.github/workflows/ci.yml`) runs the suite on Python 3.9–3.13, plus lint and
`mypy --strict`. Coverage is measured against the pure-Python engine on every run
on `main` and reported by the **coverage** badge (updated via the `badges`
branch). API documentation is built with **Sphinx** and published to GitHub Pages
by `docs.yml` on every push to `main` (the **Docs** badge links to it).

## Benchmarks

`bench/perfbench.py` mirrors the C / C++ / Rust / Go corelib benchmarks — same
messages, workloads, ids and values — so the implementations can be compared
directly:

```bash
python bench/perfbench.py time            # throughput on this machine, MB/s (MB = 1e6)
python bench/perfbench.py encode_typical  # run one named workload (for the Callgrind harness)
```

`bench/compare_protobuf.py` runs the workloads against the native accelerator,
the pure-Python fallback, and — for an external yardstick — `protobuf`'s Python
runtime (upb C backend) on an equivalent message, with **full materialization**
on both sides so it is apples-to-apples with the SofaBuffers pull API:

```bash
pip install protobuf                      # optional; the column is dropped if absent
python bench/compare_protobuf.py          # best-of-5 MB/s table
```

Representative result (throughput MB/s, higher is better; one x86-64 host,
CPython 3.12 — absolute numbers vary by machine, the *ratios* are the point; see
`PERFORMANCE.md`):

| Workload | sofab **native** | sofab pure | protobuf (upb) | native vs protobuf |
|----------|-----------------:|-----------:|---------------:|:------------------:|
| encode: u64 array (1000) | **≈300** | ≈9 | ≈190 | **≈1.6× faster** |
| encode: typical message  | **≈14**  | ≈4.5 | ≈10 | **≈1.5× faster** |
| decode: u64 array (1000) | **≈285** | ≈6.7 | ≈180 | **≈1.6× faster** |
| decode: typical message  | ≈7.3     | ≈2.0 | ≈8.6 | ≈0.85× (see note) |

The native accelerator is **~15–45× faster than the pure-Python fallback** and
beats protobuf's Python runtime on encode and on array-heavy decode. The one
workload where protobuf edges ahead is decoding a *tiny* (36-byte) mixed message:
SofaBuffers' streaming **pull** API crosses the Python↔C boundary twice per field
(`next()` then a typed read), whereas protobuf parses the whole message in a
single C call. That is an inherent pull-vs-parse-tree trade-off, it only shows on
very small messages, and the pull API is what lets SofaBuffers decode a stream
larger than RAM. For any message with arrays or a non-trivial payload, the native
accelerator wins comfortably.

For a **CPU-speed-independent** cost metric, `bench/run_callgrind.sh` runs each
workload under Callgrind and reports **instructions retired per operation** —
deterministic and comparable across machines (and against the other corelibs):

```bash
bash bench/run_callgrind.sh               # needs valgrind
```

Because the workloads are Python functions rather than C symbols, the script runs
each at two rep counts and subtracts the instruction counts, cancelling
interpreter startup and one-time setup to isolate the per-operation cost.
