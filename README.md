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

A **streaming**, **pure-Python**, **dependency-free** implementation of the
SofaBuffers (*Sofab*) serialization format — a compact, TLV-like binary format.
It is the **runtime stream core** (equivalent to the C `corelib`'s `istream` /
`ostream`), meant to be driven by **generated code**: a schema-driven code
generator emits one class per message plus `marshal` / `unmarshal` methods that
call the primitives here, the same way protobuf's generated Python code calls its
runtime.

The wire format is specified, language-neutrally, in the
[SofaBuffers documentation](https://github.com/sofa-buffers/documentation). The
unit tests here validate against the shared, language-agnostic conformance suite
(`assets/test_vectors.json`, copied verbatim from that repo) to guarantee
byte-for-byte interoperability with the C, C++, Rust, Go and Java
implementations.

Distribution: `sofabuffers` · import package `sofab`. Requires Python 3.9+.

```bash
pip install sofabuffers
```

## Why this design

| Goal | How |
|------|-----|
| Streaming **out** | [`Encoder`] writes to any binary stream (file, socket, `BytesIO`), so a message can exceed RAM and stream straight to the wire. `Encoder.over_buffer` drains a small caller buffer through a flush sink when it fills. |
| Streaming **in** | [`Decoder`] is a pull parser over any `read(n)` reader; `next()` returns one field header at a time, never materializing the whole message. Large string / blob / array payloads are read in bulk. |
| Pure Python, no dependencies | Standard library only (`struct`, `io`). No third-party modules, no build step — a single universal wheel that runs on CPython and PyPy. |
| Accelerator-ready | The varint / zigzag / IEEE-754 hot path lives behind `_varint` / `_core`, so a native backend (mypyc / Cython, or PyO3 over `corelib-rs`) can be dropped in later **without any public API change**. |
| Sticky errors | The encoder can record the first failure and turn later writes into no-ops (`Encoder(sticky=True)`), so generated `marshal` code can issue a run of writes and check `enc.error` once. |
| Reserve-offset | `Encoder.over_buffer(buf, offset=…)` leaves room at the front of the buffer for a lower-layer protocol header (saves a copy). |
| Typed | Fully type-annotated and ships a `py.typed` marker (PEP 561); clean under `mypy --strict`. |
| Forward/backward compatible | Unknown fields are consumed with `skip()` — old readers tolerate new fields, new readers tolerate missing ones. |
| 64-bit value type | Matches the C default configuration, so varint lengths and bytes are identical across languages. |

## Usage

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
while (field := dec.next()) is not None:   # None == EOF
    if   field.id == 1: v = dec.unsigned()
    elif field.id == 2: v = dec.signed()
    elif field.id == 3: s = dec.string()
    elif field.id == 4: a = dec.read_unsigned_array()
    else:               dec.skip()         # unknown field
```

### Streaming a message larger than the buffer

`Encoder.over_buffer` writes into a small scratch buffer and calls a flush sink
whenever it fills, so the "buffer" can be a socket, a pipe, or a file — nothing
is held whole in memory:

```python
from sofab import Encoder

out = bytearray()                                    # or a socket / file write
enc = Encoder.over_buffer(bytearray(16), offset=0, flush=out.extend)  # tiny buffer
for i in range(1_000_000):
    enc.write_unsigned(i % 128, i)
enc.flush()                                          # push the tail
```

The decoder is symmetric: hand `Decoder` any object with `read(n)` (socket,
`sys.stdin.buffer`, `gzip.GzipFile`, ...) and pull fields with `next()` as they
arrive — it works correctly even when fed one byte at a time.

## API summary

**Encoder** — constructors `Encoder(writer=None, *, sticky=False)` and
`Encoder.over_buffer(buf, offset=0, flush=None, *, sticky=False)`; methods
`write_unsigned`, `write_signed`, `write_bool`, `write_float32`, `write_float64`,
`write_string`, `write_bytes`, `write_unsigned_array`, `write_signed_array`,
`write_float32_array`, `write_float64_array`, `write_sequence_begin` /
`write_sequence_end`, `bytes_used`, `flush`, `getvalue`, and the `error`
property.

`Encoder.buffer_set(buf, offset=0)` installs a fresh output buffer mid-stream
(typically from inside the flush sink).

**Decoder** — `next`, `field`, `unsigned`, `signed`, `bool`, `float32`,
`float64`, `string`, `bytes`, `read_unsigned_array`, `read_signed_array`,
`read_float32_array`, `read_float64_array`, `skip`, and `drive` (visitor).

**Module** — `API_VERSION` (currently `1`), and the limits `ID_MAX`,
`ARRAY_MAX`, `UNSIGNED_MAX`, `SIGNED_MIN`, `SIGNED_MAX`.

> **Note on value width:** like the C default configuration, the scalar value
> type is 64-bit, so varint encodings match byte-for-byte across the C, C++,
> Rust, Go, Java and Python implementations.

### Read operations

Decoding is a two-step pull: `next()` returns a `Field` header (or `None` at
clean EOF), then exactly one matching `read` accessor hands you the value. Each
accessor checks that the pending field's wire type matches the read you ask for
and raises `SofaStateError` otherwise; calling `next()` again skips any value
you did not consume. Every accessor **returns a freshly built Python object** —
there is no caller-supplied destination buffer (see *Memory handling*).

| Method | Returns | For wire type |
|--------|---------|---------------|
| `next() -> Field \| None` | next field header, or `None` at EOF | (any) — also yields `SEQUENCE_START` / `SEQUENCE_END` markers |
| `field -> Field \| None` (property) | the most recently returned `Field` | (any) |
| `unsigned() -> int` | a non-negative `int` (full uint64 range) | `UNSIGNED` |
| `signed() -> int` | a zigzag-decoded `int` (int64 range) | `SIGNED` |
| `bool() -> bool` | `True`/`False` (an unsigned `0`/`1` on the wire) | `UNSIGNED` |
| `float32() -> float` | a Python `float` widened from IEEE-754 binary32 | `FIXLEN` / `FP32` |
| `float64() -> float` | a Python `float` from IEEE-754 binary64 | `FIXLEN` / `FP64` |
| `string() -> str` | a freshly decoded UTF-8 `str` (copy) | `FIXLEN` / `STRING` |
| `bytes() -> bytes` | a fresh, immutable `bytes` (copy) | `FIXLEN` / `BLOB` |
| `read_unsigned_array() -> list[int]` | a new `list[int]` | `ARRAY_UNSIGNED` |
| `read_signed_array() -> list[int]` | a new `list[int]` (zigzag-decoded) | `ARRAY_SIGNED` |
| `read_float32_array() -> list[float]` | a new `list[float]` | `ARRAY_FIXLEN` / `FP32` |
| `read_float64_array() -> list[float]` | a new `list[float]` | `ARRAY_FIXLEN` / `FP64` |
| `skip() -> None` | consumes the current value, or the **whole** nested sequence if positioned on a `SEQUENCE_START` | (any) |

**Sequences** are descended directly via the pull loop: when `next()` returns a
`SEQUENCE_START` field, keep calling `next()` to read the children until the
matching `SEQUENCE_END` marker (depth is tracked for you), or call `skip()` on
the start to drop the entire sub-tree in one go.

**Visitor driver.** `Decoder.drive(visitor)` pulls the whole stream and
dispatches each field to the typed hooks of a `sofab.Visitor` subclass
(`on_unsigned`, `on_signed`, `on_float32`, `on_float64`, `on_string`,
`on_bytes`, `on_unsigned_array`, `on_signed_array`, `on_float32_array`,
`on_float64_array`, plus `on_sequence_begin` / `on_sequence_end`). All hooks
default to a no-op that still consumes the value, so unhandled fields are
skipped safely. The control hooks `on_field` and `on_sequence_begin` can return
`False` to decline a field or an entire sub-tree *before* it is decoded, so
skipping a large array or deep sequence costs nothing.

### Supported types

Python has no generics, so the supported value types are spelled out as
distinct methods rather than template parameters. The complete matrix:

| Logical type | Encode | Decode | Notes |
|--------------|--------|--------|-------|
| Unsigned integer | `write_unsigned` | `unsigned` | 0 .. `UNSIGNED_MAX` (uint64); base-128 varint |
| Signed integer | `write_signed` | `signed` | `SIGNED_MIN` .. `SIGNED_MAX` (int64); zigzag + varint |
| Boolean | `write_bool` | `bool` | encoded as an unsigned `0`/`1` |
| 32-bit float | `write_float32` | `float32` | IEEE-754 binary32 (fixlen, FP32 subtype) |
| 64-bit float | `write_float64` | `float64` | IEEE-754 binary64 (fixlen, FP64 subtype) |
| String | `write_string` | `string` | UTF-8 bytes (fixlen, STRING subtype) |
| Blob | `write_bytes` | `bytes` | raw bytes (fixlen, BLOB subtype) |
| Unsigned array | `write_unsigned_array` | `read_unsigned_array` | `list[int]` of uint64 varints |
| Signed array | `write_signed_array` | `read_signed_array` | `list[int]` of zigzag varints |
| fp32 array | `write_float32_array` | `read_float32_array` | `list[float]`, packed binary32 |
| fp64 array | `write_float64_array` | `read_float64_array` | `list[float]`, packed binary64 |

The four `*_array` writers accept any `Iterable`, and `write_bytes` accepts
`bytes`, `bytearray`, or `memoryview`. Per-element range checks apply to the
integer arrays exactly as to the scalar writers.

**What is disallowed.** Fixlen arrays carry **only fixed-width element subtypes
— fp32 or fp64**. Dynamically sized subtypes (string, blob) are *not* permitted
as fixlen-array elements: on encode there is no `write_string_array` /
`write_bytes_array`, and on decode a fixlen-array header whose subtype is not
FP32/FP64 is rejected with `SofaDecodeError`. Model a list of strings or blobs
as a `SEQUENCE` of individual string/blob fields instead. Integer-array values
outside the uint64 / int64 range, ids above `ID_MAX`, and array counts outside
`1 .. ARRAY_MAX` raise `SofaRangeError` on encode.

### Memory handling

This is the part that differs most from the C / embedded ports, where the
caller hands the library fixed destination buffers. **In Python the library
allocates the result objects for you — the caller never pre-allocates a value
buffer.**

**Decoder — the library allocates every result.** Each `read` accessor returns
a *newly constructed* Python object: a fresh `int` / `float` for scalars, a
freshly decoded `str` for strings, an immutable `bytes` for blobs, and a brand
new `list` for arrays. There is **no bind-target / caller-buffer mode** and **no
zero-copy aliasing** into the input — `string()` and `bytes()` return an
independent copy, never a `memoryview` over the decoder's internal buffer, so
the value stays valid after the decoder advances or the input is reused. (The
decoder keeps a single contiguous internal buffer that it refills from the
reader and advances a cursor over; that buffer is an implementation detail and
is never handed out.) Practically: you decode by *receiving* values, not by
providing storage for them.

**Encoder — allocates by default, or drains a caller buffer.** Two ownership
models, both first-class:

* **Library-owned (default).** `Encoder()` / `Encoder(writer)` grows an internal
  `bytearray` as you write. `getvalue()` returns the accumulated bytes (one
  copy) for the in-memory model; `flush()` drains the buffer to the `writer`'s
  `.write()` when one is supplied. No size has to be known up front. The
  in-memory varint path appends straight into the `bytearray` with no
  intermediate `bytes` object.
* **Caller-owned, bounded (`Encoder.over_buffer(buf, offset, flush)`).** You
  provide a fixed-size `bytearray`; the encoder wraps it in a `memoryview` and
  writes **in place**, reserving `offset` bytes at the front for a lower-layer
  header. When the buffer fills it is passed to the `flush` sink (a socket /
  file / `bytearray.extend`, ...) and reused, so an arbitrarily large message
  streams out through bounded memory — nothing is held whole. `buffer_set(buf,
  offset)` swaps in a fresh buffer mid-stream (typically from inside the sink),
  and `bytes_used()` reports how much of the current buffer is filled.

The only `memoryview` in the API is this internal in-place wrapper around the
caller's encode buffer; it is a write target, not a value handed back to you.

## Layering vs. the C library

| C file | Python module | Status |
|--------|---------------|--------|
| `sofab.h` (types / constants) | `types.py` (`WireType`, `FixlenSubtype`, `Field`, errors, limits) | ported |
| `ostream.c` | `encoder.py` ([`Encoder`]) | ported |
| `istream.c` | `decoder.py` ([`Decoder`]) | ported (pull-parser model instead of bind-target callbacks) |
| `object.c` (descriptor transcoder) | — | not ported. The idiomatic Python equivalent is generated message classes from a schema-driven generator; the streaming core above already covers serialize / deserialize. |
| — | `_varint.py` / `_core.py` | varint / zigzag + IEEE-754 helpers, isolated as the hot path so a native accelerator can replace them without an API change. |

## Feature flags / build options

The C library exposes compile-time `SOFAB_DISABLE_*` switches (fixlen, fp64,
array, sequence, overflow checks) to strip whole code paths for tiny
microcontrollers. **Python has no equivalent** — the package always builds the
full format (unsigned / signed varints, fp32 / fp64, strings, blobs, arrays and
nested sequences), since the desktop, server and cloud targets it runs on are
not code-size constrained. The scalar value type is 64-bit, matching the C
default configuration so the wire image and varint lengths are identical.

## Build & test

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest ruff mypy
pytest                       # vectors + roundtrip + streaming + malformed
ruff check src/sofab tests   # lint
mypy --strict src/sofab      # type-check
```

Tests are split by concern; the conformance suite validates against the shared,
language-agnostic vectors copied verbatim from the `documentation` repo:

- `test_conformance_vectors.py` — every vector in `assets/test_vectors.json`, encode (byte-exact) **and** decode
- `test_vectors_ostream.py` / `test_vectors_istream.py` — extra encoder/decoder coverage incl. the full-scale example
- `test_roundtrip.py` — encode → decode value preservation (scalars, arrays, strings/blobs, sequences, boundary values)
- `test_streaming.py` — 1-byte-granularity decode + tiny-scratch-buffer encode match the one-shot path
- `test_malformed.py` — malformed-input decode errors + encoder range / state errors + sticky mode
- `test_varint.py` — varint / zigzag codec

Coverage is measured on every CI run on `main` and reported by the **coverage**
badge above (updated automatically via the `badges` branch). API documentation
is built with **Sphinx** (`sphinx-apidoc` + the HTML builder) and published to
GitHub Pages on every push to `main` (the **docs** badge links to it).

## Benchmarks

`bench/perfbench.py` mirrors the C / C++ / Rust / Go corelib benchmarks — same
messages, workloads, ids and values — so the implementations can be compared
directly. Two complementary views:

```bash
python bench/perfbench.py time            # throughput on this machine, MB/s (MB = 1e6)
python bench/perfbench.py encode_typical  # one workload, for the Callgrind harness
```

The `time` mode reports throughput measured against **process CPU time** (not
wall-clock), so it reflects the cost of the library rather than scheduling
noise; absolute numbers still vary with CPU speed, load and the Python
implementation (CPython vs. PyPy).

For a **CPU-speed-independent** cost metric, `bench/run_callgrind.sh` runs each
workload under Callgrind and reports **instructions retired per operation** —
deterministic and comparable across machines (and against the other corelibs):

```bash
bash bench/run_callgrind.sh               # needs valgrind
```

Because the workloads are Python functions rather than C symbols, the script
runs each at two rep counts and subtracts the instruction counts, cancelling
interpreter startup and one-time setup to isolate the per-operation cost.

[`Encoder`]: src/sofab/encoder.py
[`Decoder`]: src/sofab/decoder.py
