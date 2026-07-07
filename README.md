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

Python 3.9 or newer (CPython or PyPy); CI runs 3.9–3.13. The optional native
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
| Streaming **in** | `Decoder` is a pull parser over any `read(n)` reader; `next()` returns one field header at a time, never materializing the whole message. |
| Native speed, zero runtime deps | The hot path ships as an optional Cython accelerator (`sofab._speedups`); when it can't be built it falls back to pure Python. No runtime third-party deps either way. |
| Runs everywhere | With no compiler or wheel, `pip` still installs a working pure-Python build (`py3-none-any`). Native and pure paths are byte-for-byte identical — falling back changes only speed. |
| Sticky errors | `Encoder(sticky=True)` records the first failure and turns later writes into no-ops, so generated `marshal` code can check `enc.error` once. |
| Reserve-offset | `Encoder.over_buffer(buf, offset=…)` leaves room at the front of the buffer for a lower-layer protocol header. |
| Typed | Fully type-annotated with a `py.typed` marker (PEP 561); clean under `mypy --strict`. |
| Forward/backward compatible | Unknown fields are consumed with `skip()`. |

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

### Streaming encoder to a stream sink

An `Encoder` constructed with a *writer* (any object with `.write(bytes)`)
accumulates in an internal buffer and drains to the writer on `flush()`:

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
sink whenever it fills, so an arbitrarily large message streams out through
bounded memory:

```python
from sofab import Encoder

out = bytearray()                                            # or a socket / file write
enc = Encoder.over_buffer(bytearray(16), offset=0, flush=out.extend)  # tiny buffer
for i in range(1_000_000):
    enc.write_unsigned(i % 128, i)
enc.flush()                                                  # push the tail
```

### Streaming / pull decoder

Hand `Decoder` any object with `read(n)` (a socket, `sys.stdin.buffer`,
`gzip.GzipFile`, ...) and pull fields with `next()` as they arrive. It refills on
demand, so it decodes correctly even when fed one byte at a time:

```python
from sofab import Decoder

with open("out.sofab", "rb") as f:
    dec = Decoder(f)                 # any read(n) source: file, socket, pipe
    while (field := dec.next()) is not None:
        ...                          # pull each field, or dec.skip()
```

For callback-style decoding, subclass `sofab.Visitor` and hand it to
`Decoder.drive(visitor)`.

### Generated object code

The most common real use is driving the library through **generated code**: a
class per message whose `marshal` / `unmarshal` methods call the primitives
above.

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
checks `enc.error` once after a run of writes.

## Memory handling

The key point for Python: **the library allocates results for you — the caller
never provides a value buffer.**

* **Decode.** `Decoder` keeps a single internal buffer, refilled from the
  `read(n)` source and never handed out, so there is **no zero-copy aliasing**:
  `string()` returns a fresh `str`, `bytes()` independent `bytes`, scalars a
  fresh `int`/`float`, and arrays a new `list` — every result stays valid after
  the decoder advances.
* **Encode.** Two ownership models. The default `Encoder()` / `Encoder(writer)`
  owns a growable `bytearray` — `getvalue()` hands back a copy, or `flush()`
  drains to the writer. `Encoder.over_buffer` is caller-owned and bounded: you
  provide a fixed `bytearray`, it writes in place via a `memoryview` and flushes
  to the sink + reuses the buffer when full.

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
pip install protobuf                      # optional; the column is dropped if absent
python bench/compare_protobuf.py          # best-of-5 MB/s table
```

Representative result (throughput MB/s, higher is better; one x86-64 host,
CPython 3.12 — the *ratios* are the point):

| Workload | sofab **native** | sofab pure | protobuf (upb) | native vs protobuf |
|----------|-----------------:|-----------:|---------------:|:------------------:|
| encode: u64 array (1000) | **≈300** | ≈9 | ≈190 | **≈1.6× faster** |
| encode: typical message  | **≈14**  | ≈4.5 | ≈10 | **≈1.5× faster** |
| decode: u64 array (1000) | **≈285** | ≈6.7 | ≈180 | **≈1.6× faster** |
| decode: typical message  | ≈7.3     | ≈2.0 | ≈8.6 | ≈0.85× (see note) |

The native accelerator is **~15–45× faster than the pure-Python fallback** and
beats protobuf on encode and array-heavy decode. The one workload where protobuf
edges ahead is decoding a *tiny* mixed message: the streaming **pull** API crosses
the Python↔C boundary twice per field (`next()` then a typed read), whereas
protobuf parses the whole message in one C call — an inherent pull-vs-parse-tree
trade-off that only shows on very small messages.
