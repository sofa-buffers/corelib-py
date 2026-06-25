<p align="center"><img src="assets/sofabuffers_logo.png" alt="SofaBuffers Logo" height="140"></p>

# SofaBuffers

<b>Structured Objects For Anyone</b><br>
<i>... so optimized, feels amazing.</i>

[Would you like to know more?](https://github.com/sofa-buffers)

## SofaBuffers Python library

[![CI](https://github.com/sofa-buffers/corelib-py/actions/workflows/ci.yml/badge.svg)](https://github.com/sofa-buffers/corelib-py/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fsofa-buffers%2Fcorelib-py%2Fbadges%2Fcoverage.json)](https://github.com/sofa-buffers/corelib-py/actions/workflows/ci.yml)

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
unit tests here use the exact byte vectors from the
[C corelib](https://github.com/sofa-buffers/corelib-c-cpp)'s reference suite
(`test/c/test_ostream.c`) to guarantee byte-for-byte interoperability with the C,
C++, Rust, Go and Java implementations.

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

## Source documentation

The language-neutral wire-format specification lives in the
[SofaBuffers documentation](https://github.com/sofa-buffers/documentation). API
documentation for this package is published to GitHub Pages on every push to
`main`.

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

**Decoder** — `next`, `field`, `unsigned`, `signed`, `bool`, `float32`,
`float64`, `string`, `bytes`, `read_unsigned_array`, `read_signed_array`,
`read_float32_array`, `read_float64_array`, `skip`.

> **Note on value width:** like the C default configuration, the scalar value
> type is 64-bit, so varint encodings match byte-for-byte across the C, C++,
> Rust, Go, Java and Python implementations.

## Layering vs. the C library

| C file | Python module | Status |
|--------|---------------|--------|
| `sofab.h` (types / constants) | `types.py` (`WireType`, `FixlenSubtype`, `Field`, errors, limits) | ported |
| `ostream.c` | `encoder.py` ([`Encoder`]) | ported |
| `istream.c` | `decoder.py` ([`Decoder`]) | ported (pull-parser model instead of bind-target callbacks) |
| `object.c` (descriptor transcoder) | — | not ported. The idiomatic Python equivalent is generated message classes from a schema-driven generator; the streaming core above already covers serialize / deserialize. |
| — | `_varint.py` / `_core.py` | varint / zigzag + IEEE-754 helpers, isolated as the hot path so a native accelerator can replace them without an API change. |

## Testing & coverage

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest ruff mypy
pytest                       # unit + roundtrip + streaming + malformed
ruff check src/sofab tests   # lint
mypy --strict src/sofab      # type-check
```

Tests are split by concern, and every wire vector is taken verbatim from the C
reference implementation:

- `test_vectors_ostream.py` — encoder, byte-exact vs. the C vectors (incl. the full-scale example)
- `test_vectors_istream.py` — decoder over the same vectors, walking every field
- `test_roundtrip.py` — encode → decode value preservation (scalars, arrays, strings/blobs, sequences, boundary values)
- `test_streaming.py` — 1-byte-granularity decode + tiny-scratch-buffer encode match the one-shot path
- `test_malformed.py` — malformed-input decode errors + encoder range / state errors + sticky mode
- `test_varint.py` — varint / zigzag codec

Coverage is measured on every CI run on `main` and reported by the **coverage**
badge above (updated automatically via the `badges` branch).

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
