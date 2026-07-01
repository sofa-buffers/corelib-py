# Performance notes — pointer-advance, the visitor pattern, and throughput

This port targets **maximum throughput** per ARCHITECTURE.md ("machines that
deal with bigger libs — no focus on footprint"). The two questions raised when
booting it were: *can protobuf's "advance a pointer over a contiguous buffer"
work in Python?* and *should the decoder move to a visitor pattern?* Both are
answered here, with measured before/after numbers.

## Is "advance a pointer over a contiguous buffer" possible in Python? — Yes.

Protobuf's hot loop keeps a raw `const uint8_t*` and advances it over a
contiguous buffer, indexing without bounds-checks-per-byte and without copying.
Python has no raw pointers, but the *same idea* maps cleanly onto:

* a single contiguous `bytes` buffer (`self._buf`), and
* an **integer cursor** (`self._pos`) that we advance with direct indexing.

`buf[pos]` returns a small int with no slice/copy, and indexing a `bytes` is a
C-level operation. The win is eliminating the **per-byte Python function call**:
the old decoder pulled each byte through a `read_byte()` callback and decoded
varints one call at a time — the dominant cost for a Python-level loop. The new
decoder inlines the varint codec directly over `(buf, pos)` with local
variables (`src/sofab/decoder.py` → `_varint`, `_read_varints`, `_read_exact`),
so an entire 1000-element integer array is decoded in one tight loop with zero
per-element calls.

**What Python *can't* match vs C:** the indexed byte is still a boxed `int`,
big integers (`>2^63`) are heap objects, and we can't dereference into a foreign
buffer truly zero-copy. So this is "pointer-advance in spirit" — the
*algorithm* (one contiguous buffer, an advancing cursor, no per-byte dispatch,
bulk slices for fixlen payloads) — not literal pointer arithmetic. That spirit
is exactly where the speedups below come from, and it leaves the hot path in a
shape a future mypyc/Cython/PyO3 accelerator can lower to real pointers without
an API change (the codec lives in `_varint.py` / `_core.py` behind stable
signatures, as those modules already note).

**Streaming is preserved.** The format requires decoding in arbitrarily small
chunks (the suite feeds one byte at a time). The cursor loop fast-paths the
fully-buffered case and, only when it runs off the end mid-value, calls `_need`
to refill from the reader and continues — so the same code serves a 10 KB
in-memory message and a dribbling socket. `chunked-decode` and
`chunked skip-ids` conformance scenarios both pass at 1-byte granularity.

## Visitor pattern — added as an additive layer, not a replacement.

ARCHITECTURE.md lists the visitor pattern as a recommended decoder shape, so it
is now offered (`sofab.Visitor` + `Decoder.drive(visitor)`,
`src/sofab/visitor.py`). It dispatches each field to a typed hook and supports
declining work *before* decoding (`on_field` / `on_sequence_begin` returning
`False` skips a value or a whole sub-tree for free).

It is **layered on the pull API rather than replacing it**, deliberately:

* The pull decoder *is* the contiguous-buffer hot path; the visitor reuses it,
  so it inherits the same throughput instead of competing with it.
* Callbacks-per-field add overhead, so the zero-overhead pull loop stays the
  default for the perf-critical path (benchmarks, generated `serialize_to`).
* It keeps the streaming/skip semantics in exactly one place.

So: visitor for ergonomics, pull for raw speed — both on one engine.

## Results (throughput, MB/s; best of 5 runs, `bench/perfbench.py time`)

| Workload                 | Baseline | Optimized | Speedup |
|--------------------------|---------:|----------:|--------:|
| encode: u64 array (1000) |     6.57 |      8.72 |  1.33×  |
| encode: typical message  |     2.25 |      3.84 |  1.71×  |
| decode: u64 array (1000) |     3.18 |      6.17 |  1.94×  |
| decode: typical message  |     0.91 |      1.21 |  1.33×  |

`MB = 1e6 bytes`, ~1 s CPU-time loop per workload, same host. Decode of large
integer arrays — the workload the pointer-advance loop targets — nearly
doubled; the per-byte-callback elimination is the source.

### What changed

* **Decoder** — varint decoding moved from a per-byte callback to an inline
  cursor over the buffer; array decode is one fused loop; fixlen payloads are
  single slices; float arrays unpack in one `struct.unpack`.
* **Encoder** — varints are appended straight into the output `bytearray`
  (`_emit_varint`) instead of allocating a throwaway `bytes` per value; float
  arrays pack in one `struct.pack`.
* **Both** stay byte-for-byte compatible: all 67 shared conformance vectors and
  the existing unit/streaming suites pass unchanged.

---

# Native acceleration (Cython) — beating protobuf while still running everywhere

The pure-Python pointer-advance work above took the algorithm as far as CPython
bytecode allows. To go further — and to clear the stated goal of being *faster
than protobuf's Python runtime* — the varint / buffer hot path is now compiled.

## Approach: a native accelerator with a pure-Python fallback

The design mirrors protobuf's own (a C/upb backend selected at import, with a
pure-Python fallback), and deliberately reuses the "advance a pointer over a
contiguous buffer" algorithm from the C++ core (`corelib-cpp/sofab.hpp`) rather
than binding to the footprint-optimized bare-metal C core:

* **`sofab._speedups`** — a Cython translation of `Encoder` / `Decoder` to
  `cdef class`es. The decoder holds a raw `const unsigned char*` and advances a
  C cursor with direct indexing; varints and ZigZag are pure C bit-ops; the
  encoder writes varints straight into a C `realloc`-doubled buffer (one
  capacity check + one write per varint, matching the C++ `pushBytes` idiom);
  floats pack/unpack via endian-independent bit ops; `Field` is a `cdef class`
  (cheap C allocation instead of a `@dataclass`).
* **Pure Python stays** as `encoder.py` / `decoder.py`, used verbatim when the
  extension is absent. `sofab.__init__` picks the engine at import
  (`sofab.IMPL`), and `SOFAB_PUREPYTHON=1` forces the pure path.
* **One source of truth for semantics.** Wire constants, enums and the error
  classes live in `types.py`, imported by both engines, so behaviour (including
  which exception type is raised) is identical by construction.

"Runs everywhere" is preserved because the extension is an *optional* build: no
compiler / unsupported platform ⇒ `pip` installs the `py3-none-any` pure-Python
wheel and everything still works — only slower.

## Byte-exactness is enforced, not assumed

* All shared conformance vectors (`assets/test_vectors.json`) run against
  whichever engine is active.
* `tests/test_native_parity.py` runs an edge-case program (u64/i64 extremes,
  empty arrays/strings/blobs, multibyte UTF-8, nested sequences, the always-present
  fixlen_word) through **both** engines and asserts byte-identical encode,
  identical decoded values, and successful cross-decoding (native→pure and
  pure→native).
* A 5,000-message differential fuzz (random field mixes) confirmed
  native-encode ≡ pure-encode byte-for-byte, and a 250k-iteration decode loop
  showed zero RSS growth (no reference leaks in the `PyList_SET_ITEM` fast path).

## Results (throughput MB/s; `bench/compare_protobuf.py`, one x86-64 host, CPython 3.12)

| Workload                 | pure Python | **native** | native speedup | protobuf (upb) | native vs protobuf |
|--------------------------|------------:|-----------:|---------------:|---------------:|:------------------:|
| encode: u64 array (1000) |        ≈9   |   **≈300** |     ~33×       |      ≈190      |   **≈1.6× faster** |
| encode: typical message  |        ≈4.5 |    **≈14** |     ~3×        |      ≈10       |   **≈1.5× faster** |
| decode: u64 array (1000) |        ≈6.7 |   **≈285** |     ~43×       |      ≈180      |   **≈1.6× faster** |
| decode: typical message  |        ≈2.0 |    **≈7.3**|     ~3.6×      |      ≈8.6      |   ≈0.85× (tiny msg)|

The protobuf comparison uses **full materialization on both sides** (protobuf's
repeated-scalar container otherwise boxes integers lazily, which would only touch
the two elements the checksum reads, not all 1000 — an unfair advantage on the
array workloads). Even so, the only workload where protobuf wins is decoding a
36-byte mixed message, where its single whole-message C parse beats a streaming
pull API that crosses the Python↔C boundary twice per field. That gap is inherent
to the pull model (which is what lets SofaBuffers decode a stream larger than
RAM) and vanishes on any array-bearing or larger message.

### What changed (native vs. the pure-Python hot path)

* **Varint encode** — a `do/while` into a 10-byte stack scratch (or straight
  into the grow buffer), then one `memcpy` + cursor advance; no per-byte Python.
* **Integer array encode** — the per-element range check is done by the C cast
  itself (`<uint64_t>value` raises `OverflowError` → `SofaRangeError`), removing
  two Python rich-comparisons per element. This alone roughly doubled u64-array
  encode throughput.
* **Decode** — the whole varint codec is inlined over a `const unsigned char*`;
  arrays decode in one fused C loop building the result `list` via
  `PyList_SET_ITEM`; fixlen payloads are single slices; `Field` is a `cdef class`
  (this ~1.6×'d small-message decode by itself).
* **Floats** — packed/unpacked with endian-independent shift/`memcpy` bit ops,
  correct on big-endian hosts without a `struct` round-trip.
