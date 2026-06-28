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
