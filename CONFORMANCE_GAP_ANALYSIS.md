# SofaBuffers `corelib-py` — Conformance Gap Analysis & Remediation Plan

Audit of the Python port against the language-independent SofaBuffers corelib
specification (`CORELIB_PLAN.md`), with primary focus on the §13 Conformance
Checklist. Each item was verified by reading source, tests, assets, and CI —
not inferred from names. The full test suite was executed during the audit:
**678 passed, 6 skipped** (`PYTHONPATH=src pytest`; coverage ~93% as configured
in CI).

## Spec revision

This is a **refresh** of a prior audit against an updated `CORELIB_PLAN.md`
(commit `dcb85d6`, 2026-06-30). The substantive delta: **zero-length arrays and
empty sequences are now explicitly legal on the wire**, where the previous
revision treated arrays as always non-empty (`count >= 1`).

- **§4.7** — `element_count` range is now `0 .. 2,147,483,647` (was `1 ..`). A
  **zero-count integer array** (unsigned/signed) is a valid, fully-specified
  empty array: exactly `[ header_varint ] [ element_count_varint = 0 ]`, nothing
  after. Absent-vs-empty distinction is now a **code-generator concern, not a
  wire-level one**.
- **§4.8** — a **zero-count fixlen array** (fp32/fp64) carries **no `fixlen_word`
  and no payload**: exactly `[ header_varint ] [ element_count_varint = 0 ]`.
- **§4.9** — an **empty sequence** (`sequence start` immediately followed by its
  `0x07` end) is legal and a decoder **MUST** accept it.

### What changed vs the previous revision

- A port that **rejects** a zero-count array or empty sequence (on encode or
  decode) is now **NON-CONFORMANT**; allowing count-0 is now the compliant
  behavior. The previous audit's "empty arrays rejected (count ≥ 1) … " note for
  item 6 was *compliant then, a defect now*.
- **Item 6 (arrays): PASS → PARTIAL.** The encoder (`_array_header`,
  `encoder.py:333`) and decoder (`decoder.py:301`, `:309`) both hard-reject
  `count == 0` with an error. Zero-count integer arrays and zero-count fixlen
  arrays are unrepresentable and unparseable. New high-severity finding.
- **Item 12 (tests/vectors): PASS → PARTIAL.** `tests/test_malformed.py:66`
  (`test_array_count_zero`) now **asserts the wrong behavior** — it requires a
  zero-count array to raise `SofaDecodeError`, which the updated spec forbids.
  The shared `assets/test_vectors.json` also has **no zero-count array vector**
  (it does ship empty-sequence vectors). Downgraded accordingly.
- **Item 7 (sequences): empty-sequence sub-requirement now PASS.** The decoder
  already accepts empty / nested-empty sequences (vectors `empty_sequence`,
  `nested_empty_sequences`, `empty_sequence_between_fields` decode and round-trip
  cleanly) and the encoder permits a `sequence_begin` immediately followed by
  `sequence_end`. Item 7 nonetheless **remains PARTIAL** for the unchanged,
  still-valid `MAX_DEPTH` defect (below).
- **All other previous findings carry forward unchanged** (MAX_DEPTH missing;
  no push `feed()`/resumable decoder; UTF-8 error leak; devcontainer
  `build.sh` tag mismatch; docs.yml uses pdoc not Sphinx; pip cache off).

## Summary

| Status  | Count |
|---------|-------|
| PASS    | 10    |
| PARTIAL | 8     |
| GAP     | 0     |
| **Total** | **18** |

No item is a total miss, but eight items are partially conformant. The
highest-impact issues are now (1) **zero-count arrays are rejected on both encode
and decode** (newly a normative MUST), (2) `MAX_DEPTH` (255) is neither defined
nor enforced anywhere, and (3) the devcontainer `build.sh` tags the image
`python-devcontainer` while `start.sh`/`attach.sh` expect `py-devcontainer`, so
the build→start flow is broken.

### Zero-length / empty-sequence support, as it stands

| Wire construct (updated spec) | Encode | Decode | Tests / vectors |
|-------------------------------|--------|--------|-----------------|
| Zero-count unsigned array (§4.7) | **Rejected** — `encoder.py:333` `count < 1` → `SofaRangeError` | **Rejected** — `decoder.py:301` `count < 1` → `SofaDecodeError` | No positive vector; `test_array_count_zero` asserts rejection (now wrong) |
| Zero-count signed array (§4.7) | **Rejected** — same `_array_header` path (`encoder.py:291`,`:333`) | **Rejected** — `decoder.py:301` (shared ARRAY_UNSIGNED/SIGNED branch) | Same as above |
| Zero-count fixlen array, no fixlen_word/payload (§4.8) | **Rejected** — `_write_float_array`→`_array_header` (`encoder.py:326`,`:333`); would also wrongly emit the elem word (`:327`) if count check were relaxed | **Rejected** — `decoder.py:309` `count < 1`; and the count-then-`elem_header` read (`:308`,`:311`) assumes a `fixlen_word` always follows, which the zero case omits | No vector |
| Empty sequence `0e 07` (§4.9) | **OK** — `write_sequence_begin`+`write_sequence_end` impose no min content (`encoder.py:340`,`:353`) | **OK** — depth push/pop only (`decoder.py:275`,`:265`) | **Covered & passing**: `empty_sequence`, `nested_empty_sequences`, `empty_sequence_between_fields` in `assets/test_vectors.json` |

Net: **empty sequences are fully conformant; all three flavours of zero-count
array are non-conformant on both sides.**

## Per-checklist-item results

| Item (§13) | Status | Evidence | Notes |
|------------|--------|----------|-------|
| 1. All public symbols under `sofab` namespace (§6) | PASS | `src/sofab/__init__.py`; package dir `src/sofab/`; `pyproject.toml` packages `["src/sofab"]` | Import namespace is exactly `sofab`. Minor: PyPI distribution name is `sofabuffers` (`pyproject.toml:6`), spec wants `SofaBuffers`; equivalent under PEP 503 normalization, not the literal casing. |
| 2. API version constant/getter returns `1` (§6) | PASS | `src/sofab/types.py:17` `API_VERSION = 1`; exported from `__init__.py` | |
| 3. Varint & zig-zag match §4.1–4.2 | PASS | `_varint.py` zigzag 64-bit, encode, decode with truncation + `shift >= 64` overflow guard; decoder mirrors guards (`decoder.py:149`,`:215`) | `tests/test_varint.py`, malformed overflow tests pass. |
| 4. Field header `(id<<3)|type` + all 8 wire types (§4.3) | PASS | encode `encoder.py:180-183`; decode `decoder.py:261-263`; `WireType` enum `types.py:35-45` covers 0x0–0x7 | Sequence-end emitted/parsed as single byte `0x07`. |
| 5. Fixlen word `(length<<3)|subtype`, LE floats, UTF-8 no terminator, blobs (§4.6) | PASS | `encoder.py:255` `(len<<3)\|subtype`; `_core.py` `struct "<f"/"<d"`; decode `decoder.py:285-297`, subtype guard `:289` | String encoded as raw UTF-8, no terminator (`encoder.py:244`). See item 10 re: invalid-UTF-8 decode error type. |
| 6. Integer arrays + fixlen arrays w/ single shared word; no dynamic subtypes; **zero-count arrays valid (§4.7–4.8, UPDATED)** | **PARTIAL** | Non-empty arrays fully correct: `encoder.py:262-336` (single elem word `:327`); decoder rejects fixlen-array subtype > FP64 `decoder.py:314`. **But zero-count arrays are rejected on encode (`encoder.py:333` `count < 1`) and decode (`decoder.py:301`,`:309` `count < 1`)**, and the fixlen-array decode path (`:308-311`) assumes a `fixlen_word` always follows the count — wrong for the new zero-count fixlen form, which omits it. | **Changed: was PASS.** Newly a normative MUST. See Remediation 1. |
| 7. Sequence framing, fresh scope, single-byte `0x07` end, skip-by-walking w/ depth, **empty sequence accepted (§4.9, UPDATED)**, reject nesting > `MAX_DEPTH`=255 | PARTIAL | Framing/scope/skip present: `encoder.py:340-366`, `decoder.py:265-278`,`:342-350`. **Empty/nested-empty sequences encode, decode, and round-trip** (vectors `empty_sequence`/`nested_empty_sequences`/`empty_sequence_between_fields` pass). **But `MAX_DEPTH` is undefined** (`types.py` has no such constant; only local `_depth` counters) and **never enforced** on encode or decode. | Empty-sequence rule now satisfied; `MAX_DEPTH` defect unchanged. See Remediation 2. |
| 8. Streaming encode into smaller-than-message buffer via flush sink + mid-stream buffer swap (§5.1) | PASS | `Encoder.over_buffer` `encoder.py:60-91`, `_put`/`_drain` `:114-136`, `buffer_set` `:93-105`; `test_streaming.py` (small buffer), `test_conformance_vectors.py` (1/3/7-byte buffers) | |
| 9. Streaming decode via `feed` of arbitrarily small chunks, push/pull, lazy binding, auto-skip (§5.2) | PARTIAL | Byte-granularity suspend/resume works for a *pull* reader: `decoder.py:_need/_varint/_read_exact`; `test_streaming.py` feeds 1 byte at a time via `ChunkReader`. **No push `feed(bytes)` entry point exists** (`grep "def feed"` → none); decoder only `read(n)`-pulls and raises on mid-field EOF (`_need` → "truncated"), so it cannot resume across non-blocking feed calls. | §6 names `feed`; §5.3 permits pull, but the push API the generated streaming-IN path (item 11) needs is absent. See Remediation 4. |
| 10. Error reporting follows §6.3 baseline codes (or idiomatic exceptions) | PARTIAL | Exception hierarchy maps the codes: `SofaStateError`=UsageError, `SofaBufferError`=BufferFull, `SofaRangeError`=InvalidArgument, `SofaDecodeError`=InvalidMessage (`types.py`). **But invalid UTF-8 leaks `UnicodeDecodeError`** (`decoder.py:457` `.decode("utf-8")`), a `ValueError`, not `SofaDecodeError` — §6.3 lists invalid UTF-8 as `InvalidMessage`. | See Remediation 5. |
| 11. Streaming primitives sufficient for a thin generated-object layer that *also* (de)serializes in chunks; one-shot helpers thin wrappers (§6.1) | PARTIAL | Streaming-OUT fully supported (over_buffer + flush + buffer_set); `getvalue()`/`io.BytesIO` give one-shot paths. **Streaming-IN per §6.1 (`dec.feed(chunk1); dec.feed(chunk2); finish()`) cannot be built** — no push/resumable feed (item 9). | Tied to Remediation 4. Corelib needn't ship generated objects, but must expose the hooks; the push-decode hook is missing. |
| 12. All shared test vectors pass (encode+decode) + chunked + roundtrip + malformed + skip (§7); coverage >90% + badge | PARTIAL | `assets/test_vectors.json` (67 vectors incl. `id_max`, specials, empty-sequence vectors); `test_conformance_vectors.py` runs encode/chunked-encode/decode/chunked-decode/skip/roundtrip; `test_malformed.py`, `test_roundtrip.py`, `test_streaming.py`. Suite: 678 passed / 6 skipped; badge in README, computed in `ci.yml`. **But `tests/test_malformed.py:66` (`test_array_count_zero`) asserts a zero-count array MUST raise — contradicting updated §4.7 — and there is no positive zero-count-array vector.** | **Changed: was PASS.** Vectors are generated upstream by `corelib-c-cpp`; the local malformed test enshrining the old rule is the immediate defect. Also no vector exercises nesting > 255 or invalid UTF-8 (consistent with gaps 7 & 10). See Remediation 1/6. |
| 13. `assets/` populated per §8 (branding + `test_vectors.json`) | PASS | `assets/sofabuffers_logo.png`, `assets/sofabuffers_icon.png`, `assets/test_vectors.json` present; logo referenced `README.md:1` | |
| 14. README follows family format with badges + required sections (§9) | PASS | `README.md`: centered header+logo+tagline, library section + CI/Coverage/Docs badges, Why-this-design table, Usage incl. larger-than-buffer streaming, API summary, Feature flags, Build & test, Benchmarks | |
| 15. `perf` (CPU-independent) + `bench` (MB/s) tools present & runnable (§10) | PASS | `bench/perfbench.py` `time` mode = MB/s (`bench`), `bench/run_callgrind.sh` = instructions/op (CPU-independent `perf`); also a per-op `perf` mode | Both capabilities present; MB/s tool invoked as `time` rather than `bench` (naming only). |
| 16. `.devcontainer/` complete; extensions incl. `anthropic.claude-code`; `.env` gitignored (§11) | PARTIAL | All six files present; `devcontainer.json` includes `anthropic.claude-code`; `.env` gitignored; `start.sh:17`/`attach.sh:2` use `py-devcontainer`. **But `build.sh:6` tags the image `python-devcontainer`** — mismatch with the `py-devcontainer` image `start.sh` runs and with §11.3. | See Remediation 3. |
| 17. `ci.yml` builds+tests on push and PR; matrix across runtime versions; coverage uploaded + badge (§12.1) | PASS | push+PR triggers; matrix `python: 3.9–3.13`, `fail-fast: false`; lint/mypy/test; coverage job + badge publish | Minor: `actions/setup-python` dependency cache (`cache: 'pip'`) not enabled (§12.1 step 2; `ci.yml:30`,`:55` have no `cache:`); coverage published via `badges` branch rather than Codecov (spec allows "or equivalent"). |
| 18. `docs.yml` generates HTML docs + publishes to Pages via Actions deployment (no `gh-pages`); Docs badge links to site (§12.2) | PARTIAL | Actions-based Pages deploy is correct: `upload-pages-artifact@v3` + `deploy-pages@v4`, `permissions: pages/id-token`; Docs badge in README. **But docs are built with `pdoc` (`docs.yml:31-35`), not Sphinx** — §12.2 mandates Sphinx (`sphinx-apidoc` + HTML builder) for Python. | See Remediation 6. |

---

## Remediation Plan

Ordered by severity (highest first).

### 1. Accept zero-count arrays on encode and decode (items 6 & 12) — HIGH

**Problem.** Updated §4.7 makes `element_count == 0` valid for unsigned and
signed integer arrays, and §4.8 makes a zero-count fixlen array valid **with no
`fixlen_word` and no payload**. The Python port rejects all three:
- Encode: `Encoder._array_header` (`encoder.py:333`) raises `SofaRangeError`
  when `count < 1`; this guards `write_unsigned_array`, `write_signed_array`,
  and (via `_write_float_array`, `encoder.py:326`) both float-array writers.
  For a zero-count fixlen array, `_write_float_array` (`encoder.py:327`) would
  also emit a `fixlen_word` even though §4.8 says none must follow.
- Decode: the integer-array branch (`decoder.py:299-301`) and the fixlen-array
  branch (`decoder.py:308-309`) both raise `SofaDecodeError` when `count < 1`.
  The fixlen branch additionally reads `elem_header = self._varint()`
  (`decoder.py:311`) unconditionally, assuming a `fixlen_word` always follows
  the count — incorrect for the zero-count form.

**Fix.**
- Change the `_array_header` lower bound to `count < 0` (i.e. allow `0`), in
  `encoder.py:333`.
- In `_write_float_array` (`encoder.py:314-330`), **skip emitting the
  `fixlen_word` and payload when `len(seq) == 0`** — emit only header + count.
- In the decoder integer-array branch (`decoder.py:301`) and fixlen-array branch
  (`decoder.py:309`), change `count < 1` to `count < 0`.
- In the decoder fixlen-array branch, **guard the `elem_header` read on
  `count > 0`** (`decoder.py:311`); for `count == 0` set `elem_size = 0` /
  subtype absent and produce an empty result without reading further.
- Ensure the read paths (`read_unsigned_array`/`read_signed_array` →
  `_read_varints(0)`; `read_float*_array` → `_read_exact(0)`) return `[]`
  cleanly for count 0.

**Files.** `src/sofab/encoder.py`, `src/sofab/decoder.py`, plus tests:
`tests/test_roundtrip.py` (zero-count round-trips for u/i/fp32/fp64 arrays).

**Tests/vectors.** `tests/test_malformed.py:66` (`test_array_count_zero`) is now
**wrong** — it asserts a zero-count array must raise. Repurpose it (or replace it
with a positive test) to assert that `04 00` and `03 00` decode to empty arrays
and that `05 00` (fixlen, no `fixlen_word`) decodes to an empty fixlen array.
When the upstream `corelib-c-cpp` regenerates `assets/test_vectors.json` with
zero-count array vectors, copy them in (§7/§8); do not hand-write a divergent copy.

**Acceptance criteria.** `write_unsigned_array(id, [])`, `write_signed_array(id,
[])`, `write_float32_array(id, [])`, `write_float64_array(id, [])` each emit
exactly `[ header ][ 0x00 ]` (no `fixlen_word`, no payload for the fixlen ones);
decoding those bytes yields empty lists; `test_array_count_zero` no longer
asserts rejection; round-trip tests cover all four zero-count array kinds.

### 2. Define and enforce `MAX_DEPTH` = 255 (item 7) — HIGH

**Problem.** §4.9/§6.2 mandate `MAX_DEPTH = 255`: an encoder must not open more
than 255 nested sequences; a decoder must reject deeper nesting with an
`InvalidMessage` error. The constant is absent from `types.py`, and neither
`Encoder.write_sequence_begin` (`encoder.py:340`) nor the decoder sequence-start
branch (`decoder.py:275-278`) checks depth against any limit. Normative MUST and
a robustness/DoS concern. (Unchanged by the spec revision.)

**Fix.**
- Add `MAX_DEPTH = 255` to `src/sofab/types.py` and export it from
  `src/sofab/__init__.py`.
- In `Encoder.write_sequence_begin` (`encoder.py:340`), raise `SofaRangeError`
  (or `SofaStateError`) when `self._depth >= MAX_DEPTH` *before* emitting the
  header / incrementing depth.
- In the decoder sequence-start branch (`decoder.py:275-278`), raise
  `SofaDecodeError` when `self._depth >= MAX_DEPTH` before incrementing.

**Files.** `src/sofab/types.py`, `src/sofab/__init__.py`, `src/sofab/encoder.py`,
`src/sofab/decoder.py`, new malformed test in `tests/test_malformed.py`.

**Acceptance criteria.** `MAX_DEPTH == 255` importable from `sofab`; a 256th
nested `sequence_begin` raises a Sofa error; decoding 256 nested sequence-starts
raises `SofaDecodeError`; a 255-deep message still round-trips.

### 3. Fix devcontainer image tag in `build.sh` (item 16) — HIGH

**Problem.** `.devcontainer/build.sh:6` builds `docker build -t
python-devcontainer`, but `start.sh:17`/`:22` and `attach.sh:2` reference
`py-devcontainer` (which §11.3 also requires). As written, `build.sh` then
`start.sh` fails (image not found).

**Fix.** Tag the image `py-devcontainer`:
`docker build -t py-devcontainer "$SCRIPT_DIR"`.

**Files.** `.devcontainer/build.sh`.

**Acceptance criteria.** `build.sh` produces image `py-devcontainer`; `start.sh`
launches it unmodified; tag matches §11.3.

### 4. Provide a push `feed(bytes)` / resumable decoder (items 9 & 11) — MEDIUM

**Problem.** The decoder is pull-only (`reader.read(n)`); it cannot suspend
mid-field and resume on a later caller-pushed chunk (`_need` returns False on EOF
→ "truncated"). §5.2 and §6 name a `feed(bytes)` push API, and §6.1's generated
streaming-IN pattern (`dec.feed(chunk1); dec.feed(chunk2); finish()`) requires a
resumable push decoder that returns control to the caller between chunks.

**Fix.** Add a push/resumable decode path that buffers fed bytes and yields field
events only when a complete header/value is available, returning "needs more
data" otherwise — without raising on a mid-item chunk boundary. Either a
generator/coroutine wrapping the existing cursor state machine, or a
`feed(chunk)`-driven facade that distinguishes "incomplete" from "truncated/EOF".
Keep the pull API intact.

**Files.** `src/sofab/decoder.py` (+ `__init__.py` export); new tests in
`tests/test_streaming.py`.

**Acceptance criteria.** A caller can push arbitrarily small chunks via `feed`,
read decoded fields incrementally, and reach the same result as one-shot decode,
with control returning between chunks (no exception on an incomplete-but-not-final
chunk); a generated-style `feed(...); finish()` flow is demonstrable in a test.

### 5. Surface invalid UTF-8 as `SofaDecodeError` (item 10) — LOW/MEDIUM

**Problem.** `Decoder.string()` (`decoder.py:457`) calls `.decode("utf-8")`,
which raises `UnicodeDecodeError` (a `ValueError`), not `SofaDecodeError`. §6.3
classifies invalid UTF-8 as `InvalidMessage`, so callers catching the Sofa
hierarchy miss it.

**Fix.** Wrap the decode in `try/except UnicodeDecodeError` and re-raise as
`SofaDecodeError` (chaining the original). Add a malformed test feeding a STRING
fixlen with invalid UTF-8 bytes.

**Files.** `src/sofab/decoder.py`, `tests/test_malformed.py`.

**Acceptance criteria.** Decoding a string field with invalid UTF-8 raises
`SofaDecodeError` (catchable as `SofaError`); valid strings unaffected.

### 6. Docs tool + minor hardening (items 12, 17, 18) — LOW

**Problem / Fix (each independent, no behavior risk):**
- **Build API docs with Sphinx, not pdoc (item 18).** §12.2 requires Sphinx
  (`sphinx-apidoc` + HTML builder) for Python; `docs.yml:31-35` installs/runs
  `pdoc`. Replace with a minimal `docs/` Sphinx config, run `sphinx-apidoc` over
  `src/sofab` then `sphinx-build -b html`, and point `upload-pages-artifact` at
  the output. Update README docs wording from pdoc to Sphinx.
- Export `FIXLEN_MAX` and the new `MAX_DEPTH` from `src/sofab/__init__.py`
  (currently `ARRAY_MAX` is exported but `FIXLEN_MAX`/`MAX_DEPTH` are not).
- Enable the `actions/setup-python` dependency cache (`cache: 'pip'`) in
  `ci.yml` (steps at `:30` and `:55`) per §12.1 step 2.
- Consider aligning the PyPI distribution name in `pyproject.toml:6`
  (`sofabuffers`) with the spec's `SofaBuffers` casing (equivalent under PEP 503,
  optional).
- After fixing items 1, 2 and 5, add the corresponding malformed/positive
  vectors (zero-count arrays, depth > 255, invalid UTF-8) so the conformance
  suite tracks the updated spec (§7.2).

**Files.** `.github/workflows/docs.yml`, new `docs/` directory, `README.md`,
`src/sofab/__init__.py`, `.github/workflows/ci.yml`, `pyproject.toml`,
`tests/test_malformed.py`.

**Acceptance criteria.** `docs.yml` generates HTML via Sphinx and deploys to
`https://sofa-buffers.github.io/corelib-py/`; public limits importable from
`sofab`; CI restores pip cache; malformed/positive suite covers the newly
enforced/relaxed conditions.
