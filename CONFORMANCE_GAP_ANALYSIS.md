# SofaBuffers `corelib-py` — Conformance Gap Analysis & Remediation Plan

Audit of the Python port against the language-independent SofaBuffers corelib
specification (`CORELIB_PLAN.md`), with primary focus on the §13 Conformance
Checklist. Each item was verified by reading source, tests, assets, and CI —
not inferred from names. The full test suite was executed during the audit:
**678 passed, 6 skipped, 93% coverage** (`pytest --cov=sofab`).

## Summary

| Status  | Count |
|---------|-------|
| PASS    | 12    |
| PARTIAL | 6     |
| GAP     | 0     |
| **Total** | **18** |

No item is a total miss, but six items are partially conformant. The two
highest-impact issues are (1) `MAX_DEPTH` (255) is neither defined nor enforced
anywhere, and (2) the devcontainer `build.sh` tags the image `python-devcontainer`
while `start.sh`/`attach.sh` expect `py-devcontainer`, so the build→start flow is
broken.

## Per-checklist-item results

| Item (§13) | Status | Evidence | Notes |
|------------|--------|----------|-------|
| 1. All public symbols under `sofab` namespace (§6) | PASS | `src/sofab/__init__.py`; package dir `src/sofab/`; `pyproject.toml:26` packages `["src/sofab"]` | Import namespace is exactly `sofab`. Minor: PyPI distribution name is `sofabuffers` (`pyproject.toml:6`), spec wants `SofaBuffers`; functionally equivalent under PEP 503 case/normalization, but not the literal casing. |
| 2. API version constant/getter returns `1` (§6) | PASS | `src/sofab/types.py:17` `API_VERSION = 1`; exported `__init__.py:13,45` | |
| 3. Varint & zig-zag match §4.1–4.2 (§13) | PASS | `_varint.py:16-23` (zigzag 64-bit), `:26-40` (encode), `:43-63` (decode with truncation + `shift >= 64` overflow guard) | `tests/test_varint.py`, malformed overflow tests pass. |
| 4. Field header `(id<<3)|type` + all 8 wire types (§4.3) | PASS | encode `encoder.py:180-183`; decode `decoder.py:261-263`; `WireType` enum `types.py:35-45` covers 0x0–0x7 | Sequence-end emitted/parsed as single byte `0x07`. |
| 5. Fixlen word `(length<<3)|subtype`, LE floats, UTF-8 no terminator, blobs (§4.6) | PASS | `encoder.py:250-258` (`(len<<3)|subtype`), `_core.py` (`struct "<f"/"<d"`), `decoder.py:285-297`, subtype guard `:289` | String encoded as raw UTF-8 with no terminator (`encoder.py:244`). See item 10 re: invalid-UTF-8 decode error type. |
| 6. Integer arrays + fixlen arrays w/ single shared word; no dynamic subtypes in fixlen arrays (§4.7–4.8) | PASS | `encoder.py:262-336`, single elem word `:327`; decoder rejects fixlen-array subtype > FP64 `decoder.py:314-315`; `test_malformed.py:72` | Empty arrays rejected (count ≥ 1) on encode `encoder.py:333` and decode `:309`. |
| 7. Sequence framing, fresh scope, single-byte `0x07` end, skip-by-walking w/ depth, reject nesting > `MAX_DEPTH`=255 (§4.9) | PARTIAL | Framing/scope/skip all present: `encoder.py:340-366`, `decoder.py:265-278,342-350`. **But `MAX_DEPTH` is undefined** (not in `types.py`, `grep` finds only local `_depth` counters) and **never enforced** on encode or decode. | High-severity gap — normative MUST. See Remediation 1. |
| 8. Streaming encode into smaller-than-message buffer via flush sink + mid-stream buffer swap (§5.1) | PASS | `Encoder.over_buffer` `encoder.py:60-91`, `_put`/`_drain` `:114-136`, `buffer_set` `:93-105`; `test_streaming.py:41` (7-byte buffer), `test_conformance_vectors.py` (1/3/7-byte buffers) | |
| 9. Streaming decode via `feed` of arbitrarily small chunks, push/pull, lazy binding, auto-skip (§5.2) | PARTIAL | Byte-granularity suspend/resume works for a *pull* reader: `decoder.py:_need/_varint/_read_exact`; `test_streaming.py:35` feeds 1 byte at a time via `ChunkReader`. **No push `feed(bytes)` entry point exists** (`grep "def feed"` → none); decoder only `read(n)`-pulls and raises on mid-field EOF (`_need` returns False → "truncated"), so it cannot resume across non-blocking feed calls. | Spec §6 Decoder names `feed`; §5.3 permits pull, but the push API the generated streaming-IN path needs (item 11) is absent. See Remediation 4. |
| 10. Error reporting follows §6.3 baseline codes (or idiomatic exceptions) | PARTIAL | Exception hierarchy maps the codes: `SofaStateError`=UsageError, `SofaBufferError`=BufferFull, `SofaRangeError`=InvalidArgument, `SofaDecodeError`=InvalidMessage (`types.py:77-97`). **But invalid UTF-8 leaks `UnicodeDecodeError`** (`decoder.py:457` `.decode("utf-8")`), a `ValueError`, not `SofaDecodeError` — §6.3 lists invalid UTF-8 as `InvalidMessage`. | §6.3 arguably permits a stdlib encoding error, but the baseline meaning should still surface as a Sofa error. See Remediation 5. |
| 11. Streaming primitives sufficient for a thin generated-object layer that *also* (de)serializes in chunks; one-shot helpers thin wrappers (§6.1) | PARTIAL | Streaming-OUT is fully supported (over_buffer + flush + buffer_set), and `getvalue()`/`io.BytesIO` give one-shot paths. **Streaming-IN per §6.1 (`dec.feed(chunk1); dec.feed(chunk2); finish()`) cannot be built** because there is no push/resumable feed (item 9). | Tied to Remediation 4. The corelib need not ship generated objects, but must expose the hooks; the push-decode hook is missing. |
| 12. All shared test vectors pass (encode+decode) + chunked + roundtrip + malformed + skip (§7); coverage >90% + badge | PASS | `assets/test_vectors.json` (67 vectors incl. `id_max`, specials, `empty_sequence`); `test_conformance_vectors.py` runs encode/chunked-encode/decode/chunked-decode/skip/roundtrip; `test_malformed.py`, `test_roundtrip.py`, `test_streaming.py`. Measured coverage **93%**; badge in `README.md:13`, computed in `ci.yml:65-89` | Note: no malformed vector exercises nesting > 255 or invalid UTF-8 (consistent with code gaps 7 & 10); add when those are fixed. |
| 13. `assets/` populated per §8 (branding + `test_vectors.json`) | PASS | `assets/sofabuffers_logo.png`, `assets/sofabuffers_icon.png`, `assets/test_vectors.json` all present; logo referenced `README.md:1` | |
| 14. README follows family format with badges + required sections (§9) | PASS | `README.md`: centered header+logo+tagline (1-8), library section + CI/Coverage/Docs badges (10-16), Why-this-design table (39-51), Usage incl. larger-than-buffer streaming (53-95), API summary (97+), Feature flags (244), Build & test (254), Benchmarks (279) | |
| 15. `perf` (CPU-independent) + `bench` (MB/s) tools present & runnable (§10) | PASS | `bench/perfbench.py` `time` mode = MB/s (`bench`), `run_callgrind.sh` = instructions/op (CPU-independent `perf`); also a `perf` per-op mode | Both capabilities present; MB/s tool is invoked as `time` rather than `bench` (naming only). |
| 16. `.devcontainer/` complete; extensions incl. `anthropic.claude-code`; `.env` gitignored (§11) | PARTIAL | All six files present; `devcontainer.json:11` includes `anthropic.claude-code`; `.env` gitignored (`.gitignore:14-15`, `.devcontainer/.gitignore:6`); `start.sh:17`/`attach.sh:1` use `py-devcontainer`. **But `build.sh:6` tags the image `python-devcontainer`** — mismatch with the `py-devcontainer` image `start.sh` runs, and with §11.3's required `py-devcontainer`. | See Remediation 2. |
| 17. `ci.yml` builds+tests on push and PR; matrix across runtime versions; coverage uploaded + badge (§12.1) | PASS | `ci.yml:3-8` push+PR; matrix `python: 3.9–3.13`, `fail-fast: false` (`:21-25`); lint/mypy/test; coverage job + badge publish (`:46-89`) | Minor: `actions/setup-python` dependency cache (`cache: 'pip'`) not enabled (§12.1 step 2); coverage published via `badges` branch rather than Codecov (spec allows "or equivalent"). |
| 18. `docs.yml` generates HTML docs + publishes to Pages via Actions deployment (no `gh-pages`); Docs badge links to site (§12.2) | PARTIAL | Actions-based Pages deploy is correct: `docs.yml` `upload-pages-artifact@v3` + `deploy-pages@v4`, `permissions: pages/id-token` (`:10-13`); Docs badge `README.md:14`. **But docs are built with `pdoc` (`docs.yml:32-35`), not Sphinx** — §12.2 mandates Sphinx (`sphinx-apidoc` + HTML builder) for Python. | See Remediation 3. |

---

## Remediation Plan

Ordered by severity (highest first).

### 1. Define and enforce `MAX_DEPTH` = 255 (item 7) — HIGH

**Problem.** The spec (§4.9, §6.2) mandates `MAX_DEPTH = 255`: an encoder must
not open more than 255 nested sequences, and a decoder must reject a message that
nests deeper with an `InvalidMessage` error. The constant is not defined in
`types.py`, and neither `Encoder.write_sequence_begin` nor
`Decoder.next` (sequence-start branch) checks the depth counter against any
limit. This is a normative MUST and a robustness/DoS concern.

**Fix.**
- Add `MAX_DEPTH = 255` to `src/sofab/types.py` (alongside the other limits) and
  export it from `src/sofab/__init__.py`.
- In `Encoder.write_sequence_begin` (`encoder.py:340`), raise `SofaRangeError`
  (or `SofaStateError`) when `self._depth >= MAX_DEPTH` *before* emitting the
  header / incrementing depth.
- In `Decoder.next` sequence-start branch (`decoder.py:275-278`), raise
  `SofaDecodeError` when `self._depth >= MAX_DEPTH` before incrementing.

**Files.** `src/sofab/types.py`, `src/sofab/__init__.py`, `src/sofab/encoder.py`,
`src/sofab/decoder.py`, and a new malformed test in `tests/test_malformed.py`.

**Acceptance criteria.** `MAX_DEPTH == 255` is importable from `sofab`; encoding a
256th nested `sequence_begin` raises a Sofa error; decoding a stream with 256
nested sequence-starts raises `SofaDecodeError`; a 255-deep message still
round-trips; new tests cover both boundaries (255 OK, 256 rejected).

### 2. Fix devcontainer image tag in `build.sh` (item 16) — HIGH

**Problem.** `.devcontainer/build.sh:6` builds `docker build -t
python-devcontainer`, but `start.sh:17` (`--name py-devcontainer ... py-devcontainer`)
and `attach.sh` reference `py-devcontainer`. The spec §11.3 requires the image
tag `py-devcontainer`. As written, `build.sh` then `start.sh` fails (image not
found).

**Fix.** Change `build.sh` to tag `py-devcontainer`:
`docker build -t py-devcontainer "$SCRIPT_DIR"`.

**Files.** `.devcontainer/build.sh`.

**Acceptance criteria.** `build.sh` produces image `py-devcontainer`; `start.sh`
launches it without modification; image tag matches §11.3. (Optionally rename the
`devcontainer.json` `"name"` to a `py-devcontainer`-consistent label, though the
spec only constrains the image/container name.)

### 3. Build API docs with Sphinx instead of pdoc (item 18) — MEDIUM

**Problem.** §12.2's language→tool table requires **Sphinx** (`sphinx-apidoc` +
HTML builder) for Python. `docs.yml:32-35` installs and runs `pdoc`. The
deployment mechanism (Actions → Pages) is otherwise correct.

**Fix.** Replace the pdoc steps in `docs.yml` with a Sphinx build: add a minimal
`docs/` Sphinx config (`conf.py`, `index.rst`), run `sphinx-apidoc` over
`src/sofab` then `sphinx-build -b html` into the output folder, and point
`upload-pages-artifact` at that folder. Update the README "Build & test"/docs
wording (currently `README.md:276`) from pdoc to Sphinx.

**Files.** `.github/workflows/docs.yml`, new `docs/` directory (`conf.py`,
`index.rst`), `README.md`.

**Acceptance criteria.** `docs.yml` generates HTML via Sphinx and deploys to
`https://sofa-buffers.github.io/corelib-py/`; the Docs badge resolves; no `pdoc`
dependency remains; README references Sphinx.

### 4. Provide a push `feed(bytes)` / resumable decoder (items 9 & 11) — MEDIUM

**Problem.** The decoder is pull-only (`reader.read(n)`); it cannot suspend
mid-field and resume on a later caller-pushed chunk (`_need` returns False on EOF
→ "truncated"). §5.2 and §6 name a `feed(bytes)` push API, and §6.1's generated
streaming-IN pattern (`dec.feed(chunk1); dec.feed(chunk2); finish()`) requires a
resumable push decoder that returns control to the caller between chunks.

**Fix.** Add a push/resumable decode path that buffers fed bytes and yields field
events only when a complete header/value is available, returning "needs more
data" otherwise — without raising on a mid-item chunk boundary. Two viable
shapes: (a) a generator/coroutine wrapping the existing cursor state machine, or
(b) a `feed(chunk)`-driven facade over the current contiguous buffer that
distinguishes "incomplete" from "truncated/EOF". Keep the existing pull API
intact. This must satisfy the §6.1 generated streaming-IN contract.

**Files.** `src/sofab/decoder.py` (and `__init__.py` export); new tests in
`tests/test_streaming.py`.

**Acceptance criteria.** A caller can push arbitrarily small chunks via `feed`,
read decoded fields incrementally, and reach the same result as one-shot decode,
with control returning to the caller between chunks (no exception on an
incomplete-but-not-final chunk); a generated-style `feed(...); finish()` flow is
demonstrable in a test.

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

### 6. Minor hardening (items 1, 12, 17) — LOW

**Problem / Fix (each independent, no behavior risk):**
- Export `FIXLEN_MAX` (and the new `MAX_DEPTH`) from `src/sofab/__init__.py`;
  both are defined/intended as public limits but not currently in `__all__`.
- Enable the `actions/setup-python` dependency cache (`cache: 'pip'`) in
  `ci.yml` per §12.1 step 2.
- Consider aligning the PyPI distribution name in `pyproject.toml:6`
  (`sofabuffers`) with the spec's `SofaBuffers` casing (functionally equivalent
  under PEP 503, so optional).
- After fixing items 1 and 5, add the corresponding malformed vectors (depth >
  255, invalid UTF-8) to keep the conformance suite aligned with the spec's
  malformed-input requirements (§7.2.5).

**Files.** `src/sofab/__init__.py`, `.github/workflows/ci.yml`,
`pyproject.toml`, `tests/test_malformed.py`.

**Acceptance criteria.** Public limits importable from `sofab`; CI restores pip
cache; malformed suite covers the newly enforced conditions.
