"""The benchmark tools against BENCH_SPEC.

BENCH_SPEC is the cross-language contract for ``bench``/``perf``/
``run_callgrind.sh``: the same workloads on the same data, printed in a grammar
a central harness parses into the comparison tables. Two things can silently
break that, and neither is visible from inside the library:

* a **dataset** that drifts — the encoded sizes (the perf message's 170 bytes,
  the blob message's 1,000,005) are BENCH_SPEC's own parity checks, and the
  composite message's 956 is this port's contribution of one;
* a **row** that goes missing or gets misspelled — the harness matches row
  labels by regex, so a renamed or absent row is dropped from the table rather
  than reported, and a workload nobody notices is missing measures nothing.

So the tools are run here (over a millisecond-scale loop, not the reportable ~1s
one) and their output is matched against the spec's regexes. This is a format
and dataset test, never a performance assertion: no timing figure is checked.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
from vectors import values

from sofab import Decoder, Encoder, WireType

BENCH = Path(__file__).resolve().parent.parent / "bench" / "perfbench.py"


def _load_perfbench():
    spec = importlib.util.spec_from_file_location("perfbench", BENCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["perfbench"] = mod
    spec.loader.exec_module(mod)
    return mod


pb = _load_perfbench()


# --- the harness's own regexes (BENCH_SPEC "Output grammar") -----------------

THROUGHPUT_HEADER = re.compile(r"=== SofaBuffers (.+?) throughput")
PEROP_HEADER = re.compile(r"=== SofaBuffers (.+?) per-op")
ROW = re.compile(
    r"^(encode|decode):\s+(u64 array \(1000\)|typical message|blob 1MB one-shot"
    r"|blob 1MB streaming|blob 1MB passthrough|blob 1MB|composite skip-all"
    r"|composite)\s+([\d.]+)$"
)

#: Every row BENCH_SPEC requires, in the order it lists them. The optional
#: ``blob 1MB passthrough`` row is absent on purpose: this port implements no
#: pass-through, and BENCH_SPEC says such a port omits the row rather than
#: printing a placeholder.
REQUIRED_ROWS = [
    "encode: u64 array (1000)",
    "encode: typical message",
    "encode: blob 1MB one-shot",
    "encode: blob 1MB streaming",
    "encode: composite",
    "decode: u64 array (1000)",
    "decode: typical message",
    "decode: blob 1MB",
    "decode: composite",
    "decode: composite skip-all",
]


@pytest.fixture
def fast_loop(monkeypatch):
    """Shrink the measurement loop to the shortest thing the clock can see.

    The numbers that come out are meaningless — which is the point: this file
    checks the shape of the output, and a ~1s loop per row would put ten seconds
    of pure benchmarking into every CI run to learn nothing extra.
    """
    monkeypatch.setattr(pb, "LOOP_SECONDS", 1e-9)
    monkeypatch.setattr(pb, "BATCH_SECONDS", 1e-9)


# --- datasets ---------------------------------------------------------------


def test_blob_payload_matches_the_literal_formula():
    # make_blob() exploits the 256-index period of (i * GOLDEN) & 0xFF; that is
    # an optimisation of the setup, so it is checked here against the literal
    # derivation BENCH_SPEC states, over the whole million bytes.
    assert pb.make_blob() == bytes(((i * pb.GOLDEN) & 0xFF) for i in range(pb.BLOB_N))


def test_blob_message_is_1000005_bytes():
    enc = Encoder()
    enc.write_bytes(1, pb.make_blob())
    enc.flush()
    msg = enc.getvalue()
    assert len(msg) == pb.BLOB_ENCODED == 1_000_005
    # BENCH_SPEC spells the framing out: a 1-byte header and a 4-byte fixlen word.
    assert msg[0] == (1 << 3) | WireType.FIXLEN
    assert msg[5:] == pb.make_blob()


def test_perf_message_is_170_bytes():
    assert len(pb.encode_perf_msg()) == 170


def test_composite_message_is_956_bytes():
    assert len(pb.encode_composite_msg()) == pb.COMPOSITE_ENCODED == 956


def _by_depth(events):
    """Annotate a flat event list with its sequence depth.

    The composite message reuses ids inside its wrapper array (element id = array
    index), so an id alone does not identify a field — the depth it sits at does.
    """
    out = []
    depth = 0
    for e in events:
        if e[0] == "seq{":
            out.append((depth, e))
            depth += 1
        elif e[0] == "seq}":
            depth -= 1
            out.append((depth, e))
        else:
            out.append((depth, e))
    return out


def test_composite_carries_what_bench_spec_asks_for():
    """Each composite field is in the suite for a reason; check each is there."""
    msg = pb.encode_composite_msg()
    ev = _by_depth(values(Decoder, msg))
    top = [e for d, e in ev if d == 0]

    # Field 4 equals its declared default, so the encoder must not write it.
    # ``seq}`` carries no id, so it is not a field.
    assert [e[1] for e in top if e[0] != "seq}"] == [1, 2, 3, 130]

    # The wrapper array: one sequence per element, element id = array index.
    elems = [(e[1], e[2]) for d, e in ev if d == 1 and e[0] == "str"]
    assert elems == list(enumerate(pb.COMPOSITE_ITEMS))
    assert len(elems) == 64  # ids 0..15 one-byte headers, 16..63 two-byte

    # A non-ASCII string through the UTF-8 validator, at the top level.
    text = next(e[2] for d, e in ev if d == 0 and e[0] == "str" and e[1] == 2)
    assert text == pb.COMPOSITE_TEXT
    assert len(text.encode("utf-8")) == 320

    # Nesting at depth 3, and the values buried in it.
    seen = [e[2] for d, e in ev if d > 0 and e[0] in ("u", "s")]
    assert seen == [7, -1]

    # Field 130: the one two-byte field header in the suite.
    assert ("u", 130, 0xDEADBEEF) in [e for d, e in ev if d == 0]

    assert msg[-7:-5] == bytes([((130 << 3) | WireType.UNSIGNED) & 0x7F | 0x80,
                                (130 << 3) >> 7])


def test_streaming_blob_encode_produces_the_one_shot_bytes():
    """The streaming row must be the *same message*, only flushed ~245 times.

    A row driven through a 4096-byte buffer that produced anything other than
    the one-shot bytes would make the pair's difference — the only number
    BENCH_SPEC asks anyone to read here — meaningless.
    """
    blob = pb.make_blob()
    one_shot = bytearray(pb.BLOB_ENCODED)
    assert pb.encode_blob_oneshot(Encoder.over_buffer(one_shot), blob) == pb.BLOB_ENCODED

    chunks: list[bytes] = []
    flushes = 0

    def sink(chunk: bytes) -> None:
        nonlocal flushes
        flushes += 1
        chunks.append(bytes(chunk))

    buf = bytearray(pb.STREAM_BUFFER)
    pb.encode_blob_stream(Encoder.over_buffer(buf, 0, sink), blob)
    assert b"".join(chunks) == bytes(one_shot)
    assert flushes == -(-pb.BLOB_ENCODED // pb.STREAM_BUFFER) == 245


def test_blob_decode_is_fed_in_chunks():
    """The decode row must actually stream: the reader hands over 4096 bytes at
    a time however much the decoder asks for."""
    reader = pb._ChunkReader(pb.make_blob(), pb.STREAM_BUFFER)
    assert len(reader.read(pb.BLOB_N)) == pb.STREAM_BUFFER


def test_discard_sink_does_not_accumulate():
    sink = pb._DiscardSink()
    sink(b"\x01\x02")
    sink(b"\x03\x04")
    assert sink.xor == 0x02
    assert not hasattr(sink, "__dict__")  # __slots__: nowhere to accumulate into


# --- output grammar ---------------------------------------------------------


def test_bench_output_matches_the_spec_grammar(fast_loop, capsys):
    pb.run_timed()
    out = capsys.readouterr().out.splitlines()

    assert THROUGHPUT_HEADER.match(out[0]), out[0]
    assert THROUGHPUT_HEADER.match(out[0]).group(1) == "Python"
    assert out[1].split() == ["Workload", "MB/s"]
    assert out[-1] == "MB = 1e6 bytes. ~1s CPU-time loop per workload."

    assert out[-2] == ""  # the rows, then a blank line, then the MB = 1e6 note
    body = out[3:-2]
    rows = [ROW.match(line) for line in body]
    assert all(rows), [line for line, m in zip(body, rows) if not m]
    assert [f"{m.group(1)}: {m.group(2)}" for m in rows] == REQUIRED_ROWS
    for m in rows:
        assert float(m.group(3)) > 0
        # label left-justified to 26, value right-justified to 12, 2 decimals
        assert m.end(3) == 39, m.string
        assert re.search(r"\.\d\d$", m.group(3)), m.group(3)


def test_perf_output_matches_the_spec_grammar(fast_loop, capsys):
    pb.run_perf()
    out = capsys.readouterr().out
    lines = out.splitlines()

    assert PEROP_HEADER.match(lines[0]), lines[0]
    assert PEROP_HEADER.match(lines[0]).group(1) == "Python"
    assert "--- perf: serialize" in out and "--- perf: deserialize" in out
    assert out.rstrip().endswith("cycles/op tracks code cost; MB/s is this machine's throughput.")

    # Five value lines per section, and CPython has no hardware cycle counter,
    # so BENCH_SPEC's parenthetical stands in for the cycles/op number.
    assert out.count("  iterations    : ") == 2
    assert out.count("  message size  : 170 bytes") == 2
    assert out.count("  cycles/op     : (cycle counter unavailable on CPython)") == 2
    assert len(re.findall(r"^  CPU time/op   : [\d.]+ ns  ", out, re.M)) == 2
    assert len(re.findall(r"^  throughput    : [\d.]+ MB/s  ", out, re.M)) == 2


# --- the Callgrind rep mode -------------------------------------------------


@pytest.mark.parametrize("name", list(pb.WORKLOADS))
def test_every_workload_runs_a_rep(name, capsys):
    """``run_callgrind.sh`` drives each workload by name at two rep counts; a
    key that no longer runs would print a dash in the table instead of failing,
    so it is checked here."""
    pb.run_workload(name, 1)
    err = capsys.readouterr().err
    assert re.fullmatch(r"sink=\d+ bytes=\d+ reps=1\n", err), err


def test_unknown_workload_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        pb.run_workload("encode_nothing", 1)
    assert exc.value.code == 2
    assert "unknown workload" in capsys.readouterr().err


def test_callgrind_script_covers_every_workload():
    """The script's workload list and the tool's registry must agree — a
    workload missing from the script is a row missing from the Ir/op table."""
    script = (BENCH.parent / "run_callgrind.sh").read_text(encoding="utf-8")
    for name, (label, _) in pb.WORKLOADS.items():
        assert re.search(rf"\b{name}\b", script), name
        assert label in script, label
