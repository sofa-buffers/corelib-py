#!/usr/bin/env bash
#
# SofaBuffers Python — machine-independent instruction cost.
#
# Runs each benchmark workload under Callgrind and reports instructions retired
# per operation (Ir/op). Unlike wall-clock or CPU time, instruction counts are
# deterministic and independent of the host's clock speed and scheduler, so the
# numbers are comparable across machines (and against the C/C++/Rust/Go tools —
# the workloads, ids and values are identical). BENCH_SPEC calls this the signal
# a CI performance-regression gate should use.
#
# Because the workloads are Python functions (not C symbols), Callgrind cannot
# `--toggle-collect` on them the way the C tool does. Instead each workload is
# run at two rep counts (R1, R2) and the instruction counts are subtracted:
#
#     Ir/op = ( Ir(R2) - Ir(R1) ) / ( R2 - R1 )
#
# which cancels every fixed cost the two legs *share* — interpreter startup,
# imports and the one-time per-workload setup — leaving the pure per-operation
# cost. Sharing them is not automatic; see the next paragraph.
#
# The two legs are separate processes, so their fixed costs are only equal if
# they are made to be. Without a pinned hash seed each gets its own — the
# interpreter startups measured here lie ~3.4e5 Ir apart across seeds — and what
# fails to cancel lands in the result divided by (R2 - R1): an ABSOLUTE error of
# a few thousand Ir on every row, however much the row is worth. Three runs of
# the same commit, before and after pinning:
#
#     encode: typical        4896 / 11464 / 6298      ->  7813 / 7813 / 7813
#     decode: typical       24810 / 26577 / 26769     -> 25551 / 25551 / 25551
#
# Pinning does not make the number exact — a handful of Ir still moves between
# runs, and a *different* fixed seed lands a few tenths of a percent away
# (7810 at seed 0 and 7, 7824 at seed 42), because the seed reaches the measured
# loop's own dict lookups too. What it removes is the term that swamps them.
#
# The same residual is ±47% on a 7.8k row and ±5% on a 25k row, which is why
# this has hidden for so long: on `decode: composite` (~246k) it reads as 1.4%
# and looks like ordinary noise. It is worse still on the blob rows, where
# BR2 - BR1 is 2 rather than 100. Since the whole point of this number is to be
# gate-able, the seed is pinned below; override it only to see how much a row
# depends on it.
#
# The blob 1MB rows use their own, much smaller rep counts (BR1/BR2, default
# 1 and 3): a megabyte of copying per op is slow under Callgrind, and with the
# seed pinned the subtraction cancels fixed cost just as well at three reps as
# at three hundred. (Without it those rows are the *worst* case, not an equal
# one: the residual is divided by 2 instead of by 100.)
#
# Those rows are the ones this tool exists for — the one-shot/streaming delta is
# the cost of §5.1's divisible-run path with the host's memory subsystem and
# scheduler taken out of it, which is not something MB/s can show.
#
# Prereqs: valgrind, and `pip install -e .` so `import sofab` works.
# Usage:   bash bench/run_callgrind.sh          # defaults R1=10 R2=110, BR1=1 BR2=3
#          R1=20 R2=520 bash bench/run_callgrind.sh
#          WORKLOADS="encode_composite decode_composite" bash bench/run_callgrind.sh
#          PYTHONHASHSEED=7 bash bench/run_callgrind.sh   # a different, still fixed seed
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
SCRIPT="$ROOT/bench/perfbench.py"
R1="${R1:-10}"
R2="${R2:-110}"
BR1="${BR1:-1}"
BR2="${BR2:-3}"
# Both legs of every subtraction must run with the same hash seed -- see the note
# at the top. Exported rather than prefixed onto the valgrind line so nothing
# invoked from here can be measured with a different one.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

if ! command -v valgrind >/dev/null 2>&1; then
    echo "error: valgrind not found (needed for instruction counts)." >&2
    echo "       install it, e.g.  apt-get install valgrind" >&2
    exit 1
fi
if (( R2 <= R1 )) || (( BR2 <= BR1 )); then
    echo "error: R2 ($R2) must be greater than R1 ($R1), and BR2 ($BR2) than BR1 ($BR1)." >&2
    exit 1
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# The BENCH_SPEC row set, in the order the table prints it. The optional
# `blob 1MB passthrough` row is absent: this port implements no pass-through,
# and BENCH_SPEC has such a port omit the row rather than print a placeholder.
WORKLOADS="${WORKLOADS:-encode_u64_array encode_typical encode_blob_oneshot \
encode_blob_stream encode_composite decode_u64_array decode_typical decode_blob \
decode_composite decode_composite_skip}"

run_cg() { # $1 workload, $2 reps, $3 tag
    valgrind --tool=callgrind --callgrind-out-file="$OUT/$3.out" \
        "$PY" "$SCRIPT" "$1" "$2" >/dev/null 2>"$OUT/$3.log"
}

ir_of()    { grep -m1 '^summary:' "$OUT/$1.out" | awk '{print $2}'; }
bytes_of() { grep -ohE 'bytes=[0-9]+' "$OUT/$1.log" | head -1 | cut -d= -f2; }

label() {
    case "$1" in
        encode_u64_array)      echo "encode: u64 array (1000)";;
        encode_typical)        echo "encode: typical message";;
        encode_blob_oneshot)   echo "encode: blob 1MB one-shot";;
        encode_blob_stream)    echo "encode: blob 1MB streaming";;
        encode_composite)      echo "encode: composite";;
        decode_u64_array)      echo "decode: u64 array (1000)";;
        decode_typical)        echo "decode: typical message";;
        decode_blob)           echo "decode: blob 1MB";;
        decode_composite)      echo "decode: composite";;
        decode_composite_skip) echo "decode: composite skip-all";;
        *)                     echo "$1";;
    esac
}

# The megabyte workloads get the small rep pair; everything else the default one.
reps_for() {
    case "$1" in
        *_blob|*_blob_oneshot|*_blob_stream) echo "$BR1 $BR2";;
        *)                                   echo "$R1 $R2";;
    esac
}

echo ">> Measuring instructions/op under Callgrind (R1=$R1, R2=$R2;" \
     "blob rows BR1=$BR1, BR2=$BR2; PYTHONHASHSEED=$PYTHONHASHSEED; this is slow) ..."
echo
echo "==============================================================================="
echo " SofaBuffers Python instruction cost   (Callgrind, Ir/op)"
echo " instructions/op: lower is better. Deterministic & machine-independent."
echo "==============================================================================="
printf "%-26s %16s %9s\n" "Workload" "instr/op" "bytes"
printf "%-26s %16s %9s\n" "--------" "--------" "-----"

for w in $WORKLOADS; do
    read -r lo_reps hi_reps <<<"$(reps_for "$w")"
    ops=$(( hi_reps - lo_reps ))
    run_cg "$w" "$lo_reps" "$w.lo"
    run_cg "$w" "$hi_reps" "$w.hi"
    lo="$(ir_of "$w.lo")"; hi="$(ir_of "$w.hi")"
    b="$(bytes_of "$w.hi")"
    iperop="$(awk -v lo="${lo:-0}" -v hi="${hi:-0}" -v ops="$ops" \
        'BEGIN{ if (ops>0) printf "%d", (hi-lo)/ops; else print "-" }')"
    printf "%-26s %16s %9s\n" "$(label "$w")" "${iperop:--}" "${b:--}"
done
echo
echo "Ir = instructions retired (Callgrind). Independent of CPU clock and OS"
echo "scheduling; depends only on the executed code, so it compares across machines."
echo "The blob 1MB rows are read against each other: one-shot -> streaming is what"
echo "the divisible-run path (CORELIB_PLAN §5.1) costs, with bandwidth taken out."
echo "Caveat for that pair on x86-64: the one-shot row is a single 1,000,000-byte"
echo "memcpy, which glibc serves from its ERMS (rep movsb) path and Valgrind counts"
echo "at ~1 instruction per byte, while the streaming row's 4096-byte copies take"
echo "the vectorised path at a fraction of that. Ir/op therefore reports one-shot as"
echo "the dearer of the two though it does strictly less work — for these two rows"
echo "read the MB/s from 'perfbench.py bench'; every other row is Ir/op's to tell."
