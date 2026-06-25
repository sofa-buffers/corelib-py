#!/usr/bin/env bash
#
# SofaBuffers Python — machine-independent instruction cost.
#
# Runs each benchmark workload under Callgrind and reports instructions retired
# per operation (Ir/op). Unlike wall-clock or CPU time, instruction counts are
# deterministic and independent of the host's clock speed and scheduler, so the
# numbers are comparable across machines (and against the C/C++/Rust/Go tools —
# the workloads, ids and values are identical).
#
# Because the workloads are Python functions (not C symbols), Callgrind cannot
# `--toggle-collect` on them the way the C tool does. Instead each workload is
# run at two rep counts (R1, R2) and the instruction counts are subtracted:
#
#     Ir/op = ( Ir(R2) - Ir(R1) ) / ( R2 - R1 )
#
# which cancels *all* fixed cost exactly — interpreter startup, imports and the
# one-time per-workload setup — leaving the pure per-operation cost.
#
# Prereqs: valgrind, and `pip install -e .` so `import sofab` works.
# Usage:   bash bench/run_callgrind.sh          # defaults R1=10 R2=110
#          R1=20 R2=520 bash bench/run_callgrind.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
SCRIPT="$ROOT/bench/perfbench.py"
R1="${R1:-10}"
R2="${R2:-110}"

if ! command -v valgrind >/dev/null 2>&1; then
    echo "error: valgrind not found (needed for instruction counts)." >&2
    echo "       install it, e.g.  apt-get install valgrind" >&2
    exit 1
fi
if (( R2 <= R1 )); then
    echo "error: R2 ($R2) must be greater than R1 ($R1)." >&2
    exit 1
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
WORKLOADS=(encode_u64_array encode_typical decode_u64_array decode_typical)

run_cg() { # $1 workload, $2 reps, $3 tag
    valgrind --tool=callgrind --callgrind-out-file="$OUT/$3.out" \
        "$PY" "$SCRIPT" "$1" "$2" >/dev/null 2>"$OUT/$3.log"
}

ir_of()    { grep -m1 '^summary:' "$OUT/$1.out" | awk '{print $2}'; }
bytes_of() { grep -ohE 'bytes=[0-9]+' "$OUT/$1.log" | head -1 | cut -d= -f2; }

label() {
    case "$1" in
        encode_u64_array) echo "encode: u64 array (1000)";;
        encode_typical)   echo "encode: typical message";;
        decode_u64_array) echo "decode: u64 array (1000)";;
        decode_typical)   echo "decode: typical message";;
    esac
}

echo ">> Measuring instructions/op under Callgrind (R1=$R1, R2=$R2; this is slow) ..."
echo
echo "==============================================================================="
echo " SofaBuffers Python instruction cost   (Callgrind, Ir/op)"
echo " instructions/op: lower is better. Deterministic & machine-independent."
echo "==============================================================================="
printf "%-26s %16s %9s\n" "Workload" "instr/op" "bytes"
printf "%-26s %16s %9s\n" "--------" "--------" "-----"

ops=$(( R2 - R1 ))
for w in "${WORKLOADS[@]}"; do
    run_cg "$w" "$R1" "$w.lo"
    run_cg "$w" "$R2" "$w.hi"
    lo="$(ir_of "$w.lo")"; hi="$(ir_of "$w.hi")"
    b="$(bytes_of "$w.hi")"
    iperop="$(awk -v lo="${lo:-0}" -v hi="${hi:-0}" -v ops="$ops" \
        'BEGIN{ if (ops>0) printf "%d", (hi-lo)/ops; else print "-" }')"
    printf "%-26s %16s %9s\n" "$(label "$w")" "${iperop:--}" "${b:--}"
done
echo
echo "Ir = instructions retired (Callgrind). Independent of CPU clock and OS"
echo "scheduling; depends only on the executed code, so it compares across machines."
