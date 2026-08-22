#!/usr/bin/env bash
#
# Decode-shape comparison in instructions/op (Callgrind).
#
# Same two-rep-count subtraction as bench/run_callgrind.sh — see that script for
# why — applied to bench/decode_shapes.py's drivers instead of the BENCH_SPEC
# rows. PYTHONHASHSEED is pinned because the subtraction only cancels the
# interpreter's fixed cost if both legs *have* the same fixed cost, and a random
# hash seed changes it by ~3e5 Ir, which lands in the result divided by R2-R1.
#
# Usage: bash bench/run_shapes_callgrind.sh [driver ...]
#        R1=20 R2=520 SOFAB_PUREPYTHON=1 bash bench/run_shapes_callgrind.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
SCRIPT="$ROOT/bench/decode_shapes.py"
R1="${R1:-10}"
R2="${R2:-110}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
DRIVERS=("$@")
if [ ${#DRIVERS[@]} -eq 0 ]; then
    DRIVERS=(pull drive feed_visitor feed_bound feed_bound_read feed_bound_bulk feed_bound_fresh)
fi

command -v valgrind >/dev/null 2>&1 || { echo "error: valgrind not found" >&2; exit 1; }
(( R2 > R1 )) || { echo "error: R2 must exceed R1" >&2; exit 1; }

OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
run_cg() { valgrind --tool=callgrind --callgrind-out-file="$OUT/$3.out" \
    "$PY" "$SCRIPT" "$1" "$2" >/dev/null 2>"$OUT/$3.log"; }
ir_of() { grep -m1 '^summary:' "$OUT/$1.out" | awk '{print $2}'; }

engine="$("$PY" -c "import sys;sys.path.insert(0,'$ROOT/src');import sofab;print(sofab.IMPL)")"
echo ">> decode shapes, Ir/op (Callgrind, R1=$R1 R2=$R2, engine=$engine)"
echo
printf "%-20s %14s %10s\n" "driver" "Ir/op" "per field"
printf "%-20s %14s %10s\n" "------" "-----" "---------"
for d in "${DRIVERS[@]}"; do
    run_cg "$d" "$R1" "$d.lo"
    run_cg "$d" "$R2" "$d.hi"
    awk -v d="$d" -v lo="$(ir_of "$d.lo")" -v hi="$(ir_of "$d.hi")" -v ops="$((R2 - R1))" \
        'BEGIN{ v=(hi-lo)/ops; printf "%-20s %14d %10.0f\n", d, v, v/36 }'
done
echo
echo "36 fields / 12 arrays / 51 elements per op; 'per field' divides by 36."
