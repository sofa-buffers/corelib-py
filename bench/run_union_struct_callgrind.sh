#!/usr/bin/env bash
#
# Struct-union-option cost in instructions/op (Callgrind), for corelib-py#169.
#
# Same two-rep-count subtraction as bench/run_callgrind.sh — see that script for
# why — applied to bench/union_struct_option_shapes.py's drivers. PYTHONHASHSEED is pinned for
# the same reason: the subtraction only cancels the interpreter's fixed cost if
# both legs have the same fixed cost.
#
# Read `struct_visitor` against `struct_bound_defaults` for what moving a union onto the
# destination table buys, and `plain_bound` against the same row on another
# revision for what a message WITHOUT a union pays for the feature existing.
#
# Usage: bash bench/run_union_struct_callgrind.sh [driver ...]
#        R1=5 R2=105 SOFAB_PUREPYTHON=1 bash bench/run_union_struct_callgrind.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
SCRIPT="$ROOT/bench/union_struct_option_shapes.py"
R1="${R1:-20}"
R2="${R2:-520}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
DRIVERS=("$@")
if [ ${#DRIVERS[@]} -eq 0 ]; then
    DRIVERS=(plain_bound struct_bound struct_bound_defaults struct_bound_no_reset struct_bound_into struct_visitor struct_visitor_field)
fi

command -v valgrind >/dev/null 2>&1 || { echo "error: valgrind not found" >&2; exit 1; }
(( R2 > R1 )) || { echo "error: R2 must exceed R1" >&2; exit 1; }

OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
run_cg() { valgrind --tool=callgrind --callgrind-out-file="$OUT/$3.out" \
    "$PY" "$SCRIPT" "$1" "$2" >/dev/null 2>"$OUT/$3.log"; }
ir_of() { grep -m1 '^summary:' "$OUT/$1.out" | awk '{print $2}'; }

engine="$("$PY" -c "import sys;sys.path.insert(0,'$ROOT/src');import sofab;print(sofab.IMPL)")"
echo ">> struct union option shapes, Ir/op (Callgrind, R1=$R1 R2=$R2, engine=$engine)"
echo
printf "%-22s %14s\n" "driver" "Ir/op"
printf "%-22s %14s\n" "------" "-----"
for d in "${DRIVERS[@]}"; do
    run_cg "$d" "$R1" "$d.lo"
    run_cg "$d" "$R2" "$d.hi"
    awk -v d="$d" -v lo="$(ir_of "$d.lo")" -v hi="$(ir_of "$d.hi")" -v ops="$((R2 - R1))" \
        'BEGIN{ printf "%-22s %14d\n", d, (hi-lo)/ops }'
done
echo
echo "8 bound scalars + one union field (struct option, only lvl sent) per op."
