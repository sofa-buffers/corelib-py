#!/usr/bin/env bash
# Ir/op for bench/allocbench.py's drivers, the way genbench.sh does it: two runs
# per driver, the low one subtracted so interpreter start-up and the message
# build fall out and what is left is the decode.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export PYTHONHASHSEED=0
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
run(){ valgrind --tool=callgrind --callgrind-out-file="$OUT/$2.out" python3 "$ROOT/bench/allocbench.py" "$1" "$3" >/dev/null 2>&1; }
ir(){ grep -m1 '^summary:' "$OUT/$1.out" | awk '{print $2}'; }
for d in "$@"; do
  run "$d" "$d.lo" 10; run "$d" "$d.hi" 110
  awk -v d="$d" -v lo="$(ir "$d.lo")" -v hi="$(ir "$d.hi")" 'BEGIN{printf "%-18s %12d Ir/op\n", d, (hi-lo)/100}'
done
