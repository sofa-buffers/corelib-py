#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export PYTHONHASHSEED=0
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
run(){ valgrind --tool=callgrind --callgrind-out-file="$OUT/$2.out" python3 "$ROOT/bench/genbench.py" "$1" "$3" >/dev/null 2>&1; }
ir(){ grep -m1 '^summary:' "$OUT/$1.out" | awk '{print $2}'; }
for d in "$@"; do
  run "$d" "$d.lo" 10; run "$d" "$d.hi" 110
  awk -v d="$d" -v lo="$(ir "$d.lo")" -v hi="$(ir "$d.hi")" 'BEGIN{printf "%-12s %10d Ir/op\n", d, (hi-lo)/100}'
done
