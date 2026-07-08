#!/usr/bin/env bash
#
# SofaBuffers — CPU-independent performance gate (Callgrind Ir/op).
#
# Paranoia gate: closes (exits non-zero) if any workload's instructions-per-op
# moves more than the threshold (default 2%) in EITHER direction versus the base
# branch. It uses only Callgrind Ir/op — the deterministic, machine-independent
# signal — never wall-clock or cycle counts. CI measures base and head in the
# same job with the same toolchain, so the comparison is immune to toolchain
# drift; only a real code change moves the number.
#
# Subcommands:
#   parse             read a run_callgrind.sh table on stdin -> "<key> <ir>" lines
#   measure           run this repo's run_callgrind.sh and parse it
#   compare B H [T]   compare base file B vs head file H at threshold T% (exit 1 = closed)
set -euo pipefail
THRESHOLD_DEFAULT="${CALLGRIND_GATE_THRESHOLD:-2.0}"
cmd="${1:-}"; shift 2>/dev/null || true

parse() {
    awk '
        /^encode: u64 array/ { print "encode_u64_array " $(NF-1) }
        /^encode: typical/   { print "encode_typical "   $(NF-1) }
        /^decode: u64 array/ { print "decode_u64_array "  $(NF-1) }
        /^decode: typical/   { print "decode_typical "    $(NF-1) }
    '
}

case "$cmd" in
    parse) parse ;;
    measure)
        DIR=bench; [ -f benches/run_callgrind.sh ] && DIR=benches
        bash "$DIR/run_callgrind.sh" | parse
        ;;
    compare)
        base="$1"; head="$2"; thr="${3:-$THRESHOLD_DEFAULT}"
        awk -v thr="$thr" '
            FNR==NR { b[$1]=$2; next }
            { h[$1]=$2 }
            END {
                split("encode_u64_array encode_typical decode_u64_array decode_typical", K, " ")
                fail=0
                printf "%-20s %12s %12s %10s  %s\n","workload","base Ir/op","head Ir/op","delta","gate"
                printf "%-20s %12s %12s %10s  %s\n","--------","---------","---------","-----","----"
                for (i=1;i<=4;i++) {
                    k=K[i]; bv=b[k]; hv=h[k]
                    if (bv=="" || hv=="" || bv+0<=0) {
                        printf "%-20s %12s %12s %10s  MISSING\n", k, (bv==""?"-":bv), (hv==""?"-":hv), "-"
                        fail=1; continue
                    }
                    d=(hv-bv)/bv*100; ad=(d<0)?-d:d
                    st=(ad>thr)?"CLOSED":"ok"; if (ad>thr) fail=1
                    printf "%-20s %12d %12d %+9.2f%%  %s\n", k, bv, hv, d, st
                }
                print ""
                if (fail) { print "GATE CLOSED: Ir/op moved > " thr "% (or a value was missing). CPU-independent perf check failed."; exit 1 }
                print "GATE OPEN: every workload within +/-" thr "% Ir/op vs base."
            }
        ' "$base" "$head"
        ;;
    *) echo "usage: callgrind_gate.sh {parse|measure|compare BASE HEAD [THRESHOLD%]}" >&2; exit 2 ;;
esac
