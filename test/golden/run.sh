#!/usr/bin/env bash
# Run golden scenarios against a repo tree.  golden/run.sh <repo_root> <out_root> [jobs=2] [scenario ...]
# Each scenario = its own Julia process, 1 thread (deterministic, no state leakage between scenarios).
set -u
G="$(cd "$(dirname "$0")" && pwd)"; ROOT="$1"; OUT="$2"; JOBS="${3:-2}"; shift 3 || shift $#
mkdir -p "$OUT"
cd "$ROOT"; [ -f wt_env.sh ] && source wt_env.sh
export PAN_DATA=/workspace/landis_region4/data_r4 PAN_OUT="$OUT" PAN_PARALLEL=soa
if [ $# -gt 0 ]; then NAMES="$*"; else
  NAMES=$(grep -oE '^  "[a-zA-Z0-9_]+" *=> *\(' "$G/scenarios.jl" | grep -oE '"[a-zA-Z0-9_]+"' | tr -d '"' | grep -v '^bs_plot_only$'); NAMES="$NAMES bs_plot_only"; fi
run1() { ./julia_gdal.sh --project=. --threads=1 "$G/run_scenario.jl" "$1" "$OUT" > "$OUT/$1.log" 2>&1; tail -n 1 "$OUT/$1.log"; }
export -f run1; export G OUT
# bs_plot_only depends on bs_igelmo: run it last, after everything else
first=$(echo $NAMES | tr ' ' '\n' | grep -v '^bs_plot_only$'); last=$(echo $NAMES | tr ' ' '\n' | grep '^bs_plot_only$')
echo "$first" | xargs -P "$JOBS" -I{} bash -c 'run1 {}'
[ -n "$last" ] && run1 bs_plot_only
echo GOLDEN_ALL_DONE
