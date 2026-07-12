#!/usr/bin/env bash
# Post-hoc analysis of a completed Pan MO run — drives tools/analyze_run.jl (convergence, front sweep with the
# 5 designated positions, p101 positions, param tables) and, with --sim, the per-candidate styled scatter/sMAPE/
# TOST plots (which fan out nested Julia jobs). Single node; NOT preemption-critical (idempotent — just re-run).
#
#     sbatch tools/analyze.sbatch <run_dir> [config.yml] [analyze_run flags...]
#     sbatch tools/analyze.sbatch $RUNSDIR/outputs/fl853_igelmo_4cell_shade_outputs                       # Phase 1 only (fast, no sim)
#     sbatch tools/analyze.sbatch $RUNSDIR/outputs/fl853_..._outputs runs/fl853_igelmo_4cell_shade.yml --sim        # + per-candidate plots (5 positions)
#     sbatch --cpus-per-task=32 tools/analyze.sbatch <run_dir> <config.yml> --sim --all --warmup 20                 # every p101 candidate
#
#   <run_dir>  : the run's output_dir (holds search_state@<gen>.jld2 + cv_val_cache.csv). Outputs land in <run_dir>/analysis/.
#   config.yml : REQUIRED only for --sim (reproduces the train/val split + drives the styled scripts). Omit for Phase-1-only.
#   flags      : forwarded verbatim to analyze_run.jl (--sim, --all, --warmup N, --outsub DIR).
# Depends on the SAME pre-warmed depot + slim DB as tools/submit.sbatch (see that file's header).

#SBATCH --job-name=pan-analyze
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24              # Phase 2 runs up to min(6, cpus/4) candidates concurrently × ~4 threads each
#SBATCH --mem=16G                       # main proc ~few GiB + up to ~6 nested sims (~1.4 GiB peak each)
#SBATCH --time=04:00:00                 # Phase 1 is minutes; --sim --all over a large p101 is the long case
#SBATCH --constraint=Genoa              # match the pre-warmed depot arch (pkgimages are CPU-arch tagged)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail

# ---------- args ----------
RUN_DIR="${1:?usage: sbatch tools/analyze.sbatch <run_dir> [config.yml] [flags...]}"
shift
# Resolve run_dir + an optional config (2nd positional if it isn't a --flag) against the SUBMIT dir NOW,
# before we cd into the repo. Everything else is forwarded to analyze_run.jl untouched.
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}/$1" 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}/$1")" ;; esac }
RUN_DIR="$(abspath "$RUN_DIR")"
CONFIG=""
if [ $# -gt 0 ]; then case "$1" in --*) ;; *)
  CONFIG="$(abspath "$1")"
  shift
  ;;
esac fi
[ -d "$RUN_DIR" ] || {
  echo "run_dir not found: $RUN_DIR"
  exit 2
}
[ -z "$CONFIG" ] || [ -f "$CONFIG" ] || {
  echo "config not found: $CONFIG"
  exit 2
}

# ---------- paths (mirror tools/submit.sbatch; EDIT for your account) ----------
export SCRATCH="/scratch/$USER"
export RUNSDIR="$SCRATCH/runs"
export PAN_ROOT="$SCRATCH/landis_parametrization_julia"
export PAN_DATA="$RUNSDIR/data"           # ${PAN_DATA}/*.duckdb|*.csv in the config expand from here (needed by --sim)
export JULIA_DEPOT_PATH="$SCRATCH/.julia" # pre-warmed, matching-arch depot

# ---------- precompile off + concurrency cap ----------
export JULIA_PKG_PRECOMPILE_AUTO=0 # never compile on the clock
export OPENBLAS_NUM_THREADS=1
export PAN_ANALYSIS_CPUS="$SLURM_CPUS_PER_TASK" # cap Phase-2 nested-job concurrency to the allocation (not the whole node)

cd "$PAN_ROOT"
echo "analyze: run_dir=$RUN_DIR  config=${CONFIG:-<none, Phase 1 only>}  flags=[$*]  cpus=$SLURM_CPUS_PER_TASK  data=$PAN_DATA"
# The main process is I/O-bound (it orchestrates nested Julia jobs via asyncmap) → 1 thread is enough.
exec ./julia_gdal.sh --project=. --threads=1 tools/analyze_run.jl "$RUN_DIR" ${CONFIG:+"$CONFIG"} "$@"
