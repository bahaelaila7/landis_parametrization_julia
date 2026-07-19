#!/usr/bin/env bash
# Generic Pan training submit script. Single node, one task, all cores; resumable + preemption-safe.
# Takes ANY config as an argument — no per-run copy-paste.
#
#     sbatch tools/submit.sbatch <config.yml> [run_name]
#     sbatch --job-name=fl853 tools/submit.sbatch runs/fl853_igelmo_4cell_shade.yml
#     sbatch --cpus-per-task=32 --export=ALL,PAN_TIMING=1 tools/submit.sbatch runs/foo.yml scan32   # scaling test
#
#   run_name (optional) = output subdir under $PAN_OUT; defaults to the config's basename.
#   Override any #SBATCH default on the command line (--job-name, --cpus-per-task, --partition, --time, ...).
#
# ONE-TIME before first submit — warm the Julia depot on a MATCHING-ARCH node so the job never compiles on the
# clock (pkgimages are CPU-arch tagged; warm on the same CPU family you'll run on), and build the slim DB:
#     JULIA_PKG_PRECOMPILE_AUTO=1 ./julia_gdal.sh --project=. -e 'using Pan'   # re-run after a src/ pull
#     python3 FIA/FIA_DATA_PREP/build_slim_db.py --ecos 8.5.3 --out runs/slim_853.duckdb

#SBATCH --job-name=pan
#SBATCH --partition=batch           # 30-day limit for the full run; use "batch" for short/scaling tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72              # candidate mode (PAN_PARALLEL below) is NUMA-local → scales past one socket:
# try 64 then 128 and pick the best per-gen time. (soa mode is bandwidth/NUMA-bound and typically plateaus ~24–32.)
#SBATCH --mem=24G                       # node has 764 GB; candidate mode allocates 1 work-SoA per thread (+val) —
# a few hundred MB at 128 threads on top of ~3.2 GiB peak RSS. Bump if you enable ramdisk-output.
#SBATCH --time=7-00:00:00
#SBATCH --constraint=Genoa              # ONE arch, matching your pre-warmed depot (pkgimages are CPU-arch tagged)
#SBATCH --signal=B:TERM@120             # SIGTERM to the batch script 120s before the limit → flush/resume trap runs
#SBATCH --requeue                       # auto-resume after preemption (runner continues from the latest checkpoint)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=bma09868@uga.edu    # Where to send mail
#SBATCH --mail-type=END,FAIL                 # Mail events (BEGIN, END, FAIL, ALL)

set -uo pipefail

# ---------- args ----------
CONFIG="${1:?usage: sbatch tools/submit.sbatch <config.yml> [run_name]}"
# Make a RELATIVE config absolute against the SUBMIT dir NOW — before we cd into the repo below. The cluster yaml
# lives outside the checkout (e.g. $SCRATCH/runs/yml/), so a bare "yml/foo.yml" would otherwise be looked up in the repo.
case "$CONFIG" in /*) ;; *) CONFIG="$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}/$CONFIG" 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}/$CONFIG")" ;; esac
[ -f "$CONFIG" ] || {
  echo "config not found: $CONFIG"
  exit 2
}
RUN_NAME="${2:-$(basename "${CONFIG%.*}")}" # output subdir; default = config basename without extension

# ---------- paths (EDIT $PAN_OUT / $JULIA_DEPOT_PATH for your account) ----------
# All honor a PRE-SET value (:= default) so a SECOND, ISOLATED environment (e.g. a git-worktree checkout of another
# branch with a struct change) can point PAN_ROOT + JULIA_DEPOT_PATH elsewhere WITHOUT disturbing the running jobs'
# checkout/depot — a preempted+requeued old job keeps using its own code+depot. Pass via --export=ALL,PAN_ROOT=...,
# JULIA_DEPOT_PATH=... (or export them before sbatch). PAN_OUT/PAN_DATA can stay shared (per-run output_dir subdirs).
export SCRATCH="${SCRATCH:-/scratch/$USER}"
export RUNSDIR="${RUNSDIR:-$SCRATCH/runs}"
export PAN_ROOT="${PAN_ROOT:-$SCRATCH/landis_parametrization_julia}"   # the repo checkout you submitted from
export PAN_OUT="${PAN_OUT:-$RUNSDIR/outputs}"                          # DURABLE output root (persistent) — outputs land here, not in the repo
export PAN_DATA="${PAN_DATA:-$RUNSDIR/data}"                           # resolves ${PAN_DATA}/*.csv|*.duckdb in the yaml (CSVs + slim DB live here)
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$SCRATCH/.julia}"         # must be pre-warmed (see header); layer as "$SCRATCH/.julia_split:$SCRATCH/.julia" for an isolated writable depot that reuses shared artifacts
SLIM_DB="${PAN_SLIM_DB:-$PAN_DATA/slim_853.duckdb}"                    # slim DB to node-stage; override per-config with PAN_SLIM_DB

# ---------- threads / BLAS ----------
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 # (julia_gdal.sh also pins this) no BLAS oversubscription
export JULIA_EXCLUSIVE=1      # pin Julia threads 1:1 to cores (helps NUMA locality)

# ---------- parallelism strategy ----------
# candidate: each candidate×rep eval runs single-threaded on its own thread-local (NUMA-local) SoA — no shared-SoA
#            cross-socket contention → scales past one socket. Recommended for SEARCH/fit runs (igelmo etc.).
# soa:       parallelize over sites WITHIN each eval — use for SPATIAL/raster (single-eval) runs.
export PAN_PARALLEL=candidate

# ---------- precompile: isolated from the run ----------
export JULIA_PKG_PRECOMPILE_AUTO=0 # never compile on the clock; rely on the pre-warmed depot

# ---------- slim DB → node-local scratch (fast; skipped if the file isn't there → yaml's cohorts_db_path is used) ----------
if [ -n "${SLURM_TMPDIR:-}" ] && [ -f "$SLIM_DB" ]; then
  cp "$SLIM_DB" "$SLURM_TMPDIR/" && export FIADB="$SLURM_TMPDIR/$(basename "$SLIM_DB")"
elif [ -f "$SLIM_DB" ]; then
  export FIADB="$SLIM_DB"
fi # else: leave FIADB unset → the config's cohorts_db_path wins

# ---------- optional levers (uncomment, or pass via --export=ALL,VAR=val) ----------
export PAN_TIMING=1 # per-gen wall breakdown → run.log. NOTE: in candidate mode the sim/loss SPLIT is racy (many
# threads share the timer); the per-gen TOTAL is still meaningful. Use soa mode for a clean split.
# export PAN_SKIP_VAL=1          # BENCH ONLY: skip held-out val re-sim (faster). Don't use for a real run — val objectives
# (cv_val_cache) are needed to pick the final model. In candidate mode val is parallel anyway.
# export PAN_RESUME_FROM=1500    # ROLL BACK: resume from search_state@1500.jld2 instead of the latest checkpoint
#                                # (e.g. a risky change corrupted later gens). Also: latest|none|/abs/path.jld2.
#                                # Gen-based CSVs (metrics.csv, reseed_trace.csv) are backed up + truncated to <=1500.
# export PAN_RAMDISK=auto        # opt-in ramdisk output ($SLURM_TMPDIR if set, else /dev/shm); default OFF = write to $PAN_OUT
# export PAN_SYNC_SEC=600        # ramdisk→durable backstop interval
#module load inotify-tools 2>/dev/null || true        # enables event-driven ramdisk sync when PAN_RAMDISK is set

cd "$PAN_ROOT"
echo "config=$CONFIG  run=$RUN_NAME  threads=$SLURM_CPUS_PER_TASK  FIADB=${FIADB:-<from yaml>}  out=$PAN_OUT/$RUN_NAME"
# exec so the runner BECOMES the batch process → it directly receives the B:TERM signal and its flush/resume trap fires.
exec tools/train_resumable.sh "$CONFIG" "$RUN_NAME" "$SLURM_CPUS_PER_TASK"
