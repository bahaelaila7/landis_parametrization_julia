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
#SBATCH --partition=batch_30d           # 30-day limit for the full run; use "batch" for short/scaling tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64              # Genoa node has 128; START at 64 and scaling-test 32/64/128 — the knee
                                        # is likely 32–64 (1775 sites, serial fraction, cross-NUMA). Not 128 blind.
#SBATCH --mem=32G                       # Genoa node has 764 GB; ~3.2 GiB peak RSS + ramdisk-output headroom
#SBATCH --time=7-00:00:00
#SBATCH --constraint=Genoa              # ONE arch, matching your pre-warmed depot (pkgimages are CPU-arch tagged)
#SBATCH --signal=B:TERM@120             # SIGTERM to the batch script 120s before the limit → flush/resume trap runs
#SBATCH --requeue                       # auto-resume after preemption (runner continues from the latest checkpoint)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail

# ---------- args ----------
CONFIG="${1:?usage: sbatch tools/submit.sbatch <config.yml> [run_name]}"
RUN_NAME="${2:-$(basename "${CONFIG%.*}")}"          # output subdir; default = config basename without extension

# ---------- paths (EDIT $PAN_OUT / $JULIA_DEPOT_PATH for your account) ----------
export PAN_ROOT="$SLURM_SUBMIT_DIR"                  # the repo checkout you submitted from
export PAN_OUT="$HOME/scratch/pan_out"               # DURABLE output root (persistent) — outputs land here, not in the repo
export JULIA_DEPOT_PATH="$HOME/.julia"               # must be pre-warmed (see header) and readable on the compute node
SLIM_DB="${PAN_SLIM_DB:-$PAN_ROOT/runs/slim_853.duckdb}"   # slim DB to node-stage; override per-config with PAN_SLIM_DB

# ---------- threads / BLAS ----------
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1                        # (julia_gdal.sh also pins this) no BLAS oversubscription
export JULIA_EXCLUSIVE=1                             # pin Julia threads 1:1 to cores (helps NUMA locality)

# ---------- precompile: isolated from the run ----------
export JULIA_PKG_PRECOMPILE_AUTO=0                   # never compile on the clock; rely on the pre-warmed depot

# ---------- slim DB → node-local scratch (fast; skipped if the file isn't there → yaml's cohorts_db_path is used) ----------
if [ -n "${SLURM_TMPDIR:-}" ] && [ -f "$SLIM_DB" ]; then
  cp "$SLIM_DB" "$SLURM_TMPDIR/" && export FIADB="$SLURM_TMPDIR/$(basename "$SLIM_DB")"
elif [ -f "$SLIM_DB" ]; then
  export FIADB="$SLIM_DB"
fi                                                   # else: leave FIADB unset → the config's cohorts_db_path wins

# ---------- optional levers (uncomment, or pass via --export=ALL,VAR=val) ----------
# export PAN_TIMING=1            # per-gen wall breakdown (startup / ask / eval[sim vs loss] / tell / final[io]) → run.log
# export PAN_RAMDISK=auto        # opt-in ramdisk output ($SLURM_TMPDIR if set, else /dev/shm); default OFF = write to $PAN_OUT
# export PAN_SYNC_SEC=600        # ramdisk→durable backstop interval
module load inotify-tools 2>/dev/null || true        # enables event-driven ramdisk sync when PAN_RAMDISK is set

cd "$PAN_ROOT"
echo "config=$CONFIG  run=$RUN_NAME  threads=$SLURM_CPUS_PER_TASK  FIADB=${FIADB:-<from yaml>}  out=$PAN_OUT/$RUN_NAME"
# exec so the runner BECOMES the batch process → it directly receives the B:TERM signal and its flush/resume trap fires.
exec tools/train_resumable.sh "$CONFIG" "$RUN_NAME" "$SLURM_CPUS_PER_TASK"
