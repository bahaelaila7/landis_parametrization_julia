#!/usr/bin/env bash
# Pan training submit template. Single node, one task, all cores; resumable + preemption-safe.
# EDIT the four paths in the "paths" block and the SBATCH partition/constraint/time for your cluster.
#
# ONE-TIME before first submit — warm the Julia depot on a MATCHING-ARCH node so the job never compiles
# on the clock (pkgimages are CPU-arch tagged; warm on the same CPU family you'll run on):
#     cd <repo> && JULIA_PKG_PRECOMPILE_AUTO=1 ./julia_gdal.sh --project=. -e 'using Pan'
# Re-run that once after any `git pull` that touches src/.  Build the slim DB once too:
#     python3 FIA/FIA_DATA_PREP/build_slim_db.py --ecos 8.5.3 --out <somewhere>/slim_853.duckdb
#
#     sbatch tools/submit.sbatch

#SBATCH --job-name=pan853
#SBATCH --partition=batch_30d           # 30-day limit for the full run; use "batch" for short/scaling tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64              # Genoa node has 128; START at 64 and scaling-test 32/64/128 — the
                                        # knee is likely 32–64 (1775 sites, serial fraction, cross-NUMA). Not 128 blind.
#SBATCH --mem=32G                       # node has 764 GB; ~3.2 GiB peak RSS + ramdisk-output headroom
#SBATCH --time=7-00:00:00
#SBATCH --constraint=Genoa              # ONE arch, matching your pre-warmed depot (pkgimages are CPU-arch tagged)
#SBATCH --signal=B:TERM@120             # send SIGTERM to the batch script 120s before the limit → flush trap runs
#SBATCH --requeue                       # auto-resume after preemption (runner continues from the latest checkpoint)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -uo pipefail

# ---------- paths (EDIT) ----------
export PAN_ROOT="$SLURM_SUBMIT_DIR"                 # the repo checkout you submitted from
export PAN_OUT="$HOME/scratch/pan_out"              # DURABLE output root (persistent) — outputs land here, not in the repo
SLIM_DB="$PAN_ROOT/runs/slim_853.duckdb"            # slim DB colocated with the config/CSVs (upload it there; gitignored). FIADB env overrides the yaml's cohorts_db_path.
export JULIA_DEPOT_PATH="$HOME/.julia"              # must be pre-warmed (see header) and readable on the compute node
CONFIG="runs/fl853_igelmo_4cell_shade.yml"

# ---------- threads / BLAS ----------
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1                       # (julia_gdal.sh also pins this) no BLAS oversubscription
export JULIA_EXCLUSIVE=1                            # pin Julia threads 1:1 to cores (helps NUMA locality)

# ---------- precompile: isolated from the run ----------
export JULIA_PKG_PRECOMPILE_AUTO=0                  # never compile on the clock; rely on the pre-warmed depot

# ---------- slim DB → node-local scratch (fast, avoids shared-FS reads) ----------
if [ -n "${SLURM_TMPDIR:-}" ] && [ -f "$SLIM_DB" ]; then
  cp "$SLIM_DB" "$SLURM_TMPDIR/" && export FIADB="$SLURM_TMPDIR/$(basename "$SLIM_DB")"
else
  export FIADB="$SLIM_DB"
fi

# ---------- optional levers (uncomment to enable) ----------
# export PAN_TIMING=1            # per-gen wall breakdown (startup / ask / eval[sim vs loss] / tell / final[io]) → run.log
# export PAN_RAMDISK=auto        # opt-in ramdisk output ($SLURM_TMPDIR if set, else /dev/shm); default OFF = write to $PAN_OUT
# export PAN_SYNC_SEC=600        # ramdisk→durable backstop interval
module load inotify-tools 2>/dev/null || true       # enables event-driven ramdisk sync when PAN_RAMDISK is set

cd "$PAN_ROOT"
# exec so the runner BECOMES the batch process → it directly receives the B:TERM signal and its flush/resume trap fires.
exec tools/train_resumable.sh "$CONFIG" pan853 "$SLURM_CPUS_PER_TASK"
