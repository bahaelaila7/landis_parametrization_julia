#!/usr/bin/env bash
# Generate a Sobol candidate POOL for seeding a search (igelmo `sobol_candidates_db`). Runs Pan with a
# `search_mode: "sobol"` config: it draws `sobol_n` Sobol params and scores EACH with the SAME fit_params
# simulator + _mo_objectives as the real search (so the pool's dominance ranking transfers directly), writing
# <output_dir>/losses.duckdb (table `sobol_results`). One-shot, no resume — if it dies, just re-submit.
#
#     sbatch tools/gen_sobol.sbatch <sobolgen_config.yml> [run_name]
#     sbatch tools/gen_sobol.sbatch runs/fl853_igelmo_4cell_shade_sobolgen.yml
#
#   The config MUST mirror the target search's loss/data/eco/tier/split settings (copy it, flip search_mode→sobol,
#   set sobol_n) so objectives land on the identical surface. Output pool → $PAN_OUT/<run_name>/losses.duckdb;
#   point the search's `sobol_candidates_db` at that path (+ `sobol_top_frac: 1.0` to expose the whole pool).
# Depends on the SAME pre-warmed depot + slim DB as tools/submit.sbatch (see that file's header).

#SBATCH --job-name=pan-sobolgen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32              # parametrize_sobol's candidate loop is SERIAL → parallelism is site-level
# (soa mode), which is bandwidth/NUMA-bound and plateaus ~24-32 cores. Going wider won't speed the sobol job.
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00               # sobol_n evals run SERIALLY (each site-parallel). Bump for large sobol_n.
#SBATCH --constraint=Genoa              # match the pre-warmed depot arch (pkgimages are CPU-arch tagged)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=bma09868@uga.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail

# ---------- args ----------
CONFIG="${1:?usage: sbatch tools/gen_sobol.sbatch <sobolgen_config.yml> [run_name]}"
case "$CONFIG" in /*) ;; *) CONFIG="$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}/$CONFIG" 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}/$CONFIG")" ;; esac
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG"; exit 2; }
RUN_NAME="${2:-$(basename "${CONFIG%.*}")}"    # output subdir under $PAN_OUT; default = config basename

# ---------- paths (mirror tools/submit.sbatch; all honor a pre-set value for an ISOLATED env) ----------
export SCRATCH="${SCRATCH:-/scratch/$USER}"
export RUNSDIR="${RUNSDIR:-$SCRATCH/runs}"
export PAN_ROOT="${PAN_ROOT:-$SCRATCH/landis_parametrization_julia}"
export PAN_OUT="${PAN_OUT:-$RUNSDIR/outputs}"
export PAN_DATA="${PAN_DATA:-$RUNSDIR/data}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$SCRATCH/.julia}"
SLIM_DB="${PAN_SLIM_DB:-$PAN_DATA/slim_853.duckdb}"

# ---------- threads / BLAS / parallelism ----------
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export JULIA_EXCLUSIVE=1
export PAN_PARALLEL=soa                 # parametrize_sobol evaluates candidates SERIALLY → parallelize over SITES
# within each eval (candidate mode would run each eval single-threaded → 2000 serial single-thread evals = slow).
export JULIA_PKG_PRECOMPILE_AUTO=0      # never compile on the clock; rely on the pre-warmed depot

# ---------- slim DB → node-local scratch (fast; skipped if absent → config's cohorts_db_path wins) ----------
if [ -n "${SLURM_TMPDIR:-}" ] && [ -f "$SLIM_DB" ]; then
  cp "$SLIM_DB" "$SLURM_TMPDIR/" && export FIADB="$SLURM_TMPDIR/$(basename "$SLIM_DB")"
elif [ -f "$SLIM_DB" ]; then
  export FIADB="$SLIM_DB"
fi

cd "$PAN_ROOT"
OUT="$PAN_OUT/$RUN_NAME"
echo "sobolgen: config=$CONFIG  run=$RUN_NAME  threads=$SLURM_CPUS_PER_TASK  FIADB=${FIADB:-<from yaml>}  → $OUT/losses.duckdb"
# Reuse the generic launcher: it overrides output_dir → $PAN_OUT/$RUN_NAME and runs Pan. parametrize_sobol has no
# checkpoints, so there's nothing to resume; a re-submit just regenerates the pool. NOT exec'd — we render UMAPs after.
tools/train_resumable.sh "$CONFIG" "$RUN_NAME" "$SLURM_CPUS_PER_TASK"; rc=$?

# ---------- UMAP of the pool (aggregate / ΣW / ΣAGB / balanced / dominance) ----------
# test/sobol_umap.jl reads sumW/sumAGB/params straight from losses.duckdb → no re-evaluation; embeds the param
# vectors and colours by each metric (yellow = lower = better; dominance = Pareto front rank over (ΣW,ΣAGB)).
DB="$OUT/losses.duckdb"
if [ "$rc" -eq 0 ] && [ -f "$DB" ]; then
  echo "sobolgen done → $DB ; rendering UMAP plots (aggregate / W / AGB / balanced / dominance) → $OUT/sobol_umap*.png"
  ./julia_gdal.sh --project=. --threads="$SLURM_CPUS_PER_TASK" test/sobol_umap.jl "$DB" "$OUT/sobol_umap" || echo "UMAP step failed (pool DB is still usable)"
else
  echo "sobolgen rc=$rc, losses.duckdb present=$([ -f "$DB" ] && echo yes || echo no) — skipping UMAP"
fi
exit $rc
