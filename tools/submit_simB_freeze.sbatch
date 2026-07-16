#!/usr/bin/env bash
# Sim-B (phased frozen-growth) submit: SELECT seeds from a finished Sim-A run, then TRAIN — resumable +
# preemption-safe. Two steps in one job: (1) build the seed DB from the Sim-A p101 front [idempotent],
# (2) launch tools/train_resumable.sh with sobol_candidates_db + igel_mu pinned to the actual seed count.
#
#     sbatch tools/submit_simB_freeze.sbatch <simB_config.yml> <sim_a_run_dir> [run_name]
#     sbatch --job-name=simBfz tools/submit_simB_freeze.sbatch \
#            runs/fl853_igelmo_4cell_shade_simB_freeze.yml "$PAN_OUT/fl853_igelmo_4cell_shade_n1000"
#
#   sim_a_run_dir = the finished Sim-A output_dir (has search_state@N.jld2 + cv_val_cache.csv).
#   run_name (opt) = Sim-B output subdir under $PAN_OUT; defaults to the config's basename.
#
# Env levers (pass via --export=ALL,VAR=val):
#   SIMB_VAL=1        select seeds on the VALIDATION p101 front (needs cv_val_cache.csv). Default: TRAIN.
#   SIMB_MU=25        total seeds = 5 positions + (SIMB_MU-5) crowding-weighted-random. Default 25.
#   SIMB_WARMUP=10    ignore Sim-A gens <= this when building the front. Default 10.
#   SIMB_SEED=1       RNG seed for the crowding-weighted sampling (reproducible). Default 1.
#   SIMB_SEED_DB=...  seed-DB path. Default $PAN_DATA/simB_seeds_<run_name>.duckdb.
#   PAN_RESELECT=1    force re-selection even if the seed DB exists (default: reuse ⇒ stable seeds on resume).
#
# The Sim-B config supplies the FREEZE/PHASE knobs (submit pins seeds + igel_mu on top):
#   search_mode: igelmo   fix_growth: true   igel_freeze_seed_growth: true
#   igel_seed_maturity: true   igel_maturity: <gens>   no_establishment: false   (+ your Sim-B loss/data knobs)
#
# ONE-TIME warmup (matching-arch node, same as submit.sbatch):
#     JULIA_PKG_PRECOMPILE_AUTO=1 ./julia_gdal.sh --project=. -e 'using Pan'

#SBATCH --job-name=panBfz
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --constraint=Genoa              # ONE arch, matching your pre-warmed depot (pkgimages are CPU-arch tagged)
#SBATCH --signal=B:TERM@120             # SIGTERM 120s before the limit → train_resumable's flush/resume trap runs
#SBATCH --requeue                       # auto-resume after preemption (selector is skipped, training continues)
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=bma09868@uga.edu
#SBATCH --mail-type=END,FAIL

set -uo pipefail

# ---------- args ----------
CONFIG="${1:?usage: sbatch tools/submit_simB_freeze.sbatch <simB_config.yml> <sim_a_run_dir> [run_name]}"
SIMA_RUN="${2:?need the finished Sim-A run dir (output_dir with search_state@N.jld2)}"
# absolutize BEFORE cd into the repo (the yaml / run dir live outside the checkout)
case "$CONFIG"   in /*) ;; *) CONFIG="$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}/$CONFIG" 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}/$CONFIG")" ;; esac
case "$SIMA_RUN" in /*) ;; *) SIMA_RUN="$(readlink -f "${SLURM_SUBMIT_DIR:-$PWD}/$SIMA_RUN" 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}/$SIMA_RUN")" ;; esac
[ -f "$CONFIG" ]   || { echo "config not found: $CONFIG"; exit 2; }
[ -d "$SIMA_RUN" ] || { echo "Sim-A run dir not found: $SIMA_RUN"; exit 2; }
RUN_NAME="${3:-$(basename "${CONFIG%.*}")}"

# ---------- paths (same layout as tools/submit.sbatch — EDIT for your account) ----------
export SCRATCH="/scratch/$USER"
export RUNSDIR="$SCRATCH/runs"
export PAN_ROOT="$SCRATCH/landis_parametrization_julia"
export PAN_OUT="$RUNSDIR/outputs"
export PAN_DATA="$RUNSDIR/data"
export JULIA_DEPOT_PATH="$SCRATCH/.julia"
SLIM_DB="${PAN_SLIM_DB:-$PAN_DATA/slim_853.duckdb}"

# ---------- threads / BLAS / parallelism (candidate mode for search) ----------
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export JULIA_EXCLUSIVE=1
export PAN_PARALLEL=candidate
export JULIA_PKG_PRECOMPILE_AUTO=0      # never compile on the clock (pre-warmed depot)

# ---------- slim DB → node-local scratch (same as submit.sbatch) ----------
if [ -n "${SLURM_TMPDIR:-}" ] && [ -f "$SLIM_DB" ]; then
  cp "$SLIM_DB" "$SLURM_TMPDIR/" && export FIADB="$SLURM_TMPDIR/$(basename "$SLIM_DB")"
elif [ -f "$SLIM_DB" ]; then
  export FIADB="$SLIM_DB"
fi

cd "$PAN_ROOT"

# ---------- STEP 1: select seeds from the Sim-A p101 front (idempotent) ----------
SIMB_MU="${SIMB_MU:-25}"; SIMB_WARMUP="${SIMB_WARMUP:-10}"; SIMB_SEED="${SIMB_SEED:-1}"
mkdir -p "$PAN_DATA"
SEED_DB="${SIMB_SEED_DB:-$PAN_DATA/simB_seeds_${RUN_NAME}.duckdb}"
CNT_FILE="$SEED_DB.count"
VALFLAG=""; [ "${SIMB_VAL:-0}" = "1" ] && VALFLAG="--val"

if [ -f "$SEED_DB" ] && [ -f "$CNT_FILE" ] && [ "${PAN_RESELECT:-0}" != "1" ]; then
  N_SEEDS="$(cat "$CNT_FILE")"
  echo "seeds: reusing $SEED_DB ($N_SEEDS seeds) — set PAN_RESELECT=1 to rebuild"
else
  echo "seeds: selecting from $SIMA_RUN (mu=$SIMB_MU warmup=$SIMB_WARMUP seed=$SIMB_SEED ${VALFLAG:-train})"
  SEL_OUT="$(./julia_gdal.sh --project=. tools/select_simB_seeds.jl \
              "$SIMA_RUN" "$SEED_DB" --mu "$SIMB_MU" --warmup "$SIMB_WARMUP" --seed "$SIMB_SEED" $VALFLAG 2>&1)"
  echo "$SEL_OUT"
  N_SEEDS="$(printf '%s\n' "$SEL_OUT" | sed -n 's/^SEEDS=//p' | tail -1)"
  [ -n "$N_SEEDS" ] && [ -f "$SEED_DB" ] || { echo "seed selection FAILED (no SEEDS= line / no DB written)"; exit 3; }
  echo "$N_SEEDS" > "$CNT_FILE"
fi
[ "$N_SEEDS" -ge 5 ] 2>/dev/null || { echo "only $N_SEEDS seeds — front too small; pick a run with a larger p101 or lower SIMB_MU"; exit 3; }
[ "$N_SEEDS" -lt "$SIMB_MU" ] && echo "NOTE: emitted $N_SEEDS < requested mu=$SIMB_MU (small p101) → igel_mu pinned to $N_SEEDS"

# ---------- STEP 2: train — pin seeds + igel_mu to the actual count (μ MUST match, else random-growth fallback) ----------
export PAN_EXTRA_OVERRIDES="\"sobol_candidates_db\"=>\"$SEED_DB\",\"sobol_top_frac\"=>1.0,\"igel_mu\"=>$N_SEEDS"
echo "config=$CONFIG  run=$RUN_NAME  seeds=$SEED_DB (igel_mu=$N_SEEDS)  threads=$SLURM_CPUS_PER_TASK  out=$PAN_OUT/$RUN_NAME"
# exec so the runner BECOMES the batch process → it directly receives B:TERM and its flush/resume trap fires.
exec tools/train_resumable.sh "$CONFIG" "$RUN_NAME" "$SLURM_CPUS_PER_TASK"
