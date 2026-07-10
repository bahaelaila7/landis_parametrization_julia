#!/usr/bin/env bash
# Per-candidate scatter/TOST/sMAPE for extracted p101 candidates, using the existing scripts exactly as
# for the L1 pruned/unpruned candidates (PAN_PARAMS + PAN_OUTSUB → p101_candidates/candidate_N/).
# Idempotent (skips candidates with smape_agebin.csv + tost_sim_obs_15pct.csv). Reduced threads to avoid
# the OOM cascade seen when run concurrently with a training job.
#   tools/run_p101_candidates.sh <config.yml> <output_dir> <n_candidates> [threads] [priority_csv]
# priority_csv = comma-separated candidate indices processed FIRST (in order), then the remainder ascending
# (used to do the 5 designated front positions first, for early assessment). Honors PAN_SCATTER_MODES.
set -u
cd /workspace/landis_parametrization_julia
CFG="$1"; OUT="$2"; NCAND="$3"; THREADS="${4:-4}"; PRIORITY="${5:-}"
J="./julia_gdal.sh --project=. --threads=$THREADS"
# build processing order: priority indices first (in given order), then the rest ascending
ORDER=""
if [ -n "$PRIORITY" ]; then
  ORDER=$(echo "$PRIORITY" | tr ',' ' ')
  for N in $(seq 1 "$NCAND"); do echo " $ORDER " | grep -q " $N " || ORDER="$ORDER $N"; done
else
  ORDER=$(seq 1 "$NCAND")
fi
for N in $ORDER; do
  SUB="p101_candidates/candidate_$N"; DIR="$OUT/$SUB"; PARAMS="$DIR/params.jld2"
  [ -f "$PARAMS" ] || { echo "MISSING $PARAMS"; continue; }
  if [ -f "$DIR/smape_agebin.csv" ] && [ -f "$DIR/tost_sim_obs_15pct.csv" ]; then
    echo "==================== candidate $N SKIP (complete) ===================="; continue
  fi
  echo "==================== candidate $N ===================="
  PAN_PARAMS="$PARAMS" PAN_OUTSUB="$SUB" $J test/scatter_sim_obs.jl "$CFG"                 > "$DIR/scatter.log" 2>&1
  PAN_PARAMS="$PARAMS" PAN_OUTSUB="$SUB" $J test/tost_sim_obs.jl   "$CFG" "0.05,0.10,0.15" > "$DIR/tost.log"    2>&1
  PAN_PARAMS="$PARAMS" PAN_OUTSUB="$SUB" $J test/smape_agebin.jl   "$CFG"                 > "$DIR/mape.log"    2>&1
  echo "candidate $N done: $(ls "$DIR"/*.png 2>/dev/null | wc -l) PNGs"
done
echo "ALL $NCAND p101 candidates complete"
