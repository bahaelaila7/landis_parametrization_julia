#!/usr/bin/env bash
# Full p101 analysis pipeline for ONE finished MO run, idempotent (skips completed steps):
#   1) train sweep  2) build val fronts  3) val sweep  4) extract p101  5) reorganize to candidate_N/
#   6) front_positions (5 roles)  7) per-candidate LINEAR scatter/TOST/sMAPE — 5 positions FIRST (assess),
#   then the rest  8) archive_candidates.csv + symlink  9) p101 front-tost {5,10,15}×{train,val}
# Runs a single Pan job at a time (data-loads OOM the box if two run concurrently — DO NOT parallelize).
#   tools/p101_pipeline.sh <config.yml> <output_dir> [threads]
set -u
cd /workspace/landis_parametrization_julia
CFG="$1"; OUT="$2"; TH="${3:-8}"; J="./julia_gdal.sh --project=. --threads=$TH"
C="$OUT/p101_candidates"
export PAN_SCATTER_MODES=linear         # candidate scatters: linear only (per request)
echo "########## p101 pipeline: $(basename "$OUT") ##########"

[ -f "$OUT/ccigel_sweep.png" ]     || $J tools/sweep_fold_percentiles.jl "$OUT" "$OUT/ccigel_sweep.png"     2>&1 | grep -viE "^\[|Warning|@ |━|Running|└|┌" | tail -3
[ -f "$OUT/cv_val_fronts.csv" ]    || $J tools/build_val_fronts.jl "$OUT"                                    2>&1 | grep -viE "^\[|Warning|@ |━|Running|└|┌" | tail -2
[ -f "$OUT/ccigel_sweep_val.png" ] || $J tools/sweep_fold_percentiles_val.jl "$OUT" "$OUT/ccigel_sweep_val.png" 2>&1 | grep -viE "^\[|Warning|@ |━|Running|└|┌" | tail -3
[ -f "$C/manifest.csv" ]           || $J tools/extract_p101.jl "$OUT"                                         2>&1 | grep -viE "^\[|Warning|@ |━|Running|└|┌" | tail -2

# reorganize cand_XX.jld2 -> candidate_N/params.jld2 (idempotent)
for f in "$C"/cand_*.jld2; do [ -e "$f" ] || continue; n=$(basename "$f" .jld2 | sed 's/cand_0*//'); mkdir -p "$C/candidate_$n"; mv "$f" "$C/candidate_$n/params.jld2"; done
NCAND=$(( $(wc -l < "$C/manifest.csv") - 1 ))

# front_positions FIRST so we can process the 5 designated positions ahead of the rest
awk -F, 'NR==1{print "candidate,A_W,A_AGB,aggregate"; next}{print $1","$2","$3","($2+$3)}' "$C/manifest.csv" > "$C/archive_candidates.csv"
ln -sfn . "$C/candidates"
bash tools/front_positions.sh "$OUT" >/dev/null
POS=$(awk -F, 'NR>1{print $2}' "$C/front_positions.csv" | awk '!seen[$0]++' | paste -sd, -)   # unique candidate idx of the 5 roles
echo "  p101 candidates: $NCAND  |  5-position priority: $POS"

# per-candidate LINEAR scatter/TOST/sMAPE — 5 positions first, then the rest (idempotent)
bash tools/run_p101_candidates.sh "$CFG" "$OUT" "$NCAND" "$TH" "$POS"

# p101 front-tost plots
for SP in train val; do for PCT in 5 10 15; do
  [ -f "$C/pareto_tost_${PCT}pct_by_stratum_${SP}.png" ] || $J test/plot_pareto_tost.jl "$SP" "$PCT" "$C" 2>&1 | grep -viE "^\[|Warning|@ |━|Running|└|┌" | tail -1
done; done
echo "########## p101 pipeline DONE: $(basename "$OUT") ##########"
