#!/usr/bin/env bash
# General reselection + analysis pipeline for any Pan MO-CMA-ES cross-validation run.
#   scripts/cv_analysis.sh <CONFIG.yml> [n_folds]
# Steps:
#   (1) generate held-out (val) objectives per fold via the CV_RESELECT hook, if not already present
#       (scores every archive candidate on that fold's val split → cv_reselect_metrics.csv + cv_val_fronts.csv)
#   (2) per-fold front-area percentile sweep plots — TRAIN and VAL (p25/p50/p75/p100, knee + "median" on p100)
#   (3) two Pareto fronts (p100-selected vs last) aggregated over folds into 8 positions — TRAIN and VAL spaces
#   (4) per-fold convergence plots (train vs val archive-min aggregate; p100 iteration marked; pop λ + archive size)
# Idempotent: step (1) skips folds already done, so re-running only regenerates plots.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG="${1:?usage: cv_analysis.sh CONFIG.yml [n_folds]}"
NF="${2:-5}"
OUTDIR=$(grep -E '^[[:space:]]*output_dir:' "$CFG" | head -1 | sed -E 's/^[[:space:]]*output_dir:[[:space:]]*"?([^"]*)"?.*/\1/')
J="./julia_gdal.sh --project=."
echo "=== cv_analysis: config=$CFG  output_dir=$OUTDIR  folds=$NF ==="

echo "--- (1) generate held-out objectives (CV_RESELECT hook) ---"
for k in $(seq 1 "$NF"); do
  if [ -f "$OUTDIR/fold_$k/cv_reselect_metrics.csv" ] && [ -f "$OUTDIR/fold_$k/cv_val_fronts.csv" ]; then
    echo "  fold_$k: already present, skipping"
  else
    echo "  fold_$k: generating (scoring archive on val)..."
    $J --threads=4 -e \
      "using Pan; Pan.CV_RESELECT[]=true; Pan.run_from_yaml(\"$CFG\"; overrides=Dict(\"n_folds\"=>$NF, \"fold_index\"=>$k))" \
      2>&1 | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | grep -aiE 'VAL-ranked|ERROR' | sed "s/^/    /"
  fi
done

echo "--- (1b) designate 5 positions on val fronts (best on val by area; extremes/knee/median/best_aggregate) ---"
for k in $(seq 1 "$NF"); do
  $J --threads=2 tools/cv_designate.jl "$OUTDIR/fold_$k" 2>&1 | grep -aiE 'fold_|ERROR' | sed "s/^/  /"
done

echo "--- (2) per-fold sweep plots (train + val) ---"
for k in $(seq 1 "$NF"); do
  $J --threads=2 tools/sweep_fold_percentiles.jl     "$OUTDIR/fold_$k" 2>&1 | grep -aiE 'wrote|ERROR' | sed "s/^/  f$k train: /"
  $J --threads=2 tools/sweep_fold_percentiles_val.jl "$OUTDIR/fold_$k" 2>&1 | grep -aiE 'wrote|ERROR' | sed "s/^/  f$k val:   /"
done

echo "--- (3) two fronts (train + val, 8 positions) ---"
$J --threads=2 tools/cv_two_fronts_val.jl "$OUTDIR" "$NF" 2>&1 | grep -aiE 'wrote|ERROR' | sed "s/^/  /"

echo "--- (4) convergence plots ---"
$J --threads=2 tools/cv_convergence.jl "$OUTDIR" "$NF" 2>&1 | grep -aiE 'fold_|wrote|ERROR' | sed "s/^/  /"

echo "=== cv_analysis complete: $OUTDIR ==="
ls "$OUTDIR"/cv_two_fronts_*.png "$OUTDIR"/cv_convergence.png 2>/dev/null | sed "s/^/  /"
