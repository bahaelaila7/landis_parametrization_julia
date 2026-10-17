#!/usr/bin/env bash
# K-fold cross-validation runner for Pan Sim A.
#
# Launches `parametrize` once per fold (same split_seed ⇒ the K folds partition the same shuffle),
# each fold writing to <output_dir>/fold_<k>. Sequential by default; override FOLDS to run a subset
# (e.g. in parallel across machines). Each fold is independently resumable via the config's resume_from.
#
#   scripts/run_cv.sh runs/fl5_..._simA_cv.yml [n_folds]
#   THREADS=8 scripts/run_cv.sh CONFIG.yml 5
#   FOLDS="3 4" scripts/run_cv.sh CONFIG.yml 5     # only folds 3 and 4
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="${1:?usage: run_cv.sh CONFIG.yml [n_folds]}"
NFOLDS="${2:-5}"
THREADS="${THREADS:-auto}"
FOLDS="${FOLDS:-$(seq 1 "$NFOLDS")}"

for k in $FOLDS; do
  echo "======================= CV fold $k / $NFOLDS ======================="
  ./julia_gdal.sh --project=. --threads="$THREADS" -e \
    "using Pan; Pan.run_from_yaml(\"$CFG\"; overrides=Dict(\"n_folds\"=>$NFOLDS, \"fold_index\"=>$k))"
done

echo "======================= CV complete: fronts under <output_dir>/fold_1..$NFOLDS ======================="
echo "Next: ./julia_gdal.sh --project=. tools/cv_aggregate.jl <output_dir>   (front collapse + held-out metrics)"
echo "      ./julia_gdal.sh --project=. tools/cv_tost.jl <output_dir>        (per-stratum TOST equivalence)"
