#!/usr/bin/env bash
# 3-fold CV over the existing train/val/test splits + all-data final fit, with the fixed calibration procedure
# (docs/gafl_calibration_log.md D8/D10). usage: tools/gafl_cv.sh <config.yml> <out_prefix> <start.jld2> [schedule]
# Env: TOST_REAL_ONLY (default 1), CALIB_SIZE_THR (default 0), THREADS (default 8); Sim B: SIMB_REPS (calibration) and
# EVAL_SIMB_REPS (held-out + final in-sample evaluation; default = SIMB_REPS). Writes <prefix>_<fit>/,
# <prefix>_<fit>_heldout_<split>_groups.csv and <prefix>.log; prints a summary line per fold.
set -u
CFG="$1"; PFX="$2"; START="$3"
SCHED="${4:-S,ratio,bmax,S,ratio,bmax,ratio~,S~,bmax~,ratio~,S~,bmax~}"
cd "$(dirname "$(readlink -f "$0")")/.." && source wt_env.sh
export TOST_REAL_ONLY="${TOST_REAL_ONLY:-1}" CALIB_SIZE_THR="${CALIB_SIZE_THR:-0}"; TH="${THREADS:-8}"
export SIMB_REPS="${SIMB_REPS:-1}"; ER="${EVAL_SIMB_REPS:-$SIMB_REPS}"
for spec in "train+test:val" "val+test:train" "train+val:test" "train+val+test:"; do
  fit=${spec%%:*}; hold=${spec##*:}; tag=$(echo $fit | tr '+' '_')
  CALIB_SPLIT=$fit ./julia_gdal.sh --project=. --threads=$TH tools/calib_coord.jl "$CFG" "$START" ${PFX}_$tag "$SCHED" > ${PFX}_$tag.log 2>&1
  echo "fit $fit: $(grep -a '^round' ${PFX}_$tag.log | tail -1)"
  if [ -n "$hold" ]; then
    SIMB_REPS=$ER ./julia_gdal.sh --project=. --threads=$TH tools/tost_select.jl "$CFG" ${PFX}_$tag/calibrated.jld2 1 $hold 0.05,0.1,0.2 > ${PFX}_${tag}_eval_$hold.log 2>&1
    cp ${PFX}_$tag/tost_select_${hold}_groups.csv ${PFX}_${tag}_heldout_${hold}_groups.csv
    echo "  held-out $hold: $(grep -a '\[1/1\]' ${PFX}_${tag}_eval_$hold.log)"
  else   # final all-data fit: in-sample evaluation with the evaluation replicate count
    SIMB_REPS=$ER ./julia_gdal.sh --project=. --threads=$TH tools/tost_select.jl "$CFG" ${PFX}_$tag/calibrated.jld2 1 $fit 0.05,0.1,0.2 > ${PFX}_${tag}_eval_insample.log 2>&1
    echo "  in-sample $fit: $(grep -a '\[1/1\]' ${PFX}_${tag}_eval_insample.log)"
  fi
done
echo CV_DONE
