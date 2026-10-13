#!/usr/bin/env bash
# 3-fold CV over the existing splits + all-data final fit (D8). Same procedure for every fold.
cd /workspace/landis_region4 && source wt_env.sh
export PAN_DATA=/workspace/landis_region4/data_r4 PAN_OUT=/workspace/landis_region4/runs_local CALIB_SIZE_THR=0 TOST_REAL_ONLY=1
SCHED='S,ratio,bmax,S,ratio,bmax,ratio~,S~,bmax~,ratio~,S~,bmax~'
START=runs_local/diag_bounds/a_best.jld2; CFG=configs/region4/gafl_A.yml
for spec in "train+test:val" "val+test:train" "train+val:test" "train+val+test:"; do
  fit=${spec%%:*}; hold=${spec##*:}; tag=$(echo $fit | tr '+' '_')
  CALIB_SPLIT=$fit ./julia_gdal.sh --project=. --threads=8 tools/calib_coord.jl $CFG $START runs_local/cvr_$tag "$SCHED" > runs_local/cvr_$tag.log 2>&1
  echo "fit $fit: $(grep -a '^round 12' runs_local/cvr_$tag.log)"
  if [ -n "$hold" ]; then
    ./julia_gdal.sh --project=. --threads=8 tools/tost_select.jl $CFG runs_local/cvr_$tag/calibrated.jld2 1 $hold 0.1,0.2 > runs_local/cvr_${tag}_eval_$hold.log 2>&1
    cp runs_local/cvr_$tag/tost_select_${hold}_groups.csv runs_local/cvr_${tag}_heldout_${hold}_groups.csv
    echo "  held-out $hold: $(grep -a '\[1/1\]' runs_local/cvr_${tag}_eval_$hold.log)"
  fi
done
echo CVR_DONE
