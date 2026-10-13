#!/usr/bin/env bash
# Re-score the v1 (D10) real-only CV folds + all-data fit at ±5/10/20 % (roster extension, D11).
cd /workspace/landis_region4 && source wt_env.sh
export PAN_DATA=/workspace/landis_region4/data_r4 PAN_OUT=/workspace/landis_region4/runs_local TOST_REAL_ONLY=1
CFG=configs/region4/gafl_A.yml; mkdir -p runs_local/v1_5pct
for spec in "train_test:val" "val_test:train" "train_val:test" "train_val_test:train+val+test"; do
  tag=${spec%%:*}; hold=${spec##*:}
  ./julia_gdal.sh --project=. --threads=2 tools/tost_select.jl $CFG runs_local/cvr_$tag/calibrated.jld2 1 $hold 0.05,0.1,0.2 > runs_local/v1_5pct/${tag}.log 2>&1
  echo "$tag held-out $hold: $(grep -a '\[1/1\]' runs_local/v1_5pct/${tag}.log)"
done
echo RESCORE_DONE
