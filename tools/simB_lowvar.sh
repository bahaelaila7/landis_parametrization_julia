#!/usr/bin/env bash
# D15 test: low-variance Sim B procedure (PROB_MORT only, Sim A growth kept) on two CV folds.
cd /workspace/landis_region4 && source wt_env.sh
export PAN_DATA=/workspace/landis_region4/data_r4 PAN_OUT=/workspace/landis_region4/runs_local CALIB_SIZE_THR=0
spec="$1"; fit=${spec%%:*}; hold=${spec##*:}; tag=$(echo $fit | tr '+' '_')
SIMB_REPS=2 CALIB_SPLIT=$fit ./julia_gdal.sh --project=. --threads=4 tools/calib_coord.jl configs/region4/gafl_B.yml runs_local/simB_dbg/s2_pe0.02.jld2 runs_local/cvBlv_$tag pmort,pmort~,pmort~ > runs_local/cvBlv_$tag.log 2>&1
echo "fit $fit: $(grep -a '^round' runs_local/cvBlv_$tag.log | tail -1)"
SIMB_REPS=4 ./julia_gdal.sh --project=. --threads=4 tools/tost_select.jl configs/region4/gafl_B.yml runs_local/cvBlv_$tag/calibrated.jld2 1 $hold 0.05,0.1,0.2 > runs_local/cvBlv_${tag}_eval_$hold.log 2>&1
cp runs_local/cvBlv_$tag/tost_select_${hold}_groups.csv runs_local/cvBlv_${tag}_heldout_${hold}_groups.csv
echo "  held-out $hold: $(grep -a '\[1/1\]' runs_local/cvBlv_${tag}_eval_$hold.log)"
