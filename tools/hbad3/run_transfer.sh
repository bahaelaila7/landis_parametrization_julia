#!/bin/bash
# transfer evaluations (tools/hbad3/transfer_eval.jl) with each fit's own structural env; args: model specs "v2:<tag>:<sigv_max>" or "h3:<tag>:<sform>"
cd /workspace/landis_hbad3 && source wt_env.sh
export HBAD2_DATA_DIR=results/hbad2_r4 HBAD3_WATER_CSV=results/hbad3/interval_water_r4.csv HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot HBAD2_VIGOUR=1
for m in "$@"; do
  IFS=: read kind tag x <<< "$m"
  if [ "$kind" = v2 ]; then env HBAD2_SIGV_MAX=$x ./julia_gdal.sh --project=. --threads=${THREADS:-2} tools/hbad3/transfer_eval.jl $kind:$tag > results/hbad3/transfer_$tag.out 2>&1
  else env HBAD2_SIGV_MAX=0.5 HBAD3_SFORM=$x ./julia_gdal.sh --project=. --threads=${THREADS:-2} tools/hbad3/transfer_eval.jl $kind:$tag > results/hbad3/transfer_$tag.out 2>&1; fi
  grep " ALL " results/hbad3/transfer_$tag.out
done
