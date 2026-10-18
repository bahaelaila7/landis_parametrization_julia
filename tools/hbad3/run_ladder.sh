#!/bin/bash
# HBAD3 S3 ablation ladder (sequential; each run: tools/hbad3/fit_tied.jl). Log per run in results/hbad3/tied_<tag>/log.txt
cd /workspace/landis_hbad3 && source wt_env.sh
export HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot HBAD2_VIGOUR=1 HBAD2_SIGV_MAX=0.5 HBAD2_EVALS=${EVALS:-6000}
run() { tag=$1; shift; echo "== $tag $*"; env "$@" ./julia_gdal.sh --project=. --threads=${THREADS:-4} tools/hbad3/fit_tied.jl $tag > results/hbad3/ladder_$tag.out 2>&1; grep "FIT " results/hbad3/tied_$tag/log.txt; }
run T1comp  HBAD3_TIE="gOD:ST,gRD:ST,gOH:ST,gRH:ST"
run T2mort  HBAD3_TIE="mbg:WD+ST,mD:WD+ST,msup:WD+ST,sdimax:g"
run T3shape HBAD3_TIE="cH:g,cD:g,Hmax:g,Dmax:g"
run T4mix   HBAD3_TIE="cH:g,cD:g,Hmax:HX,Dmax:DX,gOD:ST,gRD:ST,gOH:ST,gRH:ST,mbg:WD+ST,mD:WD+ST,msup:WD+ST,sdimax:g"
run T5trait HBAD3_TIE="rH:HX+ST+SW,rD:WD+ST+SW,cH:g,cD:g,Hmax:HX,Dmax:DX,gOD:ST,gRD:ST,gOH:ST,gRH:ST,mbg:WD+ST,mD:WD+ST,msup:WD+ST,sdimax:g"
echo LADDER_DONE
