#!/bin/bash
# HBAD3P pipeline driver — every stage of the model, in order, for one region (docs/hbad3p_pipeline.md; configuration: tools/hbad3/pipeline3p.env).
#   tools/hbad3/pipeline3p.sh <REGION> <TAG> [stage ...]       REGION: GAFL | APP | IP | WSP | NCSC | L3:x.y.z      default stages: all, in this order
# Stages (R4-wide data products are built once and skipped when present; FORCE=1 rebuilds):
#   data       interval product (Pan R4 builder cache) -> results/hbad2_r4/*.arrow, checked byte-identical to the Pan cache
#   climate    scripts/hbad3/climate_join.py   -> plot / interval climate normals + period values
#   water      scripts/hbad3/water_balance.py  -> interval AET/PET (period, plot normal)
#   forest     scripts/hbad3/subplot_forest.py -> forested fraction of every subplot / microplot visit
#   vdata      tools/hbad3/verify3p.jl data    -> data-product checks (region)
#   fit        tools/hbad3/fit3p.jl            -> results/hbad3/fit3p_<TAG>/ (Sim A growth + mortality)
#   recruit    tools/hbad3/recruit3p.jl        -> fit3p_<TAG>/recruitment.csv (recruitment, probe-exposure Poisson fit)
#   freerun    tools/hbad3/freerun3p.jl        -> fit3p_<TAG>/freerun_val.csv (+ log)
#   crossing   tools/hbad3/crossing3p.jl       -> in-growth diagnostic (log)
#   vmodel     tools/hbad3/verify3p.jl model   -> kernel / likelihood / fit / recruitment checks of fit3p_<TAG>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source wt_env.sh; source tools/hbad3/pipeline3p.env
REGION=${1:?region}; TAG=${2:?tag}; shift 2
STAGES=${*:-data climate water forest vdata fit recruit freerun crossing vmodel}
export HBAD2_REGION="$REGION"
FD=results/hbad3/fit3p_$TAG; LOGD=results/hbad3/pipeline_$TAG; mkdir -p "$LOGD"
JL="./julia_gdal.sh --project=. --threads=$THREADS"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGD/pipeline.log"; }
need() { [ "${FORCE:-0}" = 1 ] || ! [ -s "$1" ]; }
run() { local st=$1; shift; say "== $st: $*"; "$@" > "$LOGD/$st.log" 2>&1 || { say "!! $st FAILED (see $LOGD/$st.log)"; tail -20 "$LOGD/$st.log"; exit 1; }; tail -n "${TAILN:-6}" "$LOGD/$st.log" | sed 's/^/   /' | tee -a "$LOGD/pipeline.log"; }
for st in $STAGES; do case $st in
  data)
    src="$HBAD3_PAN_R4/results/hbad2_r4/intervals"
    [ -f "$src/cache_key.txt" ] || { say "!! no Pan interval cache at $src (build it in Pan: load_hbad2_intervals)"; exit 1; }
    mkdir -p "$HBAD2_DATA_DIR"
    for f in intervals.arrow interval_trees.arrow interval_ingrowth.arrow; do
      need "$HBAD2_DATA_DIR/$f" && cp "$src/$f" "$HBAD2_DATA_DIR/$f"
      cmp -s "$src/$f" "$HBAD2_DATA_DIR/$f" || { say "!! $HBAD2_DATA_DIR/$f differs from the Pan cache $src/$f"; exit 1; }
    done
    say "== data: interval product = Pan cache ($(tr '\n' ' ' < "$src/cache_key.txt"))";;
  climate) need "$HBAD3_CLIM_CSV" && run climate "$HBAD3_PYTHON" -I scripts/hbad3/climate_join.py . "$HBAD2_DATA_DIR/intervals.arrow" _r4 || say "== climate: present";;
  water)   need "$HBAD3_WATER_CSV" && run water "$HBAD3_PYTHON" -I scripts/hbad3/water_balance.py . "$HBAD2_DATA_DIR/intervals.arrow" _r4 || say "== water: present";;
  forest)  need "$HBAD3_FOREST_CSV" && run forest "$HBAD3_PYTHON" -I scripts/hbad3/subplot_forest.py "$HBAD3_FIADB" || say "== forest: present";;
  vdata)   run vdata $JL tools/hbad3/verify3p.jl data;;
  fit)     run fit $JL tools/hbad3/fit3p.jl "$TAG";;
  recruit) run recruit $JL tools/hbad3/recruit3p.jl "$TAG";;
  freerun) TAILN=12 run freerun $JL tools/hbad3/freerun3p.jl "$TAG" val;;
  crossing) TAILN=2 run crossing $JL tools/hbad3/crossing3p.jl "$TAG";;
  vmodel)  TAILN=40 run vmodel $JL tools/hbad3/verify3p.jl model "$TAG";;
  *) say "!! unknown stage $st"; exit 1;;
esac; done
say "DONE $REGION $TAG: $STAGES"
