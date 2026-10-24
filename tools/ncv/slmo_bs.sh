#!/usr/bin/env bash
# Species-level multi-objective experiment on one nested-CV role of Biomass Succession (user request 2026-10-24; docs/nested_cv_gafl.md §6).
# The trade-off surface per species: from each of the role's three MO archive picks (min_A, knee, min_W) the per-group minimize refinement
# (tools/calib_coord.jl CALIB_MODE=minimize: every group accepts its own value of obj_g = max(lam a_g/a_ref_g, (1-lam) w_g/w_ref_g)) at
# lam = 0, 1/6, ..., 1  -> 21 sets, each with a per-group (a_g, w_g) on TRAIN. Selection on the inner VAL split, TOST-targeted, PER GROUP
# (tools/ncv/slmo_assemble.jl), assembled into one set; val then decides between the assembled set and the best single set (same rule).
# The outer TEST fold reports the val-chosen set once (plus, labelled as comparisons, the other one and the production selection of the role).
# Resumable (step markers in <out>/steps). Runs beside the main campaign (THREADS default 4).
#   tools/ncv/slmo_bs.sh [role=k1]
set -uo pipefail
NCV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; WS="$(dirname "$NCV")"; cd "$NCV"; source wt_env.sh
ROLE="${1:-k1}"; R="$NCV/runs_local/ncv"; D="$R/bs/$ROLE"; X="$NCV/runs_local/ncv_slmo/bs_$ROLE"; mkdir -p "$X"
export PAN_DATA="$WS/landis_region4/data_r4" PAN_OUT="$X" TOST_REAL_ONLY=1 CALIB_SIZE_THR=0 CALIB_SPLIT=train
PY="$WS/standmap/.venv/bin/python"; JL="./julia_gdal.sh --project=. --threads=${THREADS:-4}"; C="$D/config.yml"
SCHED="ratio~,S~,bmax~,ratio~,S~,bmax~"; LAMS="0 0.1667 0.3333 0.5 0.6667 0.8333 1"
[ -f "$D/picks.txt" ] && [ -f "$R/steps/bs/$ROLE/select.done" ] || { echo "role $ROLE of the main campaign is not complete"; exit 1; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$X/slmo.log"; }
step() { local id=$1; shift; local mk="$X/steps/$id.done" lg="$X/logs/$id.log"; [ -f "$mk" ] && return 0; mkdir -p "$(dirname "$mk")" "$(dirname "$lg")"
         log "run  $id"; local t0=$SECONDS; if ( "$@" ) < /dev/null >> "$lg" 2>&1; then date > "$mk"; log "done $id ($((SECONDS - t0)) s)"; else log "FAIL $id ($lg)"; tail -15 "$lg"; exit 1; fi; }
while read -r role pj lam aref wref; do
  for l in $LAMS; do
    step sweep_${role}_$l env CALIB_MODE=minimize CALIB_LAMBDA=$l CALIB_AREF=$aref CALIB_WREF=$wref $JL tools/calib_coord.jl "$C" "$D/mo_${role}_raw.jld2" "$X/sweep_${role}_$l" "$SCHED"
  done
done < "$D/picks.txt"
step cands bash -c "set -e; mkdir -p $X/cands; : > $X/cands/list.txt; for d in $X/sweep_*; do n=\$(basename \$d); cp \$d/calibrated.jld2 $X/cands/\$n.jld2; echo $X/cands/\$n.jld2 >> $X/cands/list.txt; done; wc -l $X/cands/list.txt"
step val bash -c "TOST_RANK=0.2,0.1,0.05,acc $JL tools/tost_select.jl $C \$(paste -sd, $X/cands/list.txt) 64 val 0.05,0.1,0.2"
step assemble $JL tools/ncv/slmo_assemble.jl "$C" "$X/cands" "$X/assembled.jld2"
step final_files bash -c "set -e; mkdir -p $X/final; cp $X/assembled.jld2 $X/final/a_assembled.jld2; cp $X/cands/selected_val/params.jld2 $X/final/b_best_single.jld2; cp $D/cands/selected_val/params.jld2 $X/final/c_production.jld2"
# val decides between the assembled and the best single set (the production set is not a candidate of this procedure: only scored on test for comparison)
step final_val bash -c "mkdir -p $X/final/val && cp $X/final/a_assembled.jld2 $X/final/b_best_single.jld2 $X/final/val/ && TOST_RANK=0.2,0.1,0.05,acc $JL tools/tost_select.jl $C $X/final/val/a_assembled.jld2,$X/final/val/b_best_single.jld2 2 val 0.05,0.1,0.2"
step final_test bash -c "mkdir -p $X/final/test/pairs && cp $X/final/a_assembled.jld2 $X/final/b_best_single.jld2 $X/final/c_production.jld2 $X/final/test/ && \
     TOST_DUMP=$X/final/test/pairs $JL tools/tost_select.jl $C $X/final/test/a_assembled.jld2,$X/final/test/b_best_single.jld2,$X/final/test/c_production.jld2 3 test 0.05,0.1,0.2"
step surface "$WS/PERSEUS_PRESENTATION/.venv/bin/python" -I tools/ncv/slmo_surface.py "$X"
log "SLMO DONE $ROLE"
