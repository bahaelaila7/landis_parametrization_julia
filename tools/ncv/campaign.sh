#!/usr/bin/env bash
# Nested-CV calibration campaign of the five Pan models on GA+FL (docs/nested_cv_gafl.md). Resumable: every step leaves
# <root>/steps/<model>/<role>/<step>.done after success and is skipped when the marker exists; gdigel and HBAD v2 searches continue from their latest
# checkpoint; after a crash or a container restart rerun the SAME command. One campaign per root at a time (flock).
#   tools/ncv/campaign.sh [models: all | bs,hbad,hbad3p,hbad2,pnet] [roles: all | k1,k2,k3,k4,k5,final]
# Env: NCV_SMOKE=1 (tiny budgets, root runs_local/ncv_smoke), NCV_ROOT, THREADS (8), PAN_DATA (landis_region4/data_r4), NCV_HBAD3 (hbad3-ncv worktree).
set -uo pipefail
# bash reads a script lazily: run from a snapshot so that editing this file while a campaign runs cannot corrupt it
if [ -z "${NCV_SNAPSHOT:-}" ]; then
  _home="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; mkdir -p "$_home/runs_local"; _snap="$(mktemp "$_home/runs_local/.campaign_snapshot_XXXXXX.sh")"
  cp "${BASH_SOURCE[0]}" "$_snap"; NCV_SNAPSHOT="$_snap" NCV_HOME="$_home" exec bash "$_snap" "$@"
fi
trap 'rm -f "$NCV_SNAPSHOT"' EXIT
NCV="$NCV_HOME"; WS="${WORKSPACE:-$(dirname "$NCV")}"
cd "$NCV"; source wt_env.sh
SMOKE="${NCV_SMOKE:-0}"
R="${NCV_ROOT:-$NCV/runs_local/ncv$([ "$SMOKE" = 1 ] && echo _smoke)}"; mkdir -p "$R"
H3="${NCV_HBAD3:-$WS/landis_hbad3_ncv}"; H3DATA="$WS/landis_hbad3/results"      # hbad3-ncv code; the R4 interval product + climate/water/forest tables
export PAN_DATA="${PAN_DATA:-$WS/landis_region4/data_r4}" PAN_OUT="$R"
DB="$PAN_DATA/fia_truth_r4.duckdb"; PY="$WS/standmap/.venv/bin/python"; TH="${THREADS:-8}"
JL="./julia_gdal.sh --project=. --threads=$TH"
MODELS="${1:-all}"; ROLES="${2:-all}"
[ "$MODELS" = all ] && MODELS="bs,hbad,hbad3p,hbad2,pnet"
[ "$ROLES" = all ] && ROLES="k1,k2,k3,k4,k5,final"
exec 9> "$R/.lock"; flock -n 9 || { echo "another campaign is running on $R (lock $R/.lock)"; exit 1; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$R/campaign.log"; }
# step <id> <command...>: run once; output in <root>/logs/<id>.log (appended, so a resumed search keeps its history); marker <root>/steps/<id>.done
step() {
  local id=$1; shift; local mk="$R/steps/$id.done" lg="$R/logs/$id.log"
  [ -f "$mk" ] && return 0
  mkdir -p "$(dirname "$mk")" "$(dirname "$lg")"; log "run  $id"; local t0=$SECONDS
  echo "=== $(date '+%F %T') $*" >> "$lg"
  if ( "$@" ) < /dev/null >> "$lg" 2>&1; then date '+%F %T' > "$mk"; log "done $id ($((SECONDS - t0)) s)"; else log "FAIL $id (log $lg)"; tail -15 "$lg"; exit 1; fi
}
# resume point in dir $1 (files matching regex $2, numbered @N): the SECOND-newest checkpoint, because the newest may be truncated by a kill during its
# (non-atomic) write; with a single checkpoint, that one. Costs at most ckpt_every generations of repeated work.
latest() { local f; f=$(ls "$1" 2>/dev/null | grep -E "$2" | sort -t@ -k2 -n); [ -z "$f" ] && return 0
           [ "$(echo "$f" | wc -l)" -ge 2 ] && echo "$f" | tail -2 | head -1 || echo "$f" | tail -1; }
# ---------------------------------------------------------------- budgets (production = the Oct-14 recipe; smoke = tiny)
if [ "$SMOKE" = 1 ]; then
  declare -A GDT=([bs]=40 [hbad]=40 [pnet]=24); GDMU="mu=4 patience=3 rel_tol=0.005 ckpt_every=1"; GDMO="mu=4 weights=4 patience=3 rel_tol=0.005 ckpt_every=1"
  declare -A SROOT=([bs]="S" [hbad]="gmult" [pnet]="foln"); declare -A SMIN=([bs]="S~" [hbad]="gmult~" [pnet]="foln~")
  H2A_T=60; H2J_T=48; H3_EVALS=48; TSK=4; H2CK="hbad2_calib.ckpt_every=1"
else
  # bias family = the "gdigel mu 16, fast restarts" run of docs/cmaes_gdigel.md (gd_bs_div: patience 15, rel_tol 0.005); mo family = mogd_full.sh
  declare -A GDT=([bs]=6400 [hbad]=6400 [pnet]=4800); GDMU="mu=16 patience=15 rel_tol=0.005"; GDMO="mu=16 weights=8 patience=15 rel_tol=0.005"
  declare -A SROOT=([bs]="S,ratio,bmax,S,ratio,bmax,ratio~,S~,bmax~,ratio~,S~,bmax~" [hbad]="gmult,gmult~,mmult~,gmult~,mmult~,gmult~" [pnet]="foln,foln~,foln~,foln~,foln~")
  declare -A SMIN=([bs]="ratio~,S~,bmax~,ratio~,S~,bmax~" [hbad]="gmult~,mmult~,gmult~,mmult~" [pnet]="foln~,foln~,foln~,foln~")
  H2A_T=40000; H2J_T=12000; H3_EVALS=6000; TSK=16; H2CK=""
fi
declare -A BASE=([bs]=configs/region4/gafl_A.yml [hbad]=configs/region4/gafl_hbad.yml [pnet]=configs/region4/gafl_pnet.yml)
SPLIT_KEYS="val_frac=0 test_frac=0 n_folds=1"
# ---------------------------------------------------------------- folds (once) and per-role inputs
folds() {
  step folds/build "$PY" -I tools/ncv/make_folds.py --db "$DB" --out "$R/folds"
  step folds/verify "$PY" -I tools/ncv/verify_folds.py "$R/folds"
}
prep() {   # role $1: the train-capped split every model uses, then the inputs derived from its train plots
  local r=$1 P="$R/prep/$1"
  # 1. split_only pass of the BS config: split_plots.csv (role file + the species-balanced train cap) and species_map.csv
  step prep/$r/split_cfg "$PY" -I tools/ncv/mkcfg.py configs/region4/gafl_A.yml "$P/split.yml" split_csv="$R/folds/split_$r.csv" $SPLIT_KEYS \
       output_dir="$P/split" split_only=true bmax_floor_from_data=false anpp_floor_from_data=false
  step prep/$r/split $JL -e "using Pan; Pan.run_from_yaml(\"$P/split.yml\")"
  # 2. Biomass Succession B_MAX / ANPP data floors from the train plots
  step prep/$r/floors $JL tools/build_floors_region.jl --run "$P/split" --db "$DB" --out "$P/floor"
  # 3. HBAD v1 empirical priors + translation from the train plots
  # (model structure fixed to the deployed HBAD v1, results/hbad/translation_selection.csv; coefficients refit on this role's train plots)
  step prep/$r/hbad_priors env PAN_HBAD_SPLIT="$P/split/split_plots.csv" PAN_HBAD_OUT="$P/hbad" PAN_HBAD_DB="$DB" PAN_HBAD_FIX_SELECTION="$NCV/results/hbad/translation_selection.csv" PAN_HBAD_TRAIN_ONLY=1 bash -c \
       "set -e; for t in hbad_cohorts hbad_translation hbad_priors; do ./julia_gdal.sh --project=. --threads=4 tools/\$t.jl; done"
  # 4. interval product with this role's sets (HBAD v2 recruitment fit, HBAD3P)
  step prep/$r/product $JL tools/ncv/fold_product.jl "$H3DATA/hbad2_r4" "$P/split/split_plots.csv" "$P/product"
}
# ---------------------------------------------------------------- BS / HBAD v1 / PnET: two candidate families, inner val selects, test reports
cohort_model() {
  local m=$1 r=$2 D="$R/$1/$2" P="$R/prep/$2"
  local extra=""
  [ "$m" = bs ] && extra="bmax_floor_csv=$P/floor_bmax.csv anpp_floor_csv=$P/floor_anpp.csv"
  [ "$m" = hbad ] && extra="hbad_priors_csv=$P/hbad/priors.csv hbad_coefs_csv=$P/hbad/translation_coefs.csv"
  step $m/$r/config "$PY" -I tools/ncv/mkcfg.py "${BASE[$m]}" "$D/config.yml" split_csv="$R/folds/split_$r.csv" $SPLIT_KEYS output_dir="$D/run" $extra
  local C="$D/config.yml"
  export TOST_REAL_ONLY=1 CALIB_SIZE_THR=0 CALIB_SPLIT=train
  # family bias: gdigel (TOST-aligned bias objective) -> best solution -> root refinement
  # candidate-parallel evaluation (each candidate sites-serial on its own thread), as the Oct-14 production runs (mogd_full.sh: PAN_PARALLEL=candidate)
  gd() { local out=$1; shift; local rs; rs=$(latest "$out" '^gd_state@[0-9]+\.jld2$'); PAN_PARALLEL=candidate $JL tools/gd_run.jl "$C" "$out" "$@" ${rs:+resume_from=$out/$rs}; }
  step $m/$r/gd_bias gd "$D/gd_bias" "${GDT[$m]}" $GDMU
  step checks/$m/$r/splits "$PY" -I tools/ncv/verify_splits.py "$R" "$m" "$r"
  step $m/$r/ref_bias bash -c "best=\$($PY -I -c \"import pandas as p; d=p.read_csv('$D/gd_bias/gd_solutions.csv'); print(d.loc[d.total.idxmin(), 'file'])\") && \
       cp $D/gd_bias/solutions/\$best $D/bias_raw.jld2 && CALIB_MODE=root $JL tools/calib_coord.jl $C $D/bias_raw.jld2 $D/ref_bias '${SROOT[$m]}'"
  # family mo: gdigel (accuracy + Wasserstein) -> knee / min_A / min_W -> minimize refinement with each pick's lambda
  step $m/$r/gd_mo gd "$D/gd_mo" "${GDT[$m]}" objective=mo $GDMO
  step $m/$r/picks bash -c "$JL tools/mo_archive_pick.jl $D/gd_mo > $D/picks.txt && test -s $D/picks.txt && cat $D/picks.txt"
  while read -r role pj lam aref wref; do
    step $m/$r/ref_mo_$role env CALIB_MODE=minimize CALIB_LAMBDA=$lam CALIB_AREF=$aref CALIB_WREF=$wref \
         bash -c "cp $pj $D/mo_${role}_raw.jld2 && $JL tools/calib_coord.jl $C $D/mo_${role}_raw.jld2 $D/ref_mo_$role '${SMIN[$m]}'"
  done < "$D/picks.txt"
  # inner-val selection among the candidates (the outer test fold is never used here)
  step $m/$r/select bash -c "set -e; mkdir -p $D/cands; cp $D/bias_raw.jld2 $D/cands/bias_raw.jld2; cp $D/ref_bias/calibrated.jld2 $D/cands/bias_ref.jld2
       for f in $D/mo_*_raw.jld2; do b=\$(basename \$f _raw.jld2); cp \$f $D/cands/\${b}_raw.jld2; cp $D/ref_\$b/calibrated.jld2 $D/cands/\${b}_ref.jld2; done
       L=\$(ls $D/cands/*.jld2 | paste -sd,); TOST_RANK=0.2,0.1,0.05,acc $JL tools/tost_select.jl $C \$L $TSK val 0.05,0.1,0.2"
  # outer test: report the val-selected set once (pairs dumped for the pooled out-of-fold TOST)
  if [ "$r" != final ]; then
    step $m/$r/test bash -c "mkdir -p $D/test_pairs && TOST_DUMP=$D/test_pairs $JL tools/tost_select.jl $C $D/cands/selected_val/params.jld2 1 test 0.05,0.1,0.2"
  fi
}
# ---------------------------------------------------------------- HBAD v2: recruitment data fit -> Sim A (cold) -> joint (v2j recipe)
hbad2() {
  local r=$1 D="$R/hbad2/$1" P="$R/prep/$1" ts=true; [ "$r" = final ] && ts=false
  step hbad2/$r/recruit bash -c "set -e; mkdir -p $D/recr; cd $H3 && source wt_env.sh && HBAD2_R2DIR=$P/product HBAD2_DATA_DIR=$P/product HBAD2_REGION=GAFL HBAD2_FORM=2 \
       HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot ./julia_gdal.sh --project=. tools/hbad2/recruitment_check.jl && \
       $PY -I -c \"import pandas as p; d=p.read_csv('$P/product/recruitment_coefs_subplot.csv'); d['D0']=2.54; d['H0']=3.0; d['A0']=5.0; d.to_csv('$D/recr/params_recruitment.csv', index=False); print(d)\""
  local hc="hbad2_calib.split_csv=$P/split/split_plots.csv hbad2_calib.priors_csv=$P/hbad/priors.csv hbad2_calib.recruitment_csv=$D/recr/params_recruitment.csv hbad2_calib.test_set=$ts $H2CK"
  step hbad2/$r/simA_cfg "$PY" -I tools/ncv/mkcfg.py configs/region4/hbad2_calib_simA.yml "$D/simA.yml" output_dir="$D/simA" trials=$H2A_T cmaes_lambda=32 cmaes_sigma0=0.3 \
       $hc hbad2_calib.start_theta= hbad2_calib.sigv_max=0.7 hbad2_calib.intervals_cache="$D/intervals"
  h2run() { local cfgf=$1 out=$2; local rs; rs=$(latest "$out" '^search_state@[0-9]+\.jld2$'); [ -n "$rs" ] && rs="$out/$rs"
            $JL -e "using Pan; Pan.run_from_yaml(\"$cfgf\"; overrides=Dict{String,Any}($( [ -n "$rs" ] && echo "\"resume_from\" => \"$rs\"" )))"; }
  step hbad2/$r/simA h2run "$D/simA.yml" "$D/simA"
  step checks/hbad2/$r/splits "$PY" -I tools/ncv/verify_splits.py "$R" hbad2 "$r"
  step hbad2/$r/joint_cfg0 "$PY" -I tools/ncv/mkcfg.py configs/region4/hbad2_calib_joint.yml "$D/joint0.yml" output_dir="$D/joint" trials=$H2J_T \
       $hc hbad2_calib.start_theta="$D/simA/best/theta.csv" hbad2_calib.intervals_cache="$D/intervals"
  step hbad2/$r/joint_refs bash -c "set -o pipefail; $JL tools/ncv/hbad2_refs.jl $D/joint0.yml | tee $D/joint_refs.txt"
  step hbad2/$r/joint_cfg bash -c "id=\$(grep -o 'ideal: \[[^]]*\]' $D/joint_refs.txt | cut -d' ' -f2-); nd=\$(grep -o 'nadir: \[[^]]*\]' $D/joint_refs.txt | cut -d' ' -f2-); \
       test -n \"\$id\" && test -n \"\$nd\" && $PY -I tools/ncv/mkcfg.py $D/joint0.yml $D/joint.yml \"hbad2_calib.ideal=\$id\" \"hbad2_calib.nadir=\$nd\""
  step hbad2/$r/joint h2run "$D/joint.yml" "$D/joint"
}
# ---------------------------------------------------------------- HBAD3P: cold Sim A fit -> recruitment -> free runs (hbad3-ncv stages)
hbad3p() {
  local r=$1 D="$R/hbad3p/$1" P="$R/prep/$1"; mkdir -p "$D"
  h3() { ( cd "$H3" && source wt_env.sh && source tools/hbad3/pipeline3p.env && export HBAD2_DATA_DIR="$P/product" HBAD3_R3DIR="$D" HBAD2_REGION=GAFL \
           HBAD3_CLIM_CSV="$H3DATA/hbad3/interval_climate_r4.csv" HBAD3_WATER_CSV="$H3DATA/hbad3/interval_water_r4.csv" HBAD3_FOREST_CSV="$H3DATA/hbad3/subplot_forest_r4.csv" \
           HBAD2_SIGMA=0.15 HBAD2_EVALS=$H3_EVALS && ./julia_gdal.sh --project=. --threads=$THREADS_H3 "$@" ); }
  THREADS_H3=$TH
  step checks/hbad3p/$r/splits "$PY" -I tools/ncv/verify_splits.py "$R" hbad3p "$r"
  step hbad3p/$r/vdata h3 tools/hbad3/verify3p.jl data
  step hbad3p/$r/fit h3 tools/hbad3/fit3p.jl "$r"
  step hbad3p/$r/recruit h3 tools/hbad3/recruit3p.jl "$r"
  step hbad3p/$r/freerun_val h3 tools/hbad3/freerun3p.jl "$r" val
  [ "$r" != final ] && step hbad3p/$r/freerun_test h3 tools/hbad3/freerun3p.jl "$r" test
  step hbad3p/$r/verify h3 tools/hbad3/verify3p.jl model "$r"
}
# ---------------------------------------------------------------- run
log "campaign start: models $MODELS roles $ROLES (root $R, smoke $SMOKE)"
folds
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra RS <<< "$ROLES"
for m in "${MS[@]}"; do
  for r in "${RS[@]}"; do
    prep "$r"
    case $m in
      bs|hbad|pnet) cohort_model "$m" "$r";;
      hbad2) hbad2 "$r";;
      hbad3p) hbad3p "$r";;
      *) log "unknown model $m"; exit 1;;
    esac
  done
done
step checks/splits_$(echo "$MODELS" | tr , _)_$(echo "$ROLES" | tr , _) "$PY" -I tools/ncv/verify_splits.py "$R" "$MODELS" "$ROLES"
log "CAMPAIGN DONE: models $MODELS roles $ROLES"
