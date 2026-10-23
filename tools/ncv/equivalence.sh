#!/usr/bin/env bash
# Equivalence tests of the nested-CV pipeline against the earlier production procedure (docs/nested_cv_gafl.md §5). Each prints PASS/FAIL.
#   tools/ncv/equivalence.sh <scratch dir>
# E1 split_csv data path == the drawn split: the split written by a split_only pass of the production config, fed back through split_csv (no train cap),
#    gives the same plot sets, and tost_select of the deployed parameter set on test gives byte-identical group tables (BS, HBAD v1, PnET).
# E2 HBAD3P interval product rewrite: a fold product whose sets are the R4 product's own sets reproduces the P3_GAFL fit's logged train/val NLL
#    (verify3p model F1) and passes every model check.
# E3 regression gate (BS 8.5.3 fixed-seed search) byte-identical to its baseline.
set -uo pipefail
NCV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; WS="$(dirname "$NCV")"; E="$1"; mkdir -p "$E"
cd "$NCV"; source wt_env.sh
export PAN_DATA="$WS/landis_region4/data_r4" PAN_OUT="$E"; PY="$WS/standmap/.venv/bin/python"; JL="./julia_gdal.sh --project=. --threads=8"
export TOST_REAL_ONLY=1
res() { echo "$1 $2" | tee -a "$E/RESULTS.txt"; }
: > "$E/RESULTS.txt"
declare -A CFG=([bs]=configs/region4/gafl_A.yml [hbad]=configs/region4/gafl_hbad.yml [pnet]=configs/region4/gafl_pnet.yml)
declare -A PAR=([bs]="$NCV/results/gafl/params_gafl_biomass_real.jld2" [hbad]="$WS/landis_hbad/results/hbad_gafl/params_gafl_hbad.jld2" [pnet]="$WS/landis_plugin/results/pnet_gafl/params_gafl_pnet.jld2")
# ---- E1
$PY -I tools/ncv/mkcfg.py configs/region4/gafl_A.yml "$E/A_split.yml" output_dir="$E/A_split" split_only=true bmax_floor_from_data=false anpp_floor_from_data=false
$JL -e "using Pan; Pan.run_from_yaml(\"$E/A_split.yml\")" > "$E/A_split.log" 2>&1
$PY -I tools/ncv/mkcfg.py "$E/A_split.yml" "$E/B_split.yml" output_dir="$E/B_split" split_csv="$E/A_split/split_plots.csv" val_frac=0 test_frac=0 train_cap=null
$JL -e "using Pan; Pan.run_from_yaml(\"$E/B_split.yml\")" > "$E/B_split.log" 2>&1
$PY -I -c "
import pandas as p, sys
k=['statecd','unitcd','countycd','plot']; a=p.read_csv('$E/A_split/split_plots.csv').sort_values(k).reset_index(drop=True); b=p.read_csv('$E/B_split/split_plots.csv').sort_values(k).reset_index(drop=True)
ok = a[k+['set']].equals(b[k+['set']]); print(ok, len(a), len(b), a.set.value_counts().to_dict()); sys.exit(0 if ok else 1)" && res PASS "E1a split_csv reproduces the drawn split's plot sets" || res FAIL "E1a split_csv plot sets"
for m in bs hbad pnet; do
  [ -f "${PAR[$m]}" ] || { res SKIP "E1b $m: no deployed params ${PAR[$m]}"; continue; }
  $PY -I tools/ncv/mkcfg.py "${CFG[$m]}" "$E/${m}_B.yml" split_csv="$E/A_split/split_plots.csv" val_frac=0 test_frac=0 train_cap=null
  for v in A B; do
    mkdir -p "$E/${m}_$v"; cp "${PAR[$m]}" "$E/${m}_$v/params.jld2"
    c="${CFG[$m]}"; [ $v = B ] && c="$E/${m}_B.yml"
    $JL tools/tost_select.jl "$c" "$E/${m}_$v/params.jld2" 1 test 0.05,0.1,0.2 > "$E/${m}_$v.log" 2>&1
  done
  cmp -s "$E/${m}_A/tost_select_test_groups.csv" "$E/${m}_B/tost_select_test_groups.csv" && [ -s "$E/${m}_A/tost_select_test_groups.csv" ] \
    && res PASS "E1b $m: test TOST group table identical ($(grep -a SELECTED "$E/${m}_A.log" | cut -c1-90))" || res FAIL "E1b $m: test TOST tables differ (see $E/${m}_{A,B}.log)"
done
# ---- E2
H3=$WS/landis_hbad3_ncv; H3D=$WS/landis_hbad3/results
$PY -I -c "
import pyarrow.ipc as ipc, pandas as p
t=ipc.open_file('$H3D/hbad2_r4/intervals.arrow').read_all().to_pandas()[['st','u','c','p','set']].dropna().drop_duplicates()
assert not t.duplicated(['st','u','c','p']).any(), 'a plot with two sets'
t.columns=['statecd','unitcd','countycd','plot','set']; t.to_csv('$E/r4_own_split.csv', index=False); print(len(t), 'plots')"
$JL tools/ncv/fold_product.jl "$H3D/hbad2_r4" "$E/r4_own_split.csv" "$E/r4_product" > "$E/r4_product.log" 2>&1
mkdir -p "$E/r3" && cp -r "$H3D/hbad3/fit3p_P3_GAFL" "$E/r3/"
( cd $H3 && source wt_env.sh && source tools/hbad3/pipeline3p.env && export HBAD2_DATA_DIR="$E/r4_product" HBAD3_R3DIR="$E/r3" HBAD2_REGION=GAFL \
  HBAD3_CLIM_CSV="$H3D/hbad3/interval_climate_r4.csv" HBAD3_WATER_CSV="$H3D/hbad3/interval_water_r4.csv" HBAD3_FOREST_CSV="$H3D/hbad3/subplot_forest_r4.csv" \
  && ./julia_gdal.sh --project=. --threads=8 tools/hbad3/verify3p.jl model P3_GAFL ) > "$E/e2_verify.log" 2>&1
grep -q "ALL CHECKS PASS" "$E/e2_verify.log" && res PASS "E2 HBAD3P fold product: P3_GAFL verify3p model all checks ($(grep -a '^PASS F1' "$E/e2_verify.log" | cut -c1-120))" || res FAIL "E2 HBAD3P (see $E/e2_verify.log)"
# ---- E3
tools/regress_gate.sh "$NCV" "$E/regress" 4 > /dev/null 2>&1
cmp -s docs/regress/baseline_853_refactor-step8.txt "$E/regress/REGRESS_ARCHIVE.txt" && res PASS "E3 regression gate byte-identical" || res FAIL "E3 regression gate differs"
echo EQUIVALENCE_DONE; cat "$E/RESULTS.txt"
