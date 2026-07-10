#!/usr/bin/env bash
# Re-run scatter+TOST for the already-extracted sanity candidates with PRE-INJECTION cache
# (injected recruits + disturbance-overrides excluded) → honest sim↔obs. Writes sanity_<role>_pred.
set -u
cd /workspace/landis_parametrization_julia
CB=runs/fl853_igelmo_cbalpct_simA_l1_8020_stdorg_outputs
CFG=runs/fl853_igelmo_cbalpct_simA_l1_8020_stdorg.yml
export PAN_PREINJECT=1 PAN_SCATTER_MODES=linear
for role in bestW bestAGB bestagg; do
  P="$CB/sanity_$role/params.jld2"; [ -f "$P" ] || { echo "MISSING $P"; continue; }
  SUB="sanity_${role}_pred"; D="$CB/$SUB"; mkdir -p "$D"
  PAN_PARAMS="$P" PAN_OUTSUB="$SUB" ./julia_gdal.sh --project=. --threads=3 test/scatter_sim_obs.jl "$CFG"                 > "$D/scatter.log" 2>&1
  PAN_PARAMS="$P" PAN_OUTSUB="$SUB" ./julia_gdal.sh --project=. --threads=3 test/tost_sim_obs.jl   "$CFG" "0.05,0.10,0.15" > "$D/tost.log" 2>&1
  echo "$role pred done: $(ls "$D"/*.png 2>/dev/null|wc -l) png | stripped: $(grep -ao 'stripped [0-9]* supplied' "$D/scatter.log" 2>/dev/null|head -1)"
done
echo "ALL PREINJECT SANITY DONE"
