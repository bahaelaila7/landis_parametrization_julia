#!/usr/bin/env bash
# Sanity check on a MID-training archive: extract the best-aggregate candidate from the latest
# search_state@N.jld2 and run scatter (linear) + TOST. Low threads (runs alongside training).
set -u
ROOT="${PAN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}}"
cd "$ROOT"
CB=runs/fl853_igelmo_cbalpct_simA_l1_8020_stdorg_outputs
CFG=runs/fl853_igelmo_cbalpct_simA_l1_8020_stdorg.yml
SUB=sanity_bestagg; D="$CB/$SUB"; mkdir -p "$D"
CK=$(ls "$CB"/search_state@*.jld2 | sed -E 's/.*@([0-9]+)\.jld2/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
echo "best-agg from $(basename "$CK")"
./julia_gdal.sh --project=. --threads=1 -e "using Pan; import JLD2;
  st=JLD2.load_object(\"$CK\"); arch=collect(st.archive);
  agg(c)=sum(Float64.(c.fx.objectives[1:2:end]))+sum(Float64.(c.fx.objectives[2:2:end]));
  b=arch[argmin(agg.(arch))]; JLD2.save_object(\"$D/params.jld2\", b.x);
  println(\"BESTAGG A_W=\", round(sum(Float64.(b.fx.objectives[1:2:end])),digits=5), \" A_AGB=\", round(sum(Float64.(b.fx.objectives[2:2:end])),digits=4), \" n_arch=\", length(arch))" 2>&1 | grep -aE "BESTAGG"
export PAN_SCATTER_MODES=linear
PAN_PARAMS="$D/params.jld2" PAN_OUTSUB="$SUB" ./julia_gdal.sh --project=. --threads=3 test/scatter_sim_obs.jl "$CFG"                 > "$D/scatter.log" 2>&1
PAN_PARAMS="$D/params.jld2" PAN_OUTSUB="$SUB" ./julia_gdal.sh --project=. --threads=3 test/tost_sim_obs.jl   "$CFG" "0.05,0.10,0.15" > "$D/tost.log"    2>&1
echo "DONE: $(ls "$D"/*.png 2>/dev/null | wc -l) png"
