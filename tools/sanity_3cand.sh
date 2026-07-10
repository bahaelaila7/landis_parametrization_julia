#!/usr/bin/env bash
# Sanity check on the CURRENT latest archive: extract best-W, best-AGB, best-agg candidates and run
# scatter(linear) + TOST for each. Low threads (runs alongside training).
set -u
cd /workspace/landis_parametrization_julia
CB=runs/fl853_igelmo_cbalpct_simA_l1_8020_stdorg_outputs
CFG=runs/fl853_igelmo_cbalpct_simA_l1_8020_stdorg.yml
CK=$(ls "$CB"/search_state@*.jld2 | sed -E 's/.*@([0-9]+)\.jld2/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
GEN=$(echo "$CK" | sed -E 's/.*@([0-9]+)\.jld2/\1/')
echo "SANITY archive = $(basename "$CK") (gen $GEN)"
./julia_gdal.sh --project=. --threads=1 -e "using Pan; import JLD2;
  st=JLD2.load_object(\"$CK\"); arch=collect(st.archive);
  W(c)=sum(Float64.(c.fx.objectives[1:2:end])); A(c)=sum(Float64.(c.fx.objectives[2:2:end]));
  roles=[(\"bestW\",argmin([W(c) for c in arch])),(\"bestAGB\",argmin([A(c) for c in arch])),(\"bestagg\",argmin([W(c)+A(c) for c in arch]))];
  for (r,i) in roles
    d=\"$CB/sanity_\$r\"; mkpath(d); JLD2.save_object(joinpath(d,\"params.jld2\"), arch[i].x);
    println(\"ROLE \$r idx=\$i A_W=\", round(W(arch[i]),digits=5), \" A_AGB=\", round(A(arch[i]),digits=4));
  end;
  println(\"n_arch=\", length(arch))" 2>&1 | grep -aE "ROLE|n_arch"
export PAN_SCATTER_MODES=linear
for role in bestW bestAGB bestagg; do
  D="$CB/sanity_$role"
  PAN_PARAMS="$D/params.jld2" PAN_OUTSUB="sanity_$role" ./julia_gdal.sh --project=. --threads=3 test/scatter_sim_obs.jl "$CFG"                 > "$D/scatter.log" 2>&1
  PAN_PARAMS="$D/params.jld2" PAN_OUTSUB="sanity_$role" ./julia_gdal.sh --project=. --threads=3 test/tost_sim_obs.jl   "$CFG" "0.05,0.10,0.15" > "$D/tost.log"    2>&1
  echo "$role done: $(ls "$D"/*.png 2>/dev/null | wc -l) png"
done
echo "ALL SANITY DONE (gen $GEN)"
