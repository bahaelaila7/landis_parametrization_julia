#!/usr/bin/env bash
# Sanity check on a specific archive gen for the site-class 3-way run: extract best-W, best-AGB, best-agg
# candidates and run scatter(linear) for each (train+val; TEST held out — PAN_EVAL_TEST unset). Low threads.
#   tools/sanity_scatter_sc.sh <gen>
set -u
cd /workspace/landis_parametrization_julia
CB=runs/fl853_igelmo_siteclass_anpp_3way_outputs
CFG=runs/fl853_igelmo_siteclass_anpp_3way.yml
GEN="${1:-611}"
CK="$CB/search_state@$GEN.jld2"
[ -f "$CK" ] || { echo "no checkpoint $CK"; exit 1; }
echo "SANITY archive = search_state@$GEN.jld2"
./julia_gdal.sh --project=. --threads=1 -e "using Pan; import JLD2;
  st=JLD2.load_object(\"$CK\"); arch=collect(st.archive);
  W(c)=sum(Float64.(c.fx.objectives[1:2:end])); A(c)=sum(Float64.(c.fx.objectives[2:2:end]));
  roles=[(\"bestW\",argmin([W(c) for c in arch])),(\"bestAGB\",argmin([A(c) for c in arch])),(\"bestagg\",argmin([W(c)+A(c) for c in arch]))];
  for (r,i) in roles
    d=\"$CB/sanity_\$(r)_$GEN\"; mkpath(d); JLD2.save_object(joinpath(d,\"params.jld2\"), arch[i].x);
    println(\"ROLE \$r idx=\$i A_W=\", round(W(arch[i]),digits=5), \" A_AGB=\", round(A(arch[i]),digits=4));
  end;
  println(\"n_arch=\", length(arch))" 2>&1 | grep -aE "ROLE|n_arch"
export PAN_SCATTER_MODES=linear
for role in bestW bestAGB bestagg; do
  D="$CB/sanity_${role}_$GEN"
  PAN_PARAMS="$D/params.jld2" PAN_OUTSUB="sanity_${role}_$GEN" ./julia_gdal.sh --project=. --threads=3 test/scatter_sim_obs.jl "$CFG" > "$D/scatter.log" 2>&1
  echo "$role done: $(ls "$D"/*.png 2>/dev/null | wc -l) png in $(basename "$D")"
done
echo "ALL SANITY DONE (gen $GEN)"
