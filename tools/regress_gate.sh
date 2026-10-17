#!/usr/bin/env bash
# Fixed-seed regression gate (see docs/region4_pnet.md §7). Runs a 40-evaluation Igel MO-CMA-ES search (igel_mu 6,
# n_reps 1, no Sobol init) of the 8.5.3 production config and dumps the archive objectives (sorted, full precision).
# PASS ⇔ byte-identical to the CURRENT baseline docs/regress/baseline_853_refactor-step8.txt (re-recorded 2026-10-17 on the
# verified refactor tree, two identical runs; the older baseline_853_split-productive-species.txt predates the refactor's
# behaviour-changing bug fixes (docs: /workspace/pan_refactor/BUGS.md) and is kept for history only).
#   tools/regress_gate.sh <repo_root> <out_dir> [threads=4]
#   cmp docs/regress/baseline_853_refactor-step8.txt <out_dir>/REGRESS_ARCHIVE.txt
# Needs the 8.5.3 config + data in the main checkout's runs/ (PAN_DATA below).
set -u
ROOT="$1"; OUT="$2"; TH="${3:-4}"
CFG=/workspace/landis_parametrization_julia/runs/fl853_igelmo_4cell_shade_n1000_s_anpp.yml
cd "$ROOT"; [ -f wt_env.sh ] && source wt_env.sh
export PAN_DATA=/workspace/landis_parametrization_julia/runs PAN_PARALLEL=candidate
rm -rf "$OUT"; mkdir -p "$OUT"
./julia_gdal.sh --project=. --threads=$TH -e "using Pan; import JLD2
Pan.run_from_yaml(\"$CFG\"; overrides=Dict{String,Any}(\"output_dir\"=>\"$OUT\",\"trials\"=>40,\"igel_mu\"=>6,
  \"igel_sobol_init\"=>false,\"n_reps\"=>1,\"cmaes_archive_cap\"=>50))
ck = sort(filter(f->occursin(r\"search_state@\\d+\\.jld2\", f), readdir(\"$OUT\")); by=f->parse(Int, match(r\"@(\\d+)\", f)[1]))
st = JLD2.load_object(joinpath(\"$OUT\", last(ck)))
open(joinpath(\"$OUT\",\"REGRESS_ARCHIVE.txt\"),\"w\") do io
  for c in sort(collect(st.archive); by=c->Tuple(Float64.(c.fx.objectives)))
    println(io, join(repr.(Float64.(c.fx.objectives)), \",\"))
  end
end
println(\"REGRESS_DONE n_arch=\", length(st.archive), \" ck=\", last(ck))" > "$OUT/regress.log" 2>&1
tail -n 1 "$OUT/regress.log"
