# Effective per-(species×stratum) error multiplier of rankw_mode=cbal_pct vs the status-quo rank weighting.
# Both weightings normalize within each stratum to sum 1, and the cross-stratum factor sw[e]/Σsw is IDENTICAL
# in both modes, so it cancels: multiplier = R_cbal_within / R_rank_within (pure within-stratum ratio).
#   R_rank  : rank species by ref-AGB desc within stratum, w = 1/sqrt(log1p(rank)), normalize to Σ=1.
#   R_cbal  : pct = max(round(10·nc/tot)/10, 0.10); neff = pct·tot; w = (1-β)/(1-β^neff), normalize to Σ=1.
#   nc      : cohort-row count per (species_id, eco_id) in the TRAIN split (== CELL_NCOH).
#   Run: ./julia_gdal.sh --project=. tools/rankw_multiplier.jl <config.yml> [beta=0.999]
using Pan
import YAML, DataFrames, Statistics
using Printf
const P = Pan; const D = P.Data; const DF = DataFrames

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
β = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : Float64(g("rankw_beta", 0.999))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false))
rng = P.RNGType(UInt64(Int(g("seed", 1))))
val_frac = Float64(g("val_frac", 0.0)); n_folds = Int(g("n_folds", 1))
split_rng = (val_frac > 0 || n_folds > 1) ? P.RNGType(UInt64(Int(g("split_seed", 42)))) : nothing
excl_plots = NTuple{4,Int}[]
let epc = g("exclude_plots_csv", nothing)
  if !(epc === nothing || epc == "null")
    for ln in Iterators.drop(eachline(String(epc)), 1); isempty(strip(ln)) && continue; v = parse.(Int, split(ln, ",")); push!(excl_plots, (v[1], v[2], v[3], v[4])); end
  end
end
splots, eco_list, species_list, eco_species_ids, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=val_frac, split_rng=split_rng,
  n_folds=n_folds, fold_index=Int(g("fold_index", 1)), exclude_plots=excl_plots,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  stratify_eco_mixed=Bool(g("stratify_eco_mixed", false)),
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  filter_extent=(haskey(cfg, "filter_extent") ? String(cfg["filter_extent"]) : nothing),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  filter_plots=NTuple{4,Int}[NTuple{4,Int}(Int.(p)) for p in get(cfg, "filter_plots", [])], RNG=rng)
ns = length(species_list); ne = length(eco_list)

# nc[gsp,e]: cohort-row count per (species_id, eco_id) over TRAIN splots (matches CELL_NCOH build)
nc = zeros(Int, ns, ne)
for r in eachrow(splots)
  s = Int(r.species_id); e = Int(r.eco_id)
  (1 <= s <= ns && 1 <= e <= ne) && (nc[s, e] += 1)
end
# A[gsp,e]: total ref AGB per cell (rank driver) — order matches _set_rankw!'s A (Σ sp_agb_sum)
A = zeros(Float64, ns, ne)
for r in eachrow(DF.subset(splots, :sim_year => DF.ByRow(>(0))))
  s = Int(r.species_id); e = Int(r.eco_id)
  (1 <= s <= ns && 1 <= e <= ne) && (A[s, e] += Float64(r.agb_sum))
end

@printf("beta = %.4f\n\n", β)
allmult = Float64[]
for e in 1:ne
  ids = [gsp for gsp in eco_species_ids[e]]
  isempty(ids) && continue
  # rank weights
  order = sort(ids; by = gsp -> -A[gsp, e])
  rankw = Dict{Int,Float64}(); tot = 0.0
  for (rk, gsp) in enumerate(order); w = 1 / sqrt(log1p(rk)); rankw[gsp] = w; tot += w; end
  for gsp in ids; rankw[gsp] /= tot; end
  # cbal weights
  totc = sum(nc[gsp, e] for gsp in ids); cbw = Dict{Int,Float64}(); totw = 0.0
  for gsp in ids
    pct = totc > 0 ? max(round(10 * nc[gsp, e] / totc) / 10, 0.10) : 0.10
    w = (1 - β) / (1 - β^(pct * totc)); cbw[gsp] = w; totw += w
  end
  for gsp in ids; cbw[gsp] /= totw; end
  println("── stratum $(eco_list[e])  (Σcohorts=$totc, $(length(ids)) species) ──")
  @printf("  %-8s %7s %6s %5s %9s %9s %9s\n", "species", "nc", "share", "rank", "R_rank", "R_cbal", "×mult")
  for gsp in sort(ids; by = gsp -> -nc[gsp, e])
    rk = findfirst(==(gsp), order)
    m = cbw[gsp] / rankw[gsp]; push!(allmult, m)
    @printf("  %-8s %7d %5.1f%% %5d %9.4f %9.4f %8.2fx\n",
      species_list[gsp], nc[gsp, e], 100 * nc[gsp, e] / max(totc, 1), rk, rankw[gsp], cbw[gsp], m)
  end
  println()
end
sort!(allmult)
@printf("OVERALL multiplier (cbal_pct / rank): min=%.2fx  p25=%.2fx  median=%.2fx  p75=%.2fx  max=%.2fx  (spread=%.1fx)\n",
  allmult[1], Statistics.quantile(allmult, 0.25), Statistics.median(allmult), Statistics.quantile(allmult, 0.75), allmult[end], allmult[end] / allmult[1])
