# Sanity-check the per-stratum RANKW change: build per-(species,stratum) reference AGB from the data,
# run Pan._rankw, and report each stratum's total weight + top species (old global ranking starved
# small strata; new per-stratum ranking gives every stratum 1/n_strata).
#   Run: ./julia_gdal.sh --project=. tools/check_rankw.jl runs/fl5_l4cover_mocmaes_Aonly_Sglobal_ipop_v2.yml
using Pan, YAML, DataFrames
const P = Pan; const D = P.Data; const PU = P.PU
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
splots, eco_list, species_list, eco_species_ids, _ = P.prepare_from_cfg(cfg)[1:5]   # same data-prep as training

# reference AGB per (global species, stratum eco_id) + plots per stratum (split size)
agb = zeros(P.FloatType, length(species_list), length(eco_list))
for r in eachrow(splots); agb[Int(r.species_id), Int(r.eco_id)] += P.FloatType(r.agb_sum); end
nplots = zeros(Int, length(eco_list))
for (_, eid) in unique(collect(zip(Int.(splots.plot_id), Int.(splots.eco_id)))); nplots[eid] += 1; end

lp = PU.LossParams(age_bins=PU.AgeBins(bins_idx=[10, 20], last_bin_open=true), smoothing_weights=P.FloatType[1.0])   # default rankw_mode = :rank
R = P._rankw(lp, agb, eco_species_ids; split_size=nplots)
println("Σ RANKW total = ", round(sum(R); digits=4), "  (should be 1.0)\n")
for e in eachindex(eco_list)
  col = R[:, e]
  tops = sort([(col[gsp], species_list[gsp]) for gsp in eco_species_ids[e] if col[gsp] > 0]; rev=true)
  println(rpad(eco_list[e], 24), " nplots=", rpad(nplots[e], 5), " stratum Σ=", rpad(round(sum(col); digits=4), 8),
    " | top: ", join(["$(n)=$(round(w;digits=3))" for (w, n) in tops[1:min(3, end)]], ", "))
end