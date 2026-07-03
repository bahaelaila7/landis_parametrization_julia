# Sanity-check the per-stratum RANKW change: build per-(species,stratum) reference AGB from the data,
# run Pan._set_rankw!, and report each stratum's total weight + top species (old global ranking starved
# small strata; new per-stratum ranking gives every stratum 1/n_strata).
#   Run: ./julia_gdal.sh --project=. tools/check_rankw.jl runs/fl5_l4cover_mocmaes_Aonly_Sglobal_ipop_v2.yml
using Pan, YAML, DataFrames
const P = Pan; const D = P.Data; const PU = P.PU
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
splots, eco_list, species_list, eco_species_ids, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]),
  output_dir=String(cfg["tablename"]), filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])), filter_plots=NTuple{4,Int}[],
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=0.0,
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  RNG=P.RNGType(UInt64(Int(g("seed", 1)))))

# reference AGB per (global species, stratum eco_id) + plots per stratum (split size)
agb = zeros(P.FloatType, length(species_list), length(eco_list))
for r in eachrow(splots); agb[Int(r.species_id), Int(r.eco_id)] += P.FloatType(r.agb_sum); end
nplots = zeros(Int, length(eco_list))
for (_, eid) in unique(collect(zip(Int.(splots.plot_id), Int.(splots.eco_id)))); nplots[eid] += 1; end

P._set_rankw!(agb, eco_species_ids; split_size=nplots)
R = PU.RANKW[]
println("Σ RANKW total = ", round(sum(R); digits=4), "  (should be 1.0)\n")
for e in eachindex(eco_list)
  col = R[:, e]
  tops = sort([(col[gsp], species_list[gsp]) for gsp in eco_species_ids[e] if col[gsp] > 0]; rev=true)
  println(rpad(eco_list[e], 24), " nplots=", rpad(nplots[e], 5), " stratum Σ=", rpad(round(sum(col); digits=4), 8),
    " | top: ", join(["$(n)=$(round(w;digits=3))" for (w, n) in tops[1:min(3, end)]], ", "))
end