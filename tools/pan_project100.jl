# Pan Sim-A 100-yr projection of a candidate: initialise each study plot from its first-measurement
# cohorts, grow forward (no establishment = Sim A), snapshot AGB (g/m²) by (plotkey, tiered-species,
# age-bin) at 25/50/75/100 yr → pan_cohorts_100.csv (next to the candidate params).
#   Run: ./julia_gdal.sh --project=. tools/pan_project100.jl <config.yml> <candidate params.jld2>
using Pan
import JLD2, YAML, DuckDB, DataFrames, CSV
const P = Pan; const D = P.Data; const BSP = P.BiomassSuccessionPlugin; const DF = DataFrames
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
candpath = ARGS[2]
const HORIZONS = [25, 50, 75, 100]; const BINS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]
binidx(a) = (for (i, b) in enumerate(BINS); a < b && return i; end; length(BINS) + 1)
binlabel(i) = i == 1 ? "≤$(BINS[1])" : i <= length(BINS) ? "$(BINS[i-1])–$(BINS[i])" : "$(BINS[end])+"

splots, eco_list, species_list, eco_species_ids, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]),
  output_dir=String(cfg["tablename"]), filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false,
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  RNG=P.RNGType(UInt64(Int(g("seed", 1)))))
println("Pan study plots: ", length(unique(splots.plot_id)), " ; species(tiered): ", length(species_list))

params = JLD2.load_object(candpath)
no_estab = Bool(g("no_establishment", true))
soa = P.make_sites(splots, eco_species_ids; rng=P.RNGType(1), spinup=false, no_establishment=no_estab)
eco_params = BSP.generate_eco_params(params); ctx = (eco_params=eco_params,)
site_pid = [Int(P.getsite(soa, i).mapcode) for i in 1:soa.n]
pid2key = Dict(Int(r.plot_id) => join((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)), "_")
               for r in eachrow(unique(DF.select(splots, [:plot_id, :statecd, :unitcd, :countycd, :plot]))))

rows = NamedTuple[]
function snap!(off)
  acc = Dict{Tuple{String,String,Int},Float64}()
  for i in 1:soa.n
    s = P.getsite(soa, i); pk = get(pid2key, site_pid[i], "?")
    for j in 1:Int(s.live)
      eff = species_list[Int(s.c_species[j])]; ab = binidx(round(Int, Float64(s.c_age[j])))
      k = (pk, eff, ab); acc[k] = get(acc, k, 0.0) + Float64(s.c_bio[j])
    end
  end
  for ((pk, eff, ab), v) in acc; push!(rows, (plotkey=pk, offset=off, eff=eff, agebin=ab, agb=v)); end
end
for y in 1:maximum(HORIZONS)
  P.PanCore.process_plugin!(soa, BSP.BiomassSuccession, y; ctx=ctx)
  y in HORIZONS && snap!(y)
end
coh = DF.DataFrame(rows); coh.agebin_label = binlabel.(coh.agebin)
out = joinpath(dirname(candpath), "pan_cohorts_100.csv"); CSV.write(out, coh)
println("wrote $out  ($(DF.nrow(coh)) rows)")
for off in HORIZONS
  s = coh[coh.offset.==off, :]; isempty(s) && continue
  byp = DF.combine(DF.groupby(s, :plotkey), :agb => sum => :tot)
  println("  yr $off: $(DF.nrow(byp)) plots, mean plot AGB = $(round(sum(byp.tot)/DF.nrow(byp))) g/m²")
end
