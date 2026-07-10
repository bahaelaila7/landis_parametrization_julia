# Batch sim-vs-obs for a whole archive: data-prep + reference built ONCE, then loop every candidate under
# <candidates_dir>/candidate_*/params.jld2. Per candidate × stratum (eco×lu, species POOLED): linear OLS of
# simulated vs observed AGB → R² (variance explained / precision) and slope (bias; 1 = unbiased). Saves one
# 4-panel linear-scatter PNG per candidate and prints an R²/slope table.
#   ./julia_gdal.sh --project=. tools/scatter_archive.jl <config.yml> <candidates_dir> [out_subdir]
using Pan
import JLD2, YAML, CairoMakie, Statistics, DataFrames, CSV
const MK = CairoMakie; const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
CANDDIR = ARGS[2]
OUTSUB = length(ARGS) >= 3 ? ARGS[3] : "archive_scatter"
outdir = joinpath(cfg["output_dir"], OUTSUB); mkpath(outdir)

P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true)); P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false)); P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false)); P.INIT_PERTURB_FRAC[] = 0.0
no_estab = Bool(g("no_establishment", false)); rng = P.RNGType(UInt64(Int(g("seed", 1))))

# honor the run's outlier exclusions so candidates are evaluated on the SAME plot universe they were fit on
excl_plots = NTuple{4,Int}[]
let epc = g("exclude_plots_csv", nothing)
  if !(epc === nothing || epc == "null")
    for r in eachrow(CSV.read(String(epc), DF.DataFrame)); push!(excl_plots, (Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot))); end
    println("exclude_plots: dropping $(length(excl_plots)) plots")
  end
end

# data-prep once (train split); reference + sites built once (candidate-independent)
splots, eco_list, species_list, eco_species_ids, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=0.0, split_rng=nothing,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])), exclude_plots=excl_plots, RNG=rng)
n_species = length(species_list)
bins = Int.(g("bins_idx", [20, 60, 120]))
loss_params = PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=true), smoothing_weights=P.FloatType[1.0], lambda=P.FloatType(g("loss_lambda", 1.0)))
sp = splots
max_age = Int(maximum(sp.age_calc))
spdf = PU.smoothen_ref_years(sp, loss_params, max_age; debug=false)
spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
inj = (no_estab || P.OVERRIDE_INJECTION[]) ? D.get_injection_cohorts(sp; all_cohorts=P.OVERRIDE_INJECTION[]) : nothing
rs = P.make_sites(sp, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)
idict = isnothing(inj) ? nothing : P._build_injection_dict(inj, rs)
iyears = isnothing(inj) ? Set{Int}() : Set(Int.(inj.sim_year))
msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
obs = DF.combine(DF.groupby(DF.subset(sp, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :eco_species_id, :eco_id]), :agb_sum => sum => :obs_agb)
DF.rename!(obs, :eco_species_id => :esp)
excl = (P.OVERRIDE_INJECTION_DISTURBANCE[] == :exclude_overwrite && !isnothing(inj)) ?
  Set((Int(r.plot_id), Int(r.sim_year), Int(r.eco_species_id)) for r in eachrow(inj) if r.disturbance_drop_pct > 0) : Set{Tuple{Int,Int,Int}}()

linfit(x, y) = begin
  mx, my = Statistics.mean(x), Statistics.mean(y); vx = sum((x .- mx) .^ 2)
  b = vx > 0 ? sum((x .- mx) .* (y .- my)) / vx : 0.0; a = my - b * mx
  ssr = sum((y .- (a .+ b .* x)) .^ 2); sst = sum((y .- my) .^ 2)
  (a, b, sst > 0 ? 1 - ssr / sst : NaN)
end

cand_files = sort(filter(f -> isfile(f), [joinpath(CANDDIR, "candidate_$i", "params.jld2") for i in 1:10_000]), by = f -> parse(Int, match(r"candidate_(\d+)", f).captures[1]))
strata = sort(unique(String.(eco_list)))
println("candidates=$(length(cand_files)) strata=$(length(strata))")
tbl = DF.DataFrame(candidate=Int[], species=String[], stratum=String[], n=Int[], R2=Float64[], slope=Float64[])

for (ci, cf) in enumerate(cand_files)
  cand = JLD2.load_object(cf)
  res = P.fit_params(rs, cand, msy, n_species, eco_species_ids, spdf_plts, ssy, false, spc, loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)])
  cached = res[1][2]
  simdf = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], agb=Float64[])
  for (pid, sy, esp, _a, bio) in cached; sy > 0 && push!(simdf, (Int(pid), Int(sy), Int(esp), Float64(bio))); end
  sim_agg = DF.combine(DF.groupby(simdf, [:plot_id, :sim_year, :esp]), :agb => sum => :sim_agb)
  paired = DF.innerjoin(obs, sim_agg, on=[:plot_id, :sim_year, :esp])
  paired = DF.filter(r -> !((r.plot_id, r.sim_year, r.esp) in excl), paired)
  paired.species = [species_list[eco_species_ids[r.eco_id][r.esp]] for r in eachrow(paired)]  # esp is eco-LOCAL → global
  splist = sort(unique(paired.species))
  fig = MK.Figure(size=(300 * length(strata) + 70, 250 * length(splist) + 50))
  MK.Label(fig[0, :], "candidate $ci — sim vs obs AGB (linear OLS) per species × stratum", fontsize=15)
  for (ri, sp2) in enumerate(splist), (ciX, st) in enumerate(strata)
    e = findfirst(==(st), eco_list)
    d = DF.subset(paired, :species => DF.ByRow(==(sp2)), :eco_id => DF.ByRow(==(e)))
    ax = MK.Axis(fig[ri, ciX], title=(ri == 1 ? st : ""), ylabel=(ciX == 1 ? sp2 : ""), xlabel=(ri == length(splist) ? "obs" : ""))
    DF.nrow(d) < 2 && continue
    x = Float64.(d.obs_agb); y = Float64.(d.sim_agb); (a, b, r2) = linfit(x, y)
    MK.scatter!(ax, x, y, markersize=4, color=(:steelblue, 0.4))
    mx = max(maximum(x), 1.0); MK.lines!(ax, [0, mx], [0, mx], color=:gray, linestyle=:dash)
    MK.lines!(ax, [0, mx], [a, a + b * mx], color=:red, linewidth=2)
    MK.text!(ax, 0.02, 0.98, text="R²=$(round(r2,digits=2)) b=$(round(b,digits=2))", space=:relative, align=(:left, :top), fontsize=9)
    push!(tbl, (ci, sp2, st, DF.nrow(d), round(r2, digits=3), round(b, digits=3)))
  end
  out = joinpath(outdir, "archive_cand$(ci)_linear.png"); MK.save(out, fig)
end
DF.sort!(tbl, [:species, :stratum, :candidate])
CSV.write(joinpath(outdir, "archive_r2_slope.csv"), tbl)
println("\n=== per-candidate R² (variance) + slope (bias), by species × stratum ==="); show(tbl, allrows=true, allcols=true); println()
println("\nwrote $(length(cand_files)) PNGs + archive_r2_slope.csv to $outdir")
