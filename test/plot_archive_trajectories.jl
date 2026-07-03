# Single-plot archive growth curves: observed cohort progression vs the simulated stand for EVERY archive
# candidate, on ONE plot, with LOCAL competition (all the plot's cohorts present). Config-driven so the
# species/eco indexing MATCHES the archive params (loading with a different eco_field misapplies per-species
# params — that was the old bug). Stand AGB is summed over live cohorts; age is the biomass-weighted mean.
#   Run:  ./julia_gdal.sh --project=. test/plot_archive_trajectories.jl <config.yml> [state unit county plot] [horizon=320]
# (default plot 12-2-73-65, a PITA-only stand.)
using Pan
import JLD2, YAML, CairoMakie, DataFrames
const MK = CairoMakie; const P = Pan; const D = P.Data; const BSP = P.BiomassSuccessionPlugin
const DF = DataFrames

cfgpath = ARGS[1]
TGT = length(ARGS) >= 5 ? (parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4]), parse(Int, ARGS[5])) : (12, 2, 73, 65)
HORIZON = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 320
cfg = YAML.load_file(cfgpath); g(k, d) = get(cfg, k, d)
outdir = cfg["output_dir"]

# --- load the FULL run data (no plot filter) so species_list/eco_species_ids match the archive params ---
splots, eco_list, species_list, eco_species_ids, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]),
  output_dir=String(cfg["tablename"]), filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])), filter_plots=NTuple{4,Int}[],
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false,
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  RNG=P.RNGType(UInt64(Int(g("seed", 1)))))

# locate the target plot (FIA key → global plot_id); fall back to the PITA-richest plot if filtered out
key = (:statecd, :unitcd, :countycd, :plot)
mask = (splots.statecd .== TGT[1]) .& (splots.unitcd .== TGT[2]) .& (splots.countycd .== TGT[3]) .& (splots.plot .== TGT[4])
if !any(mask)
  @warn "target plot $(TGT) not in the run data (filtered out); falling back to the cohort-richest plot of its kind"
  # pick the plot with the most cohort rows (proxy for a clean multi-visit stand)
  cnt = DF.combine(DF.groupby(splots, :plot_id), DF.nrow => :n)
  pid = cnt.plot_id[argmax(cnt.n)]
  mask = splots.plot_id .== pid
end
plot_id = Int(first(splots.plot_id[mask]))
eco_id = Int(first(splots.eco_id[mask]))
plotlbl = "$(join(TGT,"-"))"
println("plot $(plotlbl): global plot_id=$(plot_id), eco_id=$(eco_id) ($(eco_list[eco_id]))")

# observed cohort progression in AGE space, per local eco_species
sub = splots[splots.plot_id .== plot_id, :]
obs = Dict{Int,Vector{Tuple{Float64,Float64}}}()
for r in eachrow(sub); push!(get!(obs, Int(r.eco_species_id), Tuple{Float64,Float64}[]), (Float64(r.age_calc), Float64(r.agb_sum))); end
for v in values(obs); sort!(v); end

# single-site SoA for just this plot (real cohorts → local competition), index-consistent with the params
soa1_ref = P.make_sites(sub, eco_species_ids; rng=P.RNGType(1), spinup=false, no_establishment=true)
@assert soa1_ref.n == 1 "expected a single site for one plot, got $(soa1_ref.n)"
# dominant species at sim start (max initial biomass) → its longevity/B_MAX label + senescence mark
s0 = P.getsite(soa1_ref, 1)
domj = argmax([Float64(s0.c_bio[j]) for j in 1:Int(s0.live)])
dom_sp_local = Int(s0.c_species[domj]); dom_gsp = eco_species_ids[eco_id][dom_sp_local]
dom_li = findfirst(==(dom_gsp), eco_species_ids[eco_id])
println("dominant species: $(species_list[dom_gsp]) (global id $(dom_gsp))")

# --- archive ---
st = JLD2.load_object(joinpath(outdir, "search_state_latest.jld2"))
archive = collect(st.archive)
losses = [Float64(m.fx.aggregate) for m in archive]
println("archive: $(length(archive)) candidates; loss $(round(minimum(losses);digits=4))–$(round(maximum(losses);digits=4))")

# step one candidate; record summed stand AGB + biomass-weighted mean age each sim year
function trajectory(params)
  soa = P.copy_and_reseed_soa(soa1_ref, UInt64(1))
  ctx = (BiomassSuccession=(eco_params=BSP.generate_eco_params(params),),)
  ages = Float64[]; agbs = Float64[]
  function rec!()
    s = P.getsite(soa, 1); live = Int(s.live); live == 0 && return
    b = sum(@view s.c_bio[1:live])
    a = b > 0 ? sum(Float64(s.c_age[j]) * Float64(s.c_bio[j]) for j in 1:live) / b : Float64(s.c_age[1])
    push!(ages, a); push!(agbs, b)
  end
  rec!()
  for y in 1:HORIZON
    P.PanCore.process_plugin!(soa, BSP.BiomassSuccession, y; ctx=ctx.BiomassSuccession)
    rec!()
  end
  (ages, agbs)
end
trajs = [trajectory(m.x) for m in archive]
LB(p) = (Float64(p.LONGEVITY[dom_gsp]), Float64(p.B_MAX_SPP[eco_id][dom_li]))   # dominant species' params

# --- plot ---
lo, hi = minimum(losses), maximum(losses)
const CMAP = MK.cgrad(:viridis; rev=true)
fig = MK.Figure(size=(1180, 720))
ax = MK.Axis(fig[1, 1]; xlabel="cohort age (years)", ylabel="stand AGB (g/m²)",
  title="plot $(plotlbl) ($(species_list[dom_gsp])) — simulated stand to $(HORIZON) yr vs observed; labels = LONGEVITY (L)/B_MAX (B), ▾ = age crosses LONGEVITY")
order = sortperm(losses; rev=true)
for k in order
  ages, agbs = trajs[k]
  lon, bmax = LB(archive[k].x)
  MK.lines!(ax, ages, agbs; color=losses[k], colormap=CMAP, colorrange=(lo, hi), linewidth=1.4, alpha=0.85)
  if !isempty(ages) && ages[1] <= lon <= ages[end]
    idx = clamp(round(Int, lon - ages[1]) + 1, 1, length(agbs))
    MK.scatter!(ax, [ages[idx]], [agbs[idx]]; color=losses[k], colormap=CMAP, colorrange=(lo, hi), marker=:dtriangle, markersize=11, strokecolor=:black, strokewidth=0.4)
  end
  MK.text!(ax, ages[end] + 1.0, agbs[end]; text="L$(round(Int,lon)) B$(round(bmax/1000,digits=1))k", fontsize=9, align=(:left, :center))
end
for (esid, pts) in obs
  MK.scatter!(ax, first.(pts), last.(pts); color=:red, markersize=14, marker=:circle, strokecolor=:black, strokewidth=1, label="observed")
end
MK.Colorbar(fig[1, 2]; colormap=CMAP, colorrange=(lo, hi), label="candidate train loss (lower=better)")
MK.axislegend(ax; position=:lt, merge=true)
out = joinpath(outdir, "cohort_trajectories.png")
MK.save(out, fig); println("wrote $out")
