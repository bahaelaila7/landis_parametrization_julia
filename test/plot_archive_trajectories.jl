# Plot the observed cohort progression vs the simulated progression for EVERY archive candidate, for a
# single-plot fit. Reuses Pan's sim machinery (make_sites / generate_eco_params / process_plugin!) to
# step each archive candidate's params year-by-year and read the cohort biomass (site.c_bio).
#   Run:  ./julia_gdal.sh --project=. test/plot_archive_trajectories.jl <output_dir>
# (defaults to the single_cohort_12_2_73_65 run; data setup mirrors that yaml.)
using Pan
import JLD2
import CairoMakie
const MK = CairoMakie
const P = Pan

outdir = isempty(ARGS) ? "runs/single_cohort_12_2_73_65_outputs" : ARGS[1]
const HORIZON = 100   # extend the simulation this many years past sim-start (well beyond the data)

# --- rebuild the data exactly as the single-plot run did ---
splots, eco_list, species_list, eco_species_ids, _ = P.Data.prepare_parametrization_data(;
  cohorts_db_path="/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb",
  eco_field="land_use", tablename="curated_cohorts_landis", output_dir="curated_cohorts_landis",
  filter_eco_field="epa_l4", filter_ecos=["8.3.5.65o"], filter_plots=[(12, 2, 73, 65)],
  min_trees=1, min_agb_frac=0.0, skip_disturbances=false, spinup=false, RNG=P.RNGType(31231))
max_sim_year = maximum(splots.sim_year)
ref_soa = P.make_sites(splots, eco_species_ids; rng=P.RNGType(1), spinup=false, no_establishment=true)

# observed cohort progression in AGE space: (age → agb) per eco_species present
obs = Dict{Int,Vector{Tuple{Float64,Float64}}}()   # eco_species_id → [(age, agb)]
for r in eachrow(splots)
  push!(get!(obs, Int(r.eco_species_id), Tuple{Float64,Float64}[]), (Float64(r.age_calc), Float64(r.agb_sum)))
end
for v in values(obs); sort!(v); end

# --- load the archive ---
st = JLD2.load_object(joinpath(outdir, "search_state_latest.jld2"))
archive = st.archive
losses = [Float64(m.fx.aggregate) for m in archive]
println("archive: $(length(archive)) candidates; loss range $(round(minimum(losses);digits=2))–$(round(maximum(losses);digits=2))")

# Step one candidate's params year-by-year; record the cohort's (age, AGB) at EVERY sim year — the
# curve the cohort traces through age-biomass space (not just the measurement years).
function trajectory(params)
  soa = P.copy_and_reseed_soa(ref_soa, UInt64(1))
  eco_params = P.BiomassSuccessionPlugin.generate_eco_params(params)
  ctx = (BiomassSuccession=(eco_params=eco_params,),)
  ages = Float64[]; agbs = Float64[]
  function rec!()
    site = P.getsite(soa, 1); live = Int(site.live); live == 0 && return
    b = sum(@view site.c_bio[1:live])
    a = b > 0 ? sum(Float64(site.c_age[j]) * Float64(site.c_bio[j]) for j in 1:live) / b : Float64(site.c_age[1])
    push!(ages, a); push!(agbs, b)
  end
  rec!()                                                 # year 0 = initial observed state (age 46)
  for y in 1:HORIZON
    P.PanCore.process_plugin!(soa, P.BiomassSuccessionPlugin.BiomassSuccession, y; ctx=ctx.BiomassSuccession)
    rec!()
  end
  (ages, agbs)
end

trajs = [trajectory(m.x) for m in archive]

# PITA per-candidate LONGEVITY (per-species) and B_MAX (per-eco-species; single eco/species here)
LB(p) = (Float64(p.LONGEVITY[1]), Float64(p.B_MAX_SPP[1][1]))

# --- plot: observed points + one simulated AGE-AGB curve per archive candidate, coloured by train loss,
#         annotated with each candidate's LONGEVITY (L) and B_MAX (B), with a tick at the senescence age ---
lo, hi = minimum(losses), maximum(losses)
cmap = MK.cgrad(:viridis)
colof(l) = cmap[clamp((l - lo) / (hi - lo + eps()), 0.0, 1.0)]
fig = MK.Figure(size=(1180, 720))
ax = MK.Axis(fig[1, 1]; xlabel="cohort age (years)", ylabel="cohort AGB (g/m²)",
  title="plot 12-2-73-65 (PITA) — simulated growth curves to $(HORIZON) yr vs observed; labels = LONGEVITY (L) / B_MAX (B), ▾ = age crosses LONGEVITY")
order = sortperm(losses; rev=true)   # draw worst first so best sit on top
for k in order
  ages, agbs = trajs[k]
  c = colof(losses[k]); lon, bmax = LB(archive[k].x)
  MK.lines!(ax, ages, agbs; color=c, linewidth=1.5, alpha=0.85)
  if ages[1] <= lon <= ages[end]                          # mark senescence onset (age == LONGEVITY)
    idx = clamp(round(Int, lon - ages[1]) + 1, 1, length(agbs))
    MK.scatter!(ax, [ages[idx]], [agbs[idx]]; color=c, marker=:dtriangle, markersize=11, strokecolor=:black, strokewidth=0.4)
  end
  MK.text!(ax, ages[end] + 1.0, agbs[end]; text="L$(round(Int,lon)) B$(round(bmax/1000,digits=1))k",
    fontsize=9, color=:black, align=(:left, :center))
end
MK.xlims!(ax, 44, 46 + HORIZON + 24)
for (esid, pts) in obs
  MK.scatter!(ax, [p[1] for p in pts], [p[2] for p in pts]; color=:red, markersize=15, marker=:circle,
    strokecolor=:black, strokewidth=1, label="observed")
end
MK.Colorbar(fig[1, 2]; colormap=:viridis, colorrange=(lo, hi), label="candidate train loss (lower=better)")
MK.axislegend(ax; position=:lt)
out = joinpath(outdir, "cohort_trajectories.png")
MK.save(out, fig)
println("wrote $out")
