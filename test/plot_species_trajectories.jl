# Follow EVERY live cohort on the plot and plot each one's simulated biomass as it ages (one panel per
# cohort), for every archive candidate. Free growth from the observed initial state: all cohorts are
# present so they COMPETE, but each panel's line is exactly that one cohort's biomass (matched by
# species + age = age0 + sim_year). No species-summing, no sync — so additions/removals never distort.
#   Run:  ./julia_gdal.sh --project=. test/plot_species_trajectories.jl <config.yml> [horizon=100]
using Pan
import JLD2, YAML, CairoMakie
const MK = CairoMakie
const P = Pan

cfgpath = ARGS[1]
HORIZON = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
cfg = YAML.load_file(cfgpath)
outdir = cfg["output_dir"]
fps = NTuple{4,Int}[NTuple{4,Int}(Int.(p)) for p in get(cfg, "filter_plots", [])]

splots, eco_list, species_list, eco_species_ids, _ = P.Data.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], eco_field=cfg["eco_field"], tablename=cfg["tablename"],
  output_dir=cfg["tablename"], filter_eco_field=cfg["filter_eco_field"], filter_ecos=String[], filter_plots=fps,
  min_trees=Int(get(cfg, "min_trees", 100)), min_agb_frac=Float64(get(cfg, "min_agb_frac", 0.05)),
  skip_disturbances=Bool(get(cfg, "skip_disturbances", true)), spinup=false,
  RNG=P.RNGType(UInt64(get(cfg, "seed", 1))))
ref_soa = P.make_sites(splots, eco_species_ids; rng=P.RNGType(1), spinup=false, no_establishment=true)
eco_id = Int(P.getsite(ref_soa, 1).eco_id)
gids = eco_species_ids[eco_id]
sp_names = String[species_list[g] for g in gids]

# every live cohort at sim_year 0: (sp_local, age0). Track each by species + age = age0+y.
cohorts = Tuple{Int,Int}[]
for r in eachrow(splots)
  Int(r.sim_year) == 0 && push!(cohorts, (Int(r.eco_species_id), Int(r.age_calc)))
end
sort!(unique!(cohorts))
N = length(cohorts)
println("$(N) live cohorts at sim start: ", [(sp_names[c[1]], c[2]) for c in cohorts])
# observed points per cohort (its own (age,agb) across visits; matched by birth = -age0)
obspts = [sort([(Int(r.age_calc), Float64(r.agb_sum)) for r in eachrow(splots)
                if Int(r.eco_species_id) == sp && (Int(r.sim_year) - Int(r.age_calc)) == -age0])
          for (sp, age0) in cohorts]

st = JLD2.load_object(joinpath(outdir, "search_state_latest.jld2"))
archive = st.archive
losses = [Float64(m.fx.aggregate) for m in archive]
println("archive: $(length(archive)) candidates; loss $(round(minimum(losses);digits=2))–$(round(maximum(losses);digits=2))")

function trajectory(params)
  soa = P.copy_and_reseed_soa(ref_soa, UInt64(1))
  ctx = (BiomassSuccession=(eco_params=P.BiomassSuccessionPlugin.generate_eco_params(params),),)
  M = fill(NaN, N, HORIZON + 1)
  function rec!(y)
    site = P.getsite(soa, 1)
    for (ci, (sp, age0)) in enumerate(cohorts)
      ta = age0 + y
      for j in 1:Int(site.live)
        if Int(site.c_species[j]) == sp && Int(round(site.c_age[j])) == ta
          M[ci, y+1] = Float64(site.c_bio[j]); break
        end
      end
    end
  end
  rec!(0)
  for y in 1:HORIZON
    P.PanCore.process_plugin!(soa, P.BiomassSuccessionPlugin.BiomassSuccession, y; ctx=ctx.BiomassSuccession)
    rec!(y)
  end
  M
end
trajs = [trajectory(m.x) for m in archive]

lo, hi = minimum(losses), maximum(losses)
order = sortperm(losses; rev=true)
const CMAP = MK.cgrad(:viridis; rev=true)   # flipped: low loss = yellow, high loss = purple
ncol = 3
nrow = cld(N, ncol)
fig = MK.Figure(size=(360 * ncol + 80, 260 * nrow + 50))
MK.Label(fig[0, 1:(ncol+1)], "plot $(join(fps[1],"-")) — every cohort tracked individually, simulated $(HORIZON) yr for all $(length(archive)) candidates (● = observed)"; fontsize=13, font=:bold)
for ci in 1:N
  sp, age0 = cohorts[ci]
  r, c = fldmod1(ci, ncol)
  ax = MK.Axis(fig[r, c]; xlabel="cohort age", ylabel="AGB g/m²", title="$(sp_names[sp])  age0=$(age0)")
  xs = age0 .+ (0:HORIZON)
  for k in order
    MK.lines!(ax, xs, trajs[k][ci, :]; color=losses[k], colormap=CMAP, colorrange=(lo, hi), linewidth=1.0, alpha=0.65)
  end
  isempty(obspts[ci]) || MK.scatter!(ax, first.(obspts[ci]), last.(obspts[ci]); color=:red, markersize=12, marker=:circle, strokecolor=:black, strokewidth=1)
end
MK.Colorbar(fig[1:nrow, ncol+1]; colormap=CMAP, colorrange=(lo, hi), label="candidate train loss (lower=better)")
out = joinpath(outdir, "all_cohort_trajectories.png")
MK.save(out, fig)
println("wrote $out")
