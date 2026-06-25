using Pan
import JLD2, YAML
const P = Pan
cfg = YAML.load_file("runs/densest_pita_22_4_103_106.yml")
fps = NTuple{4,Int}[NTuple{4,Int}(Int.(p)) for p in cfg["filter_plots"]]
splots, eco_list, species_list, eco_species_ids, _ = P.Data.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], eco_field=cfg["eco_field"], tablename=cfg["tablename"],
  output_dir=cfg["tablename"], filter_eco_field=cfg["filter_eco_field"], filter_ecos=String[],
  filter_plots=fps, min_trees=1, min_agb_frac=0.0, skip_disturbances=false, spinup=false, RNG=P.RNGType(31231))
ref_soa = P.make_sites(splots, eco_species_ids; rng=P.RNGType(1), spinup=false, no_establishment=true)
eco_id = Int(P.getsite(ref_soa, 1).eco_id)
gids = eco_species_ids[eco_id]; sp_names = [species_list[g] for g in gids]
sp_local = findfirst(==("LIST2"), sp_names)
println("species order: $sp_names ; LIST2 local idx=$sp_local")
# sync injection setup
P.OVERRIDE_INJECTION[] = true; P.OVERRIDE_INJECTION_SYNC[] = true; P.OVERRIDE_INJECTION_DISTURBANCE[] = :off
ic = P.Data.get_injection_cohorts(splots; all_cohorts=true)
injd = P._build_injection_dict(ic, ref_soa); iyears = Set(Int.(ic.sim_year))
println("inject years=$(sort(collect(iyears)))")
# observed LIST2 (eco_species_id == sp_local) cohorts per sim_year
println("\nobserved LIST2 cohorts (sim_year, age, agb):")
for r in eachrow(splots)
  Int(r.eco_species_id) == sp_local && println("  y=$(r.sim_year) age=$(r.age_calc) agb=$(round(r.agb_sum))")
end

st = JLD2.load_object("runs/densest_pita_22_4_103_106_outputs/search_state_latest.jld2")
m = st.archive[argmin([Float64(c.fx.aggregate) for c in st.archive])]
println("\n=== best candidate (loss=$(round(Float64(m.fx.aggregate);digits=2))), LIST2 sync trace ===")
soa = P.copy_and_reseed_soa(ref_soa, UInt64(1))
ctx = (BiomassSuccession=(eco_params=P.BiomassSuccessionPlugin.generate_eco_params(m.x),),)
for y in 0:30
  if y > 0
    P.PanCore.process_plugin!(soa, P.BiomassSuccessionPlugin.BiomassSuccession, y; ctx=ctx.BiomassSuccession)
    y in iyears && haskey(injd, y) && P._inject_observed_cohorts!(soa, injd[y]; override=true, sync=true)
  end
  site = P.getsite(soa, 1); ages = Float64[]; bios = Float64[]
  for j in 1:Int(site.live)
    Int(site.c_species[j]) == sp_local || continue
    push!(ages, Float64(site.c_age[j])); push!(bios, Float64(site.c_bio[j]))
  end
  tot = sum(bios; init=0.0); mage = tot > 0 ? sum(ages .* bios) / tot : NaN
  tag = y in iyears ? " <-- INJECT(sync)" : ""
  println("  y=$(lpad(y,2))  n=$(length(ages))  total=$(lpad(round(Int,tot),6))  mean_age=$(round(mage;digits=1))  ages=$(round.(Int,sort(ages;rev=true)))$tag")
end
