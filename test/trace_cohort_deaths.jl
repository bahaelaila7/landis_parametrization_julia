# Diagnose the abrupt jumps in the free-growth species trajectories: trace PITA cohorts year-by-year
# for a couple of archive candidates and flag when a cohort disappears (and at what age vs LONGEVITY).
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
pita_local = findfirst(==("PITA"), sp_names)
st = JLD2.load_object("runs/densest_pita_22_4_103_106_outputs/search_state_latest.jld2")
archive = st.archive

function trace(m, HORIZON=100)
  p = m.x
  println("\n=== candidate loss=$(round(Float64(m.fx.aggregate);digits=2))  PITA LONGEVITY=$(Int(p.LONGEVITY[gids[pita_local]]))  B_MAX=$(Int(p.B_MAX_SPP[eco_id][pita_local])) ===")
  soa = P.copy_and_reseed_soa(ref_soa, UInt64(1))
  ctx = (BiomassSuccession=(eco_params=P.BiomassSuccessionPlugin.generate_eco_params(p),),)
  prev_n = -1
  for y in 0:HORIZON
    y > 0 && P.PanCore.process_plugin!(soa, P.BiomassSuccessionPlugin.BiomassSuccession, y; ctx=ctx.BiomassSuccession)
    site = P.getsite(soa, 1)
    ages = Float64[]; bios = Float64[]
    for j in 1:Int(site.live)
      Int(site.c_species[j]) == pita_local || continue
      push!(ages, Float64(site.c_age[j])); push!(bios, Float64(site.c_bio[j]))
    end
    n = length(ages); tot = sum(bios; init=0.0)
    mage = tot > 0 ? sum(ages .* bios) / tot : NaN
    if n != prev_n || y in (0, 7, 14)
      flag = n < prev_n && prev_n >= 0 ? "  <-- a PITA cohort DIED" : (y in (0,7,14) ? "  (measurement)" : "")
      println("  y=$(lpad(y,3))  PITA: n_cohorts=$n  total=$(lpad(round(Int,tot),7))  mean_age=$(round(mage;digits=1))  ages=$(round.(Int,sort(ages;rev=true)))$flag")
      prev_n = n
    end
  end
end

# best candidate + a short-longevity one
losses = [Float64(m.fx.aggregate) for m in archive]
trace(archive[argmin(losses)])
shortL = archive[argmin([Float64(m.x.LONGEVITY[gids[pita_local]]) for m in archive])]
trace(shortL)
