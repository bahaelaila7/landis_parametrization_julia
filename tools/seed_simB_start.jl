# Build a Sim B starting parameter set from a (calibrated) Sim A set (docs/gafl_calibration_log.md, Sim B).
# Sim B's species/eco lists come from its own (timber-only) data, so growth params are REMAPPED by name
# (eco cell string × species), not copied by index. Establishment-related traits are set from data:
# SHADE_TOL (shade_tol_csv; catch-all groups by tier: *_HST→4, *_LST→2, else 3), MATURITY (maturity_csv),
# PROB_ESTAB_SPP flat (default 0.10, calibrated afterwards), MIN_REL_BIOMASS = pinned data default.
#   ./julia_gdal.sh --project=. tools/seed_simB_start.jl <simB_config.yml> <simA_params.jld2> <out.jld2> [pestab0=0.10]
using Pan; import JLD2
const P = Pan; const BSP = P.BiomassSuccessionPlugin
cfg = Pan.load_config(ARGS[1]); simA = JLD2.load_object(ARGS[2]); out = ARGS[3]
pe0 = length(ARGS) >= 4 ? parse(Float32, ARGS[4]) : 0.10f0
splots, eco_list, species_list, eco_species_ids, _, _ = P.prepare_from_cfg(cfg)
BSP.SHADE_TOL_TABLE[] = P._load_shade_tol_table(cfg["shade_tol_csv"])
BSP.MATURITY_TABLE[] = P._load_maturity_table(cfg["maturity_csv"])
rng = P.RNGType(UInt64(1))
p = P.BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=rng, no_establishment=false)
# --- remap Sim A growth by name
ai = Dict((simA.ECO_LIST[e], simA.SPECIES_LIST[Int(gid)]) => (e, j) for (e, ids) in enumerate(simA.ECO_SPECIES_IDS) for (j, gid) in enumerate(ids))
asp = Dict(s => k for (k, s) in enumerate(simA.SPECIES_LIST))
nhit = 0; nmiss = 0
for (e, ids) in enumerate(eco_species_ids), (j, gid) in enumerate(ids)
  key = (eco_list[e], species_list[gid])
  if haskey(ai, key)
    ea, ja = ai[key]
    p.S[e][j] = simA.S[ea][ja]; p.B_MAX_SPP[e][j] = simA.B_MAX_SPP[ea][ja]; p.ANPP_MAX_SPP[e][j] = simA.ANPP_MAX_SPP[ea][ja]
    p.PROB_MORT_SPP[e][j] = simA.PROB_MORT_SPP[ea][ja]   # NOT left at the random prior draw (median 0.56/yr → mass die-off)
    global nhit += 1
  else
    p.PROB_MORT_SPP[e][j] = 0f0
    global nmiss += 1
  end
end
for (k, s) in enumerate(species_list)
  haskey(asp, s) || continue
  a = asp[s]; p.D[k] = simA.D[a]; p.LONGEVITY[k] = simA.LONGEVITY[a]
end
# --- establishment-related traits from data
st = BSP.SHADE_TOL_TABLE[]
for (k, s) in enumerate(species_list)
  v = st === nothing ? nothing : get(st, uppercase(s), nothing)
  p.SHADE_TOL[k] = v !== nothing ? v : (endswith(s, "_HST") ? 4 : endswith(s, "_LST") ? 2 : 3)
end
P._seed_maturity!(p, species_list)
# PROB_MORT_SPP is remapped with growth above (cells new to Sim B keep 0: no catastrophic mortality, as in Sim A).
# PROB_ESTAB from data (prob_estab_all_species.csv: species, category, l3, …, prob_estab): match (category==name, l3),
# else (species==name, l3), else the species' median over L3s, else the global median; pe0 only if the table is absent.
import CSV, DataFrames, Statistics
pet = isfile(cfg["maturity_csv"]) ? CSV.read(cfg["maturity_csv"], DataFrames.DataFrame) : nothing
npe = 0
for (e, ids) in enumerate(eco_species_ids), (j, gid) in enumerate(ids)
  if pet === nothing || length(ARGS) >= 4; p.PROB_ESTAB_SPP[e][j] = pe0; continue; end   # explicit pe0 arg = flat override
  nm = uppercase(species_list[gid]); l3 = String(first(split(eco_list[e], "|lu=")))
  sel = pet[(uppercase.(String.(pet.category)) .== nm) .& (String.(string.(pet.l3)) .== l3), :prob_estab]
  isempty(sel) && (sel = pet[(uppercase.(String.(pet.species)) .== nm) .& (String.(string.(pet.l3)) .== l3), :prob_estab])
  isempty(sel) && (sel = pet[uppercase.(String.(pet.species)) .== nm, :prob_estab])
  v = isempty(sel) ? Statistics.median(skipmissing(pet.prob_estab)) : Statistics.median(skipmissing(sel))
  p.PROB_ESTAB_SPP[e][j] = Float32(clamp(v, 0.0, 1.0)); global npe += 1
end
println("PROB_ESTAB from data for $npe (eco,species) cells; median ", round(Statistics.median(reduce(vcat, p.PROB_ESTAB_SPP)), digits=3))
P._seed_min_rel!(p, eco_list, P.param_space_options(cfg))
JLD2.save_object(out, p)
println("Sim B start: growth remapped for $nhit (eco,species) cells, $nmiss new (defaults); species $(length(species_list)), ecos $(length(eco_list)); PROB_ESTAB=$pe0 → $out")
