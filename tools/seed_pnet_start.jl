# Deterministic PnET start for a config: plant-functional-type defaults (params.jl pft_spec, no jitter), TOwood 0.01,
# RD_MULT 1, LONGEVITY copied by species name from a BiomassSuccession parameter set (same data-derived longevity).
#   ./julia_gdal.sh --project=. tools/seed_pnet_start.jl <pnet_config.yml> <bs_params.jld2> <out.jld2>
using Pan; import JLD2
const P = Pan; const PN = P.PnETPlugin
cfg = Pan.load_config(ARGS[1]); bs = JLD2.load_object(ARGS[2]); out = ARGS[3]
splots, eco_list, species_list, eco_species_ids, _, _ = P.prepare_from_cfg(cfg)
p = PN.generate_pnet_params(P.BiomassSuccessionPlugin.BSTables(), Dict{String,String}(), species_list,   # no climate needed here: default tables, genus-heuristic PFTs (as before)
   eco_list, eco_species_ids; rng=P.RNGType(UInt64(1)))
for i in eachindex(species_list)
  s = PN.pft_spec(p.PFT[i])
  p.MaxFracFol[i] = s.maxfracfol; p.FrActWd[i] = s.fractwd; p.SLWmax[i] = s.slwmax; p.TOwood[i] = 0.01f0; p.PsnAgeRed[i] = s.psnagered
end
for (e, ids) in enumerate(eco_species_ids), (j, gid) in enumerate(ids); p.FolN[e][j] = PN.pft_spec(p.PFT[Int(gid)]).folN; end
p.RD_MULT .= 1f0
bsi = Dict(s => k for (k, s) in enumerate(bs.SPECIES_LIST)); nl = 0
for (k, s) in enumerate(species_list); haskey(bsi, s) && (p.LONGEVITY[k] = bs.LONGEVITY[bsi[s]]; global nl += 1); end
JLD2.save_object(out, p)
println("PnET start: $(length(species_list)) species ($(nl) longevity from BS), $(length(eco_list)) ecos → $out")
