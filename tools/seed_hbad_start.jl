# Deterministic HBAD start: G_MULT = M_MULT = 1 (empirical priors as fitted), SDIMAX = species P95, LONGEVITY copied by species
# name from a BiomassSuccession parameter set (same data-derived longevity as the BS calibration).
#   ./julia_gdal.sh --project=. tools/seed_hbad_start.jl <hbad_config.yml> <bs_params.jld2> <out.jld2>
using Pan; import YAML, JLD2
const P = Pan; const HB = P.HBADPlugin
cfg = YAML.load_file(ARGS[1]); bs = JLD2.load_object(ARGS[2]); out = ARGS[3]
splots, eco_list, species_list, eco_species_ids, _, _ = P.prepare_from_cfg(cfg)
P.setup_plugin_from_cfg(cfg, eco_list; species_list=species_list, eco_species_ids=eco_species_ids)
p = HB.generate_hbad_params(species_list, eco_list, eco_species_ids; rng=P.RNGType(UInt64(1)), mode=:default)
bsi = Dict(s => k for (k, s) in enumerate(bs.SPECIES_LIST)); nl = 0
for (k, s) in enumerate(species_list); haskey(bsi, s) && (p.LONGEVITY[k] = bs.LONGEVITY[bsi[s]]; global nl += 1); end
JLD2.save_object(out, p)
println("HBAD start: $(length(species_list)) species ($(nl) longevity from BS), $(length(eco_list)) ecos → $out")
