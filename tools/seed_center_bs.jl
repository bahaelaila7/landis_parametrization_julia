# Neutral Biomass Succession start for a config (the GA+FL coordinate calibration started from the same box centre,
# docs/gafl_calibration_log.md D2/D4): S 0.5, D 15, B_MAX 23,500, ANPP = B_MAX / 27.5, PROB_MORT 0, LONGEVITY copied by
# species name from a reference set (else 275). tools/calib_coord.jl applies the data floors on load.
#   ./julia_gdal.sh --project=. tools/seed_center_bs.jl <config.yml> <ref_params.jld2> <out.jld2>
using Pan; import YAML, JLD2
const P = Pan
cfg = YAML.load_file(ARGS[1]); ref = JLD2.load_object(ARGS[2]); out = ARGS[3]
splots, eco_list, species_list, eco_species_ids, _, _ = P.prepare_from_cfg(cfg)
p = P.BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=P.RNGType(UInt64(1)), no_establishment=true)
for (e, ids) in enumerate(eco_species_ids), j in eachindex(ids)
  p.S[e][j] = 0.5f0; p.B_MAX_SPP[e][j] = 23500f0; p.ANPP_MAX_SPP[e][j] = 23500f0 / 27.5f0; p.PROB_MORT_SPP[e][j] = 0f0
end
ri = Dict(s => k for (k, s) in enumerate(ref.SPECIES_LIST)); nl = 0
for (k, s) in enumerate(species_list)
  p.D[k] = 15f0
  if haskey(ri, s); p.LONGEVITY[k] = ref.LONGEVITY[ri[s]]; global nl += 1; else; p.LONGEVITY[k] = 275f0; end
end
JLD2.save_object(out, p)
println("BS centre start: $(length(species_list)) species ($(nl) longevity from reference), $(length(eco_list)) ecos → $out")
