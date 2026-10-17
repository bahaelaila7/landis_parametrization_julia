"""
    Reference

The scoring reference of one split (train, val, or a tool's plot set), built by `build_reference`:

  - `splots`: the split's cohort table.
  - `spdf`: the smoothed, binned observed age distributions (`PU.smoothen_ref_years`).
  - `spdf_plts`: `(plot_id, eco_id) → sim_year → SPDFGroundTruth`, the loss reference.
  - `site_sim_years`: the measurement years of every site.
  - `spinup_cohorts`: the year-deficit cohorts of the Sim B spinup.
  - `injection_cohorts`: the observed cohorts of the injection (and the disturbance data), or `nothing` when the run does not inject.
  - `ref_soa`: the initial `SiteSoA` (`nothing` for `make_soa=false`).
"""
struct Reference{SY,SC,I,R}
  splots::DataFrame
  spdf::DataFrame
  spdf_plts::Dict{Tuple{UIntType,UIntType},Dict{Int,PU.SPDFGroundTruth}}
  site_sim_years::SY
  spinup_cohorts::SC
  injection_cohorts::I
  ref_soa::R
end

"The smoothed observed distributions of `rows` and their `(plot_id, eco_id) → sim_year → SPDFGroundTruth` index (the smoothing / binning step of a reference)."
function reference_tables(rows::DataFrame, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams; debug::Bool=false)
  spdf = PU.smoothen_ref_years(rows, loss_params, Int(maximum(rows.age_calc)); debug=debug)
  (spdf, Data.make_spdf_dict(spdf, eco_species_ids))
end

"""
    build_reference(splots, eco_species_ids, loss_params; scoring=nothing, simb=nothing, no_establishment=false, plugin=nothing,
                    rng=nothing, spinup=true, debug=false) -> Reference

The one place that builds the scoring reference of a split: smoothing and binning, the `(plot, eco) → year` loss index, the site years, the
spinup cohorts, the injection cohorts (only when the run injects: `no_establishment`, `scoring.override` or `simb.simb_disturb_only`) and, when a
`plugin` is given, the initial `SiteSoA` (`make_sites`; consumes `rng`).
"""
function build_reference(splots::DataFrame, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams;
                         scoring::Union{Nothing,ScoringOptions}=nothing, simb::Union{Nothing,SimBOptions}=nothing, no_establishment::Bool=false,
                         plugin::Union{Nothing,PluginConfig}=nothing, rng::Union{Nothing,Random.AbstractRNG}=nothing, spinup::Bool=true, debug::Bool=false)
  spdf, spdf_plts = reference_tables(splots, eco_species_ids, loss_params; debug=debug)
  ssy = Data.get_site_sim_years(spdf)
  spc = Data.get_spinup_cohorts(splots)
  # simb_disturb_only builds the injection cohorts purely as the disturbance-data source (all_cohorts=true so the drops cover the whole stand);
  # it never injects.
  all_coh = (scoring !== nothing && scoring.override) || (simb !== nothing && simb.simb_disturb_only)
  inj = (no_establishment || all_coh) ? Data.get_injection_cohorts(splots; all_cohorts=all_coh) : nothing
  soa = plugin === nothing ? nothing : make_sites(splots, eco_species_ids; rng=rng, spinup=spinup, no_establishment=no_establishment, plugin=plugin)
  Reference(splots, spdf, spdf_plts, ssy, spc, inj, soa)
end
