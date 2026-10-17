module Pan
ENV["GKSwstype"] = "nul" # this is to suppress window opening upon plotting
include("PanCore.jl")
include("Plugins.jl")
include("Parametrization.jl")
include("Search.jl")
include("Data.jl")
include("SpatialData.jl")
include("Spatial.jl")
#for file in filter(f -> endswith(f,"Plugin.jl"), readdir("src/plugins";join=false))
#    println("including plugin $(file)")
#    include(joinpath("plugins", file))
#end
#include("plugins/BiomassSuccessionPlugin.jl")
using .PanCore
using .Plugins: BaseSitePlugin, BiomassSuccessionPlugin, PnETPlugin, HBADPlugin
import .Parametrization as PU
import .Parametrization.BiomassSuccessionParametrization as BSP
import .Parametrization.PnETParametrization as PNP
import .Parametrization.HBADParametrization as HBP
using .Search: SA, LBSA, MOLBSA, CMAES, MOCMAES, IgelMOCMAES, CMAMAE, NSGA2, CCIgel, GDIgel, Common
import .Data as Data
import .SpatialData
import .Spatial
import Dates
import LinearAlgebra
import CSV

import Random
import Statistics
import Distributions as Dists
import Term.Progress as TProgress
import JLD2
import Serialization
import YAML
using DataFrames
import CairoMakie
import DuckDB
import HypothesisTests




const RNGType = Random.Xoshiro #Random.MersenneTwister
const ActivePlugins = (BaseSitePlugin.BaseSite, BiomassSuccessionPlugin.BiomassSuccession)
const ActiveSoA = SiteSoA{Tuple{(ActivePlugins)...}}
include("succession_api.jl")
include("config/params_yaml.jl")
include("config/schema.jl")
include("config/options.jl")
include("config/param_space.jl")
include("config/plugin_config.jl")

include("sites.jl")

include("reference.jl")

include("scoring/injection.jl")

include("scoring/shock.jl")

include("scoring/exclusion.jl")

include("scoring/death.jl")

include("scoring/wdiag.jl")

include("scoring/plan.jl")

include("io/plots.jl")

include("io/writer.jl")

include("io/outputs.jl")

# search drivers reachable from `parametrize` (sobol returns early; plot_only* are handled in run_from_yaml)
const SEARCH_MODES = ("lbsa", "sa", "molbsa", "cmaes", "mocmaes", "igelmo", "cmame", "ccigel", "nsga2", "gdigel")
function _check_search_mode(m::AbstractString)
  (m in SEARCH_MODES || m == "sobol") && return nothing
  error("unknown search_mode $(repr(m)); valid: $(join(SEARCH_MODES, ", ")), sobol, plot_only, plot_only_sobol")
end

function parametrize(; data::Data.DataConfig,
  output_dir::String="./outputs",
  bins_idx::Vector{Int64}=1:180 .|> Int64,
  smoothing_window::Vector{FloatType}=FloatType[one(FloatType)],
  unbinned_w::Bool=false,   # Sim A: compute W as the EXACT per-year EMD (no binning, no smoothing). bins_idx applies to Sim B only.
  w_count_balance::Bool=false,          # count-balance reweight of the W1 term (survivorship under-representation fix)
  w_count_balance_mode::String="both",  # which sim(s) get reweighted: "a" / "b" / "both"
  w_count_beta::Float64=0.99,           # effective-number-of-samples temper (→1 ≈ 1/n, →0 ≈ uniform)
  w_count_pct_floor::Float64=0.10,      # floor on the per-bin count-share → effective #cohorts (neff=pct·tot); lower ⇒ rare bins get MORE weight
  rankw_mode::String="rank",            # per-(species,stratum) objective weight: "rank" (1/√ln(rank) by AGB) or "cbal_pct" (percentile-floored class-balanced by cohort count)
  rankw_beta::Float64=0.999,            # β for rankw_mode=cbal_pct
  rankw_pct_floor::Float64=0.10,        # cohort-count-share floor+grid for rankw_mode=cbal_pct (species×strata; twin of w_count_pct_floor)
  param_split_species::AbstractDict=Dict{String,Vector{String}}(),  # {param_name => [species]} fit per-eco; others tied across ecos
  param_tier_merge::AbstractDict=Dict{String,Any}(),  # {param_name => {species => {cell => group}}} tie split site-cells per species
  param_pooling::AbstractDict=Dict{String,Any}(),     # region runs: L3→L2→region pooling of EcoSpecies params (see _build_pool_labels)
  succession_plugin::String="biomass",                # "biomass" (BiomassSuccession, default) | "pnet" (PnET ecophysiology)
  tables::BiomassSuccessionPlugin.BSTables=BiomassSuccessionPlugin.BSTables(),   # Biomass Succession data tables (see bs_tables_from_cfg); shared by every plugin's generators
  pnet_climate_cache::String="",                       # pnet: PRISM L4 monthly cache (JLD2)
  pnet_co2_csv::String="",                             # pnet: annual CO2 csv (year,co2_ppm)
  pnet_generic::AbstractDict=Dict{String,Any}(),       # pnet: PnETConst field overrides
  pnet_soil_texture::AbstractDict=Dict{String,Any}(),  # pnet: {L3 => Saxton-Rawls texture code}; default LOAM
  pnet_soil_csv::String="",                            # pnet: per-plot SOLUS soil (tools/build_plot_soil.jl); "" = texture path (legacy)
  hbad_priors_csv::String="",                          # hbad: fixed priors (results/hbad/priors.csv); "" = repo default
  hbad_coefs_csv::String="",                           # hbad: translation coefficients (results/hbad/translation_coefs.csv); "" = repo default
  hbad_seed_rain::Bool=false,                          # hbad: every species of the ecoregion is a seed source (ESTAB_SEED_RAIN)
  hbad_prob_estab_csv::String="",                      # hbad: per-cycle (species, L3) PROB_ESTAB table (prob_estab_all_species.csv); "" = constant fallback
  hbad_cycle_years::Float64=8.0,                       # hbad: remeasurement interval for the per-cycle -> annual PROB_ESTAB conversion
  hbad_init::String="jitter",                          # hbad: start candidates default | jitter | prior (see HBADPlugin.INIT_MODE)
  cmaes_blocks::String="eco",                          # CMA-ES covariance blocks: "eco" (per cell, legacy) | "l3" (per L3 + tail)
  search_mode::String="lbsa",
  context_only::Bool=false,   # prepare the training context exactly as a search would, hand it to `post_search_hook` (if any), skip the search
  post_search_hook=nothing,   # optional diagnostic callback `f(ctx)`: called with the prepared training context right after the search driver returns (tools, test/pnet_smoke_diag.jl); nothing ⇒ no call
  tier::Int=3,
  TRIALS::Int=30000,
  resume_from::Union{Nothing,String}=nothing,
  start_from::Union{Nothing,String}=nothing,
  force_restart_from_random::Bool=false,
  sobol_n::Int=100,
  saltelli::Bool=false,
  n_reps::Int=5,
  sobol_candidates_db::Union{Nothing,String}=nothing,
  sobol_top_frac::Float64=0.5,
  n_output_plots::Int=0,
  no_establishment::Bool=false,

  split_only::Bool=false,                  # write split_plots.csv + species_map.csv (+ pool_map.csv) and stop (floor-building pass)
  loss_lambda::Float64=1.0,
  loss_alpha::Float64=1.0,
  agb_hinge::Bool=false,
  agb_hinge_threshold::Float64=10.0,
  agb_hinge_pct::Float64=0.0,
  agb_hinge_pct_min::Float64=0.0,
  agb_hinge_pct_max::Float64=Inf,
  agb_hinge_l2::Bool=false,
  agb_hinge_p::Float64=2.0,
  agb_hinge_beta::Float64=1.0,
  w_hinge::Bool=false,
  w_hinge_pct::Float64=0.01,
  w_hinge_pct_min::Float64=2.0,
  w_hinge_pct_max::Float64=5.0,
  agb_normalize::Bool=false,
  w_normalize::Bool=false,
  cell_normalize::Bool=false,
  w_scale_factor::Float64=1.0,   # multiplies W_SCALE_{A,B}; <1 shrinks the W divisor → amplifies ΣW (rebalance vs AGB)
  w_smooth::Bool=false,
  w_smooth_band::Float64=0.05,
  w_smooth_conc::Float64=0.5,
  w_smooth_beta::Float64=1.0,          # fixed β for the W softplus (used when w_smooth_auto_knee=false, the default)
  w_smooth_auto_knee::Bool=false,      # true → derive β_W from band·conc·wmax (reweight-aware per-cell); W only, AGB unaffected
  w_p::Float64=1.0,
  loss_piecewise::Bool=false,
  w_pivot::Float64=1.0,
  agb_pivot::Float64=1.0,
  scoring::ScoringOptions=ScoringOptions(),   # the scoring rule + initial-biomass perturbation (see `scoring_from_config`)
  simb::SimBOptions=SimBOptions(),            # Sim B / dual-mode options
  space_opts::PU.ParamSpaceOptions=PU.ParamSpaceOptions(),   # which parameters are searched and their boxes (see `param_space_options`)
  diagnose::Bool=false,
  cycle_years::Real=8,
  cmaes_lambda::Union{Nothing,Int}=nothing,
  cmaes_sigma0::Float64=0.3,
  cmaes_warmstart_seeds::Int=20,   # MO-CMA-ES: # top Sobol seeds to warm-start mean(recombination)+C(covariance shape); 1 ⇒ single-point + C=I (old behavior)
  ipop::Bool=false,
  ipop_stagnation::Int=20,
  ipop_max_barren_restarts::Int=4,   # stop after this many consecutive IPOP restarts with no new best (0 ⇒ disabled)
  cmaes_archive_cap::Int=200,
  seed_archive_from::Union{Nothing,String}=nothing,
  cmaes_integer_handling::Bool=false,
  cmaes_integer_std_factor::Float64=0.3,
  cmaes_single_cov::Bool=false,   # CMA-ES family: one full covariance over ALL params (no eco×lu block split)
  nsga2_pop::Int=48, nsga2_offspring::Union{Nothing,Int}=nothing, nsga2_eta_c::Float64=20.0,
  nsga2_eta_m::Float64=20.0, nsga2_pc::Float64=0.9, nsga2_pm::Float64=-1.0, nsga2_sobol_init::Bool=true,
  igel_mu::Int=20,
  igel_sigma0::Float64=0.3,
  igel_sobol_init::Bool=true,
  igel_sobol_pool_k::Int=1,
  igel_sobol_raw_mult::Int=2,
  igel_niche_radius::Float64=0.0,
  igel_reseed_sigma::Float64=0.0,
  igel_reseed_random_frac::Float64=0.0,
  igel_maturity::Int=0,
  igel_seed_maturity::Bool=false,   # igelmo phase-1: shield the μ seeds for `igel_maturity` gens before they compete
  igel_warm_start::Bool=false,      # igelmo: initial population = the start_from params + (μ−1) Gaussian perturbations (σ=igel_sigma0, u-space)
  igel_freeze_seed_growth::Bool=false,   # igelmo: each seed lineage keeps its own frozen {D,S,B_MAX,ANPP_MAX} (needs fix_growth + seeds)
  cc_group_size::Int=2,
  cc_spec_gens::Int=20,
  cc_integ_gens::Int=20,
  cc_cycles::Int=5,
  cc_fix_others::Bool=false,
  gdigel::AbstractDict=Dict{String,Any}(),   # search_mode gdigel: group-decomposed (1+1)-CMA options (docs/cmaes_gdigel.md)
  cmame_alpha::Float64=0.02,
  cmame_grid::Int=15,
  cmame_reseed_explore::Float64=1.0,
  cmame_restart_patience::Int=6,
  cmame_sobol_reseed::Bool=false,
  cmame_mo_rank::Bool=false,
  balanced_quality::Bool=false,
  archive_by_sp::Bool=false,
  bounds_by_sobol::Bool=false,
  top_seeds::Float64=0.2,
  dominance_species::String="exact",
  exec::ExecOptions=ExecOptions(),   # how the run executes (parallel mode, timing, validation switches); see ExecOptions
  search::SearchOptions=SearchOptions(),   # config-derived knobs of the search engines (LBSA, Igel, MO-CMA-ES ranking); see SearchOptions
  rng::Random.AbstractRNG)
  _check_search_mode(search_mode)   # fail fast, before data preparation
  spinup = data.spinup; cohorts_db_path = data.cohorts_db_path; tablename = data.tablename   # (the rest of `data` is read by Data.prepare_parametrization_data)

  # Pin BLAS to 1 thread: the search does hundreds of tiny per-offspring covariance eigendecomps per gen —
  # multi-threaded OpenBLAS spawns all cores per call and oversubscribes catastrophically at high --threads.
  LinearAlgebra.BLAS.set_num_threads(1)
  @info "search starting" julia_threads=Threads.nthreads() blas_threads=LinearAlgebra.BLAS.get_num_threads()
  mkpath(output_dir)
  # [5, 10, 20, 40, 60, 80]
  #bins_idx = vcat(5:5:30, 40:10:80, 100:20:160)
  @info bins_idx
  @info smoothing_window
  scoring.init_perturb_frac > 0 && @info "Initial-condition perturbation: ±$(round(100*scoring.init_perturb_frac;digits=2))% over $(n_reps) reps (best-of), shared seed; perturbed = max(2, bio + clamp(bio·pct, ±$(scoring.init_perturb_cap))) [max deviation $(scoring.init_perturb_cap), floor 2]"
  # unbinned_w (Sim A): per-year age grid (1:400) + no age-smoothing → W₁ is the EXACT per-year EMD.
  # Setting the bins at the SOURCE means the softplus β and the cell-norm W scales
  # (compute_loss_scales) both recompute on this same grid — W₁, its normalization, and the softplus knee
  # stay consistent (no per-year-W₁ ÷ stale-binned-scale mismatch). bins_idx then only governs Sim B (tier-4).
  # The ONE place the loss options are read: every loss knob of the run lives in this immutable LossParams (and the per-reference,
  # data-derived normalisation in LossScales, see compute_loss_scales). No module-level loss state.
  _loss_params(age_bins, smoothing_weights) = PU.LossParams(; age_bins=age_bins, smoothing_weights=smoothing_weights, lambda=FloatType(loss_lambda),
    loss_alpha=FloatType(loss_alpha),                         # 0 → optimize L2/AGB-level only (zero Wasserstein)
    agb_hinge=agb_hinge,                                      # AGB term: hinge-L1 (tolerance band) vs sqrt-difference
    agb_hinge_threshold=FloatType(agb_hinge_threshold),
    agb_hinge_pct=FloatType(agb_hinge_pct),                   # >0 ⇒ band = clamp(obs*pct, min, max)
    agb_hinge_pct_min=FloatType(agb_hinge_pct_min), agb_hinge_pct_max=FloatType(agb_hinge_pct_max),
    agb_hinge_p=FloatType(agb_hinge_p),                       # hinge exponent pen(h)=h^p (default 2 = L2)
    agb_hinge_beta=FloatType(agb_hinge_beta),                 # softplus sharpness (constant; 0 ⇒ hard hinge)
    agb_normalize=agb_normalize,                              # AGB term = relative error (÷ observed total)
    w_normalize=w_normalize,                                  # W1 term = ÷ per-reference max → [0,1] ratio
    cell_norm=cell_normalize,                                 # per-(eco×lu×sp) ÷scale + AGB-rank rescale (overrides global)
    w_scale_factor=FloatType(w_scale_factor),                 # scale the W divisor (rebalance ΣW vs ΣAGB)
    w_softplus=w_smooth,                                      # softplus-smooth the per-term W1
    w_smooth_band=FloatType(w_smooth_band), w_smooth_conc=FloatType(w_smooth_conc),
    w_smooth_beta=FloatType(w_smooth_beta), w_smooth_auto_knee=w_smooth_auto_knee,
    w_p=FloatType(w_p),                                       # square the W term (aggression)
    w_hinge=w_hinge,                                          # Sim A: ±band "benefit of the doubt" forgiveness before the age CDF
    w_hinge_pct=FloatType(w_hinge_pct), w_hinge_pct_min=FloatType(w_hinge_pct_min), w_hinge_pct_max=FloatType(w_hinge_pct_max),
    loss_piecewise=loss_piecewise, w_pivot=FloatType(w_pivot), agb_pivot=FloatType(agb_pivot),
    rankw_mode=Symbol(lowercase(rankw_mode)), rankw_beta=rankw_beta, rankw_pct_floor=rankw_pct_floor,
    cbal_on=w_count_balance, cbal_mode=Symbol(lowercase(w_count_balance_mode)), cbal_beta=w_count_beta, cbal_pct_floor=w_count_pct_floor)
  loss_params = unbinned_w ?
    _loss_params(PU.AgeBins(bins_idx=collect(1:400), last_bin_open=true), FloatType[one(FloatType)]) :
    _loss_params(PU.AgeBins(bins_idx=bins_idx .|> Int, last_bin_open=true), smoothing_window)
  unbinned_w && @info "Sim-A W = UNBINNED per-year EMD (bins 1:400, no age-smoothing); softplus β + cell W_SCALE recompute on this grid"
  splots, eco_list, species_list, eco_species_ids, splots_val_raw, splots_test_raw, data_outs =
    Data.prepare_parametrization_data(data)
  _write_split_plots(output_dir, eco_list, splots, splots_val_raw, splots_test_raw, data_outs)
  n_species = length(species_list)
  # --- succession plugin selection (static type; everything downstream dispatches on succession_plugin_of(soa)) ---
  SUCC = succession_plugin_type(succession_plugin)
  if SUCC === PnETPlugin.PnET
    spinup && error("succession_plugin=pnet requires spinup: false (cohorts start from the observed state)")
    simb.dual_mode === :off || error("succession_plugin=pnet does not support dual_mode yet (Sim B needs establishment)")
  end
  if SUCC === HBADPlugin.HBAD
    spinup && error("succession_plugin=hbad requires spinup: false (cohorts start from the observed state)")
    # Establishment (natural recruitment, docs/hbad_plugin.md): ON when no_establishment is false (Sim B / free runs); Sim A keeps no_establishment: true.
    no_establishment || @info "succession_plugin=hbad: natural establishment is ON (PROB_ESTAB x SUFFICIENT_LIGHT(shade class from RD), seed rain=$(hbad_seed_rain))"
  end
  # The plugin of the run: an immutable context (BS tables | PnET climate + constants | HBAD priors + options) and the site-attribute builder, passed
  # explicitly to the drivers, `make_sites` and `fit_params` - no module-level plugin state.
  plugin = plugin_config(succession_plugin, species_list, eco_species_ids, eco_list; tables=tables, cohorts_db_path=cohorts_db_path, tablename=tablename,
    pnet_climate_cache=pnet_climate_cache, pnet_co2_csv=pnet_co2_csv, pnet_generic=pnet_generic, pnet_soil_texture=pnet_soil_texture, pnet_soil_csv=pnet_soil_csv,
    hbad_priors_csv=hbad_priors_csv, hbad_coefs_csv=hbad_coefs_csv, hbad_init=hbad_init, hbad_seed_rain=hbad_seed_rain,
    hbad_prob_estab_csv=hbad_prob_estab_csv, hbad_cycle_years=hbad_cycle_years)
  n_species = length(species_list)
  n_ecoregions = length(eco_list)
  n_plots = maximum(splots.plot_id)
  # count-balance reweight: build per-agebin W1 weights ONCE from the TRAIN reference (frozen), on the COARSE
  # age_idx bins; Sim A (per-year) looks them up via CBAL_COARSE. Applied in calculate_species_loss_acc!.
  # RANKW mode: :rank (default, 1/√ln(rank) by AGB) or :cbal_pct (percentile-floored class-balanced by cohort
  # count — species play the role of age-bins in build_cbal_tables; 10% floor caps rare/empty cells).
  # The run-wide TRAIN tables the reference scales are built from: per-cell cohort counts (rankw_mode = :cbal_pct) and the count-balance
  # weights. They travel as `scale_seed` (a LossScales holding only these) to every search driver, which derives the scales of each reference.
  _nc = zeros(Int, 0, 0)
  if loss_params.rankw_mode === :cbal_pct
    _nc = _cell_ncoh(splots, n_species, length(eco_species_ids))
    @info "RANKW mode = cbal_pct (percentile-floored class-balanced, β=$rankw_beta)"
  end
  _cbal = PU.NO_CBAL
  if w_count_balance
    _cbal = PU.build_cbal_tables(splots, PU.AgeBins(bins_idx=bins_idx .|> Int, last_bin_open=true), loss_params.age_bins, eco_species_ids, n_species; beta=w_count_beta, pct_floor=w_count_pct_floor)
    @info "Count-balance reweight ON (mode=$(loss_params.cbal_mode) β=$(w_count_beta) pct_floor=$(w_count_pct_floor)); coarse=$(length(bins_idx)) bins+open, Sim-A W bins=$(length(loss_params.age_bins.bin_widths))"
  end
  # Which species drive DOMINANCE (MO Pareto / CMA-MAE measure / breadth axis / RANKW). Default exact (SPCD only). Travels in the scale seed.
  _dom = dominance_mask(species_list, dominance_species)
  scale_seed = PU.LossScales(cbal=_cbal, ncoh=_nc, dom=_dom)
  # What shares a u-dimension (split sets, tier merges, pooling); writes pool_map.csv. Part of the run's ParamSpace (see param_space.jl).
  tying = build_param_tying(param_split_species, param_tier_merge, param_pooling, splots, eco_list, species_list, eco_species_ids, output_dir)
  if split_only
    @info "split_only: wrote split_plots.csv / species_map.csv / pool_map.csv to $(output_dir) — stopping before search"
    return nothing
  end
  println("Plots:$n_plots, Ecos:$n_ecoregions, Species:$n_species, Measurements: $(size(splots))")
  dominance_species != "all" && @info "Dominance species ($dominance_species): $(count(_dom))/$n_species → $([species_list[s] for s in findall(_dom)])"
  # The parameter space of the run: distributions + slot layout + pooling + covariance blocks (data floors are added below, after the sobol branch).
  space = make_param_space(plugin.ctx, species_list, eco_list, eco_species_ids; no_establishment=no_establishment, fit_establishment=(simb.dual_mode != :off),
    opts=space_opts, tying=tying, blocks=cma_blocks(cmaes_blocks, eco_list))
  println(species_list)

  if diagnose
    Data.print_cycle_coverage(splots; cycle_years=cycle_years, eco_list=eco_list)
    plot_biomass_bin_deltas(splots, loss_params; output_dir=output_dir)
  end

  # Val preprocessing — each half is self-contained with its own contiguous plot_ids
  val_splots = nothing
  val_ref_soa = nothing
  val_spdf_plts = nothing
  val_site_sim_years = nothing
  val_spinup_cohorts = nothing
  val_injection_cohorts = nothing
  if !isnothing(splots_val_raw)
    vref = build_reference(splots_val_raw, eco_species_ids, loss_params; scoring=scoring, simb=simb, no_establishment=no_establishment, plugin=plugin, rng=rng, spinup=spinup)
    val_spdf_plts = vref.spdf_plts
    val_site_sim_years = vref.site_sim_years
    val_spinup_cohorts = vref.spinup_cohorts   # needed for the val Sim-B spinup (dual val)
    val_injection_cohorts = vref.injection_cohorts
    val_ref_soa = vref.ref_soa
    val_splots = splots_val_raw
    println("Val: $(length(unique(splots_val_raw.plot_id))) plots, $(nrow(splots_val_raw)) rows")
  end
  # 3-way split: the test set is carved out of the data (so train is the intended fraction) and HELD OUT of
  # training + val monitoring. It is not fed to the optimizer; evaluate the final model on it post-hoc
  # (the diagnostic scripts reproduce the same split via split_seed/val_frac/test_frac and target the test set).
  if !isnothing(splots_test_raw)
    println("Test (HELD OUT — not used in training/val): $(length(unique(splots_test_raw.plot_id))) plots, $(nrow(splots_test_raw)) rows")
  end

  println("Initiating param distributions")
  #greet()
  #println(splots)
  #println(splots)
  println("Preprocessing plot results (smoothing and binning)")
  debug = (get(ENV, "JULIA_DEBUG", "") ∈ ["Pan", "all"])
  ref = build_reference(splots, eco_species_ids, loss_params; scoring=scoring, simb=simb, no_establishment=no_establishment, plugin=plugin, rng=rng, spinup=spinup, debug=debug)
  @assert minimum(ref.spdf.sim_year) == 0 "$(ref.spdf.sim_year)"
  spdf_plts = ref.spdf_plts; site_sim_years = ref.site_sim_years; spinup_cohorts = ref.spinup_cohorts
  injection_cohorts = ref.injection_cohorts; ref_soa = ref.ref_soa
  total_err = test_spdf(spdf_plts, n_species, eco_species_ids, loss_params)
  println(total_err)
  println(PU.get_total_loss(total_err, loss_params))
  println(sum(length(xs) for xs in eco_species_ids))
  println("beginning trials")
  if search_mode == "sobol"
    return parametrize_sobol(; ref_soa=ref_soa, space=space, plugin=plugin,
      output_dir=output_dir,
      splots=splots,
      spdf_plts=spdf_plts,
      spinup_cohorts=spinup_cohorts,
      site_sim_years=site_sim_years,
      species_list=species_list,
      eco_list=eco_list,
      eco_species_ids=eco_species_ids,
      loss_params=loss_params,
      scoring=scoring,
      simb=simb,
      scale_seed=scale_seed,
      spinup=spinup,
      no_establishment=no_establishment,
      injection_cohorts=injection_cohorts,
      rng=rng,
      debug=debug,
      N=sobol_n,
      saltelli=saltelli,
      M=n_reps,
      eval_tier=tier,
      n_output_plots=n_output_plots,
      cycle_years=cycle_years)
  end
  # per-(eco,species) data floors of B_MAX / ANPP / PROB_ESTAB (each nothing if off); the sobol branch above searches the unfloored space
  space = PU.with_floors(space, build_param_floors(bs_tables(plugin), space_opts, eco_list, species_list, eco_species_ids))
  if context_only   # plot_only / plot_only_sobol and context tools: the training context, no search
    ctx = (; space, plugin, ref_soa, splots, spdf_plts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, scale_seed, injection_cohorts, spinup_cohorts, output_dir, n_species, tier, spinup, no_establishment)
    post_search_hook === nothing || post_search_hook(ctx)
    return ctx
  end
  driver = search_mode == "sa" ? parametrize_SA :
           search_mode == "molbsa" ? parametrize_MOLBSA :
           search_mode == "cmaes" ? parametrize_CMAES :
           search_mode == "mocmaes" ? parametrize_MOCMAES :
           search_mode == "igelmo" ? parametrize_IgelMOCMAES :
           search_mode == "cmame" ? parametrize_CMAMAE :
           search_mode == "ccigel" ? parametrize_CCIgel :
           search_mode == "nsga2" ? parametrize_NSGA2 :
           search_mode == "gdigel" ? parametrize_GDIgel : parametrize_LBSA   # "lbsa"
  # CMA-ES-only knobs; LBSA/MOLBSA don't accept these, so only splat them for the (MO)CMA-ES drivers.
  cmaes_kw = search_mode == "molbsa" ? (archive_cap=cmaes_archive_cap, seed_archive_from=seed_archive_from) :
             search_mode == "cmaes" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, ipop=ipop, ipop_stagnation=ipop_stagnation, ipop_max_barren_restarts=ipop_max_barren_restarts, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov) :
             search_mode == "mocmaes" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, cmaes_warmstart_seeds=cmaes_warmstart_seeds, ipop=ipop, ipop_stagnation=ipop_stagnation, ipop_max_barren_restarts=ipop_max_barren_restarts, archive_cap=cmaes_archive_cap, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov, seed_archive_from=seed_archive_from) :
             search_mode == "igelmo" ? (archive_cap=cmaes_archive_cap, igel_mu=igel_mu, igel_sigma0=igel_sigma0, igel_sobol_init=igel_sobol_init, igel_sobol_pool_k=igel_sobol_pool_k, igel_sobol_raw_mult=igel_sobol_raw_mult, igel_niche_radius=igel_niche_radius, igel_reseed_sigma=igel_reseed_sigma, igel_reseed_random_frac=igel_reseed_random_frac, igel_maturity=igel_maturity, igel_seed_maturity=igel_seed_maturity, igel_warm_start=igel_warm_start, igel_freeze_seed_growth=igel_freeze_seed_growth, single_cov=cmaes_single_cov) :
             search_mode == "ccigel" ? (archive_cap=cmaes_archive_cap, igel_sigma0=igel_sigma0, igel_sobol_init=igel_sobol_init, igel_niche_radius=igel_niche_radius, igel_reseed_sigma=igel_reseed_sigma, igel_maturity=igel_maturity, single_cov=cmaes_single_cov, cc_group_size=cc_group_size, cc_spec_gens=cc_spec_gens, cc_integ_gens=cc_integ_gens, cc_cycles=cc_cycles, cc_fix_others=cc_fix_others) :
             search_mode == "gdigel" ? (gd_opts=gdigel,) :
             search_mode == "nsga2" ? (archive_cap=cmaes_archive_cap, nsga2_pop=nsga2_pop, nsga2_offspring=nsga2_offspring, nsga2_eta_c=nsga2_eta_c, nsga2_eta_m=nsga2_eta_m, nsga2_pc=nsga2_pc, nsga2_pm=nsga2_pm, sobol_init=nsga2_sobol_init) :
             search_mode == "cmame" ? (cmaes_lambda=cmaes_lambda, cmaes_sigma0=cmaes_sigma0, cmame_alpha=cmame_alpha, cmame_grid=cmame_grid, cmame_reseed_explore=cmame_reseed_explore, cmame_restart_patience=cmame_restart_patience, cmame_sobol_reseed=cmame_sobol_reseed, balanced_quality=balanced_quality, cmame_mo_rank=cmame_mo_rank, archive_by_sp=archive_by_sp, bounds_by_sobol=bounds_by_sobol, top_seeds=top_seeds, integer_handling=cmaes_integer_handling, integer_std_factor=cmaes_integer_std_factor, single_cov=cmaes_single_cov) :
             NamedTuple()
  _driver_result = driver(; ref_soa=ref_soa, space=space, plugin=plugin,
      output_dir=output_dir,
      splots=splots,
      spdf_plts=spdf_plts,
      spinup_cohorts=spinup_cohorts,
      site_sim_years=site_sim_years,
      species_list=species_list,
      eco_list=eco_list,
      eco_species_ids=eco_species_ids,
      spinup=spinup,
      search_tier=tier,
      TRIALS=TRIALS,
      resume_from=resume_from,
      start_from=start_from,
      force_restart_from_random=force_restart_from_random,
      n_reps=n_reps,
      sobol_candidates_db=sobol_candidates_db,
      sobol_top_frac=sobol_top_frac,
      loss_params=loss_params,
      scoring=scoring,
      simb=simb,
      scale_seed=scale_seed,
      no_establishment=no_establishment,
      injection_cohorts=injection_cohorts,
      n_output_plots=n_output_plots,
      val_splots=val_splots,
      val_ref_soa=val_ref_soa,
      val_spdf_plts=val_spdf_plts,
      val_site_sim_years=val_site_sim_years,
      val_spinup_cohorts=val_spinup_cohorts,
      val_injection_cohorts=val_injection_cohorts,
      cycle_years=cycle_years,
      rng=rng,
      debug=debug,
      exec=exec,
      search=search,
      cmaes_kw...)
  post_search_hook === nothing || post_search_hook((; space, plugin, ref_soa, splots, spdf_plts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, scale_seed, injection_cohorts, spinup_cohorts, output_dir, n_species, tier, spinup, no_establishment))
  return _driver_result
end

include("loss/aggregate.jl")

include("loss/scales.jl")

include("sim/fit_params.jl")

include("drivers/sobol.jl")

include("drivers/common.jl")

include("io/losses_db.jl")

include("drivers/lbsa.jl")

include("drivers/cmaes.jl")

include("drivers/mo_common.jl")

include("drivers/mocmaes.jl")

include("drivers/igelmo.jl")

include("drivers/ccigel.jl")

include("drivers/nsga2.jl")

include("drivers/cmame.jl")

include("drivers/molbsa.jl")

include("io/checkpoints.jl")

include("drivers/sa.jl")

include("config/tables.jl")

include("config/from_yaml.jl")

function run_from_yaml(yaml_path::String; overrides::AbstractDict=Dict{String,Any}(), exec::Union{Nothing,ExecOptions}=nothing, post_search_hook=nothing)
  exec = exec === nothing ? exec_options_from_env() : exec   # env PAN_TIMING / PAN_SKIP_VAL / PAN_PARALLEL / PAN_NEWBEST_STATS, read once for the whole run
  get!(ENV, "PAN_DATA", "runs")          # data dir for ${PAN_DATA}/*.csv|*.duckdb in the yaml; default = repo runs/ (local),
                                         # set PAN_DATA=<dir> on the cluster (e.g. $SCRATCH/runs/data) → one yaml, both layouts
  get!(ENV, "PAN_OUT", "runs/outputs")   # output root for a ${PAN_OUT}/... output_dir in the yaml; default local, set on cluster.
                                         # (When launched via train_resumable.sh, output_dir is overridden anyway — this is the bare-run fallback.)
  # The one validated reader: unknown/retired keys are errors, values are coerced, defaults filled, ${VAR}/$VAR/~ expanded.
  # `overrides` are runner/CLI values (e.g. per-fold fold_index/output_dir).
  cfg = load_config(yaml_path; overrides=overrides)

  seed = cfg["seed"]
  Random.seed!(seed)
  rng = RNGType(rand(UInt64))

  # Data-prep inputs (filter_plots, exclude_plots[_csv], split/region/site-class keys, ...) are parsed by the SAME
  # function the analysis scripts use (`data_config`), so training and re-derivations can never drift.
  din = _cfg_data_inputs(cfg)
  filter_plots = din.filter_plots
  exclude_plots = din.exclude_plots
  isempty(exclude_plots) || @info "exclude_plots: dropping $(length(exclude_plots)) plots from the loaded set"

  sw_size = cfg["smoothing_window_size"]
  smoothing_window = sw_size > 0 ?
                     PU.get_smoothing_window(; smoothing_window=sw_size, smoothing_variance=FloatType(cfg["smoothing_variance"])) :
                     FloatType[one(FloatType)]

  # K-fold: each fold writes to <output_dir>/fold_<fold_index> so the 5 fold runs don't collide.
  _n_folds = cfg["n_folds"]; _fold_index = cfg["fold_index"]
  _output_dir = cfg["output_dir"]
  _n_folds > 1 && (_output_dir = joinpath(_output_dir, "fold_$(_fold_index)"))
  isdir(_output_dir) || mkpath(_output_dir)

  _db_path = din.cohorts_db_path   # (yaml value, with the FIADB env override applied by load_config)

  # Fields shared by parametrize, plot_sample, and plot_sample_sobol
  data = data_config(cfg)   # every data-preparation input of the run (selection, split, strata, tiering, disturbance parquet, on_missing), passed to parametrize
  common_kw = (
    data=data,
    output_dir=_output_dir,
    no_establishment=cfg["no_establishment"],
    split_only=cfg["split_only"],
    bins_idx=cfg["bins_idx"],
  )
  n_output_plots = cfg["n_output_plots"]
  search_mode = cfg["search_mode"]

  scoring, simb = scoring_from_config(cfg)   # scoring rule + Sim B options (shared with the analysis tools), passed explicitly below
  search = search_options_from_config(cfg)   # LBSA / Igel / MO-CMA-ES-ranking knobs of the run, passed to the drivers explicitly
  search.mocmaes_nds && @info "MO-CMA-ES offspring ranking: TRUE NSGA-II non-dominated sorting (Pareto fronts + crowding)"
  tables = bs_tables_from_cfg(cfg, _db_path)   # the Biomass Succession data tables of the run, passed explicitly (parametrize(tables=...))

  # plot modes (B8): the plots use the training context (same data prep, splits, strata, loss and scoring flags as a search)
  plot_mode = search_mode in ("plot_only", "plot_only_sobol")
  plot_params_path = plot_mode && search_mode == "plot_only" ? cfg["params_path"] : ""
  plot_sobol_db = search_mode == "plot_only_sobol" ? something(cfg["sobol_candidates_db"], "") : ""
  search_mode == "plot_only" && isempty(plot_params_path) && error("search_mode=plot_only requires params_path in yaml")
  search_mode == "plot_only_sobol" && isempty(plot_sobol_db) && error("search_mode=plot_only_sobol requires sobol_candidates_db in yaml")

  _res = parametrize(;
    common_kw...,
    search_mode=plot_mode ? "igelmo" : search_mode,
    context_only=plot_mode,
    tier=cfg["tier"],
    smoothing_window=smoothing_window,
    unbinned_w=cfg["unbinned_w"],   # Sim A: exact per-year EMD (no binning/age-smoothing)
    w_count_balance=cfg["w_count_balance"],
    w_count_balance_mode=cfg["w_count_balance_mode"],
    w_count_beta=cfg["w_count_beta"],
    w_count_pct_floor=cfg["w_count_pct_floor"],
    rankw_mode=cfg["rankw_mode"],
    rankw_beta=cfg["rankw_beta"],
    rankw_pct_floor=cfg["rankw_pct_floor"],
    param_split_species=cfg["param_split_species"],
    param_pooling=cfg["param_pooling"],
    succession_plugin=cfg["succession_plugin"],
    tables=tables,
    pnet_climate_cache=cfg["pnet_climate_cache"],
    pnet_co2_csv=cfg["pnet_co2_csv"],
    pnet_generic=cfg["pnet_generic"],
    pnet_soil_texture=cfg["pnet_soil_texture"],
    pnet_soil_csv=cfg["pnet_soil_csv"],
    hbad_priors_csv=cfg["hbad_priors_csv"],
    hbad_coefs_csv=cfg["hbad_coefs_csv"],
    hbad_init=cfg["hbad_init"],
    hbad_seed_rain=cfg["hbad_seed_rain"],
    hbad_prob_estab_csv=cfg["hbad_prob_estab_csv"],
    hbad_cycle_years=cfg["cycle_years"],
    cmaes_blocks=cfg["cmaes_blocks"],
    param_tier_merge=cfg["param_tier_merge"],
    TRIALS=cfg["trials"],
    resume_from=cfg["resume_from"],
    start_from=cfg["start_from"],
    force_restart_from_random=cfg["force_restart_from_random"],
    sobol_n=cfg["sobol_n"],
    saltelli=cfg["saltelli"],
    n_reps=cfg["n_reps"],
    sobol_candidates_db=cfg["sobol_candidates_db"],
    sobol_top_frac=cfg["sobol_top_frac"],
    n_output_plots=n_output_plots,
    diagnose=cfg["diagnose"],
    cycle_years=cfg["cycle_years"],
    loss_lambda=cfg["loss_lambda"],
    loss_alpha=cfg["loss_alpha"],
    agb_hinge=cfg["agb_hinge"],
    agb_hinge_threshold=cfg["agb_hinge_threshold"],
    agb_hinge_pct=cfg["agb_hinge_pct"],
    agb_hinge_pct_min=cfg["agb_hinge_pct_min"],
    agb_hinge_pct_max=cfg["agb_hinge_pct_max"],
    agb_hinge_l2=cfg["agb_hinge_l2"],
    agb_hinge_p=cfg["agb_hinge_p"],   # hinge exponent pen(h)=h^p, default 2 (L2)
    agb_hinge_beta=cfg["agb_hinge_beta"],
    agb_normalize=cfg["agb_normalize"],
    w_normalize=cfg["w_normalize"],
    cell_normalize=cfg["cell_normalize"],
    w_scale_factor=cfg["w_scale_factor"],
    w_smooth=cfg["w_smooth"],
    w_smooth_band=cfg["w_smooth_band"],
    w_smooth_conc=cfg["w_smooth_conc"],
    w_smooth_beta=cfg["w_smooth_beta"],
    w_smooth_auto_knee=cfg["w_smooth_auto_knee"],
    w_p=cfg["w_p"],
    w_hinge=cfg["w_hinge"],
    w_hinge_pct=cfg["w_hinge_pct"],
    w_hinge_pct_min=cfg["w_hinge_pct_min"],
    w_hinge_pct_max=cfg["w_hinge_pct_max"],
    loss_piecewise=cfg["loss_piecewise"],
    w_pivot=cfg["w_pivot"],
    agb_pivot=cfg["agb_pivot"],
    scoring=scoring,
    simb=simb,
    space_opts=param_space_options(cfg),   # which parameters are searched and their boxes
    cmaes_lambda=cfg["cmaes_lambda"],
    cmaes_sigma0=cfg["cmaes_sigma0"],
    cmaes_warmstart_seeds=cfg["cmaes_warmstart_seeds"],
    ipop=cfg["ipop"],
    ipop_stagnation=cfg["ipop_stagnation"],
    ipop_max_barren_restarts=cfg["ipop_max_barren_restarts"],
    cmaes_archive_cap=cfg["cmaes_archive_cap"],
    seed_archive_from=cfg["seed_archive_from"],
    cmaes_integer_handling=cfg["cmaes_integer_handling"],
    cmaes_integer_std_factor=cfg["cmaes_integer_std_factor"],
    cmaes_single_cov=cfg["cmaes_single_cov"],
    nsga2_pop=cfg["nsga2_pop"],
    nsga2_offspring=cfg["nsga2_offspring"],
    nsga2_eta_c=cfg["nsga2_eta_c"],
    nsga2_eta_m=cfg["nsga2_eta_m"],
    nsga2_pc=cfg["nsga2_pc"],
    nsga2_pm=cfg["nsga2_pm"],
    nsga2_sobol_init=cfg["nsga2_sobol_init"],
    igel_mu=cfg["igel_mu"],
    igel_sigma0=cfg["igel_sigma0"],
    igel_sobol_init=cfg["igel_sobol_init"],
    igel_sobol_pool_k=cfg["igel_sobol_pool_k"],        # >1 ⇒ oversample: keep best μ·k by dominance, random-pick μ
    igel_sobol_raw_mult=cfg["igel_sobol_raw_mult"],    # raw Sobol pool = (μ·pool_k)·raw_mult candidates evaluated
    igel_niche_radius=cfg["igel_niche_radius"],
    igel_reseed_sigma=cfg["igel_reseed_sigma"],
    igel_reseed_random_frac=cfg["igel_reseed_random_frac"],
    igel_maturity=cfg["igel_maturity"],
    igel_seed_maturity=cfg["igel_seed_maturity"],
    igel_warm_start=cfg["igel_warm_start"],
    igel_freeze_seed_growth=cfg["igel_freeze_seed_growth"],
    cc_group_size=cfg["cc_group_size"],
    cc_spec_gens=cfg["cc_spec_gens"],
    cc_integ_gens=cfg["cc_integ_gens"],
    cc_cycles=cfg["cc_cycles"],
    cc_fix_others=cfg["cc_fix_others"],
    gdigel=cfg["gdigel"],
    cmame_alpha=cfg["cmame_alpha"],
    cmame_grid=cfg["cmame_grid"],
    cmame_reseed_explore=cfg["cmame_reseed_explore"],
    cmame_restart_patience=cfg["cmame_restart_patience"],
    cmame_sobol_reseed=cfg["cmame_sobol_reseed"],
    cmame_mo_rank=cfg["cmame_mo_rank"],
    balanced_quality=cfg["balanced_quality"],
    archive_by_sp=cfg["archive_by_sp"],
    bounds_by_sobol=cfg["bounds_by_sobol"],
    top_seeds=cfg["top_seeds"],
    dominance_species=cfg["dominance_species"],
    exec=exec,
    search=search,
    post_search_hook=post_search_hook,
    rng=rng)
  plot_mode || return _res
  return search_mode == "plot_only" ?
    plot_sample(_res; params_path=plot_params_path, n_output_plots=max(1, n_output_plots), rng_seed=seed) :
    plot_sample_sobol(_res; sobol_candidates_db=plot_sobol_db, n_output_plots=max(1, n_output_plots),
                      n_sobol_params_to_plot=max(1, cfg["n_sobol_params_to_plot"]), rng_seed=seed)
end

function main()
  if length(ARGS) >= 1 && (endswith(ARGS[1], ".yaml") || endswith(ARGS[1], ".yml"))
    return run_from_yaml(ARGS[1])
  end
  seed = 12312315 #404 #1337
  Random.seed!(seed)
  rng = RNGType(rand(UInt64))
  #filter_ecos = String["8.5.3.75e", "8.5.3.75f", "8.5.3.75a", "8.5.3.75c", "8.5.3.75g", "8.3.5.65o", "8.5.3.75d", "8.5.3.75h", "8.3.5.65h", "8.3.5.65f", "8.3.5.65g", "15.4.1.76b", "8.5.3.75b", "8.5.3.75i", "9.4.7.32b", "8.3.7.35b", "8.3.7.35e", "8.5.1.63h", "8.3.7.35g", "8.3.7.35f", "8.3.5.65l", "8.3.5.65c", "9.5.1.34a"]
  #filter_ecos = String["8.5.3", "8.3.5", "15.4.1", "9.4.7", "8.3.7", "8.5.1", "9.5.1"]
  #filter_ecos=String["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
  #filter_ecos = String["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
  #filter_ecos = String["8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
  #filter_ecos = String["8.5.3.75f"]
  #filter_ecos = String["8.5.3.75g"]
  #filter_ecos = ["8.5.3.75g"]
  filter_ecos = ["8.5.3"]
  filter_plots = NTuple{4,Int}[]#(13, 1, 49, 26)]  # e.g. [(statecd,unitcd,countycd,plot), ...]
  filter_species = String[] #String["PIEL"]        # e.g. ["ACRU", "QURU"]
  # bins_idx = vcat(10:10:40, 60:20:120)
  t1_bins_idx = [20, 60, 120]
  # smoothing_window = PU.get_smoothing_window(; smoothing_window=2, smoothing_variance=FloatType(1.2f0))
  t1_smoothing_window = PU.get_smoothing_window(; smoothing_window=2, smoothing_variance=FloatType(1.2f0))

  resume_from = "./outputs/search_state_latest.jld2" #nothing             # set to e.g. "./outputs/search_state_latest.jld2" to resume
  force_restart_from_random = false # set to true to restart from random params when resuming
  parametrize(;
    data=Data.DataConfig(; cohorts_db_path="../data_eco_cohorts.duckdb", filter_eco_field="epa_l3", eco_field="epa_l3", tablename="data_eco_cohorts",
      filter_ecos=filter_ecos, filter_plots=filter_plots, filter_species=filter_species, skip_disturbances=true, spinup=false),
    output_dir="./outputs",
    search_mode="lbsa",
    tier=1,
    bins_idx=t1_bins_idx,
    smoothing_window=t1_smoothing_window,
    TRIALS=1000000,
    resume_from=resume_from,
    force_restart_from_random=force_restart_from_random,
    rng=rng)
end

function julia_main()::Cint
  main()
  return 0
end

# ---------------------------------------------------------------------------
# Standalone plot helpers
# ---------------------------------------------------------------------------

function _load_params_from_path(params_path::String, pc=nothing)
  raw = JLD2.load_object(params_path)
  if raw isa SA.SAState
    @info "Loaded SAState — using best params" loss = convert(Float64, raw.best.fx) iter = raw.best_iteration
    return raw.best.x
  elseif raw isa LBSA.LBSAState
    @info "Loaded LBSAState — using best params" loss = convert(Float64, raw.best.fx) iter = raw.best_iteration
    return raw.best.x
  elseif raw isa MOLBSA.MOLBSAState
    @info "Loaded MOLBSAState — using representative params" agg = raw.representative.fx.aggregate iter = raw.best_iteration
    return raw.representative.x
  elseif raw isa CMAES.CMAESState
    @info "Loaded CMAESState — using best params" loss = convert(Float64, raw.best.fx) iter = raw.best_iteration
    return raw.best.x
  elseif raw isa MOCMAES.MOCMAESState
    @info "Loaded MOCMAESState — using representative params" agg = raw.representative.fx.aggregate iter = raw.best_iteration archive = length(raw.archive)
    return raw.representative.x
  elseif raw isa IgelMOCMAES.IgelState
    @info "Loaded IgelState — using representative params" agg = raw.representative.fx.aggregate iter = raw.best_iteration archive = length(raw.archive)
    return raw.representative.x
  elseif raw isa CMAMAE.CMAMAEMOState
    @info "Loaded CMAMAEMOState — using representative params" agg = raw.representative.fx.aggregate iter = raw.best_iteration archive = length(raw.archive)
    return raw.representative.x
  elseif raw isa JLD2.ReconstructedMutable && occursin("HBADParams", String(typeof(raw).parameters[1]))
    @info "Loaded HBADParams saved before the establishment fields existed - upgraded with the defaults"
    return HBADPlugin.upgrade_params(raw, pc isa HBADPlugin.HBADContext ? pc : nothing)
  elseif raw isa Vector  # sobol results
    @info "Loaded sobol results — using rank-1 params" mean_loss = raw[1].mean_loss
    return raw[1].params
  else
    @info "Loaded params directly"
    return raw
  end
end

# One simulation of `bio_params` on the training context `ctx` (parametrize(context_only=true)), then the sampled plots' figures.
function _run_and_plot(bio_params, label, ctx; sampled_ids, emp_sample, output_dir, rng)
  injection = build_injection_plan(ctx.injection_cohorts, ctx.ref_soa, ctx.eco_species_ids, ctx.loss_params, ctx.scoring)
  max_sim_year = maximum(maximum.(filter(!isempty, ctx.site_sim_years.sim_years)))
  result = only(fit_params(ctx.ref_soa, bio_params, max_sim_year, ctx.n_species, ctx.eco_species_ids,
    ctx.spdf_plts, ctx.site_sim_years, ctx.spinup, ctx.spinup_cohorts, ctx.loss_params;
    debug=false, search_tier=3, seeds=[rand(rng, UInt64)], cache_preinject=!isnothing(injection), collect_cohorts=true,
    injection=injection,
    scales=compute_loss_scales(ctx.loss_params, 3, ctx.spdf_plts, nothing, ctx.eco_species_ids, ctx.n_species; per_plot_mean=ctx.simb.t4_per_plot_mean, cbal=ctx.scale_seed.cbal, ncoh=ctx.scale_seed.ncoh, dom=ctx.scale_seed.dom), scoring=ctx.scoring, simb=ctx.simb, plugin=ctx.plugin))
  loss = convert(Float64, PU.get_total_loss(result[1], ctx.loss_params))
  sim_sample = _filter_cached_to_df(result[2], sampled_ids)
  generate_plots(emp_sample, sim_sample, label, loss, output_dir)
  @info "Plots saved [$label]" loss n_plots = length(sampled_ids)
end

"""
plot_sample — run one simulation with given params and plot n_output_plots forest plots.
Accepts best_params@X.jld2, search_state@X.jld2, or sobol_results@X.jld2.
"""
function plot_sample(ctx; params_path::String, n_output_plots::Int=20, rng_seed::Int=1337)
  output_dir = ctx.output_dir; mkpath(output_dir)
  rng = RNGType(UInt64(rng_seed))
  bio_params = _load_params_from_path(params_path, ctx.plugin.ctx)
  all_ids = UIntType.(unique(ctx.splots.plot_id))
  sampled_ids = _sample_plot_ids(all_ids, n_output_plots, rng; injection_cohorts=ctx.injection_cohorts)
  emp_sample = _make_emp_df(ctx.splots, sampled_ids)
  _run_and_plot(bio_params, "plot_only", ctx; sampled_ids, emp_sample, output_dir, rng)
end

"""
plot_sample_sobol — load saved sobol results and plot a stratified sample of param sets.
Selection: 1 absolute top + 60% of remaining from top half + rest from bottom half.
All param sets are visualised on the same fixed set of n_output_plots forest plots.
"""
function plot_sample_sobol(ctx; sobol_candidates_db::String, n_output_plots::Int=10, n_sobol_params_to_plot::Int=5, rng_seed::Int=1337)
  output_dir = ctx.output_dir
  mkpath(output_dir)
  rng = RNGType(UInt64(rng_seed))

  db = DuckDB.DB(sobol_candidates_db)
  con = DuckDB.connect(db)
  raw = DuckDB.execute(con, "SELECT mean_loss, params_blob FROM sobol_results ORDER BY mean_loss ASC, sobol_idx ASC") |> DataFrame
  close(db)
  isempty(raw) && (@warn "No sobol results found in $sobol_candidates_db"; return)
  results = [(params=Serialization.deserialize(IOBuffer(row.params_blob)), mean_loss=row.mean_loss)
             for row in eachrow(raw)]
  N = length(results)
  @info "Loaded $N sobol results from $sobol_candidates_db"

  # Stratified param-set selection
  n_extra = n_sobol_params_to_plot - 1
  n_top = round(Int, 0.6 * n_extra)
  n_bot = n_extra - n_top
  top_half = results[2:max(2, N ÷ 2)]
  bot_half = results[max(2, N ÷ 2)+1:end]
  selected = vcat(
    [results[1]],
    Random.shuffle(rng, top_half)[1:min(n_top, length(top_half))],
    Random.shuffle(rng, bot_half)[1:min(n_bot, length(bot_half))],
  )
  @info "plot_sample_sobol: plotting $(length(selected)) param sets from $N sobol results"

  all_ids = UIntType.(unique(ctx.splots.plot_id))
  sampled_ids = _sample_plot_ids(all_ids, n_output_plots, rng; injection_cohorts=ctx.injection_cohorts)
  emp_sample = _make_emp_df(ctx.splots, sampled_ids)

  for (rank, res) in enumerate(selected)
    label = rank == 1 ? "sobol_rank1_loss$(round(res.mean_loss,digits=4))" :
            "sobol_rank$(rank)_loss$(round(res.mean_loss,digits=4))"
    _run_and_plot(res.params, label, ctx; sampled_ids, emp_sample, output_dir, rng)
  end
end

# ---------------------------------------------------------------------------
# Statistical testing: simulated vs reference plot distributions
# ---------------------------------------------------------------------------


function simulate_and_test(;
  splots::DataFrame,
  bio_params,
  plugin::PluginConfig,
  eco_list::Vector{String},
  species_list::Vector{String},
  eco_species_ids::Vector{Vector{Int}},
  loss_params::PU.LossParams,
  site_sim_years,
  M::Int=10,
  equiv_margin::Float64=0.2,   # equivalence band = ±equiv_margin · mean(ref)
  tost_alpha::Float64=0.05,
  no_establishment::Bool=false,
  rng::Random.AbstractRNG,
)::DataFrame
  n_ecos = length(eco_list)
  n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
  max_sim_year = maximum(maximum.(filter(!isempty, site_sim_years.sim_years)))
  ref_soa = make_sites(splots, eco_species_ids; plugin=plugin, rng=rng, spinup=false, no_establishment=no_establishment)
  SP = succession_plugin_of(ref_soa)
  eco_params = generate_eco_params(plugin.ctx, bio_params)
  ctx = make_ctx(plugin.ctx, eco_params)

  # Reference: per (eco_id, sp_eco, bin) → [AGB per (plot, measurement_year)], excluding the
  # initial year (sim_year=0) since that's the state we initialize from.
  ref_vals = [[[FloatType[] for _ in 1:n_bins] for _ in 1:length(eco_species_ids[e])] for e in 1:n_ecos]
  for gdf in groupby(splots, [:plot_id, :eco_id, :eco_species_id, :measdate])
    r = gdf[1, :]
    sim_year = Int(round(Dates.value(Dates.Day(r.measdate - r.start_measdate)) / 365.25))
    sim_year > 0 || continue
    eco_id = Int(r.eco_id)
    sp_eco = Int(r.eco_species_id)
    (1 <= eco_id <= n_ecos && 1 <= sp_eco <= length(eco_species_ids[eco_id])) || continue
    bin_agbs = zeros(FloatType, n_bins)
    for row in eachrow(gdf)
      b = PU.find_age_bin(max(1, Int(round(Float64(row.age_calc)))), loss_params.age_bins)
      b == 0 && continue
      bin_agbs[b] += FloatType(row.agb_sum)
    end
    for b in 1:n_bins
      push!(ref_vals[eco_id][sp_eco][b], bin_agbs[b])
    end
  end

  # Simulate M times; accumulate per (site_idx, eco_id, sp_eco, bin) to average across reps.
  sim_sum = Dict{NTuple{4,Int},Float64}()
  sim_cnt = Dict{NTuple{4,Int},Int}()

  for seed in (rand(rng, UInt64) for _ in 1:M)
    soa = copy_and_reseed_soa(ref_soa, seed)
    for current_sim_year in 1:max_sim_year
      step!(SP, soa, current_sim_year, ctx)
      for i in 1:soa.n
        site = getsite(soa, i)
        !site.active && continue
        sim_years = site_sim_years.sim_years[site.mapcode]
        if current_sim_year in sim_years
          eco_id = Int(site.eco_id)
          n_sp_eco = length(eco_species_ids[eco_id])
          bin_agbs = zeros(FloatType, n_sp_eco, n_bins)
          for j in 1:site.live
            sp_eco = Int(site.c_species[j])
            b = PU.find_age_bin(max(1, Int(ceil(Float64(site.c_age[j])))), loss_params.age_bins)
            b == 0 && continue
            bin_agbs[sp_eco, b] += site.c_bio[j]
          end
          for sp_eco in 1:n_sp_eco
            for b in 1:n_bins
              key = (i, eco_id, sp_eco, b)
              sim_sum[key] = get(sim_sum, key, 0.0) + Float64(bin_agbs[sp_eco, b])
              sim_cnt[key] = get(sim_cnt, key, 0) + 1
            end
          end
        end
        if current_sim_year == last(sim_years)
          site.active = false
        end
      end
    end
  end

  # Per (eco_id, sp_eco, bin): one mean-across-M value per (site, measurement-year) pair.
  sim_vals = [[[FloatType[] for _ in 1:n_bins] for _ in 1:length(eco_species_ids[e])] for e in 1:n_ecos]
  for (key, total) in sim_sum
    (_, eco_id, sp_eco, b) = key
    push!(sim_vals[eco_id][sp_eco][b], FloatType(total / sim_cnt[key]))
  end

  # TOST (two one-sided tests) for equivalence per (eco, species, bin), on log1p(AGB)
  # so the band is multiplicative (handles skew + zeros). Δ = log1p(equiv_margin) is a
  # fixed log-scale band ≈ a factor-(1+equiv_margin) tolerance on the original scale.
  # Equivalent at tost_alpha iff BOTH one-sided Welch tests reject — i.e. the geometric-
  # mean ratio sim/ref is provably inside [1/(1+m), (1+m)] (approximate under log1p).
  bin_label(b) = b <= length(loss_params.age_bins.bins_idx) ?
                 "<$(loss_params.age_bins.bins_idx[b])" : ">=$(loss_params.age_bins.bins_idx[end])"

  Δ = log1p(equiv_margin)   # fixed band on the log1p scale

  rows = NamedTuple[]
  for eco_id in 1:n_ecos
    for sp_eco in eachindex(eco_species_ids[eco_id])
      gsp = eco_species_ids[eco_id][sp_eco]
      for b in 1:n_bins
        rawref = Float64.(ref_vals[eco_id][sp_eco][b])
        rawsim = Float64.(sim_vals[eco_id][sp_eco][b])
        (length(rawref) < 2 || length(rawsim) < 2) && continue   # need a variance per side

        ref = log1p.(rawref)
        sim = log1p.(rawsim)
        d = Statistics.mean(sim) - Statistics.mean(ref)

        if Statistics.var(ref) == 0 && Statistics.var(sim) == 0
          # both bins constant → no sampling uncertainty; decide directly (Welch SE would be 0)
          equiv = abs(d) < Δ
          p_tost, ci_lo, ci_hi = (equiv ? 0.0 : 1.0), d, d
        else
          # two one-sided Welch tests against the shifted nulls ∓Δ (SE > 0 here)
          p_lower = HypothesisTests.pvalue(HypothesisTests.UnequalVarianceTTest(sim .+ Δ, ref); tail=:right)  # H1: d > −Δ
          p_upper = HypothesisTests.pvalue(HypothesisTests.UnequalVarianceTTest(sim .- Δ, ref); tail=:left)   # H1: d <  Δ
          p_tost = max(p_lower, p_upper)
          ci = HypothesisTests.confint(HypothesisTests.UnequalVarianceTTest(sim, ref); level=1 - 2 * tost_alpha)
          ci_lo, ci_hi = ci[1], ci[2]
        end

        push!(rows, (
          eco=eco_list[eco_id],
          species=species_list[gsp],
          bin=b,
          age_class=bin_label(b),
          n_ref=length(rawref),
          n_sim=length(rawsim),
          mean_ref=Statistics.mean(rawref),   # original scale, for context
          log_diff=d,                          # mean(log1p sim) − mean(log1p ref)
          ratio=exp(d),                        # ≈ geometric-mean ratio sim/ref
          margin=Δ,                            # log-scale band
          ci_lo=ci_lo, ci_hi=ci_hi,            # CI for log_diff
          p_tost=p_tost,
          equivalent=(p_tost < tost_alpha),
        ))
      end
    end
  end

  return isempty(rows) ? DataFrame() : DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Spatial (forward) simulation
# ---------------------------------------------------------------------------

function make_sites_from_communities(
  splots::DataFrame,
  eco_species_ids::Vector{Vector{Int}},
  rng::Random.AbstractRNG,
)
  site_df = unique(select(splots, [:mapcode, :eco_id]))
  sort!(site_df, :mapcode)
  n = nrow(site_df)

  splots_dict = Dict(
    key.mapcode => DataFrame(rows)
    for (key, rows) in pairs(groupby(splots, :mapcode, sort=false))
  )

  cohort_counts = Int32[nrow(splots_dict[row.mapcode]) for row in eachrow(site_df)]
  species_counts = Int32[length(eco_species_ids[row.eco_id]) for row in eachrow(site_df)]

  @info "Pre-SoA RSS: $(round(Sys.maxrss()/1e9, digits=2)) GB"
  soa = ActiveSoA((cohort=cohort_counts, species=species_counts))
  base_seed = rand(rng, UInt64)   # per-site RNG = RNGType(hash((base_seed, i))) below → thread-count-invariant & reproducible

  Threads.@threads :static for i in 1:n
    @inbounds begin
      row = site_df[i, :]
      cohorts = splots_dict[row.mapcode]
      site = getsite(soa, i)

      site.active = true
      site.rng = RNGType(hash((base_seed, i)))
      site.mapcode = row.mapcode
      site.ecocode = 0
      site.eco_id = row.eco_id
      site.ref_cn = row.mapcode
      site.old = 0
      site.live = 0
      site.B = zero(FloatType)
      site.AGNPP = zero(FloatType)
      site.harvestCapacityReduction = zero(FloatType)
      site.growthReduction = zero(FloatType)
      site.prevYearMortality = zero(FloatType)
      site.shade_class = 1
      site.c_species .= zero(UIntType)
      site.c_age .= zero(FloatType)
      site.c_bio .= zero(FloatType)
      site.c_m_tot .= zero(FloatType)
      site.c_comp .= zero(FloatType)
      site.no_establish = false
      site.sp_mature .= false
      site.sp_sprout .= false
      site.sp_seed .= false
      site.sp_serotiny .= false
      site.sp_plant .= false

      for cohort_row in eachrow(cohorts)
        BiomassSuccessionPlugin.add_cohort!(
          site,
          UIntType(cohort_row.eco_species_id),
          FloatType(cohort_row.age_calc),
          FloatType(cohort_row.agb_sum),
        )
      end
    end
  end
  return soa
end

function simulate_spatial_treemap(;
  data_dir::String,
  output_dir::String,
  eco_raster::String,
  eco_ecocode_mapping::String,
  biomass_params_path::String,
  treemap_raster::Union{String,Nothing}=nothing,
  communities_csv::Union{String,Nothing}=nothing,
  communities_db::Union{String,Nothing}=nothing,
  treemap_version::Int=2022,
  treemap_db_path::String="../data_eco_cohorts.duckdb",
  rng_seed::Int=113387,
  timehorizon_years::Int=50,
  output_every_years::Int=5,
  simple_match::Bool=true,
  stratify_eco_mixed::Bool=false,
)
  rng = RNGType(UInt64(rng_seed))

  println("Loading params: $biomass_params_path")
  params = JLD2.load_object(joinpath(data_dir, biomass_params_path))

  println("Loading eco raster")
  eco_raster_data = SpatialData.load_eco_raster(joinpath(data_dir, eco_raster))

  eco_mapping_path = joinpath(data_dir, eco_ecocode_mapping)
  eco_mapping_df = CSV.read(eco_mapping_path, DataFrame)

  local splots_pixels, mod_params, eco_species_ids

  if !isnothing(treemap_raster)
    println("Loading treemap raster: $treemap_raster")
    @time cn_raster, _ = SpatialData.load_treemap_raster(
      joinpath(data_dir, treemap_raster); treemap_version=treemap_version)
    @assert size(cn_raster) == size(eco_raster_data) "Raster size mismatch: treemap $(size(cn_raster)) ≠ eco $(size(eco_raster_data))"

    println("Extracting cohorts from DuckDB (treemap path)")
    @time splots, eco_list, eff_eco_list, species_list = (simple_match ? SpatialData.load_treemap_cohorts_simple : SpatialData.load_treemap_cohorts)(
      cn_raster, eco_raster_data, treemap_db_path, eco_mapping_path, params; stratify_eco_mixed=stratify_eco_mixed)
    println("Plots: $(length(unique(splots.plt_cn))), Ecos: $(length(eco_list)), Species: $(length(species_list))")
    println(species_list)

    println("Remapping params to data eco/species")
    @time mod_params, mapped_splots, eco_species_ids = SpatialData.map_params_to_data_treemap(
      params, eco_list, eff_eco_list, species_list, splots)

    println("Expanding cohorts to raster pixels")
    @time splots_pixels = SpatialData.expand_to_pixels(
      mapped_splots, cn_raster, eco_raster_data)

  elseif !isnothing(communities_csv)
    println("Loading initial communities from CSV: $communities_csv")
    ic_df = SpatialData.load_csv_communities(joinpath(data_dir, communities_csv))
    splots_pixels, mod_params, eco_species_ids = SpatialData.prepare_general_splots(
      ic_df, eco_raster_data, eco_mapping_df, params)

  elseif !isnothing(communities_db)
    println("Loading initial communities from DuckDB: $communities_db")
    ic_df = SpatialData.load_duckdb_communities(communities_db)
    splots_pixels, mod_params, eco_species_ids = SpatialData.prepare_general_splots(
      ic_df, eco_raster_data, eco_mapping_df, params)

  else
    error("Specify one of: treemap_raster, communities_csv, communities_db")
  end

  n_sites = length(unique(splots_pixels.mapcode))
  println("Initializing SoA: $n_sites sites")
  @time ref_soa = make_sites_from_communities(splots_pixels, eco_species_ids, rng)

  eco_params = BiomassSuccessionPlugin.generate_eco_params(mod_params)

  println("Running simulation: $timehorizon_years years, output every $output_every_years")
  let ctx = (eco_params=eco_params,),
    bufs = [BiomassSuccessionPlugin._new_buf() for _ in 1:Threads.maxthreadid()],
    chunks = zeros(Int, Threads.maxthreadid())

    Spatial.run_spatial!(ref_soa, BiomassSuccessionPlugin.BiomassSuccession, ctx, output_dir,
      (soa, year) -> BiomassSuccessionPlugin.emit_year!(bufs, chunks, soa, year, output_dir);
      timehorizon=timehorizon_years, output_every=output_every_years)
  end

  println("Generating output rasters")
  ref_raster_path = !isnothing(treemap_raster) ?
                    joinpath(data_dir, treemap_raster) :
                    joinpath(data_dir, eco_raster)
  BiomassSuccessionPlugin.generate_rasters_from_output(; output_dir=output_dir, ef_raster_path=ref_raster_path)

  println("Coalescing Arrow files to DuckDB")
  BiomassSuccessionPlugin.coalesce_to_duckdb(; output_dir=output_dir, db_path=joinpath(output_dir, "cohorts.duckdb"))
end

function spatial_main()
  simulate_spatial_treemap(
    data_dir="../",
    output_dir="./outputs/spatial",
    eco_raster="eco_raster.tif",
    eco_ecocode_mapping="eco_ecocode_mapping.csv",
    biomass_params_path="landis_parametrization_julia/outputs/best_params.jld2",
    treemap_raster="treemap.tif",
    treemap_version=2022,
    treemap_db_path="../data_eco_cohorts.duckdb",
    timehorizon_years=50,
    output_every_years=5,
  )
end

function simulate_spatial_landis(;
  output_dir::String,
  initial_communities_tif::String,
  ecoregion_tif::String,
  initial_communities_csv::String,
  core_species_data::String,
  spp_ecoregion_data::String,
  species_data::String,
  eco_ecocode_mapping::String,
  spp_eco_year::Int=0,
  min_rel_biomass::Vector{Float32}=Float32[0.25, 0.45, 0.56, 0.70, 0.90],
  rng_seed::Int=1337,
  timehorizon_years::Int=50,
  output_every_years::Int=1,
)
  rng = RNGType(UInt64(rng_seed))

  println("Loading LANDIS rasters")
  @time communities_raster = SpatialData.load_landis_mapcode_raster(initial_communities_tif)
  @time eco_raster = SpatialData.load_eco_raster(ecoregion_tif)
  @assert size(communities_raster) == size(eco_raster) "Raster size mismatch"

  println("Loading LANDIS tables")
  println("eco ecocode mapping")
  eco_ecocode_df = CSV.read(eco_ecocode_mapping, DataFrame)
  println("eco ecocode mapping, done")

  println("initial communities csv")
  ic_df = CSV.read(initial_communities_csv, DataFrame)
  println("initial communities csv: done")

  println("Core species data")
  core_sp_df = SpatialData.load_landis_core_species(core_species_data)
  println("Core species data done")
  println("Core biomass spp")
  spp_eco_df = SpatialData.load_landis_spp_ecoregion(spp_ecoregion_data; year=spp_eco_year)
  println("Core biomass spp done")
  println("species data")
  species_df = CSV.read(species_data, DataFrame)
  println("species data done")

  println("Building BiomassSuccessionParams from LANDIS files")
  @time params = SpatialData.make_landis_params(
    core_sp_df, species_df, spp_eco_df;
    min_rel_biomass=min_rel_biomass,
  )
  println("Ecos: $(length(params.ECO_LIST)), Species: $(length(params.SPECIES_LIST))")

  println("Expanding IC to pixels")
  @time splots = SpatialData.expand_landis_pixels(
    communities_raster, eco_raster, ic_df, eco_ecocode_df, params)
  println("Sites: $(length(unique(splots.mapcode))), Cohort rows: $(nrow(splots))")

  eco_species_ids = Vector{Vector{Int}}(params.ECO_SPECIES_IDS)

  println("Initializing SoA")
  @time ref_soa = make_sites_from_communities(splots, eco_species_ids, rng)

  eco_params = BiomassSuccessionPlugin.generate_eco_params(params)

  println("Running simulation: $timehorizon_years years, output every $output_every_years")
  let ctx = (eco_params=eco_params,),
    bufs = [BiomassSuccessionPlugin._new_buf() for _ in 1:Threads.maxthreadid()],
    chunks = zeros(Int, Threads.maxthreadid())

    Spatial.run_spatial!(ref_soa, BiomassSuccessionPlugin.BiomassSuccession, ctx, output_dir,
      (soa, year) -> BiomassSuccessionPlugin.emit_year!(bufs, chunks, soa, year, output_dir);
      timehorizon=timehorizon_years, output_every=output_every_years)
  end

  println("Generating output rasters")
  BiomassSuccessionPlugin.generate_rasters_from_output(; output_dir=output_dir, ref_raster_path=ecoregion_tif)

  println("Coalescing Arrow files to DuckDB")
  BiomassSuccessionPlugin.coalesce_to_duckdb(; output_dir=output_dir, db_path=joinpath(output_dir, "cohorts.duckdb"))
end

function export_landis_main(; output_dir::String="/workspace/best_params_landis")
  jld2_files = filter(f -> startswith(f, "best_params@") && endswith(f, ".jld2"),
    readdir("./outputs"))
  isempty(jld2_files) && error("No best_params JLD2 files found in ./outputs/")
  latest = last(sort(jld2_files))
  jld2_path = joinpath("./outputs", latest)
  println("Exporting from $jld2_path")
  params = JLD2.load_object(jld2_path)
  BiomassSuccessionPlugin.export_landis_params(params; output_dir=output_dir)
end

# ---------------------------------------------------------------------------
# Export LANDIS-II scenario files (same loading path as spatial simulation)
# ---------------------------------------------------------------------------

# Copy a biomass-climate config file and any data files it references (ClimateFile /
# SpinUpClimateFile lines) into output_dir. Returns the basename for use in
# biomass_succession.txt. Does not read CSV content — just copies bytes.
function _copy_climate_files(src_path::String, output_dir::String)::String
  src_abs = abspath(src_path)
  src_dir = dirname(src_abs)
  dst_name = basename(src_abs)
  cp(src_abs, joinpath(output_dir, dst_name); force=true)
  for line in eachline(src_abs)
    m = match(r"^\s*(?:ClimateFile|SpinUpClimateFile)\s+(\S+)", line)
    isnothing(m) && continue
    ref = m.captures[1]
    ref_src = isabspath(ref) ? ref : joinpath(src_dir, ref)
    isfile(ref_src) || continue
    ref_dst = joinpath(output_dir, basename(ref))
    isfile(ref_dst) || cp(ref_src, ref_dst)
    println("  $(basename(ref))")
  end
  return dst_name
end

function export_landis_scenario(;
  data_dir::String,
  output_dir::String,
  eco_raster::String,
  eco_ecocode_mapping::String,
  biomass_params_path::String,
  climate_config_file::String,
  treemap_raster::Union{String,Nothing}=nothing,
  communities_csv::Union{String,Nothing}=nothing,
  communities_db::Union{String,Nothing}=nothing,
  treemap_version::Int=2022,
  treemap_db_path::String="../data_eco_cohorts.duckdb",
  duration_years::Int=40,
  cell_length_m::Int=30,
  timestep::Int=5,
  downsample::Int=1,
  # Ignored parameters (for signature compatibility with simulate_spatial_treemap)
  rng_seed::Int=1337,
  timehorizon_years::Int=50,
  output_every_years::Int=5,
  simple_match::Bool=true,
  stratify_eco_mixed::Bool=false,
)
  mkpath(output_dir)

  println("Loading params: $biomass_params_path")
  params = JLD2.load_object(joinpath(data_dir, biomass_params_path))

  println("Loading eco raster")
  eco_raster_path = joinpath(data_dir, eco_raster)
  eco_raster_data = SpatialData.load_eco_raster(eco_raster_path)

  eco_mapping_path = joinpath(data_dir, eco_ecocode_mapping)
  eco_mapping_df = CSV.read(eco_mapping_path, DataFrame)

  local communities_df, combo_to_mapcode, mod_params, eco_species_ids, cn_raster
  local mapcode_raster, coarse_eco_raster
  # When stratifying, the exported ecoregion files encode the stratum in the ecocode
  # (base*10 + mixed?1:2); otherwise these stay at the base raster/mapping (copy as-is).
  strat_eco_raster = nothing
  eco_export_mapping = eco_mapping_df

  if !isnothing(treemap_raster)
    println("Loading treemap raster: $treemap_raster")
    @time cn_raster, _ = SpatialData.load_treemap_raster(
      joinpath(data_dir, treemap_raster); treemap_version=treemap_version)
    @assert size(cn_raster) == size(eco_raster_data) "Raster size mismatch: treemap $(size(cn_raster)) ≠ eco $(size(eco_raster_data))"

    # Optional spatial downsample: group n×n fine pixels into one coarse cell. The eco raster
    # used for species matching is replaced by its block-majority so a coarse cell has a single
    # ecoregion. With stratify_eco_mixed, the per-plot stratum is baked into the ecocode FIRST
    # (base*10 + mixed?1:2) and then majority-voted together with the base eco — so the cell's
    # stratum is a majority vote too, and the loader matches every cohort in the block against
    # that single stratified ecoregion (consistent base+stratum majority vote).
    eco_match_raster = eco_raster_data
    eco_loader_override = nothing      # stratified mapping df handed to the loader
    loader_stratify = stratify_eco_mixed
    mapcode_raster = nothing
    coarse_eco_raster = nothing
    if downsample > 1
      println("Downsampling rasters $(downsample)x ($(cell_length_m)m → $(cell_length_m * downsample)m), eco by majority vote")
      if stratify_eco_mixed
        cn_mixed = SpatialData.load_plot_mixed(cn_raster, treemap_db_path)
        strat_fine = SpatialData.stratify_eco_raster(eco_raster_data, cn_raster, cn_mixed)
        eco_export_mapping = SpatialData.stratified_eco_mapping(eco_mapping_df)
        eco_match_raster, coarse_eco_raster = SpatialData.block_majority_eco(strat_fine, downsample, Set(Int.(eco_export_mapping.ecocode)))
        eco_loader_override = eco_export_mapping  # ecocodes already stratified → loader runs unstratified
        loader_stratify = false
      else
        eco_match_raster, coarse_eco_raster = SpatialData.block_majority_eco(eco_raster_data, downsample, Set(Int.(eco_mapping_df.ecocode)))
      end
    end

    println("Extracting cohorts from DuckDB (treemap path)")
    @time splots, eco_list, eff_eco_list, species_list = (simple_match ? SpatialData.load_treemap_cohorts_simple : SpatialData.load_treemap_cohorts)(
      cn_raster, eco_match_raster, treemap_db_path, eco_mapping_path, params;
      stratify_eco_mixed=loader_stratify, eco_ecocode_override=eco_loader_override)
    println("Plots: $(length(unique(splots.plt_cn))), Ecos: $(length(eco_list)), Species: $(length(species_list))")
    println(species_list)

    println("Remapping params to data eco/species")
    @time mod_params, mapped_splots, eco_species_ids = SpatialData.map_params_to_data_treemap(
      params, eco_list, eff_eco_list, species_list, splots)

    if downsample > 1
      println("Aggregating cohorts into $(downsample)x coarse cells (combine species×age, biomass / $(downsample^2))")
      @time communities_df, mapcode_raster = SpatialData.aggregate_coarse_communities(
        mapped_splots, cn_raster, eco_match_raster, downsample)
      combo_to_mapcode = nothing
    else
      println("Deduplicating cohorts by (plt_cn, ecocode)")
      @time communities_df, combo_to_mapcode = SpatialData.deduplicate_for_export(
        mapped_splots, cn_raster, eco_raster_data)
    end

    # Non-downsample stratification: build a full-res stratified ecoregion raster + mapping.
    # (The downsample+stratify case already produced coarse_eco_raster/eco_export_mapping above.)
    if stratify_eco_mixed && downsample == 1
      println("Building stratified ecoregion raster + mapping (base*10 + mixed?1:2)")
      cn_mixed = SpatialData.load_plot_mixed(cn_raster, treemap_db_path)
      strat_eco_raster = SpatialData.stratify_eco_raster(eco_raster_data, cn_raster, cn_mixed)
      eco_export_mapping = SpatialData.stratified_eco_mapping(eco_mapping_df)
    end

  elseif !isnothing(communities_csv) || !isnothing(communities_db)
    if !isnothing(communities_csv)
      println("Loading initial communities from CSV: $communities_csv")
      ic_df = SpatialData.load_csv_communities(joinpath(data_dir, communities_csv))
    else
      println("Loading initial communities from DuckDB: $communities_db")
      ic_df = SpatialData.load_duckdb_communities(communities_db)
    end
    # prepare_general_splots maps to internal IDs; for export we use ic_df directly.
    # We still call it to get mod_params and eco_species_ids.
    _, mod_params, eco_species_ids = SpatialData.prepare_general_splots(
      ic_df, eco_raster_data, eco_mapping_df, params)
    cn_raster = nothing
    combo_to_mapcode = nothing
    mapcode_raster = nothing
    coarse_eco_raster = nothing

    # ic_df already has semantic columns (mapcode, species, age/CohortAge, biomass/CohortBiomass).
    # Normalize column names and filter to species present in mod_params.
    sp_set = Set(string.(mod_params.SPECIES_LIST))
    sp_col = hasproperty(ic_df, :SpeciesName) ? ic_df.SpeciesName : ic_df.species
    age_col = hasproperty(ic_df, :CohortAge) ? ic_df.CohortAge : ic_df.age
    bio_col = hasproperty(ic_df, :CohortBiomass) ? ic_df.CohortBiomass : ic_df.biomass
    mc_col = hasproperty(ic_df, :MapCode) ? ic_df.MapCode : ic_df.mapcode
    valid = [s in sp_set for s in sp_col]
    communities_df = DataFrame(
      mapcode=Int.(mc_col[valid]),
      species=String.(sp_col[valid]),
      age_calc=FloatType.(age_col[valid]),
      agb_sum=FloatType.(bio_col[valid]),
    )

  else
    error("Specify one of: treemap_raster, communities_csv, communities_db")
  end

  n_sites = length(unique(communities_df.mapcode))
  println("Prepared $n_sites unique mapcodes for export")

  # ---------------------------------------------------------------------------
  println("\nExporting LANDIS-II scenario files to $output_dir ...")

  # 1. scenario.txt
  eff_cell_length = downsample > 1 ? cell_length_m * downsample : cell_length_m
  println("Exporting scenario.txt")
  BiomassSuccessionPlugin.export_scenario_file(;
    output_path=joinpath(output_dir, "scenario.txt"),
    duration_years=duration_years,
    cell_length_m=eff_cell_length,
    rng_seed=rng_seed,
  )

  # 2. ecoregion.txt + ecoregion.tif (stratified, downsampled, or copied as-is)
  println("Exporting ecoregion.txt")
  BiomassSuccessionPlugin.export_ecoregions_txt(
    mod_params, eco_export_mapping;
    output_path=joinpath(output_dir, "ecoregion.txt"))
  if !isnothing(strat_eco_raster)
    println("Writing stratified ecoregion.tif")
    BiomassSuccessionPlugin.export_coarse_raster(
      strat_eco_raster, eco_raster_path, 1;
      output_path=joinpath(output_dir, "ecoregion.tif"),
      dtype=Int16, nodata=0)
  elseif downsample > 1 && !isnothing(coarse_eco_raster)
    println("Writing downsampled ecoregion.tif")
    BiomassSuccessionPlugin.export_coarse_raster(
      coarse_eco_raster, eco_raster_path, downsample;
      output_path=joinpath(output_dir, "ecoregion.tif"),
      dtype=Int16, nodata=0)
  else
    println("Copying ecoregion.tif")
    cp(realpath(eco_raster_path), joinpath(output_dir, "ecoregion.tif"); force=true)
    println("  ecoregion.tif")
  end

  # 3. Climate config file + any data files it references
  climate_ref = nothing  # filename to embed in biomass_succession.txt
  if !isnothing(climate_config_file)
    println("Copying climate files")
    climate_ref = _copy_climate_files(climate_config_file, output_dir)
    println("  $climate_ref")
  end

  # 4. CoreSpeciesData.txt, SpeciesData.csv, SppEcoregionData.csv, biomass_succession.txt
  BiomassSuccessionPlugin.export_landis_params(mod_params;
    output_dir=output_dir,
    climate_config_file=climate_ref
  )

  # 4. initial_communities.csv
  println("Exporting initial_communities.csv")
  BiomassSuccessionPlugin.export_initial_communities_csv(
    communities_df;
    output_path=joinpath(output_dir, "initial_communities.csv"))

  # 5. initial_communities.tif (treemap path only; mapcode raster)
  if downsample > 1 && !isnothing(mapcode_raster)
    println("Exporting downsampled initial_communities.tif")
    ref_raster = joinpath(data_dir, treemap_raster)
    BiomassSuccessionPlugin.export_coarse_raster(
      mapcode_raster, ref_raster, downsample;
      output_path=joinpath(output_dir, "initial_communities.tif"),
      dtype=Int32, nodata=0)
  elseif !isnothing(combo_to_mapcode)
    println("Exporting initial_communities.tif")
    ref_raster = joinpath(data_dir, treemap_raster)
    BiomassSuccessionPlugin.export_initial_communities_tif(
      combo_to_mapcode, cn_raster, eco_raster_data, ref_raster;
      output_path=joinpath(output_dir, "initial_communities.tif"))
  end

  # 6. eco_ecocode_mapping.csv (lookup table, not read by LANDIS directly)
  BiomassSuccessionPlugin.export_eco_ecocode_mapping(
    mod_params, eco_export_mapping;
    output_path=joinpath(output_dir, "eco_ecocode_mapping.csv"))
  println("  eco_ecocode_mapping.csv")

  println("\nLANDIS-II scenario exported to: $output_dir")
  println("Files:")
  for f in sort(readdir(output_dir))
    println("  $f")
  end
end

function export_landis_scenario_main()
  export_landis_scenario(
    data_dir="../poster/fl5_strat4_prob_spinup/",
    output_dir="../poster/fl5_strat4_prob_spinup/landis_90_2",
    eco_raster="ecoregion.tif",
    eco_ecocode_mapping="eco_ecocode_l3_mapping.csv",
    biomass_params_path="params.jld2",
    treemap_raster="FL5_22.tif",
    treemap_version=2022,
    treemap_db_path="../FIASQLITE2PGSQL/FIADB.duckdb",
    climate_config_file="../poster/fl5/biomass-climate.txt",
    downsample=3,
    stratify_eco_mixed=true,
  )
end

function landis_poster_main()
  prefix = joinpath("../poster/fl5_strat4_prob_spinup/landis_90_2")
  simulate_spatial_landis(
    output_dir="../poster/fl5_strat4_prob_spinup/pan_outputs_90_2",
    initial_communities_tif=joinpath(prefix, "initial_communities.tif"),
    ecoregion_tif=joinpath(prefix, "ecoregion.tif"),
    initial_communities_csv=joinpath(prefix, "initial_communities.csv"),
    core_species_data=joinpath(prefix, "CoreSpeciesData.txt"),
    spp_ecoregion_data=joinpath(prefix, "SppEcoregionData.csv"),
    species_data=joinpath(prefix, "SpeciesData.csv"),
    eco_ecocode_mapping=joinpath(prefix, "eco_ecocode_mapping.csv"),
    spp_eco_year=0,
    timehorizon_years=50,
    output_every_years=5,
  )
end

# ---------------------------------------------------------------------------
# Figure: forest species composition by ecoregion (stacked bars) from the
# coalesced cohorts.duckdb. One subplot per ecoregion; a stacked bar per year
# (default 0/25/50), stacked by species. Species labelled with FIA common names.
#
# The coalesced cohorts table stores integer (eco_id, species_id) where species_id
# is ECO-LOCAL; we rebuild the sim's params from the scenario dir to resolve
# species_id → ECO_SPECIES_IDS[eco_id][species_id] → SPECIES_LIST → symbol, then
# map symbols to common names (REF_SPECIES / REF_SPECIES_GROUP).
# ---------------------------------------------------------------------------
function plot_species_composition(;
  cohorts_db::String,
  scenario_dir::String,          # exported LANDIS files (CoreSpeciesData.txt, SpeciesData.csv, SppEcoregionData.csv)
  fia_db::String,                # DB holding REF_SPECIES (+ REF_SPECIES_GROUP)
  output_path::String,
  years::Vector{Int}=[0, 25, 50],
  spp_eco_year::Int=0,
  relative::Bool=false,          # normalize each bar to fractions of total biomass
  species_order::Vector{String}=String[],  # canonical symbol order for stable colors across plots
)
  # 1. Reconstruct the sim's id scheme.
  core_sp_df = SpatialData.load_landis_core_species(joinpath(scenario_dir, "CoreSpeciesData.txt"))
  species_df = CSV.read(joinpath(scenario_dir, "SpeciesData.csv"), DataFrame)
  spp_eco_df = SpatialData.load_landis_spp_ecoregion(joinpath(scenario_dir, "SppEcoregionData.csv"); year=spp_eco_year)
  params = SpatialData.make_landis_params(core_sp_df, species_df, spp_eco_df)
  SPECIES_LIST = params.SPECIES_LIST
  ECO_LIST = params.ECO_LIST
  eco_species_ids = params.ECO_SPECIES_IDS

  # 2. Aggregate biomass per (eco_id, species_id, year).
  db = DuckDB.DB(cohorts_db)
  con = DuckDB.connect(db)
  agg = DuckDB.execute(
    con,
    """
  SELECT year, eco_id, species_id, SUM(biomass) AS biomass
  FROM cohorts WHERE year IN ($(join(years, ",")))
  GROUP BY year, eco_id, species_id
  ORDER BY year, eco_id, species_id
"""
  ) |> DataFrame
  DuckDB.close(db)
  isempty(agg) && error("No cohorts for years $(years) in $(cohorts_db)")
  present_years = sort(unique(Int.(agg.year)))
  miss = setdiff(years, present_years)
  isempty(miss) || @warn "requested years absent in cohorts.duckdb (skipped)" absent = miss

  # 3. Common-name labels + softwood/hardwood class (REF_SPECIES_GROUP.CLASS).
  fdb = DuckDB.DB(fia_db)
  fcon = DuckDB.connect(fdb)
  ref_sp = DuckDB.execute(fcon, "SELECT r.SPECIES_SYMBOL AS sym, r.COMMON_NAME AS cn, r.SFTWD_HRDWD AS sh FROM REF_SPECIES r ORDER BY r.SPCD") |> DataFrame
  grp = DataFrame(spgrpcd=Int[], name=String[], class=String[])
  try
    grp = DuckDB.execute(fcon, "SELECT g.SPGRPCD AS spgrpcd, g.NAME AS name, g.CLASS AS class FROM REF_SPECIES_GROUP g ORDER BY g.SPGRPCD") |> DataFrame
  catch e
    @warn "REF_SPECIES_GROUP unavailable; _GRP_ labels stay raw and their soft/hard defaults to hardwood" exception = e
  end
  DuckDB.close(fdb)
  common = Dict(uppercase(strip(String(r.sym))) => String(r.cn) for r in eachrow(ref_sp))
  sym_sh = Dict(uppercase(strip(String(r.sym))) => uppercase(strip(String(r.sh))) for r in eachrow(ref_sp) if !ismissing(r.sh))
  grpname = Dict(Int(r.spgrpcd) => String(r.name) for r in eachrow(grp))
  grpclass = Dict(Int(r.spgrpcd) => String(r.class) for r in eachrow(grp))

  # softwood (blue–green) vs hardwood (yellow–red):
  #   real species → REF_SPECIES.SFTWD_HRDWD ; _GRP_<spgrpcd> → REF_SPECIES_GROUP.CLASS ; _S/_H explicit.
  function is_soft(sym::AbstractString)
    s = uppercase(strip(sym))
    s == "_S" && return true
    s == "_H" && return false
    if startswith(s, "_GRP_")
      g = tryparse(Int, s[6:end])
      return startswith(uppercase(isnothing(g) ? "" : get(grpclass, g, "")), "S")
    end
    return get(sym_sh, s, "H") == "S"   # default (unknown) → hardwood
  end

  function label_for(sym::AbstractString)
    if startswith(sym, "_GRP_")
      g = tryparse(Int, sym[6:end])
      return (!isnothing(g) && haskey(grpname, g)) ? "Other $(grpname[g])" : "Other group $(sym[6:end])"
    elseif sym == "_H"
      return "Other hardwood"
    elseif sym == "_S"
      return "Other softwood"
    else
      cn = get(common, uppercase(strip(sym)), nothing)
      return isnothing(cn) ? sym : "$(cn) ($(sym))"
    end
  end

  # 4. id (eco-local) → global symbol → label; eco_id → name. Drop any out-of-range ids.
  neco = length(ECO_LIST)
  keep = [1 <= Int(r.eco_id) <= neco && 1 <= Int(r.species_id) <= length(eco_species_ids[Int(r.eco_id)])
          for r in eachrow(agg)]
  agg = agg[keep, :]
  agg.symbol = [string(SPECIES_LIST[Int(eco_species_ids[Int(r.eco_id)][Int(r.species_id)])]) for r in eachrow(agg)]
  agg.label = label_for.(agg.symbol)
  agg.eco = [ECO_LIST[Int(r.eco_id)] for r in eachrow(agg)]
  agg = combine(groupby(agg, [:eco, :year, :symbol, :label]), :biomass => sum => :biomass)

  # 5. Color & stack by each species' CANONICAL position so the same species gets the same
  #    color across different plots. Default order = SPECIES_LIST (identical across runs of the
  #    same scenario); pass `species_order` (a shared symbol list) for cross-scenario consistency.
  #    Any present symbol not in the order is appended deterministically (sorted) at the end.
  order = String[string(s) for s in (isempty(species_order) ? SPECIES_LIST : species_order)]
  seen = Set(order)
  for s in sort(unique(agg.symbol))
    s in seen || (push!(order, s); push!(seen, s))
  end
  # Split canonically into softwoods (blue→green) and hardwoods (yellow→red); shade each
  # species by its position WITHIN its class (over the full order, so shades are plot-stable).
  soft_all = [s for s in order if is_soft(s)]
  hard_all = [s for s in order if !is_soft(s)]
  soft_grad = CairoMakie.cgrad([:navy, :dodgerblue, :darkturquoise, :seagreen, :limegreen])
  hard_grad = CairoMakie.cgrad([:gold, :orange, :orangered, :red, :darkred])
  shade(i, n) = n <= 1 ? 0.5 : (i - 1) / (n - 1)
  color_of = Dict{String,CairoMakie.RGBAf}()
  for (i, s) in enumerate(soft_all)
    color_of[s] = CairoMakie.RGBAf(soft_grad[shade(i, length(soft_all))])
  end
  for (i, s) in enumerate(hard_all)
    color_of[s] = CairoMakie.RGBAf(hard_grad[shade(i, length(hard_all))])
  end
  # stack order: softwoods (bottom) then hardwoods, each in canonical order.
  stack_order = vcat(soft_all, hard_all)
  sym_idx = Dict(s => i for (i, s) in enumerate(stack_order))

  # 6. figure: one axis per ecoregion + shared legend column.
  ecos = sort(unique(agg.eco))
  yx = Dict(y => i for (i, y) in enumerate(present_years))
  ncols = max(1, ceil(Int, sqrt(length(ecos))))   # square-ish grid (4 ecoregions → 2×2)
  nrows = cld(length(ecos), ncols)
  f = CairoMakie.Figure(size=(360 * ncols + 340, 80 + 320 * nrows))
  CairoMakie.Label(f[0, 1:(ncols+1)],
    "Forest species composition by ecoregion" * (relative ? " (relative)" : " (biomass g/m²)");
    fontsize=16, font=:bold, tellwidth=false)

  for (ei, eco) in enumerate(ecos)
    r = cld(ei, ncols)
    c = mod1(ei, ncols)
    ax = CairoMakie.Axis(f[r, c]; title="Ecoregion: $(eco)", xlabel="year",
      ylabel=relative ? "biomass fraction" : "biomass (g/m²)",
      xticks=(collect(1:length(present_years)), string.(present_years)))
    sub = agg[agg.eco.==eco, :]
    ymap = relative ? Dict(row.year => row.ytot for row in eachrow(combine(groupby(sub, :year), :biomass => sum => :ytot))) : Dict()
    xs = Int[]
    ys = Float64[]
    stk = Int[]
    cs = CairoMakie.RGBAf[]
    for row in eachrow(sub)
      haskey(yx, Int(row.year)) || continue
      h = relative ? Float64(row.biomass) / max(ymap[row.year], eps()) : Float64(row.biomass)
      push!(xs, yx[Int(row.year)])
      push!(ys, h)
      push!(stk, sym_idx[row.symbol])
      push!(cs, color_of[row.symbol])
    end
    isempty(xs) || CairoMakie.barplot!(ax, xs, ys; stack=stk, color=cs)
  end

  # Legend: present species in stack order (softwoods then hardwoods), each with its fixed color.
  present = sort(unique(agg.symbol); by=s -> sym_idx[s])
  label_of = Dict(r.symbol => r.label for r in eachrow(agg))
  elems = [CairoMakie.PolyElement(color=color_of[s]) for s in present]
  CairoMakie.Legend(f[1:nrows, ncols+1], elems, [label_of[s] for s in present],
    "Species  (softwood: blue–green, hardwood: yellow–red)"; framevisible=false, labelsize=10)

  mkpath(dirname(output_path))
  CairoMakie.save(output_path, f)
  println("Wrote $output_path  ($(length(ecos)) ecoregions, $(length(present)) species, years $(present_years))")
  return output_path
end

function species_composition_main()
  prefix = joinpath("../poster/fl5_strat4/landis_90_2")
  plot_species_composition(
    cohorts_db=joinpath("../poster/fl5_strat4_prob_spinup/pan_outputs_90_2", "cohorts.duckdb"),
    scenario_dir=prefix,
    fia_db="../FIASQLITE2PGSQL/FIADB.duckdb",
    output_path=joinpath("../poster/fl5_strat4_prob_spinup/pan_outputs_90_2", "species_composition.png"),
    years=[0, 25, 50],
  )
end

function landis_main()
  prefix = joinpath("/home/bahaa/Downloads/", "Extension-Biomass-Succession-master/testings/CoreV8.0-BiomassSuccession7.0/")
  simulate_spatial_landis(
    output_dir="./outputs/landis_test",
    initial_communities_tif=joinpath(prefix, "initial-communities.tif"),
    ecoregion_tif=joinpath(prefix, "ecoregions.tif"),
    initial_communities_csv=joinpath(prefix, "biomass-succession_InitialCommunities.csv"),
    core_species_data=joinpath(prefix, "CoreSpeciesData.txt"),
    spp_ecoregion_data=joinpath(prefix, "SppEcoregionData.csv"),
    species_data=joinpath(prefix, "SpeciesData.csv"),
    eco_ecocode_mapping=joinpath(prefix, "eco_ecocode_mapping.csv"),
    spp_eco_year=0,
    timehorizon_years=50,
    output_every_years=1,
  )
end

include("drivers/search_loop.jl")
include("drivers/gdigel.jl")

include("climate_scenario.jl")
include("scenario.jl")
include("scenario_site.jl")
include("scenario_volume.jl")

end
