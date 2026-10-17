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
  #filter_ecos=String["8.3.5.65o", "8.5.3.75e", "8.5.3.75f", "8.5.3.75g"]
  filter_ecos = ["8.5.3"]
  filter_plots = NTuple{4,Int}[]#(13, 1, 49, 26)]  # e.g. [(statecd,unitcd,countycd,plot), ...]
  filter_species = String[] #String["PIEL"]        # e.g. ["ACRU", "QURU"]
  t1_bins_idx = [20, 60, 120]
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
