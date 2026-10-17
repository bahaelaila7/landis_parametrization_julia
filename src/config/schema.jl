# One validated configuration reader for every Pan entry point (run_from_yaml, the analysis tools, the plot paths).
#
#   cfg = Pan.load_config("configs/region4/gafl_A.yml"; overrides=Dict("n_reps" => 1))
#   cfg["tier"]        # resolved value: coerced to the schema type, default filled, ${VAR}/~ expanded
#
# The SCHEMA below is the only place where a key's type and default live. Unknown keys are an error (with the closest
# known keys), retired keys are an error naming the replacement. The reader is setup-time code: it is dynamic on purpose
# and nothing in the simulation hot path touches it.

# ---------------------------------------------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------------------------------------------

"""
    KeySpec(name, type, default, doc; nullable=false, choices=(), null_str=true)

One config key. `type` is one of `:bool :int :float :string :path :symbol :intlist :strlist :floatlist :plots :dict :dict_strlist
:dict_tiermerge :dict_float :intorstr :dual_mode :disturb_path :gdigel`. `:string` and `:path` are both \${VAR}/\$VAR/~ expanded
(every top-level string value always was); `:path` marks file-system paths. `nullable` keys accept `nothing`, and (when `null_str`)
the strings "null" / "none" (any case). `choices` restricts string/symbol values.
"""
struct KeySpec
  name::String
  type::Symbol
  default::Any
  doc::String
  nullable::Bool
  choices::Tuple
  null_str::Bool
end
KeySpec(name, type, default, doc; nullable=false, choices=(), null_str=true) =
  KeySpec(String(name), type, default, String(doc), nullable, Tuple(choices), null_str)

const _CHOICES_W_SCOPE = ("all", "survivors")
const _CHOICES_DEATH_TERM = ("none", "likelihood")

# Top-level keys, in documentation order. Defaults are the values run_from_yaml used before this reader existed.
const SCHEMA_LIST = KeySpec[
  # ---- data source -------------------------------------------------------------------------------------------------
  KeySpec("cohorts_db_path", :path, "../data_eco_cohorts.duckdb", "cohort DuckDB; env FIADB overrides it"),
  KeySpec("tablename", :string, "data_eco_cohorts", "cohort table in the DB"),
  KeySpec("eco_field", :string, "epa_l3", "column that defines the ecoregion cell"),
  KeySpec("filter_eco_field", :string, "epa_l3", "column that filter_ecos applies to"),
  KeySpec("filter_ecos", :strlist, String[], "keep only these ecoregion codes (prefix match; empty = all)"),
  KeySpec("filter_plots", :plots, NTuple{4,Int}[], "keep only these plots, [statecd, unitcd, countycd, plot] (empty = all)"),
  KeySpec("exclude_plots", :plots, NTuple{4,Int}[], "drop these plots, [statecd, unitcd, countycd, plot]"),
  KeySpec("exclude_plots_csv", :path, nothing, "CSV (statecd,unitcd,countycd,plot) of plots to drop, merged with exclude_plots"; nullable=true),
  KeySpec("filter_species", :strlist, String[], "keep only these species codes (empty = all)"),
  KeySpec("filter_extent", :string, nothing, "state-extent filter (e.g. FL); null = none"; nullable=true),
  KeySpec("filter_states", :intlist, Int[], "region = state FIPS list intersected with filter_ecos (empty = no state filter)"),
  KeySpec("skip_disturbances", :bool, true, "drop plots with recorded disturbances at load"),
  KeySpec("spinup", :bool, false, "start sites from a 2-cohort spinup instead of the observed first-visit cohorts"),
  KeySpec("by_subplot", :bool, false, "treat each subplot as a plot"),
  KeySpec("drop_phantom", :bool, false, "drop intro_type=phantom_backfill rows at load"),
  KeySpec("data_on_missing", :string, "warn", "data preparation: a default substituted for missing data (species without shade tier / tree stats / softwood-hardwood class, plots without site class, NULL land_use, filter values matching nothing) is logged (warn) or stops the run (error)"; choices=("warn", "error")),
  KeySpec("min_trees", :int, 100, "minimum trees per plot"),
  KeySpec("min_agb_frac", :float, 0.05, "minimum share of plot AGB a species needs to count"),
  # ---- split / strata / tiering -------------------------------------------------------------------------------------
  KeySpec("val_frac", :float, 0.0, "validation fraction of the single split"),
  KeySpec("test_frac", :float, 0.0, "held-out test fraction (>0 gives a 3-way split; exclusive with n_folds>1)"),
  KeySpec("split_seed", :int, 42, "seed of the split shuffle (shared by all folds)"),
  KeySpec("n_folds", :int, 1, "k-fold CV folds (each fold writes to output_dir/fold_<i>)"),
  KeySpec("fold_index", :int, 1, "which fold this run trains on"),
  KeySpec("stratify_eco_mixed", :bool, false, "stratify the split on mixed-ecoregion plots"),
  KeySpec("single_ecoregion", :bool, false, "collapse all ecoregions into one cell"),
  KeySpec("stratify_landuse", :bool, false, "add the natural/artificial land-use stratum"),
  KeySpec("site_class_strata", :bool, false, "replace land_use by a per-plot site-productivity tier (COND.SITECLCD)"),
  KeySpec("siteclass_hi_max", :int, 4, "site class <= this is 'hi', above is 'lo'"),
  KeySpec("siteclass_scheme", :string, "2way", "site-tier labels: 2way (hi/lo) | 4cell"),
  KeySpec("siteclass_l3s", :strlist, String[], "site-class cells only in these L3 ecoregions (empty = all)"),
  KeySpec("shade_tier_csv", :path, nothing, "species->shade_class CSV; splits the grouping key by LST/HST"; nullable=true),
  KeySpec("max_exact", :int, 12, "species tiering: max own-parameter species"),
  KeySpec("max_group", :int, 4, "species tiering: max pooled groups"),
  KeySpec("train_cap", :intorstr, nothing, "species-balanced training cap: integer, or \"auto\"; surplus plots go to val/test"; nullable=true),
  KeySpec("train_cap_k", :float, 1.5, "\"auto\" cap = max(train_cap_min, ceil(k * median stratum size))"),
  KeySpec("train_cap_min", :int, 20, "lower bound of the \"auto\" cap"),
  KeySpec("split_only", :bool, false, "write split_plots.csv / species_map.csv (+ pool_map.csv) and stop"),
  # ---- run control ---------------------------------------------------------------------------------------------------
  KeySpec("search_mode", :string, "lbsa", "lbsa | sa | molbsa | cmaes | mocmaes | igelmo | cmame | ccigel | nsga2 | gdigel | sobol | plot_only | plot_only_sobol"),
  KeySpec("tier", :int, 1, "Sim A scoring tier (1-5)"),
  KeySpec("trials", :int, 1000000, "evaluation budget"),
  KeySpec("seed", :int, 404, "master RNG seed"),
  KeySpec("output_dir", :path, "./outputs", "output directory (fold_<i> is appended when n_folds>1)"),
  KeySpec("resume_from", :path, nothing, "search-state checkpoint to resume"; nullable=true),
  KeySpec("start_from", :path, nothing, "parameter set / checkpoint to start from"; nullable=true),
  KeySpec("force_restart_from_random", :bool, false, "ignore the resumed state's parameters and restart from random"),
  KeySpec("n_reps", :int, 5, "stochastic replicates per evaluation"),
  KeySpec("no_establishment", :bool, false, "disable establishment (no new cohorts)"),
  KeySpec("diagnose", :bool, false, "print diagnostics during the run"),
  KeySpec("cycle_years", :float, 8.0, "FIA remeasurement interval in years (cycle map; hbad PROB_ESTAB conversion)"),
  KeySpec("n_output_plots", :int, 0, "number of per-plot trajectory plots to write"),
  KeySpec("n_sobol_params_to_plot", :int, 5, "plot_only_sobol: number of Sobol parameter sets to plot"),
  KeySpec("params_path", :path, "", "plot_only: parameter checkpoint to plot"),
  KeySpec("sobol_candidates_db", :path, nothing, "Sobol candidate DB (sobol output, seeds, plot_only_sobol)"; nullable=true),
  KeySpec("fia_cycle", :bool, false, "tier-4/5 cycles from the FIA `cycle` column instead of the computed split"),
  KeySpec("fia_cycle_merge", :int, 3, "merge this many consecutive FIA cycles into one group"),
  KeySpec("bins_idx", :intlist, collect(vcat(10:10:40, 60:20:120)), "age-bin upper edges (years) of the Wasserstein CDFs"),
  KeySpec("smoothing_window_size", :int, 0, "age-smoothing half window (0 = none)"),
  KeySpec("smoothing_variance", :float, 1.0, "variance of the age-smoothing kernel"),
  # ---- scoring rule / injection --------------------------------------------------------------------------------------
  KeySpec("override_injection", :bool, false, "inject observed cohorts at each measurement year"),
  KeySpec("override_injection_sync", :bool, false, "sync injection: replace the simulated cohorts by the observed ones after scoring"),
  KeySpec("override_injection_replace", :bool, false, "replace injection (instead of appending)"),
  KeySpec("override_injection_noise", :float, 0.0, "relative noise added to injected observed biomass"),
  KeySpec("override_injection_disturbance", :symbol, :off, "disturbance handling of injected cohorts: off | scale | exclude_noscale | exclude_overwrite ..."),
  KeySpec("loss_before_injection", :bool, nothing, "score before injecting (shock -> score -> sync); default true for sync injection, otherwise the legacy order"; nullable=true),
  KeySpec("exclude_recruits_at_entry", :bool, true, "exclude recruits (cohorts absent at the previous visit) from the score"),
  KeySpec("w_scope", :symbol, :all, "W scope: all | survivors"; choices=_CHOICES_W_SCOPE),
  KeySpec("death_term", :symbol, :none, "death term: none | likelihood"; choices=_CHOICES_DEATH_TERM),
  KeySpec("disturbance_full_removals", :disturb_path, nothing, "parquet of fully-removed cohorts for the shock; null = repo default, false/\"\"/none/off = disabled"; nullable=true, null_str=false),
  KeySpec("dual_mode", :dual_mode, :off, "off (Sim A) | joint (A + B) | b (B only, seeded from A); true = joint"),
  KeySpec("tier_b", :int, 4, "Sim B's tier"),
  KeySpec("simB_spinup", :bool, true, "Sim B spins up (false = starts from observed cohorts)"),
  KeySpec("simB_per_plot_mean", :bool, false, "tier 4: AGB as mean biomass per plot by age"),
  KeySpec("simB_rep_mean", :bool, false, "tier 4: score the mean-over-replicates histogram"),
  KeySpec("simB_disturb_only", :bool, false, "Sim B purely free (no injection); disturbance scaling on all plots"),
  # ---- loss ----------------------------------------------------------------------------------------------------------
  KeySpec("loss_lambda", :float, 1.0, "weight of the AGB term"),
  KeySpec("loss_alpha", :float, 1.0, "weight of the Wasserstein term"),
  KeySpec("unbinned_w", :bool, false, "Sim A: exact per-year EMD (no binning / smoothing)"),
  KeySpec("agb_hinge", :bool, false, "AGB term: hinge-L1 with a tolerance band"),
  KeySpec("agb_hinge_threshold", :float, 10.0, "hinge band (absolute)"),
  KeySpec("agb_hinge_pct", :float, 0.0, ">0: band = clamp(obs*pct, pct_min, pct_max)"),
  KeySpec("agb_hinge_pct_min", :float, 0.0, "lower clamp of the percentage band"),
  KeySpec("agb_hinge_pct_max", :float, Inf, "upper clamp of the percentage band"),
  KeySpec("agb_hinge_l2", :bool, false, "legacy L2 hinge"),
  KeySpec("agb_hinge_p", :float, 2.0, "hinge exponent pen(h)=h^p"),
  KeySpec("agb_hinge_beta", :float, 1.0, "softplus sharpness of the hinge (0 = hard)"),
  KeySpec("agb_normalize", :bool, false, "AGB term = relative error"),
  KeySpec("agb_pivot", :float, 1.0, "piecewise loss: AGB pivot"),
  KeySpec("w_normalize", :bool, false, "divide W by the per-reference maximum"),
  KeySpec("cell_normalize", :bool, false, "per-(eco x land-use x species) scale and AGB-rank rescale"),
  KeySpec("w_scale_factor", :float, 1.0, "multiplier of the W divisor"),
  KeySpec("w_smooth", :bool, false, "softplus-smooth the per-term W1"),
  KeySpec("w_smooth_band", :float, 0.05, "softplus band"),
  KeySpec("w_smooth_conc", :float, 0.5, "softplus concentration"),
  KeySpec("w_smooth_beta", :float, 1.0, "fixed softplus beta (when w_smooth_auto_knee is off)"),
  KeySpec("w_smooth_auto_knee", :bool, false, "derive the softplus beta from band, conc and the W maximum"),
  KeySpec("w_p", :float, 1.0, "exponent of the W term"),
  KeySpec("w_hinge", :bool, false, "hinge on the W term"),
  KeySpec("w_hinge_pct", :float, 0.01, "W hinge percentage"),
  KeySpec("w_hinge_pct_min", :float, 2.0, "W hinge lower clamp"),
  KeySpec("w_hinge_pct_max", :float, 5.0, "W hinge upper clamp"),
  KeySpec("loss_piecewise", :bool, false, "piecewise-linear loss around the pivots"),
  KeySpec("w_pivot", :float, 1.0, "piecewise loss: W pivot"),
  KeySpec("w_count_balance", :bool, false, "count-balance reweight of the W1 term"),
  KeySpec("w_count_balance_mode", :string, "both", "which sims get reweighted: a | b | both"),
  KeySpec("w_count_beta", :float, 0.99, "effective-number-of-samples temper"),
  KeySpec("w_count_pct_floor", :float, 0.10, "floor on the per-bin count share"),
  KeySpec("rankw_mode", :string, "rank", "per-(species, stratum) weight: rank | cbal_pct"),
  KeySpec("rankw_beta", :float, 0.999, "beta of rankw_mode=cbal_pct"),
  KeySpec("rankw_pct_floor", :float, 0.10, "count-share floor of rankw_mode=cbal_pct"),
  KeySpec("init_perturb_frac", :float, 0.0, "relative perturbation of the initial cohort biomass"),
  KeySpec("init_perturb_cap", :float, 50.0, "max absolute biomass deviation per cohort"),
  # ---- parameter structure -------------------------------------------------------------------------------------------
  KeySpec("param_split_species", :dict_strlist, Dict{String,Vector{String}}(), "{param => [species]} fitted per ecoregion; the others are tied"),
  KeySpec("param_pooling", :dict, Dict{String,Any}(), "region runs: L3 -> L2 -> region pooling of EcoSpecies parameters (min_plots, floor_pooled, cell_merge, overrides, params)"),
  KeySpec("param_tier_merge", :dict_tiermerge, Dict{String,Any}(), "{param => {species => {cell => group}}} ties split site cells per species"),
  KeySpec("cmaes_blocks", :string, "eco", "CMA-ES covariance blocks: eco | l3"),
  # ---- succession plugins --------------------------------------------------------------------------------------------
  KeySpec("succession_plugin", :string, "biomass", "biomass | pnet | hbad"),
  KeySpec("pnet_climate_cache", :path, "", "pnet: PRISM monthly climate cache (JLD2)"),
  KeySpec("pnet_co2_csv", :path, "", "pnet: annual CO2 csv (year, co2_ppm)"),
  KeySpec("pnet_generic", :dict, Dict{String,Any}(), "pnet: PnETConst field overrides"),
  KeySpec("pnet_soil_texture", :dict, Dict{String,Any}(), "pnet: {L3 => Saxton-Rawls texture code}"),
  KeySpec("pnet_soil_csv", :path, "", "pnet: per-plot SOLUS soil csv (empty = texture path)"),
  KeySpec("hbad_priors_csv", :path, "", "hbad: priors csv (empty = repo default)"),
  KeySpec("hbad_coefs_csv", :path, "", "hbad: translation coefficients csv (empty = repo default)"),
  KeySpec("hbad_init", :string, "jitter", "hbad: start candidates default | jitter | prior"),
  KeySpec("hbad_seed_rain", :bool, false, "hbad: every species of the ecoregion is a seed source"),
  KeySpec("hbad_prob_estab_csv", :path, "", "hbad: per-cycle (species, L3) PROB_ESTAB table (empty = constant fallback)"),
  # ---- BiomassSuccession data tables ---------------------------------------------------------------------------------
  KeySpec("fix_longevity", :float, nothing, "fixed LONGEVITY for all species"; nullable=true),
  KeySpec("longevity_from_data", :bool, false, "per-species LONGEVITY from the species_longevity_ref table"),
  KeySpec("longevity_overrides", :dict_float, nothing, "{species => longevity} applied on top of the data table"; nullable=true),
  KeySpec("shade_tol_from_data", :bool, false, "per-species shade tolerance from data"),
  KeySpec("shade_tol_csv", :path, "./runs/shadetol_all_species.csv", "shade tolerance table"),
  KeySpec("prob_estab_from_data", :bool, false, "PROB_ESTAB from data"),
  KeySpec("prob_estab_csv", :path, "./runs/prob_estab_from_data.csv", "PROB_ESTAB table (natural)"),
  KeySpec("prob_estab_csv_artificial", :path, "./runs/prob_estab_from_data_artificial.csv", "PROB_ESTAB table (artificial)"),
  KeySpec("prob_estab_lower_bound", :bool, false, "use the PROB_ESTAB data value as a search lower bound (needs prob_estab_from_data)"),
  KeySpec("prob_estab_lower_bound_mult", :float, 1.0, "lower bound = mult x data value"),
  KeySpec("maturity_from_data", :bool, false, "MATURITY from data"),
  KeySpec("maturity_csv", :path, "./runs/prob_estab_all_species.csv", "MATURITY table"),
  KeySpec("bmax_floor_from_data", :bool, false, "B_MAX floor from data"),
  KeySpec("bmax_floor_csv", :path, "./runs/bmax_floor.csv", "B_MAX floor table"),
  KeySpec("anpp_floor_from_data", :bool, false, "ANPP_MAX floor from data"),
  KeySpec("anpp_floor_csv", :path, "./runs/anpp_floor_853.csv", "ANPP_MAX floor table"),
  KeySpec("fix_pmort", :bool, false, "pin the mortality parameters"),
  KeySpec("pmort_upper", :float, 0.02, "upper bound of the mortality parameter (Sim B)"),
  KeySpec("pmort_upper_simA", :float, 0.2, "upper bound of the mortality parameter (Sim A)"),
  KeySpec("fix_growth", :bool, false, "stage B: pin {D, S, ANPP_MAX, B_MAX}, fit establishment only"),
  KeySpec("fix_maturity", :bool, false, "pin MATURITY out of the search"),
  KeySpec("fix_min_rel", :bool, false, "pin MIN_REL_BIOMASS out of the search"),
  KeySpec("bmax_upper", :int, 35000, "upper bound of the B_MAX search box (lower stays 12000)"),
  KeySpec("anpp_ratio_bounds", :floatlist, [20.0, 35.0], "B_MAX / ANPP_MAX ratio box [lo, hi]"),
  # ---- Sobol ---------------------------------------------------------------------------------------------------------
  KeySpec("sobol_n", :int, 100, "Sobol sample count"),
  KeySpec("saltelli", :bool, false, "Saltelli sampling for Sobol indices"),
  KeySpec("sobol_top_frac", :float, 0.5, "top fraction of Sobol candidates kept as seeds"),
  # ---- LBSA ----------------------------------------------------------------------------------------------------------
  KeySpec("lbsa_stretch_len", :int, 150, "LBSA stretch length"),
  KeySpec("lbsa_stale_ratio", :float, 0.95, "LBSA freeze-detection uphill-attempt ratio"),
  KeySpec("lbsa_temp_list_len", :int, 150, "LBSA temperature-list length"),
  KeySpec("lbsa_replace_oldest", :bool, true, "false = overwrite the maximum each stretch (responsive temperature)"),
  KeySpec("lbsa_cooling_only", :bool, false, "true = monotone cooling so the search can freeze and reheat"),
  # ---- SA (classic simulated annealing with a cooling schedule) -----------------------------------------------------
  KeySpec("sa_initial_t", :float, 1000.0, "SA initial temperature"),
  KeySpec("sa_alpha", :float, 0.9992, "SA geometric cooling factor per trial (t <- alpha t)"),
  KeySpec("sa_min_t", :float, 0.01, "SA stops when the temperature falls below this"),
  KeySpec("sa_acceptance", :string, "SA", "SA acceptance rule: SA (Metropolis) | TA (threshold accepting)"; choices=("SA", "TA")),
  KeySpec("sa_reheat_prob", :float, 0.01, "SA reheats when the running acceptance probability stays below this"),
  KeySpec("sa_reheat_after", :int, 100, "SA reheats after this many consecutive trials below sa_reheat_prob"),
  # ---- CMA-ES family -------------------------------------------------------------------------------------------------
  KeySpec("mocmaes_true_nds", :bool, false, "MO-CMA-ES ranks offspring by NSGA-II fronts instead of win count"),
  KeySpec("cmaes_lambda", :int, nothing, "CMA-ES offspring count (null = default)"; nullable=true),
  KeySpec("cmaes_sigma0", :float, 0.3, "initial step size"),
  KeySpec("cmaes_warmstart_seeds", :int, 20, "top Sobol seeds that warm-start mean and covariance"),
  KeySpec("ipop", :bool, false, "IPOP restarts"),
  KeySpec("ipop_stagnation", :int, 20, "IPOP stagnation window"),
  KeySpec("ipop_max_barren_restarts", :int, 4, "stop after this many consecutive restarts without a new best (0 = off)"),
  KeySpec("cmaes_archive_cap", :int, 200, "archive size cap of every driver"),
  KeySpec("seed_archive_from", :path, nothing, "archive to seed from"; nullable=true),
  KeySpec("cmaes_integer_handling", :bool, false, "CMA-ES integer handling"),
  KeySpec("cmaes_integer_std_factor", :float, 0.3, "integer-handling std factor"),
  KeySpec("cmaes_single_cov", :bool, false, "one full covariance over all parameters (no block split)"),
  # ---- NSGA-II -------------------------------------------------------------------------------------------------------
  KeySpec("nsga2_pop", :int, 48, "population size"),
  KeySpec("nsga2_offspring", :int, nothing, "offspring per generation (null = population size)"; nullable=true),
  KeySpec("nsga2_eta_c", :float, 20.0, "SBX distribution index"),
  KeySpec("nsga2_eta_m", :float, 20.0, "polynomial-mutation distribution index"),
  KeySpec("nsga2_pc", :float, 0.9, "crossover probability"),
  KeySpec("nsga2_pm", :float, -1.0, "mutation probability (<0 = 1/n)"),
  KeySpec("nsga2_sobol_init", :bool, true, "initial population from Sobol"),
  # ---- Igel MO-CMA-ES ------------------------------------------------------------------------------------------------
  KeySpec("igel_mu", :int, 20, "number of lineages"),
  KeySpec("igel_sigma0", :float, 0.3, "initial step size"),
  KeySpec("igel_sobol_init", :bool, true, "initial lineages from Sobol"),
  KeySpec("igel_sobol_pool_k", :int, 1, ">1: oversample, keep best mu*k by dominance and pick mu at random"),
  KeySpec("igel_sobol_raw_mult", :int, 2, "raw Sobol pool = (mu*pool_k)*raw_mult candidates"),
  KeySpec("igel_niche_radius", :float, 0.0, "niching radius"),
  KeySpec("igel_reseed_sigma", :float, 0.0, "reseed a lineage below this step size"),
  KeySpec("igel_reseed_random_frac", :float, 0.0, "random fraction of reseeds"),
  KeySpec("igel_maturity", :int, 0, "generations a new lineage is shielded"),
  KeySpec("igel_seed_maturity", :bool, false, "phase 1: shield the mu seeds for igel_maturity generations"),
  KeySpec("igel_warm_start", :bool, false, "initial population = start_from + (mu-1) Gaussian perturbations"),
  KeySpec("igel_freeze_seed_growth", :bool, false, "each seed lineage keeps its own frozen growth parameters (needs fix_growth and seeds)"),
  KeySpec("igel_cholesky", :bool, false, "sample with Cholesky instead of eigen"),
  KeySpec("igel_reseed_sigma_override", :float, -1.0, ">=0 retunes the reseed threshold on resume"),
  KeySpec("igel_maturity_override", :int, -1, ">=0 retunes the maturity shield on resume"),
  # ---- CC-Igel -------------------------------------------------------------------------------------------------------
  KeySpec("cc_group_size", :int, 2, "species per cooperative group"),
  KeySpec("cc_spec_gens", :int, 20, "specialisation generations per cycle"),
  KeySpec("cc_integ_gens", :int, 20, "integration generations per cycle"),
  KeySpec("cc_cycles", :int, 5, "cycles"),
  KeySpec("cc_fix_others", :bool, false, "fix the other groups while one group is optimised"),
  # ---- CMA-MAE -------------------------------------------------------------------------------------------------------
  KeySpec("cmame_alpha", :float, 0.02, "archive learning rate"),
  KeySpec("cmame_grid", :int, 15, "archive grid resolution"),
  KeySpec("cmame_reseed_explore", :float, 1.0, "reseed exploration"),
  KeySpec("cmame_restart_patience", :int, 6, "restart patience"),
  KeySpec("cmame_sobol_reseed", :bool, false, "reseed from Sobol"),
  KeySpec("cmame_mo_rank", :bool, false, "multi-objective ranking"),
  # ---- misc search ---------------------------------------------------------------------------------------------------
  KeySpec("balanced_quality", :bool, false, "balanced-quality archive"),
  KeySpec("archive_by_sp", :bool, false, "archive per species"),
  KeySpec("bounds_by_sobol", :bool, false, "search bounds from Sobol"),
  KeySpec("top_seeds", :float, 0.2, "fraction of top seeds"),
  KeySpec("dominance_species", :string, "exact", "species dominance rule"),
  KeySpec("gdigel", :gdigel, nothing, "gdigel driver options (own schema: GDIGEL_LIST)"),
]

# gdigel sub-dict (search_mode: gdigel); keys read by src/drivers/gdigel.jl and the tools. `weights` and the
# m/objective cross-check are resolved in `resolve_gdigel`.
const GDIGEL_LIST = KeySpec[
  KeySpec("mu", :int, 8, "lineages (one per scalarisation weight)"),
  KeySpec("sigma0", :float, 0.15, "initial step size"),
  KeySpec("sigma_stop", :float, 0.004, "stop a block below this step size"),
  KeySpec("sigma_min", :float, 0.002, "step-size floor"),
  KeySpec("patience", :int, 40, "generations without improvement before a restart"),
  KeySpec("size_weight", :float, 0.5, "weight of group size in the block score"),
  KeySpec("real_only", :bool, true, "score real (non-phantom) pairs only"),
  KeySpec("include_start", :bool, false, "include the start point among the lineages"),
  KeySpec("ckpt_every", :int, 10, "checkpoint every n generations"),
  KeySpec("min_pairs", :int, 20, "minimum observed pairs per group"),
  KeySpec("archive_min_pass_frac", :float, 0.8, "archive admission: fraction of groups that must pass"),
  KeySpec("objective", :string, "bias", "bias | mo"; choices=("bias", "mo")),
  KeySpec("m", :int, 2, "number of objectives of objective mo (2 or 3)"),
  KeySpec("death_fold", :bool, false, "fold the death term into W (m=2)"),
  KeySpec("mode", :string, "optimize", "optimize | explore"; choices=("optimize", "explore")),
  KeySpec("single_block", :bool, false, "ablation: one block over all u-dims"),
  KeySpec("group_param", :string, "", "EcoSpecies parameter that defines the groups (empty = the first one)"),
  KeySpec("archive_cap", :int, 2000, "archive size cap"),
  KeySpec("weights", :int, nothing, "scalarisation weights (default 8, or 4 for objective mo with m=3)"; nullable=true),
  KeySpec("explore_bias", :float, 0.07, "mode explore: tolerated bias"),
  KeySpec("explore_slack", :float, 1.5, "mode explore: slack on the anchor"),
  KeySpec("explore_start_sigma", :float, 0.02, "mode explore: spread of the start points"),
  KeySpec("rel_tol", :float, 1e-3, "relative tolerance of the block stop test"),
  KeySpec("seed", :int, 1, "GDIgel RNG seed"),
  KeySpec("kappa", :float, nothing, "death_fold: override kappa = W_ref/D_ref"; nullable=true),
  KeySpec("aref", :float, nothing, "override the A reference scale"; nullable=true),
  KeySpec("wref", :float, nothing, "override the W reference scale"; nullable=true),
  KeySpec("dref", :float, nothing, "override the D reference scale"; nullable=true),
]

const SCHEMA = Dict{String,KeySpec}(s.name => s for s in SCHEMA_LIST)
const GDIGEL_SCHEMA = Dict{String,KeySpec}(s.name => s for s in GDIGEL_LIST)

# Keys that older configs used and that no longer exist: loading them is an error that names the replacement.
const RETIRED_KEYS = Dict{String,String}(
  "eval_tier" => "use `tier`",
  "search_tier" => "use `tier`",
  "sobol_m" => "Sobol replicates are `n_reps`",
)

# ---------------------------------------------------------------------------------------------------------------
# Environment expansion
# ---------------------------------------------------------------------------------------------------------------

# Data / output roots that default to the repo-local layout when the variable is not set (the cluster sets them). This is a
# lookup fallback only: the process environment is never modified by config loading.
const _ENV_FALLBACK = Dict("PAN_DATA" => "runs", "PAN_OUT" => "runs/outputs")
const _WARNED_ENV = Set{String}()

function _envvar(name::AbstractString)
  haskey(ENV, name) && return ENV[name]
  haskey(_ENV_FALLBACK, name) && return _ENV_FALLBACK[name]
  if !(name in _WARNED_ENV)
    push!(_WARNED_ENV, name)
    @warn "config: environment variable $name is not defined; it expands to an empty string"
  end
  return ""
end

# Expand ${VAR}/$VAR (from ENV) and a leading ~ in a config path string, so yaml paths are portable across
# machines (e.g. cohorts_db_path: "${FIADB}", output_dir: "${SCRATCH}/out", "~/data.duckdb"). An undefined variable
# expands to "" (with a one-time warning). Plain/relative paths pass through unchanged.
function _expandenv(s::AbstractString)
  isempty(s) && return String(s)
  out = replace(String(s), r"\$\{(\w+)\}" => m -> _envvar(m[3:end-1]))   # ${VAR}
  out = replace(out, r"\$(\w+)" => m -> _envvar(m[2:end]))               # $VAR
  return expanduser(out)
end

# ---------------------------------------------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------------------------------------------

struct ConfigError <: Exception
  msg::String
end
Base.showerror(io::IO, e::ConfigError) = print(io, "ConfigError: ", e.msg)

_isnullish(v, spec::KeySpec) = v === nothing || (spec.null_str && v isa AbstractString && lowercase(strip(v)) in ("null", "none"))

_describe(v) = repr(v)

function _bad(where_, spec::KeySpec, v, want)
  throw(ConfigError("$where_$(spec.name): expected $want, got $(_describe(v))"))
end

function _to_bool(w, spec, v)
  v isa Bool && return v
  v isa Integer && v in (0, 1) && return v == 1
  if v isa AbstractString
    s = lowercase(strip(v))
    s == "true" && return true
    s == "false" && return false
    s in ("0", "1") && return s == "1"
  end
  _bad(w, spec, v, "a boolean (true/false, 0/1)")
end

function _to_int(w, spec, v)
  v isa Bool && _bad(w, spec, v, "an integer")
  v isa Integer && return Int(v)
  if v isa AbstractFloat
    isfinite(v) && v == round(v) && return Int(v)
    _bad(w, spec, v, "an integer (got a non-integral number)")
  end
  if v isa AbstractString
    s = strip(v)
    i = tryparse(Int, s)
    i !== nothing && return i
    f = tryparse(Float64, s)
    f !== nothing && isfinite(f) && f == round(f) && return Int(f)
  end
  _bad(w, spec, v, "an integer")
end

function _to_float(w, spec, v)
  v isa Bool && _bad(w, spec, v, "a number")
  v isa Real && return Float64(v)
  if v isa AbstractString
    f = tryparse(Float64, strip(v))
    f !== nothing && return f
  end
  _bad(w, spec, v, "a number")
end

function _to_string(w, spec, v)
  v isa AbstractString || v isa Symbol || _bad(w, spec, v, "a string")
  s = _expandenv(String(v))
  isempty(spec.choices) || s in spec.choices || throw(ConfigError("$w$(spec.name): must be $(join(spec.choices, " | ")), got $(repr(s))"))
  return s
end

function _to_symbol(w, spec, v)
  s = _to_string(w, spec, v)
  return Symbol(s)
end

_list(w, spec, v) = (v isa AbstractVector || v isa Tuple) ? v : _bad(w, spec, v, "a list")

_to_intlist(w, spec, v) = Int[_to_int(w, spec, x) for x in _list(w, spec, v)]
_to_floatlist(w, spec, v) = Float64[_to_float(w, spec, x) for x in _list(w, spec, v)]
function _to_strlist(w, spec, v)
  out = String[]
  for x in _list(w, spec, v)
    (x isa AbstractString || x isa Symbol || x isa Real) && !(x isa Bool) || _bad(w, spec, x, "a list of strings")
    push!(out, String(string(x)))
  end
  return out
end

# [statecd, unitcd, countycd, plot] entries: integers or numeric strings.
function _to_plots(w, spec, v)
  out = NTuple{4,Int}[]
  for p in _list(w, spec, v)
    (p isa AbstractVector || p isa Tuple) && length(p) == 4 || _bad(w, spec, p, "a plot [statecd, unitcd, countycd, plot]")
    q = ntuple(i -> p[i] isa AbstractString ? (tryparse(Int, strip(p[i])) === nothing ? _bad(w, spec, p, "a plot of integers") : parse(Int, strip(p[i]))) :
                    _to_int(w, spec, p[i]), 4)
    push!(out, q)
  end
  return out
end

function _to_dict(w, spec, v)
  v isa AbstractDict || _bad(w, spec, v, "a mapping")
  return Dict{String,Any}(String(k) => x for (k, x) in v)
end

function _to_dict_strlist(w, spec, v)
  v isa AbstractDict || _bad(w, spec, v, "a mapping")
  return Dict{String,Vector{String}}(String(k) => _to_strlist(w, spec, x) for (k, x) in v)
end

# {param => {species => {cell => group}}}
function _to_tiermerge(w, spec, v)
  v isa AbstractDict || _bad(w, spec, v, "a mapping")
  return Dict{String,Any}(String(k) => Dict{String,Any}(String(sp) => Dict{String,String}(String(c) => String(g) for (c, g) in cm) for (sp, cm) in x) for (k, x) in v)
end

function _to_dict_float(w, spec, v)
  v isa AbstractDict || _bad(w, spec, v, "a mapping")
  return Dict{String,Float64}(String(k) => _to_float(w, spec, x) for (k, x) in v)
end

# Int stays Int (integral floats too), any string stays a string (e.g. "auto").
function _to_intorstr(w, spec, v)
  v isa AbstractString && return _expandenv(String(v))
  return _to_int(w, spec, v)
end

# nothing = repo default parquet; false/""/"none"/"off" = disabled (""); otherwise a path.
function _to_disturb_path(w, spec, v)
  v === false && return ""
  v isa AbstractString || _bad(w, spec, v, "a path, false, or null")
  s = _expandenv(String(v))
  return s in ("", "none", "off") ? "" : s
end

function _coerce(w, spec::KeySpec, v)
  t = spec.type
  t === :bool && return _to_bool(w, spec, v)
  t === :int && return _to_int(w, spec, v)
  t === :float && return _to_float(w, spec, v)
  (t === :string || t === :path) && return _to_string(w, spec, v)
  t === :symbol && return _to_symbol(w, spec, v)
  t === :intlist && return _to_intlist(w, spec, v)
  t === :strlist && return _to_strlist(w, spec, v)
  t === :floatlist && return _to_floatlist(w, spec, v)
  t === :plots && return _to_plots(w, spec, v)
  t === :dict && return _to_dict(w, spec, v)
  t === :dict_strlist && return _to_dict_strlist(w, spec, v)
  t === :dict_tiermerge && return _to_tiermerge(w, spec, v)
  t === :dict_float && return _to_dict_float(w, spec, v)
  t === :intorstr && return _to_intorstr(w, spec, v)
  t === :disturb_path && return _to_disturb_path(w, spec, v)
  t === :dual_mode && return (try _parse_dual_mode(v) catch e; e isa ErrorException ? throw(ConfigError("$w$(spec.name): " * e.msg)) : rethrow() end)
  t === :gdigel && return resolve_gdigel(v)
  error("config schema: unknown type $t for $(spec.name)")
end

# ---------------------------------------------------------------------------------------------------------------
# Key lookup errors
# ---------------------------------------------------------------------------------------------------------------

function _editdist(a::AbstractString, b::AbstractString)
  a, b = collect(a), collect(b)
  prev = collect(0:length(b))
  for (i, ca) in enumerate(a)
    cur = [i; zeros(Int, length(b))]
    for (j, cb) in enumerate(b)
      cur[j+1] = min(prev[j+1] + 1, cur[j] + 1, prev[j] + (ca == cb ? 0 : 1))
    end
    prev = cur
  end
  return prev[end]
end

function _suggest(key::AbstractString, known)
  k = lowercase(key)
  sc = [(min(_editdist(k, lowercase(n)), occursin(k, lowercase(n)) || occursin(lowercase(n), k) ? 1 : typemax(Int)), n) for n in known]
  sort!(sc)
  lim = max(2, length(key) ÷ 3)
  return [n for (d, n) in sc[1:min(3, end)] if d <= lim]
end

function _unknown_key_error(key::AbstractString, known, where_::AbstractString="")
  sug = _suggest(key, known)
  hint = isempty(sug) ? "" : " Did you mean " * join(("`$s`" for s in sug), " or ") * "?"
  return ConfigError("unknown config key `$where_$key`.$hint")
end

# ---------------------------------------------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------------------------------------------

_default_of(spec::KeySpec) = deepcopy(spec.default)

function _resolve_value(where_::AbstractString, spec::KeySpec, v)
  if _isnullish(v, spec)
    spec.nullable || spec.type === :gdigel && v === nothing || throw(ConfigError("$where_$(spec.name): must not be null"))
    spec.type === :gdigel && return resolve_gdigel(Dict{String,Any}())
    return nothing
  end
  return _coerce(where_, spec, v)
end

function _resolve_table(d::AbstractDict, specs::Dict{String,KeySpec}, speclist::Vector{KeySpec}, where_::AbstractString)
  # report every bad key at once: retired keys first (they name the replacement), then unknown keys with suggestions
  bad = [String(k) for k in keys(d) if !haskey(specs, String(k))]
  retired = where_ == "" ? [k for k in bad if haskey(RETIRED_KEYS, k)] : String[]
  isempty(retired) || throw(ConfigError("retired config key" * (length(retired) > 1 ? "s" : "") * ": " *
                                        join(("`$k` ($(RETIRED_KEYS[k]))" for k in sort(retired)), "; ")))
  if !isempty(bad)
    errs = [_unknown_key_error(k, keys(specs), where_).msg for k in sort(bad)]
    throw(ConfigError(join(errs, "\n")))
  end
  vals = Dict{String,Any}()
  given = Set{String}()
  for (k0, v) in d
    k = String(k0)
    vals[k] = _resolve_value(where_, specs[k], v)
    push!(given, k)
  end
  for s in speclist
    haskey(vals, s.name) && continue
    vals[s.name] = s.type === :gdigel ? resolve_gdigel(Dict{String,Any}()) : _default_of(s)
  end
  return vals, given
end

"""
    resolve_gdigel(d::AbstractDict) -> Dict{String,Any}

Validate a `gdigel:` sub-dict against `GDIGEL_LIST` and fill the defaults. Idempotent (a resolved dict resolves to itself).
"""
function resolve_gdigel(d)
  d isa AbstractDict || throw(ConfigError("gdigel: expected a mapping, got $(repr(d))"))
  vals, given = _resolve_table(d, GDIGEL_SCHEMA, GDIGEL_LIST, "gdigel.")
  vals["m"] in (2, 3) || throw(ConfigError("gdigel.m: must be 2 or 3, got $(vals["m"])"))
  if vals["weights"] === nothing
    vals["weights"] = (vals["objective"] == "mo" && vals["m"] == 3) ? 4 : 8
  end
  return vals
end

"""
    PanConfig <: AbstractDict{String,Any}

Read-only dictionary of RESOLVED config values (every schema key is present). `cfg[key]` of an unknown key throws a
`ConfigError` with suggestions, so a typo in a tool cannot silently fall back to a default. `isgiven(cfg, key)` tells whether
the yaml / overrides set the key explicitly; `cfg.path` is the source yaml ("" for in-memory configs).
"""
struct PanConfig <: AbstractDict{String,Any}
  vals::Dict{String,Any}
  given::Set{String}
  path::String
end

isgiven(c::PanConfig, k::AbstractString) = String(k) in c.given

function Base.getindex(c::PanConfig, k::AbstractString)
  haskey(c.vals, k) && return c.vals[k]
  throw(_unknown_key_error(k, keys(c.vals)))
end
Base.get(c::PanConfig, k::AbstractString, ::Any) = c[k]   # the schema owns the defaults: the call-site default is ignored
Base.haskey(c::PanConfig, k::AbstractString) = haskey(c.vals, k)
Base.length(c::PanConfig) = length(c.vals)
Base.iterate(c::PanConfig, s...) = iterate(c.vals, s...)
Base.keys(c::PanConfig) = keys(c.vals)
Base.setindex!(::PanConfig, ::Any, ::Any) = throw(ConfigError("PanConfig is read-only; pass overrides to load_config"))
Base.show(io::IO, c::PanConfig) = print(io, "PanConfig(", isempty(c.path) ? "<memory>" : c.path, ", ", length(c.given), " keys set)")

_load_yaml_dict(path::AbstractString) = begin
  isfile(path) || throw(ConfigError("config file not found: $path"))
  y = YAML.load_file(path)
  y === nothing && return Dict{String,Any}()
  y isa AbstractDict || throw(ConfigError("$path: top level must be a mapping"))
  y
end

"""
    load_config(path_or_dict; overrides=Dict()) -> PanConfig

The one config reader. Reads the yaml (or takes a Dict), applies `overrides` (runner / CLI values), rejects unknown and retired
keys, expands `\${VAR}`/`\$VAR`/`~` in string values, coerces every value to its schema type and fills the defaults.
`loss_before_injection` is resolved to a Bool (its default depends on the injection mode) and the `FIADB` environment variable
overrides `cohorts_db_path`.
"""
function load_config(d::AbstractDict; overrides::AbstractDict=Dict{String,Any}(), path::AbstractString="")
  merged = Dict{String,Any}(String(k) => v for (k, v) in d)
  for (k, v) in overrides; merged[String(k)] = v; end
  vals, given = try
    _resolve_table(merged, SCHEMA, SCHEMA_LIST, "")
  catch e
    e isa ConfigError && !isempty(path) && throw(ConfigError("$path: " * e.msg))
    rethrow()
  end
  # cross-key resolution
  vals["loss_before_injection"] = try
    _resolve_loss_before_injection((k, dflt) -> get(vals, k, dflt))
  catch e
    e isa ErrorException ? throw(ConfigError((isempty(path) ? "" : "$path: ") * e.msg)) : rethrow()
  end
  get(ENV, "FIADB", "") != "" && (vals["cohorts_db_path"] = ENV["FIADB"])   # env FIADB wins (portable across machines)
  return PanConfig(vals, given, String(path))
end
load_config(path::AbstractString; overrides::AbstractDict=Dict{String,Any}()) = load_config(_load_yaml_dict(path); overrides=overrides, path=path)
load_config(c::PanConfig; overrides::AbstractDict=Dict{String,Any}()) =
  load_config(Dict{String,Any}(k => c.vals[k] for k in c.given); overrides=overrides, path=c.path)

# Accept an already-loaded PanConfig, or load a raw Dict (programmatic callers).
_as_config(c::PanConfig) = c
_as_config(d::AbstractDict) = load_config(d)

"""
    config_docs() -> String

Markdown-style listing of every key: name, type, default and doc line (for `docs/` and `--help`).
"""
function config_docs()
  io = IOBuffer()
  for s in SCHEMA_LIST
    println(io, "- `", s.name, "` (", s.type, s.nullable ? ", nullable" : "", "; default ", s.type === :gdigel ? "{}" : repr(s.default), "): ", s.doc)
  end
  println(io, "\ngdigel sub-keys:")
  for s in GDIGEL_LIST
    println(io, "- `gdigel.", s.name, "` (", s.type, s.nullable ? ", nullable" : "", "; default ", repr(s.default), "): ", s.doc)
  end
  String(take!(io))
end
