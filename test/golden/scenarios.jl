# Golden-harness scenarios: name => (base yaml relative to the repo, overrides).
# Every scenario is small: about 80 fixed GA+FL plots (plots.csv), seed 1337, n_reps 1, a few evaluations.
# The harness runs each one in its own Julia process with 1 thread, so module state cannot leak between them.
# Keys that are not valid for an older tree are simply not used there; the scenario list is shared by both trees.

const GOLDEN_DIR = @__DIR__

function golden_plots()
  rows = readlines(joinpath(GOLDEN_DIR, "plots.csv"))[2:end]
  [parse.(Int, split(r, ',')[1:4]) for r in rows]
end

const A = "configs/region4/gafl_A.yml"
const B = "configs/region4/gafl_B.yml"
const PN = "configs/region4/gafl_pnet.yml"
const HB = "configs/region4/gafl_hbad.yml"

common() = Dict{String,Any}(
  "filter_plots" => golden_plots(), "seed" => 1337, "n_reps" => 1, "min_trees" => 20,
  "igel_mu" => 4, "igel_sobol_init" => true, "igel_sobol_pool_k" => 1, "cmaes_archive_cap" => 50,
  # 80 plots ⇒ most split strata are singletons; the 3-way split sends those to val/test (train ≈ 5 plots). 2-fold CV
  # fold 1 keeps singletons in train (≈ 60 train / 20 val). The production 3-way + train_cap split is covered by bs_split3.
  "n_folds" => 2, "fold_index" => 1, "test_frac" => 0.0, "train_cap" => nothing)

gd(; kw...) = Dict{String,Any}("search_mode" => "gdigel",
  "gdigel" => merge(Dict{String,Any}("mu" => 4, "ckpt_every" => 2, "min_pairs" => 3), Dict{String,Any}(String(k) => v for (k, v) in kw)))

# name => (base, overrides, trials)
const SCENARIOS = Dict{String,Tuple{String,Dict{String,Any},Int}}(
  # ---- search drivers (BS, Sim A tier 3, sync, exclude_overwrite = the production rule)
  "bs_igelmo"      => (A, Dict{String,Any}(), 12),
  "bs_split3"      => (A, Dict{String,Any}("n_folds" => 1, "test_frac" => 0.25, "train_cap" => "auto"), 8),
  "bs_gd_mo"       => (A, gd(objective="mo"), 12),
  "bs_gd_bias"     => (A, gd(objective="bias"), 12),
  "bs_lbsa"        => (A, Dict{String,Any}("search_mode" => "lbsa"), 10),
  "bs_molbsa"      => (A, Dict{String,Any}("search_mode" => "molbsa"), 10),
  "bs_cmaes"       => (A, Dict{String,Any}("search_mode" => "cmaes", "cmaes_lambda" => 4), 10),
  "bs_mocmaes"     => (A, Dict{String,Any}("search_mode" => "mocmaes", "cmaes_lambda" => 4), 10),
  "bs_cmame"       => (A, Dict{String,Any}("search_mode" => "cmame", "cmaes_lambda" => 4), 10),
  "bs_ccigel"      => (A, Dict{String,Any}("search_mode" => "ccigel", "cc_group_size" => 2, "cc_spec_gens" => 1, "cc_integ_gens" => 1, "cc_cycles" => 1), 12),
  "bs_nsga2"       => (A, Dict{String,Any}("search_mode" => "nsga2", "nsga2_pop" => 4, "nsga2_offspring" => 4), 10),
  "bs_sobol"       => (A, Dict{String,Any}("search_mode" => "sobol", "sobol_n" => 4), 4),
  "bs_sa"          => (A, Dict{String,Any}("search_mode" => "sa"), 10),          # currently unreachable (falls back to LBSA)
  # ---- tiers / dual
  "bs_tier1"       => (A, Dict{String,Any}("tier" => 1), 8),
  "bs_tier2"       => (A, Dict{String,Any}("tier" => 2), 8),
  "bs_tier4"       => (A, Dict{String,Any}("tier" => 4, "unbinned_w" => false), 8),
  "bs_tier5"       => (A, Dict{String,Any}("tier" => 5, "unbinned_w" => false), 8),
  "bs_dual_joint"  => (A, Dict{String,Any}("dual_mode" => true, "unbinned_w" => false, "no_establishment" => false), 8),
  "bs_spinup"      => (A, Dict{String,Any}("spinup" => true), 8),
  # ---- injection / scoring rule
  "bs_dist_scale"  => (A, Dict{String,Any}("override_injection_disturbance" => "scale"), 8),
  "bs_dist_off"    => (A, Dict{String,Any}("override_injection_disturbance" => "off"), 8),
  "bs_dist_noscale"=> (A, Dict{String,Any}("override_injection_disturbance" => "exclude_noscale"), 8),
  "bs_legacy_order"=> (A, Dict{String,Any}("loss_before_injection" => false), 8),
  "bs_recruits_in" => (A, Dict{String,Any}("exclude_recruits_at_entry" => false), 8),
  "bs_replace"     => (A, Dict{String,Any}("override_injection_replace" => true), 8),
  "bs_noise"       => (A, Dict{String,Any}("override_injection_noise" => 0.1), 8),
  "bs_no_sync"     => (A, Dict{String,Any}("override_injection_sync" => false), 8),
  "bs_no_inject"   => (A, Dict{String,Any}("override_injection" => false), 8),
  "bs_w_surv"      => (A, Dict{String,Any}("w_scope" => "survivors"), 8),                       # campaign option O1
  "bs_death"       => (A, Dict{String,Any}("w_scope" => "survivors", "death_term" => "likelihood"), 8),   # O2
  # ---- loss / strata knobs
  "bs_single_cov"  => (A, Dict{String,Any}("cmaes_single_cov" => true), 8),
  "bs_rankw_rank"  => (A, Dict{String,Any}("rankw_mode" => "rank"), 8),
  "bs_autoknee"    => (A, Dict{String,Any}("w_smooth_auto_knee" => true), 8),
  "bs_hinge_p2"    => (A, Dict{String,Any}("agb_hinge_p" => 2.0), 8),
  "bs_reps3"       => (A, Dict{String,Any}("n_reps" => 3, "init_perturb_frac" => 0.05), 8),
  "bs_single_eco"  => (A, Dict{String,Any}("single_ecoregion" => true), 8),
  "bs_no_strat"    => (A, Dict{String,Any}("stratify_landuse" => false, "site_class_strata" => false), 8),
  # ---- plugins
  "pnet_igelmo"    => (PN, Dict{String,Any}(), 8),
  "pnet_gd_mo"     => (PN, gd(objective="mo"), 8),
  "hbad_igelmo"    => (HB, Dict{String,Any}(), 8),
  "hbad_gd_mo"     => (HB, gd(objective="mo"), 8),
  # ---- Sim B
  "bs_simB"        => (B, Dict{String,Any}(), 8),
  "bs_simB_spinup" => (B, Dict{String,Any}("simB_spinup" => true), 8),
  # ---- analysis entry points (params from bs_igelmo's output; see run_scenario.jl)
  "bs_plot_only"   => (A, Dict{String,Any}("search_mode" => "plot_only", "n_output_plots" => 2), 1),
)
