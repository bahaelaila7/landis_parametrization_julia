# Run options that define HOW a parameter set is scored and simulated (Step 3b). Immutable, concretely typed, built once per run by
# `scoring_from_config` (Pan.jl) and passed explicitly to every function that needs them: no module-level state, so two runs (or a tool
# and a run) in one process cannot leak into each other.
#
# Fields are Bool / Float / Symbol only, all concrete: the site loop reads them without boxing or dynamic dispatch.

"""
    ScoringOptions

The scoring rule of a Sim A run (what is injected from the data, when the loss is read, what is excluded from it) plus the
initial-condition perturbation. Defaults are the legacy module defaults (no injection, new scoring order, recruits excluded).

  - `override`, `replace`, `sync`, `noise`, `disturbance`: the injection mode (yaml `override_injection[_replace|_sync|_noise|_disturbance]`).
  - `loss_before_injection`: score the simulated state BEFORE injecting the observed cohorts (resolved by `_resolve_loss_before_injection`).
  - `exclude_recruits_at_entry`: cohorts observed at T with no counterpart at the previous visit are excluded from the loss.
  - `w_scope` (`:all` | `:survivors`) and `death_term` (`:none` | `:likelihood`): the natural-death part of the Sim A score.
  - `w_diag` (`:none` | `:stats` | `:nodeath` | `:oracle`): diagnostic mode of tools/w_decompose.jl (no effect by default).
  - `init_perturb_frac`, `init_perturb_cap`: initial-biomass perturbation reps (see `_init_scales`).
  - `cycle`: how tier-4/5 measurements are bucketed into calendar cycles (`Data.CycleOptions`: FIA `cycle` column vs computed, merge width).
"""
Base.@kwdef struct ScoringOptions
  override::Bool = false
  replace::Bool = false
  sync::Bool = false
  noise::Float64 = 0.0
  disturbance::Symbol = :off
  loss_before_injection::Bool = true
  exclude_recruits_at_entry::Bool = true
  w_scope::Symbol = :all
  death_term::Symbol = :none
  w_diag::Symbol = :none
  init_perturb_frac::Float64 = 0.0
  init_perturb_cap::FloatType = FloatType(Inf)
  cycle::Data.CycleOptions = Data.CycleOptions()
end

"""
    SimBOptions

Sim B (dual mode) and tier-4 options.

  - `dual_mode`: `:off` (Sim A only) | `:joint` (A + B) | `:b` (Sim B only, seeded from A).
  - `tier_b`: Sim B's loss tier (A's tier is the config `tier`).
  - `t4_primary`: the primary simulation is scored at tier 4 (dual_mode `:b`, or tier 4 without a dual): its per-cell loss scales live in the
    B slots (the tier-4 LossScales of dual_b / the primary reference), so `_group_scales` must normalise group 0 with them.
  - `simb_spinup`: Sim B starts from a spun-up deficit (true) or from the observed year-0 cohorts (false).
  - `simb_disturb_only`: Sim B is purely free (no injection); observed disturbance drops are applied to every plot.
  - `t4_per_plot_mean`: tier-4 AGB as mean biomass per plot by age.
  - `t4_mean_over_reps`: tier-4 loss scored once on the mean-over-reps simulated histogram.
"""
Base.@kwdef struct SimBOptions
  dual_mode::Symbol = :off
  tier_b::Int = 4
  t4_primary::Bool = false
  simb_spinup::Bool = true
  simb_disturb_only::Bool = false
  t4_per_plot_mean::Bool = false
  t4_mean_over_reps::Bool = false
end

"The disturbance SHOCK order: score the simulated state first, with the shock (and exclusions) applied before scoring. Needs the sync injection."
@inline shock_order(o::ScoringOptions) = o.loss_before_injection && o.override && o.sync && !o.replace

"A copy of the options `o` with the given fields replaced: `with_options(scoring; init_perturb_frac=0.0)`."
with_options(o::T; kw...) where {T<:Union{ScoringOptions,SimBOptions}} =
  T(; pairs(merge(NamedTuple{fieldnames(T)}(ntuple(i -> getfield(o, i), fieldcount(T))), NamedTuple(kw)))...)

# ── Execution options (Step 3e) ─────────────────────────────────────────────────────────────────────────────
"""
    ExecOptions

HOW a run executes (nothing here changes what is computed): built once by `exec_options_from_env` (or by hand in a tool) and passed
explicitly to `parametrize` and the search drivers.

  - `parallel_mode`: `:soa` (default; sites of one evaluation run threaded) | `:candidate` (the driver runs candidate×rep evaluations in parallel,
    each sites-serial on a thread-local work SoA; honoured by igelmo and gdigel). env `PAN_PARALLEL`.
  - `timing`: print the per-generation wall-time breakdown (igelmo). env `PAN_TIMING`. Off ⇒ no timer object exists and no probe runs.
  - `skip_val`: skip every held-out validation re-simulation during training (benchmarks only). env `PAN_SKIP_VAL`.
  - `newbest_stats`: run the diagnostic `simulate_and_test` on the train split at each new best. env `PAN_NEWBEST_STATS`.
  - `store_val_obj`: val-score every newly admitted archive member into cv_val_cache.csv (train-argmin ≠ val-argmin).
  - `cv_reselect`: the MO-CMA-ES driver skips the search and re-scores the checkpoints on the val split (`_cv_reselect_dump`).
  - `t0`: `time_ns()` at construction, the origin of the "startup → first generation" span of the timing output.
"""
Base.@kwdef struct ExecOptions
  parallel_mode::Symbol = :soa
  timing::Bool = false
  skip_val::Bool = false
  newbest_stats::Bool = false
  store_val_obj::Bool = true
  cv_reselect::Bool = false
  t0::UInt64 = time_ns()
end

"True when the driver should parallelise over candidates (and therefore run every evaluation sites-serial)."
@inline candidate_parallel(e::ExecOptions) = e.parallel_mode === :candidate

_env_flag(env, k) = get(env, k, "") in ("1", "true", "yes")

"""
    exec_options_from_env(; env=ENV, kw...) -> ExecOptions

The options a run launched from the environment uses: `PAN_TIMING`, `PAN_SKIP_VAL`, `PAN_NEWBEST_STATS` (1|true|yes) and `PAN_PARALLEL`
(`soa` | `candidate`). Keyword arguments override fields (`exec_options_from_env(cv_reselect=true)`).
"""
function exec_options_from_env(; env=ENV, kw...)
  mode = Symbol(get(env, "PAN_PARALLEL", "soa"))
  mode in (:soa, :candidate) || error("PAN_PARALLEL must be 'soa' or 'candidate' (got '$mode')")
  return ExecOptions(; parallel_mode=mode, timing=_env_flag(env, "PAN_TIMING"), skip_val=_env_flag(env, "PAN_SKIP_VAL"),
                     newbest_stats=_env_flag(env, "PAN_NEWBEST_STATS"), kw...)
end

"""
    RunTimer

Per-run accumulators of the timing output: nanoseconds in the simulation step (`sim`, site-parallel evaluations only) and blocked handing jobs to the
checkpoint writer (`io`), both within the current generation. `nothing` instead of a RunTimer means timing is off: every probe is a
`timer === nothing` branch on a `Union{Nothing,RunTimer}` that the compiler splits statically.
"""
mutable struct RunTimer
  sim::UInt64
  io::UInt64
end
RunTimer() = RunTimer(UInt64(0), UInt64(0))
"A fresh timer when `e.timing`, else `nothing`."
run_timer(e::ExecOptions) = e.timing ? RunTimer() : nothing
reset!(t::RunTimer) = (t.sim = 0; t.io = 0; t)

"`put!(ch, job)`, accounting the time blocked on the channel into `timer.io` when a timer is given."
@inline timed_put!(::Nothing, ch, job) = put!(ch, job)
@inline function timed_put!(t::RunTimer, ch, job)
  t0 = time_ns(); put!(ch, job); t.io += time_ns() - t0
  return nothing
end

# ── Search-driver options (Step 3e) ────────────────────────────────────────────────────────────────────────
"""
    LBSAOptions

The List-Based SA knobs of `parametrize_LBSA` (yaml `lbsa_*`): `stretch_len` (shorter ⇒ faster freeze detection), `stale_ratio`
(`up_attempt_stale_ratio`; lower ⇒ freezes/reheats sooner), `temp_list_len` (the length L of the temperature list), `replace_oldest`
(false ⇒ overwrite the MAX each stretch, so the temperature adapts every stretch instead of after a full L-cycle) and `cooling_only`
(true ⇒ the max temperature can only decrease, so the search actually freezes and the freeze/reheat/restart chain fires).
"""
Base.@kwdef struct LBSAOptions
  stretch_len::Int = 150
  stale_ratio::Float64 = 0.95
  temp_list_len::Int = 150
  replace_oldest::Bool = true
  cooling_only::Bool = false
end

"""
    SAOptions

The simulated-annealing knobs of `parametrize_SA` (yaml `sa_*`): the geometric cooling schedule `t ← alpha·t` from `initial_t` (the run ends when
`t < min_t` or the trial budget is spent), the `acceptance` rule ("SA" Metropolis exp(-Δ/t) or "TA" threshold accepting Δ < t) and the reheat
(back to `initial_t` when the running acceptance probability stays below `reheat_prob_threshold` for `reheat_after_iters` trials). The defaults
are those of the original driver.
"""
Base.@kwdef struct SAOptions
  initial_t::Float64 = 1000.0
  alpha::Float64 = 0.9992
  min_t::Float64 = 0.01
  acceptance::String = "SA"
  reheat_prob_threshold::Float64 = 0.01
  reheat_after_iters::Int = 100
end

"""
    SearchOptions

The config-derived knobs of the search engines, bundled and passed to the drivers as `search=`: `lbsa` (LBSA), `sa` (simulated annealing), `igel` (the Igel engine's
resume-retunable knobs, see `IgelMOCMAES.IgelOptions`; the Igel driver adds its niching maps and reseed fraction on top) and `mocmaes_nds`
(MO-CMA-ES and CMA-ME rank offspring by TRUE NSGA-II non-dominated sorting instead of the net win count; yaml `mocmaes_true_nds`).
"""
Base.@kwdef struct SearchOptions
  lbsa::LBSAOptions = LBSAOptions()
  sa::SAOptions = SAOptions()
  igel::IgelMOCMAES.IgelOptions{typeof(identity),typeof(identity)} = IgelMOCMAES.IgelOptions()
  mocmaes_nds::Bool = false
end

"The `SearchOptions` of a validated config (see `load_config`)."
search_options_from_config(cfg::AbstractDict) = SearchOptions(
  lbsa=LBSAOptions(stretch_len=cfg["lbsa_stretch_len"], stale_ratio=cfg["lbsa_stale_ratio"], temp_list_len=cfg["lbsa_temp_list_len"],
                   replace_oldest=cfg["lbsa_replace_oldest"], cooling_only=cfg["lbsa_cooling_only"]),
  sa=SAOptions(initial_t=cfg["sa_initial_t"], alpha=cfg["sa_alpha"], min_t=cfg["sa_min_t"], acceptance=cfg["sa_acceptance"],
               reheat_prob_threshold=cfg["sa_reheat_prob"], reheat_after_iters=cfg["sa_reheat_after"]),
  igel=IgelMOCMAES.IgelOptions(use_cholesky=cfg["igel_cholesky"], reseed_sigma_override=cfg["igel_reseed_sigma_override"],
                               maturity_override=cfg["igel_maturity_override"]),
  mocmaes_nds=cfg["mocmaes_true_nds"])
