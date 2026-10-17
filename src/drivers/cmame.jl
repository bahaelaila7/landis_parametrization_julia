# CMA-MAE driver (Fontaine & Nikolaidis 2022): one archive-driven CMA-ES emitter over a 2-D MAP-Elites
# grid. Behaviour descriptor = (Σ Wasserstein loss, Σ AGB loss) — the two halves of the MO objective
# vector; quality = the aggregate loss. The archive keeps one elite parameter set per behaviour cell
# (a quality-diversity set of fits), the representative is the lowest-aggregate elite. Same MO objective
# vector, writer, archive logging and losses.duckdb schema as parametrize_MOCMAES; the emitter re-seeds
# itself on convergence/stagnation (no external IPOP). Grid bounds are auto-scaled from the seed loss.
function parametrize_CMAMAE(; ref_soa::SiteSoA, space::PU.ParamSpace, plugin::PluginConfig, output_dir::AbstractString, splots, spdf_plts, spinup_cohorts::DataFrame, site_sim_years, species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}, loss_params::PU.LossParams, scoring::ScoringOptions, simb::SimBOptions, spinup::Bool, TRIALS::Int, rng::Random.AbstractRNG, debug::Bool, search_tier::Int=1, resume_from::Union{Nothing,String}=nothing, start_from::Union{Nothing,String}=nothing, force_restart_from_random::Bool=false, n_reps::Int=1, sobol_candidates_db::Union{Nothing,String}=nothing, sobol_top_frac::Float64=0.5, n_output_plots::Int=0, no_establishment::Bool=false, injection_cohorts=nothing, val_splots=nothing, val_ref_soa=nothing, val_spdf_plts=nothing, val_site_sim_years=nothing, val_spinup_cohorts=nothing, val_injection_cohorts=nothing, cycle_years::Real=8, cmaes_lambda::Union{Nothing,Int}=nothing, cmaes_sigma0::Float64=0.3, cmame_alpha::Float64=0.02, cmame_grid::Int=15, cmame_reseed_explore::Float64=1.0, cmame_restart_patience::Int=6, cmame_sobol_reseed::Bool=false, balanced_quality::Bool=false, cmame_mo_rank::Bool=false, archive_by_sp::Bool=false, bounds_by_sobol::Bool=false, top_seeds::Float64=0.2, integer_handling::Bool=false, integer_std_factor::Float64=0.3, single_cov::Bool=false, scale_seed::PU.LossScales=PU.LossScales(), exec::ExecOptions=ExecOptions(), search::SearchOptions=SearchOptions())
  env = mo_env(; ref_soa, plugin, splots, spdf_plts, spinup_cohorts, site_sim_years, species_list, eco_list, eco_species_ids, loss_params, scoring, simb, spinup, rng, debug,
    search_tier, n_reps, n_output_plots, no_establishment, injection_cohorts, cycle_years, scale_seed, exec,
    val_splots, val_ref_soa, val_spdf_plts, val_site_sim_years, val_spinup_cohorts, val_injection_cohorts)
  _sobol_cands = isnothing(sobol_candidates_db) ? [] : load_sobol_candidates(sobol_candidates_db; top_frac=sobol_top_frac)
  n_species = env.n_species
  _run(p; ws=env.work_soa) = run_candidate(env, p; ws)
  _fitness(rep_results) = candidate_fitness(env, rep_results)
  sobol_idx = Ref(1)
  next_candidate() =
    if sobol_idx[] <= length(_sobol_cands)
      p = _sobol_cands[sobol_idx[]]; sobol_idx[] += 1; p
    else
      generate_params(plugin.ctx, species_list, eco_list, eco_species_ids; rng=rng, no_establishment=no_establishment)
    end

  # ---- per-species-breadth archive (archive_by_sp) + Sobol-calibrated (ΣW,ΣAGB) bounds (bounds_by_sobol) ----
  # 3rd grid axis = breadth LEVEL 1..n_sp: a candidate non-dominated on b species fills its (W,AGB) column
  # up to level b (each cell keeps min-aggregate). breadth = #species whose running (W_s,AGB_s) Pareto
  # front the candidate would join. A broad+low-aggregate candidate wins many levels.
  n_sp = length(species_list)
  _sp_wa(eco_losses) = begin                          # per-global-species (ΣW_s, ΣAGB_s) over all eco blocks
    W = zeros(Float64, n_sp); A = zeros(Float64, n_sp)
    @inbounds for el in eco_losses, s in 1:n_sp
      W[s] += Float64(el.sp_w_loss[s]); A[s] += Float64(el.sp_agb_loss[s])
    end
    (W, A)
  end
  _nd_on(front, w, a) = !any(p -> p[1] <= w && p[2] <= a && (p[1] < w || p[2] < a), front)
  _ins_front!(front, w, a) = (_nd_on(front, w, a) && (filter!(p -> !(w <= p[1] && a <= p[2] && (w < p[1] || a < p[2])), front); push!(front, (w, a))); nothing)
  inc_sp = isempty(scale_seed.dom) ? collect(1:n_sp) : findall(scale_seed.dom)   # dominance species (excl _H/_S etc.)
  n_inc = length(inc_sp)
  sp_fronts = [Tuple{Float64,Float64}[] for _ in 1:n_inc]   # one running (W_s,A_s) Pareto front per dominance species
  gwlo = 0.0; gwhi = 1.0; galo = 0.0; gahi = 1.0; _calibrated = false
  if (bounds_by_sobol || archive_by_sp) && !isnothing(sobol_candidates_db)
    cal = load_sobol_candidates(sobol_candidates_db; top_frac=top_seeds)
    if !isempty(cal)
      @info "CMA-MAE: calibrating (ΣW,ΣAGB) bounds from $(length(cal)) top-$(round(Int,100*top_seeds))% Sobol seeds…"
      Wg = Float64[]; Ag = Float64[]
      for cc in cal
        fxc, _, _ = _fitness(_run(cc)); mc = CMAMAE.measure(fxc)
        push!(Wg, Float64(mc[1])); push!(Ag, Float64(mc[2]))
      end
      gwlo, gwhi = minimum(Wg), max(maximum(Wg), minimum(Wg) + 1e-9)
      galo, gahi = minimum(Ag), max(maximum(Ag), minimum(Ag) + 1e-9)
      _calibrated = true
      @info "  bounds ΣW∈[$(round(gwlo,sigdigits=3)),$(round(gwhi,sigdigits=3))] ΣAGB∈[$(round(galo,sigdigits=3)),$(round(gahi,sigdigits=3))]"
    end
  end

  if isnothing(resume_from)
    bio_params = if !isnothing(start_from)
      @info "Seeding initial candidate from $start_from (fresh search_state)"
      p = _load_params_from_path(start_from, plugin.ctx)
      (p.SPECIES_LIST == species_list && p.ECO_LIST == eco_list) ||
        error("start_from params are incompatible with this run: their SPECIES_LIST/ECO_LIST differ from the loaded data.")
      p
    else
      next_candidate()
    end
    slots = space.slots
    fx0, _, _ = _fitness(_run(bio_params))
    mean0 = PU.params_to_u(bio_params, space, slots)
    rep = MOLBSA.MOCandidate(bio_params, fx0)
    # auto-scale the 2-D measure grid from the seed candidate's loss (worse than any fit we'll keep),
    # so the archive cells bracket the reachable (Wasserstein, AGB) range; quality ceiling t0 above it.
    m0 = CMAMAE.measure(fx0)
    if !_calibrated                                   # no Sobol bounds → fall back to 2× the seed's measure
      gwlo, galo = 0.0, 0.0
      gwhi = max(2.0 * Float64(m0[1]), 1e-9); gahi = max(2.0 * Float64(m0[2]), 1e-9)
    end
    t0 = 1.5 * convert(Float64, fx0.aggregate)
    groups = (single_cov ? Vector{Int}[collect(1:length(slots))] : PU.build_groups(space, slots))   # block-diagonal CMA-MAE emitter
    @info "CMA-MAE block-diagonal: $(length(groups)) covariance blocks (sizes $(length.(groups)))"
    if archive_by_sp
      grid_dims_v = [cmame_grid, cmame_grid, n_inc]   # (ΣW, ΣAGB, breadth-level 1..n_inc over dominance species)
      meas_lo_v = [0.0, 0.0, 0.0]; meas_hi_v = [1.0, 1.0, 1.0]   # position normalized in the tell loop; level∈[0,1)
      @info "CMA-MAE per-species-breadth archive: $(cmame_grid)×$(cmame_grid)×$(n_inc) (ΣW,ΣAGB,breadth over $(n_inc) dominance species); ideal cell (1,1,$(n_inc))"
    else
      grid_dims_v = [cmame_grid, cmame_grid]
      meas_lo_v = [gwlo, galo]; meas_hi_v = [gwhi, gahi]
      @info "CMA-MAE grid $(cmame_grid)×$(cmame_grid) over ΣW∈[$(round(gwlo,sigdigits=3)),$(round(gwhi,sigdigits=3))] ΣAGB∈[$(round(galo,sigdigits=3)),$(round(gahi,sigdigits=3))], λ=auto, α=$(cmame_alpha)"
    end
    search_state = CMAMAE.CMAMAEMOState(mean0, cmaes_sigma0, rep, rng; meas_lo=meas_lo_v, meas_hi=meas_hi_v,
      grid_dims=grid_dims_v, lambda=cmaes_lambda, alpha=cmame_alpha, t0=t0, reseed_explore=cmame_reseed_explore,
      restart_patience=cmame_restart_patience, sobol_reseed=cmame_sobol_reseed, mo_rank=cmame_mo_rank, balanced=balanced_quality, blocks=groups, max_iter=typemax(Int))
    search_state.n_evals = 1
  else
    @info "Resuming from $resume_from"
    search_state = JLD2.load_object(resume_from)
    bio_params = search_state.representative.x
    slots = space.slots
  end
  search_state.engine.emitter.u_min_std = integer_handling ? PU.integer_u_min_std(space, slots, integer_std_factor) : Float64[]   # Hansen integer handling on the CMA-MAE emitter
  if TRIALS < 1 || search_state.n_evals >= TRIALS
    return search_state
  end

  # Validation mirrors the TRAINING tier so the held-out loss ranks candidates by the SAME objective (tier 5 ⇒ tier-5 val; tier 4/5 need
  # per-cycle bins built from the validation split) and, in dual modes, the same A⊕B objective.
  mo_val!(env; dual=true, tier=search_tier)
  loop = mo_loop_open(env, search_state; output_dir, resuming=!isnothing(resume_from))
  gio = loop.gio

  evals_done = search_state.n_evals
  est_gens = max(1, cld(TRIALS - evals_done, search_state.engine.emitter.lambda))
  run_search_loop(loop; get_state=() -> search_state, after_exit=() -> begin
    # Re-evaluate every archive elite on the held-out validation set so each carries train+val loss
    # (aligned to the saved archive order). train_loss is the elite's stored aggregate.
    if env.have_val
      try
        open(joinpath(output_dir, "archive_eval.csv"), "w") do io
          println(io, "candidate,train_loss,val_loss")
          for (ci, m) in enumerate(search_state.archive)
            vr = env.val_fit(m.x)
            println(io, "$ci,$(Float64(m.fx.aggregate)),$(_mo_val_loss(vr, eco_species_ids, loss_params, env.val_scales, env.val_dual_b))")
          end
        end
        @info "Wrote archive_eval.csv (train+val loss per archive elite)"
      catch e; @warn "archive validation eval failed" exception = (e, catch_backtrace()); end
    end
  end) do
      TProgress.@track for _gen in 1:est_gens
        evals_done >= TRIALS && break
        xs_u = CMAMAE.ask(search_state)
        λ = search_state.engine.emitter.lambda
        fxs = Vector{MOLBSA.MOFitness}(undef, λ)
        cands = Vector{MOLBSA.MOCandidate{typeof(bio_params)}}(undef, λ)
        meas = Vector{Tuple{Float64,Float64}}(undef, λ)
        sp_meas = Vector{Vector{Vector{Float64}}}(undef, λ)   # archive_by_sp: per-candidate breadth-column cells
        sp_qual = Vector{Vector{Float64}}(undef, λ)
        gen_best_agg = Inf
        local gb_run, gb_eco, gb_cached, gb_idx
        for k in 1:λ                                    # SERIAL (fit_params threads internally over sites)
          cand = PU.u_to_params(xs_u[k], space, slots, bio_params)
          rep_results = _run(cand)
          fx_k, run_k, eco_k = _fitness(rep_results)
          cands[k] = MOLBSA.MOCandidate(cand, fx_k)
          fxs[k] = fx_k
          evals_done += 1
          if archive_by_sp
            Ws, As = _sp_wa(eco_k)
            b = 0
            @inbounds for (p, s) in enumerate(inc_sp); _nd_on(sp_fronts[p], Ws[s], As[s]) && (b += 1); end   # breadth vs running fronts
            @inbounds for (p, s) in enumerate(inc_sp); _ins_front!(sp_fronts[p], Ws[s], As[s]); end          # then update the fronts
            gW = sum(Ws[s] for s in inc_sp); gA = sum(As[s] for s in inc_sp)   # position over the dominance species (matches measure())
            wn = clamp((gW - gwlo) / (gwhi - gwlo), 0.0, 1 - 1e-9); an = clamp((gA - galo) / (gahi - galo), 0.0, 1 - 1e-9)
            # cell quality: balanced (rescaled ΣW+ΣAGB ≡ wn+an, scale-fair) OR raw aggregate (legacy)
            aggq = balanced_quality ? (wn + an) : Float64(fx_k.aggregate)
            sp_meas[k] = [[wn, an, (j - 0.5) / n_inc] for j in 1:b]  # fill breadth-levels 1..b at this (W,AGB) cell
            sp_qual[k] = fill(aggq, b)                               # quality at every filled level
          else
            m = CMAMAE.measure(fx_k); meas[k] = (Float64(m[1]), Float64(m[2]))
          end
          if k == 1 || isless(fx_k.aggregate, gen_best_agg)   # k==1 ⇒ gb_* always defined (all-NaN gen); NaN ranks worst
            gen_best_agg = fx_k.aggregate; gb_idx = k; gb_run = run_k; gb_eco = eco_k; gb_cached = _median_rep_cached(rep_results, loss_params, scoring)
          end
        end

        is_new_best = archive_by_sp ? CMAMAE.tell_species_mo!(search_state, cands, sp_meas, sp_qual, xs_u) :
                      CMAMAE.tell_mo!(search_state, cands, meas, xs_u; nds=search.mocmaes_nds)
        search_state.n_evals = evals_done
        search_state.current = cands[gb_idx]

        mo_gen_finalize!(gio, search_state, search_state.engine.emitter.lambda, is_new_best, gb_run, gb_eco, gb_cached)   # shared: ckpt on ANY archive change (train+val)
      end
  end
  return search_state
end
