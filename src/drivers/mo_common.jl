# Multi-objective CMA-ES driver. Stands to parametrize_CMAES as parametrize_MOLBSA stands to
# parametrize_LBSA: same generational CMA-ES loop and u-space bridge, but each candidate is scored by
# the per-(eco,species,{w,agb}) objective VECTOR (MOLBSA.MOFitness via _mo_objectives), offspring are
# ranked multi-objectively (MOCMAES.mo_sortperm) for the distribution update, and "best" is a Pareto
# archive + min-aggregate representative (reusing start_mo_writer). TRIALS is the total eval budget.
# End-of-run CV dump: collapse the archive to best / extreme-W / extreme-AGB / knee, then evaluate each on the
# HELD-OUT val set using the LIVE (this fold's train-frozen) cell-norm / RANKW / count-balance metric — so the
# held-out loss is on the SAME objective the candidates were selected under. Writes per-fold cv_front_metrics.csv
# (rep × W/AGB/total), cv_front_plots.csv (per-plot sim vs ref AGB, for TOST) and cv_eco_map.csv.
function _dump_cv_front(search_state, output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
    val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params, scoring, simb,
    inj_val, val_dual_b, val_scales, fixed_seeds, val_splots, val_injection_cohorts, plugin)
  arch = collect(search_state.archive)
  isempty(arch) && (@warn "cv dump: empty archive"; return)
  wtot(c) = sum(@view c.fx.objectives[1:2:end]); atot(c) = sum(@view c.fx.objectives[2:2:end])   # ΣW (odd) / ΣAGB (even)
  best = argmin(c -> c.fx.aggregate, arch); eW = argmin(wtot, arch); eA = argmin(atot, arch)
  ws = wtot.(arch); as = atot.(arch)
  nrm(x, v) = (hi = maximum(v); lo = minimum(v); hi > lo ? (x - lo) / (hi - lo) : 0.0)
  p1 = (nrm(wtot(eW), ws), nrm(atot(eW), as)); p2 = (nrm(wtot(eA), ws), nrm(atot(eA), as))
  d12 = hypot(p2[1] - p1[1], p2[2] - p1[2])   # knee = max perpendicular distance from the eW–eA line (normalized space)
  perp(c) = d12 <= 0 ? 0.0 : abs((p2[1]-p1[1])*(p1[2]-nrm(atot(c),as)) - (p1[1]-nrm(wtot(c),ws))*(p2[2]-p1[2])) / d12
  knee = argmax(perp, arch)
  reps = [("best", best), ("extW", eW), ("extAGB", eA), ("knee", knee)]
  obs = combine(groupby(subset(val_splots, :sim_year => ByRow(>(0))), [:plot_id, :sim_year]), :agb_sum => sum => :ref_agb)
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(select(val_splots, [:plot_id, :eco_id]))))
  excl = (scoring.disturbance == :exclude_overwrite && !isnothing(val_injection_cohorts) && hasproperty(val_injection_cohorts, :disturbance_drop_pct)) ?
    Set((Int(r.plot_id), Int(r.sim_year)) for r in eachrow(val_injection_cohorts) if r.disturbance_drop_pct > 0) : Set{Tuple{Int,Int}}()
  mrows = NamedTuple[]; prows = NamedTuple[]
  for (kind, c) in reps
    rr = val_fit_params(c.x; val_ref_soa, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
      injection=inj_val, dual_b=val_dual_b, seeds=fixed_seeds, scales=val_scales, scoring, simb, plugin, collect_cohorts=true)
    run_result, eco_losses, idx = _agg_reps(rr, loss_params, scoring)
    objs = _mo_objectives(eco_losses, eco_species_ids, loss_params, val_scales, val_dual_b)
    push!(mrows, (rep_kind=kind, W=Float64(sum(objs[1:2:end])), AGB=Float64(sum(objs[2:2:end])), total=Float64(_mo_aggregate(objs, run_result, loss_params)), front_size=length(arch)))
    sim = DataFrame(plot_id=Int[], sim_year=Int[], agb=Float64[])
    for (pid, sy, _esp, _a, bio) in rr[idx][2]; sy > 0 && push!(sim, (Int(pid), Int(sy), Float64(bio))); end
    paired = innerjoin(obs, combine(groupby(sim, [:plot_id, :sim_year]), :agb => sum => :sim_agb), on=[:plot_id, :sim_year])
    for row in eachrow(paired)
      (Int(row.plot_id), Int(row.sim_year)) in excl && continue
      push!(prows, (rep_kind=kind, plot_id=Int(row.plot_id), sim_year=Int(row.sim_year), eco_id=get(plot2eco, Int(row.plot_id), 0), sim_agb=Float64(row.sim_agb), ref_agb=Float64(row.ref_agb)))
    end
  end
  CSV.write(joinpath(output_dir, "cv_front_metrics.csv"), DataFrame(mrows))
  CSV.write(joinpath(output_dir, "cv_front_plots.csv"), DataFrame(prows))
  CSV.write(joinpath(output_dir, "cv_eco_map.csv"), DataFrame(eco_id=collect(1:length(eco_list)), eco_label=eco_list))
  @info "cv dump: front held-out metrics ($(length(reps)) reps, $(length(arch))-member front) → $output_dir"
end

# Post-hoc reselection RANKED ON VALIDATION. Evaluate EVERY archive candidate (deduped by train-objective id
# across checkpoints) on the held-out val set → val (A_W,A_AGB) with frozen-train normalization. Rank each
# checkpoint's archive by its VALIDATION front area (shared-rectangle rule: min area, tie max knee, tie max
# count) → p100 = best-on-val archive; last = final checkpoint. Designate 5 positions {extreme_w, extreme_agb,
# knee, "median", best_aggregate} on each archive's VAL front. Writes cv_reselect_metrics.csv (10 rows, val + train objectives).
function _cv_reselect_dump(output_dir, eco_list, eco_species_ids, n_species, max_sim_year,
    val_ref_soa, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params, scoring, simb,
    inj_val, val_dual_b, val_scales, fixed_seeds, val_splots, species_list, plugin)
  ckpts = Tuple{Int,Any}[]
  for f in readdir(output_dir; join=true)
    m = match(r"search_state@(\d+)\.jld2$", f); isnothing(m) && continue
    push!(ckpts, (parse(Int, m.captures[1]), JLD2.load_object(f)))
  end
  isempty(ckpts) && (@warn "reselect: no checkpoints in $output_dir"; return)
  sort!(ckpts, by=first)
  wtot(c)=Float64(sum(@view c.fx.objectives[1:2:end])); atot(c)=Float64(sum(@view c.fx.objectives[2:2:end]))
  ckey(c)=Tuple(round.(Float64.(collect(c.fx.objectives)), digits=10))   # dedup id (non-dominated ⇒ distinct)
  val_obj(x) = begin
    rr = val_fit_params(x; val_ref_soa, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
      injection=inj_val, dual_b=val_dual_b, seeds=fixed_seeds, scales=val_scales, scoring, simb, plugin)
    _, eco_losses, _ = _agg_reps(rr, loss_params, scoring); objs = _mo_objectives(eco_losses, eco_species_ids, loss_params, val_scales, val_dual_b)
    (Float64(sum(objs[1:2:end])), Float64(sum(objs[2:2:end])))
  end
  # evaluate EVERY unique archive candidate on val once
  vcache = Dict{Any,Tuple{Float64,Float64}}()
  for (_, st) in ckpts, c in collect(st.archive)
    k = ckey(c); haskey(vcache, k) || (vcache[k] = val_obj(c.x))
  end
  allv = collect(values(vcache)); Wm = minimum(p[1] for p in allv); Am = minimum(p[2] for p in allv)
  vnd(arch) = begin                                                   # val-NON-DOMINATED members, (candidate,valpt), sorted by val W asc
    prs = [(c, vcache[ckey(c)]) for c in arch]
    nd = filter(p -> !any(q -> q[2][1] <= p[2][1] && q[2][2] <= p[2][2] && q[2] != p[2], prs), prs)
    sort(nd, by = p -> (p[2][1], -p[2][2]))
  end
  function area_pts(pts)                                               # shared-rectangle area on val points
    p = sort(pts, by=q->(q[1], -q[2])); p1=p[1]; a=(p1[1]-Wm)*(p1[2]-Am)
    for i in 1:length(p)-1; a += (p[i+1][1]-p[i][1])*((p[i][2]-Am)+(p[i+1][2]-Am))/2; end
    a
  end
  function knee_pts(pts)
    length(pts) < 3 && return 0.0
    ws=[q[1] for q in pts]; as=[q[2] for q in pts]
    nw(w)=(hi=maximum(ws);lo=minimum(ws); hi>lo ? (w-lo)/(hi-lo) : 0.0); na(a)=(hi=maximum(as);lo=minimum(as); hi>lo ? (a-lo)/(hi-lo) : 0.0)
    eW=pts[argmin(ws)]; eA=pts[argmin(as)]; p1=(nw(eW[1]),na(eW[2])); pk=(nw(eA[1]),na(eA[2])); d12=hypot(pk[1]-p1[1],pk[2]-p1[2])
    d12<=0 ? 0.0 : maximum(abs((pk[1]-p1[1])*(p1[2]-na(q[2])) - (p1[1]-nw(q[1]))*(pk[2]-p1[2]))/d12 for q in pts)
  end
  ranked = sort(ckpts, by = kv -> (pts=[p[2] for p in vnd(collect(kv[2].archive))]; (area_pts(pts), -knee_pts(pts), -length(pts))))
  p100_arch = collect(ranked[1][2].archive); p100_gen = ranked[1][1]
  last_arch  = collect(ckpts[end][2].archive); last_gen = ckpts[end][1]
  function positions(arch)                       # designate 5 positions on the archive's val-NON-DOMINATED front
    nd = vnd(arch); vp = [p[2] for p in nd]; cs = [p[1] for p in nd]
    ws=[q[1] for q in vp]; as=[q[2] for q in vp]; iW=argmin(ws); iA=argmin(as); iB=argmin(ws .+ as)   # extremes + best aggregate
    n = length(nd)
    n == 1 && return [("extreme_w",cs[1],vp[1]), ("extreme_agb",cs[1],vp[1]), ("knee",cs[1],vp[1]), ("median",cs[1],vp[1]), ("best_aggregate",cs[1],vp[1])]
    nw(w)=(hi=maximum(ws);lo=minimum(ws); hi>lo ? (w-lo)/(hi-lo) : 0.0); na(a)=(hi=maximum(as);lo=minimum(as); hi>lo ? (a-lo)/(hi-lo) : 0.0)
    p1=(nw(vp[iW][1]),na(vp[iW][2])); pk=(nw(vp[iA][1]),na(vp[iA][2])); d12=hypot(pk[1]-p1[1],pk[2]-p1[2])
    M=((p1[1]+pk[1])/2,(p1[2]+pk[2])/2)
    segi(i)=(qn=(nw(vp[i][1]),na(vp[i][2])); ABx=-M[1];ABy=-M[2];d2=ABx^2+ABy^2; t=d2<=0 ? 0.0 : clamp(((qn[1]-M[1])*ABx+(qn[2]-M[2])*ABy)/d2,0,1); hypot(qn[1]-(M[1]+t*ABx),qn[2]-(M[2]+t*ABy)))
    im = argmin(segi(i) for i in 1:n)                                     # "median" = closest to midpoint→corner line
    n == 2 && return [("extreme_w",cs[iW],vp[iW]), ("extreme_agb",cs[iA],vp[iA]), ("knee",cs[im],vp[im]), ("median",cs[im],vp[im]), ("best_aggregate",cs[iB],vp[iB])]  # knee=median with 2 pts
    perpi(i)= d12<=0 ? 0.0 : abs((pk[1]-p1[1])*(p1[2]-na(vp[i][2])) - (p1[1]-nw(vp[i][1]))*(pk[2]-p1[2]))/d12
    ik = argmax(perpi(i) for i in 1:n)
    [("extreme_w",cs[iW],vp[iW]), ("extreme_agb",cs[iA],vp[iA]), ("knee",cs[ik],vp[ik]), ("median",cs[im],vp[im]), ("best_aggregate",cs[iB],vp[iB])]
  end
  n_val = length(unique(val_splots.plot_id))
  rows = NamedTuple[]
  for (atype, arch, agen) in (("p100", p100_arch, p100_gen), ("last", last_arch, last_gen))
    for (nm, c, vp) in positions(arch)
      push!(rows, (archive=atype, position=nm, A_W=vp[1], A_AGB=vp[2],
                   A_W_train=wtot(c), A_AGB_train=atot(c), n_val=n_val, sel_gen=agen, front_size=length(arch)))
    end
  end
  # p101 = union non-dominated front over EVERY unique candidate across ALL checkpoints (on val). Its members are
  # REAL candidates (each from some checkpoint) — extracted to p101_candidates/candidate_<i>/params.jld2 (ordered
  # by val A_W) for the per-candidate scatter/sMAPE/TOST scripts. cv_p101_positions.csv maps the 5 positions →
  # candidate index. sel_gen=-1 marks the synthetic union front.
  seen101 = Set{Any}(); uniq101 = Any[]
  for (_, st) in ckpts, c in collect(st.archive)
    k = ckey(c); k in seen101 && continue; push!(seen101, k); push!(uniq101, c)
  end
  p101_nd = vnd(uniq101)
  idx_of = Dict(ckey(p[1]) => i for (i, p) in enumerate(p101_nd))
  p101dir = joinpath(output_dir, "p101_candidates"); isdir(p101dir) && rm(p101dir, recursive=true)
  for (i, (c, _vp)) in enumerate(p101_nd)
    d = joinpath(p101dir, "candidate_$i"); mkpath(d); JLD2.save_object(joinpath(d, "params.jld2"), c.x)
  end
  posrows = NamedTuple[]
  for (nm, c, vp) in positions(uniq101)
    ci = get(idx_of, ckey(c), 0)
    push!(rows, (archive="p101", position=nm, A_W=vp[1], A_AGB=vp[2],
                 A_W_train=wtot(c), A_AGB_train=atot(c), n_val=n_val, sel_gen=-1, front_size=length(p101_nd)))
    push!(posrows, (position=nm, candidate=ci, A_W=vp[1], A_AGB=vp[2], A_W_train=wtot(c), A_AGB_train=atot(c)))
  end
  CSV.write(joinpath(output_dir, "cv_p101_positions.csv"), DataFrame(posrows))
  @info "p101: $(length(p101_nd)) union non-dominated candidates → $p101dir  (5 positions → cv_p101_positions.csv)"
  CSV.write(joinpath(output_dir, "cv_reselect_metrics.csv"), DataFrame(rows))
  # every checkpoint's per-candidate val objectives (for the val-ranked percentile sweep plots)
  frows = NamedTuple[]
  for (g, st) in ckpts, c in collect(st.archive)
    vp = vcache[ckey(c)]; push!(frows, (gen=g, A_W=vp[1], A_AGB=vp[2], A_W_train=wtot(c), A_AGB_train=atot(c)))
  end
  CSV.write(joinpath(output_dir, "cv_val_fronts.csv"), DataFrame(frows))
  # Save the p100 front's 5 representative candidate params (JLD2 + JSON) so per-species sims are runnable post-hoc.
  for (nm, c, vp) in positions(p100_arch)
    try
      JLD2.save_object(joinpath(output_dir, "cv_p100_$(nm).jld2"), c.x)
      PU.save_json(joinpath(output_dir, "cv_p100_$(nm).json"), c.x)
    catch e; @warn "cv_p100 param save failed" position=nm exception=(e, catch_backtrace()); end
  end
  # Per-(plot, species, year) simulated vs observed AGB for the p100 front's 5 positions on the held-out set —
  # the backbone for per-species linear plots / MAPE / per-(species,stratum) TOST. esp is eco-LOCAL; the global
  # species is eco_species_ids[eco_id][esp]. Uses the SAME representative rep _agg_reps selects (matches objective).
  try
    obs = combine(groupby(filter(r -> r.sim_year > 0, val_splots),
      [:plot_id, :sim_year, :eco_species_id, :eco_id]), :agb_sum => sum => :obs_agb)
    prows = NamedTuple[]
    for (nm, c, _vp) in positions(p100_arch)
      rr = val_fit_params(c.x; val_ref_soa, max_sim_year, n_species, eco_species_ids, val_spdf_plts, val_site_sim_years, spinup, val_spinup_cohorts, loss_params,
      injection=inj_val, dual_b=val_dual_b, seeds=fixed_seeds, scales=val_scales, scoring, simb, plugin, collect_cohorts=true)
      av = _agg_reps(rr, loss_params, scoring); cached = rr[av[3]][2]
      simdf = DataFrame(plot_id=Int[], sim_year=Int[], eco_species_id=Int[], sim_agb=Float64[])
      for (pid, sy, esp, _age, bio) in cached
        sy > 0 && push!(simdf, (Int(pid), Int(sy), Int(esp), Float64(bio)))
      end
      sim_agg = combine(groupby(simdf, [:plot_id, :sim_year, :eco_species_id]), :sim_agb => sum => :sim_agb)
      paired = innerjoin(obs, sim_agg, on=[:plot_id, :sim_year, :eco_species_id])
      for r in eachrow(paired)
        gsp = eco_species_ids[r.eco_id][r.eco_species_id]
        push!(prows, (position=nm, plot_id=r.plot_id, sim_year=r.sim_year, eco_id=r.eco_id,
          eco_label=eco_list[r.eco_id], species=species_list[gsp], sim_agb=r.sim_agb, obs_agb=Float64(r.obs_agb)))
      end
    end
    CSV.write(joinpath(output_dir, "cv_p100_perspecies.csv"), DataFrame(prows))
    @info "cv_p100_perspecies.csv: $(length(prows)) (position,plot,species,year) rows on val"
  catch e
    @warn "per-species dump failed" exception = (e, catch_backtrace())
  end
  @info "cv reselect (VAL-ranked): p100=@$p100_gen (best val-area), last=@$last_gen, $(length(vcache)) unique cands scored on val, n_val=$n_val → $output_dir/cv_reselect_metrics.csv (+cv_val_fronts.csv, +cv_p100_*.jld2/json)"
end

# Order-invariant signature of a MO archive: hash of its members' objective vectors, sorted so archive
# ordering doesn't matter. Two archives with the same SET of objective points hash equal; ANY membership
# change (add / drop / swap) flips it. parametrize_MOCMAES checkpoints whenever this changes, so every
# distinct archive state is pullable for the CV reselection — not just representative improvements.
_mo_archive_sig(state) = hash(sort!([collect(c.fx.objectives) for c in state.archive]))

# ── Shared per-generation checkpoint + val bookkeeping for EVERY MO search engine ────────────────────────
# MOCMAES / IgelMOCMAES / CMAMAE / MOLBSA all drive their per-generation output through `mo_gen_open` (once)
# + `mo_gen_finalize!` (every generation) so they behave identically:
#   • checkpoint search_state@N.jld2 + best_params@N.* on ANY change to the (train) Pareto archive — an add,
#     drop OR swap — not merely when the representative improves;
#   • when a val split exists, val-score every newly admitted archive member into cv_val_cache.csv (so every
#     checkpoint's val front is recoverable) and export best_val_params@N whenever the representative's val
#     loss improves;
#   • append the per-generation metrics.csv row (iteration,best_train_loss,best_val_loss,pop_size,archive_size).
# `pop_size` and the val-fit closure are the only per-engine differences and are passed in by each driver.
_vkey(c) = Tuple(round.(Float64.(collect(c.fx.objectives)), digits=10))          # dedup archive members by train objectives
_twt(c) = Float64(sum(@view c.fx.objectives[1:2:end]))                            # train W-half (Σ Wasserstein objectives)
_tat(c) = Float64(sum(@view c.fx.objectives[2:2:end]))                            # train AGB-half (Σ AGB objectives)
# val_cache key = the (Σ train-W, Σ train-AGB) pair — which IS what's persisted (cv_val_cache's first two columns)
# and what analyze_run matches on, so the cache round-trips through the CSV and can be reloaded on resume.
_rkey(c) = (round(_twt(c), digits=10), round(_tat(c), digits=10))

mutable struct MOGenIO
  output_dir::String
  writer_ch::Any
  losses_db::Any
  metrics_io::IO
  splots::Any
  eco_list::Vector{String}
  species_list::Vector{String}
  eco_species_ids::Vector{Vector{Int}}
  n_species::Int
  site_sim_years::Any
  loss_params::Any
  no_establishment::Bool
  n_reps::Int
  rng::Any
  have_val::Bool
  n_output_plots::Int
  sampled_ids::Any
  sampled_ids_val::Any
  emp_sample::Any
  emp_sample_val::Any
  val_scales::Any                   # LossScales of the val reference (RANKW = the train one); nothing when !have_val
  val_dual_b::Any                   # val Sim B (carries its own .scales) or nothing
  val_fit::Any                      # x -> (run_result, cached_sites, eco_losses) on the val split (nothing when !have_val)
  plugin::PluginConfig              # the run's plugin (diagnostic simulate_and_test)
  exec::ExecOptions                 # execution options of the run (store_val_obj, newbest_stats, parallel_mode)
  timer::Union{Nothing,RunTimer}    # io-time accounting of the writer hand-offs (nothing ⇒ timing off)
  last_ckpt_sig::UInt
  val_cache::Dict{Any,NTuple{4,Float64}}
  rep_val_loss::Float64
  rep_val_best::Float64
end

_persist_val_cache!(gio::MOGenIO) = try
  vs = collect(values(gio.val_cache))
  isempty(vs) || CSV.write(joinpath(gio.output_dir, "cv_val_cache.csv"),
    DataFrame(A_W_train=[v[1] for v in vs], A_AGB_train=[v[2] for v in vs], A_W=[v[3] for v in vs], A_AGB=[v[4] for v in vs]))
catch e; @warn "persist val cache failed" exception=(e, catch_backtrace()); end

# Reload a persisted cv_val_cache.csv into the in-memory cache on RESUME (else _persist_val_cache!'s overwrite would
# drop every pre-resume entry → the val curve would only start at the resume generation). Keyed like _rkey.
function _load_val_cache(output_dir, losses_db=nothing)
  d = Dict{Any,NTuple{4,Float64}}(); f = joinpath(output_dir, "cv_val_cache.csv")
  if isfile(f)
    try
      for r in CSV.File(f)
        d[(round(Float64(r.A_W_train), digits=10), round(Float64(r.A_AGB_train), digits=10))] =
          (Float64(r.A_W_train), Float64(r.A_AGB_train), Float64(r.A_W), Float64(r.A_AGB))
      end
      @info "Loaded $(length(d)) val-cache entries from cv_val_cache.csv (resume)"
    catch e; @warn "load val cache failed" exception=(e, catch_backtrace()); end
  end
  if isempty(d) && losses_db !== nothing       # CSV missing/empty → recover from the losses.duckdb mirror
    try
      for r in DuckDB.execute(losses_db, "SELECT A_W_train, A_AGB_train, A_W, A_AGB FROM val_cache")
        d[(round(Float64(r.A_W_train), digits=10), round(Float64(r.A_AGB_train), digits=10))] =
          (Float64(r.A_W_train), Float64(r.A_AGB_train), Float64(r.A_W), Float64(r.A_AGB))
      end
      isempty(d) || @info "Recovered $(length(d)) val-cache entries from losses.duckdb (cv_val_cache.csv absent)"
    catch e; @warn "val_cache DB recover failed (table may not exist yet)" exception=(e, catch_backtrace()); end
  end
  d
end

# Open metrics.csv (append on resume), seed rep val-loss + best_val_params@0 from the initial representative,
# and stamp the initial archive signature. `state` is the fully-built search_state; `val_fit` may be nothing.
function mo_gen_open(state; output_dir, writer_ch, losses_db, resuming::Bool, splots, eco_list, species_list,
    eco_species_ids, n_species, site_sim_years, loss_params, no_establishment, n_reps, rng, have_val,
    n_output_plots, sampled_ids, sampled_ids_val, emp_sample, emp_sample_val, val_fit, plugin::PluginConfig, exec::ExecOptions=ExecOptions(), timer::Union{Nothing,RunTimer}=nothing, val_scales=nothing, val_dual_b=nothing)
  metrics_io = open(joinpath(output_dir, "metrics.csv"), resuming ? "a" : "w")
  resuming || println(metrics_io, "iteration,best_train_loss,best_val_loss,pop_size,archive_size")
  gio = MOGenIO(output_dir, writer_ch, losses_db, metrics_io, splots, eco_list, species_list, eco_species_ids,
    n_species, site_sim_years, loss_params, no_establishment, n_reps, rng, have_val, n_output_plots,
    sampled_ids, sampled_ids_val, emp_sample, emp_sample_val, val_scales, val_dual_b, val_fit, plugin, exec, timer, _mo_archive_sig(state),
    resuming ? _load_val_cache(output_dir, losses_db) : Dict{Any,NTuple{4,Float64}}(), Inf, Inf)   # RESUME: keep the pre-resume val cache (CSV, or losses.duckdb mirror if the CSV is gone)
  if have_val && val_fit !== nothing
    try
      _r, _c, _el = val_fit(state.representative.x)
      gio.rep_val_loss = _mo_val_loss((_r, nothing, _el), eco_species_ids, loss_params, val_scales, val_dual_b)
      gio.rep_val_best = gio.rep_val_loss
      if isfinite(gio.rep_val_best)
        mkpath(output_dir)
        PU.save_json(joinpath(output_dir, "best_val_params@0.json"), state.representative.x)
        JLD2.save_object(joinpath(output_dir, "best_val_params@0.jld2"), state.representative.x)
      end
    catch e; @warn "initial val-loss init failed" exception=(e, catch_backtrace()); end
  end
  return gio
end

# The per-generation tail every MO engine shares. `gb_run/gb_eco/gb_cached` describe this generation's best
# offspring (for losses.duckdb + the train sim sample). Returns `save_ckpt`.
function mo_gen_finalize!(gio::MOGenIO, state, pop_size::Int, is_new_best::Bool, gb_run, gb_eco, gb_cached)
  _sig = _mo_archive_sig(state)
  save_ckpt = _sig != gio.last_ckpt_sig               # ANY archive change (add/drop/swap); improved ⟹ changed
  save_ckpt && (gio.last_ckpt_sig = _sig)
  archive_changed = save_ckpt
  _newvals = NTuple{4,Float64}[]         # val entries scored THIS gen → the writer mirrors them into losses.duckdb.val_cache
  if gio.have_val && gio.exec.store_val_obj && (archive_changed || is_new_best)   # val-score any NEW archive members
    _newc = [c for c in collect(state.archive) if !haskey(gio.val_cache, _rkey(c))]
    _vscore = c -> try
        _r, _c, _el = gio.val_fit(c.x)
        _o = _mo_objectives(_el, gio.eco_species_ids, gio.loss_params, gio.val_scales, gio.val_dual_b)
        (_rkey(c), (_twt(c), _tat(c), Float64(sum(@view _o[1:2:end])), Float64(sum(@view _o[2:2:end]))))
      catch e; @warn "val score failed" exception=(e, catch_backtrace()); nothing; end
    if candidate_parallel(gio.exec)      # score new members concurrently (each val_fit is sites-serial); store after
      _vout = Vector{Any}(undef, length(_newc))
      if !isempty(_newc)
        _vout[1] = _vscore(_newc[1])      # the first member serially, the rest concurrently (the val scales are explicit, nothing to race on)
        Threads.@threads :static for i in 2:length(_newc); _vout[i] = _vscore(_newc[i]); end
      end
      for r in _vout; r === nothing || (gio.val_cache[r[1]] = r[2]; push!(_newvals, r[2])); end
    else
      for c in _newc; r = _vscore(c); r === nothing || (gio.val_cache[r[1]] = r[2]; push!(_newvals, r[2])); end
    end
  end
  val_sim_sample = nothing
  _losses_payload = nothing            # snapshot of per-new-best losses; the DB write is offloaded to the writer
  if is_new_best
    iter = state.best_iteration
    total = Float64(state.representative.fx.aggregate)
    @info "New best @ gen $iter | agg=$total | archive=$(length(state.archive))"
    if gio.exec.newbest_stats   # opt-in diagnostic only (PAN_NEWBEST_STATS=1) — otherwise skip this whole extra train sim
      try
        test_df = simulate_and_test(; splots=gio.splots, bio_params=state.representative.x, eco_list=gio.eco_list,
          species_list=gio.species_list, eco_species_ids=gio.eco_species_ids, loss_params=gio.loss_params,
          site_sim_years=gio.site_sim_years, M=gio.n_reps, no_establishment=gio.no_establishment, rng=_diag_rng(UInt64[gio.n_reps], :train), plugin=gio.plugin)
        println("Train stats:"); show(test_df; allrows=true, allcols=true); println()
      catch e; @warn "simulate_and_test (train) failed" exception=(e, catch_backtrace()); end
    end
    if gio.have_val && gio.val_fit !== nothing
      try
        # The representative is an archive member, so ExecOptions.store_val_obj already val-scored it into val_cache. Under
        # cell-norm the aggregate is Σobjs = val_W + val_AGB, so reuse the cached score and SKIP the redundant sim —
        # unless we need the val plot sample (n_output_plots>0) or can't reconstruct (non-cell-norm): then re-sim.
        _rk = _rkey(state.representative)
        if gio.loss_params.cell_norm && gio.n_output_plots == 0 && haskey(gio.val_cache, _rk)
          _cv = gio.val_cache[_rk]; gio.rep_val_loss = _cv[3] + _cv[4]
          @info "Val loss @ gen $iter | loss=$(gio.rep_val_loss) (cached)"
        else
          _r, _c, _el = gio.val_fit(state.representative.x)
          gio.rep_val_loss = _mo_val_loss((_r, nothing, _el), gio.eco_species_ids, gio.loss_params, gio.val_scales, gio.val_dual_b)
          @info "Val loss @ gen $iter | loss=$(gio.rep_val_loss)"
          val_sim_sample = gio.n_output_plots > 0 ? _filter_cached_to_df(_c, gio.sampled_ids_val) : nothing
        end
        if gio.rep_val_loss < gio.rep_val_best
          gio.rep_val_best = gio.rep_val_loss
          try
            mkpath(gio.output_dir)
            PU.save_json(joinpath(gio.output_dir, "best_val_params@$(state.i).json"), state.representative.x)
            JLD2.save_object(joinpath(gio.output_dir, "best_val_params@$(state.i).jld2"), state.representative.x)
            @info "New best VAL @ gen $(state.i) | val=$(gio.rep_val_loss) train=$total → best_val_params@$(state.i)"
          catch e; @warn "best_val checkpoint failed" exception=(e, catch_backtrace()); end
        end
      catch e; @warn "val fit_params failed" exception=(e, catch_backtrace()); end
    end
    # snapshot the losses as plain values on the MAIN thread; the batched DB write runs on the writer thread
    _blob = let buf = IOBuffer(); Serialization.serialize(buf, state.representative.x); take!(buf) end
    _losses_payload = (iter=iter, ns=gb_run.num_sites, no=gb_run.num_obs, total=total, arch=length(state.archive), blob=_blob,
      eco=[(name=_eco_name(gio.eco_list, eco_id), ns=el.num_sites, no=el.num_obs, tot=convert(Float64, PU.get_total_loss(el, gio.loss_params)),
            sp=[(s=gio.species_list[gsp], w=el.sp_w_loss[gsp]/max(1, el.num_sites), a=el.sp_agb_loss[gsp]/max(1, el.num_sites))
                for gsp in 1:gio.n_species if el.sp_w_loss[gsp] != 0f0])
           for (eco_id, el) in enumerate(gb_eco)])
  end
  # OFFLOAD to the async writer: metrics line (every gen) + BATCHED losses insert (one transaction, new-best only).
  # No fsync on the search's critical path; drained/flushed on normal exit AND interrupt by stop_writer's finally.
  _mline = string(state.i, ",", Float64(state.representative.fx.aggregate), ",", isfinite(gio.rep_val_loss) ? gio.rep_val_loss : "", ",", pop_size, ",", length(state.archive))
  _lossjob = MOLossesJob(gio.losses_db, gio.metrics_io, _mline, _losses_payload, _newvals)
  timed_put!(gio.timer, gio.writer_ch, _lossjob)
  cached_sites_state_df = gb_cached === nothing ?    # nothing when n_output_plots==0 (candidate mode skips the best re-sim); the MO writer never reads this field anyway
    DataFrame(plot_id=Int[], sim_year=Int[], species_id=Int[], age=Int[], agb=Float64[]) :
    _cohorts_df(gb_cached)
  sim_sample = (is_new_best && gio.n_output_plots > 0) ? _filter_cached_to_df(gb_cached, gio.sampled_ids) : nothing
  # The async writer above persists search_state@N + best_params@N whenever save_ckpt is set, and
  # save_ckpt == archive_changed, so EVERY archive-changed state is already saved by the writer. (A former
  # synchronous save for the changed-but-not-improved case was redundant and raced the writer on the same
  # path — removed.)
  _ckptjob = WriterJob(save_ckpt, deepcopy(state), gio.splots, cached_sites_state_df, gio.emp_sample, sim_sample, is_new_best ? gio.emp_sample_val : nothing, val_sim_sample)  # deepcopy (CPU) built before the timed put! so io = channel BLOCK only
  timed_put!(gio.timer, gio.writer_ch, _ckptjob)
  (gio.have_val && gio.exec.store_val_obj && (is_new_best || archive_changed)) && _persist_val_cache!(gio)
  return save_ckpt
end

# Flatten the per-ecoregion SiteLoss breakdown into the multi-objective vector:
# for each (ecoregion, species-in-eco) two coordinates — the age-distribution
# (Wasserstein) loss and the AGB-level loss. Order is fixed across candidates, so
# the vector is coordinate-comparable in MOLBSA.mo_delta / dominance.
# Eco name for a per-eco loss index. Tier 5 returns 2·n_eco losses (tier-3 block then tier-4 block);
# indices past n_eco name the tier-4 copy with a suffix so per-eco logs stay unique.
function _eco_name(eco_list, idx::Integer)   # per-eco loss vectors can be k·n_eco long (tier-5: 2×; dual: 3×)
  ne = length(eco_list); blk = (idx - 1) ÷ ne; base = eco_list[(idx - 1) % ne + 1]
  blk == 0 ? base : base * "·blk$blk"
end

# Global-species mask for DOMINANCE/diversity: which global species enter the MO objective vector (and
# thus Pareto dominance in MO-CMA-ES/MOLBSA/Igel, the CMA-MAE measure descriptor, the per-species breadth axis and the RANKW weights).
# Empty mask = all species. Built once by parametrize() from the `dominance_species` flag (exact = SPCD only · exact+grp = + _GRP groups ·
# all = incl. _H/_S catch-alls) and carried in `LossScales.dom`. The scalar aggregate (get_total_loss) is unaffected — _H/_S still count
# toward overall fit, just not toward dominance.
function dominance_mask(species_list, mode::AbstractString)::Vector{Bool}
  lowercase(mode) == "all" && return Bool[]
  m = Bool[_dom_include_name(sp, lowercase(mode)) for sp in species_list]
  return any(m) ? m : Bool[]            # an empty selection means "all species" (as the former empty Set did)
end
@inline _dom_included(dom::Vector{Bool}, gsp::Int) = isempty(dom) || (gsp <= length(dom) && @inbounds dom[gsp])
# Decide a species name's membership for a dominance_species mode.
_dom_include_name(name::AbstractString, mode::AbstractString) =
  mode == "all" ? true :
  startswith(name, "_GRP") ? (mode == "exact+grp") :
  startswith(name, "_") ? false :          # _H, _S (and other "_" catch-alls) — excluded unless "all"
  true                                      # SPCD exact species

# Safe per-cell table lookup: 0 when the matrix isn't sized for this cell (e.g. an empty A table under a
# B-only dual) → the caller then falls back to the raw (un-normalized) term instead of indexing out of range.
# per-(species, eco) entry of a scale / rank-weight matrix. A matrix that does not cover the cell means the scales were not
# set for this reference (B11: this used to return 0, silently mixing normalised and raw terms or zeroing the objective).
@inline function _cell(M::Matrix{FloatType}, gsp::Int, e::Int)::FloatType
  (size(M, 1) >= gsp && size(M, 2) >= e) || _cell_error(M, gsp, e)
  @inbounds M[gsp, e]
end
@noinline _cell_error(M, gsp, e) = error("loss scale/weight matrix of size $(size(M)) does not cover (species=$gsp, eco=$e): " *
                                         "the cell scales / RANKW were not set for this reference (call _set_loss_scales! first)")   # (wording kept: the golden status.txt of the tier-5 scenario records it)

# The one accumulation of the MO objective vector. `target_gsp == 0` sums every dominance species; `target_gsp > 0` restricts the inner loop to that
# global species (CC-Igel specialization: still summed over ecoregions / dual groups).
function _mo_accumulate(eco_losses::Vector{PU.SiteLoss}, eco_species_ids::Vector{Vector{Int}}, lp::PU.LossParams, scales::PU.LossScales, dual_b, target_gsp::Int)::Vector{FloatType}
  # 4 GROUPED objectives for the dual: [ΣA_W, ΣA_AGB, ΣB_W, ΣB_AGB] — sum each over its eco-blocks and
  # dominance species. eco_losses = [A's n_eco blocks ; B's n_eco blocks] (2·n_eco ⇒ ngroups=2 ⇒ 4 obj);
  # a single sim (n_eco blocks) ⇒ ngroups=1 ⇒ 2 obj [ΣW, ΣAGB]. Keeping the MO vector low-dimensional
  # makes A↔B and W↔AGB first-class objectives (clean 4-D Pareto / net-win) instead of diluting them
  # across ~n_ess per-species entries. Species breadth is handled POST-HOC (the sweep), not here.
  # `scales` are those of the primary reference and `dual_b.scales` Sim B's (see _group_scales); each block is normalised by the scales of the
  # reference it was scored on, and RANKW is the train one carried by both.
  ne = length(eco_species_ids)
  ngroups = max(1, length(eco_losses) ÷ ne)
  objs = zeros(FloatType, 2 * ngroups)
  cell = lp.cell_norm
  sc0, sc1 = _group_scales(scales, dual_b)
  dom = scales.dom                          # dominance species mask (empty = all), a field of the reference scales: allocation-free
  for (idx, el) in enumerate(eco_losses)
    g = (idx - 1) ÷ ne                       # 0-based group: 0 = Sim A (tier-3), 1 = Sim B (tier-4)
    e = (idx - 1) % ne + 1
    if cell
      # Per-cell: divide each (eco,lu,sp) cell by its OWN reference scale, rescale by the AGB-rank weight,
      # then sum into the group. A and B keep separate scales; RANKW is shared. NOTE: the AGB exponent is
      # applied ONCE, per-cohort in the species loss (_agb_pow on each hinged excess) — do NOT
      # re-apply it here or it double-powers for p≠1 (W's power stays here since W is not pre-powered).
      sc = g == 0 ? sc0 : sc1
      Ws = sc.w_cell; As = sc.agb_cell; Rw = sc.rankw
      @inbounds for gsp in eco_species_ids[e]
        (target_gsp == 0 || gsp == target_gsp) || continue
        _dom_included(dom, gsp) || continue
        rw = _cell(Rw, gsp, e); ws = _cell(Ws, gsp, e); as = _cell(As, gsp, e)
        objs[2 * g + 1] += rw * PU._w_pow(lp, ws > 0 ? el.sp_w_loss[gsp] / ws : el.sp_w_loss[gsp])
        objs[2 * g + 2] += rw * (as > 0 ? el.sp_agb_loss[gsp] / as : el.sp_agb_loss[gsp])
      end
    else
      @inbounds for gsp in eco_species_ids[e]
        (target_gsp == 0 || gsp == target_gsp) || continue
        _dom_included(dom, gsp) || continue         # restrict to the selected dominance species
        objs[2 * g + 1] += el.sp_w_loss[gsp]   # ΣW   for this group
        objs[2 * g + 2] += el.sp_agb_loss[gsp] # ΣAGB for this group
      end
    end
  end
  return objs
end
_mo_objectives(eco_losses::Vector{PU.SiteLoss}, eco_species_ids::Vector{Vector{Int}}, lp::PU.LossParams, scales::PU.LossScales, dual_b=nothing)::Vector{FloatType} =
  _mo_accumulate(eco_losses, eco_species_ids, lp, scales, dual_b, 0)
# Per-species variant for CC-Igel specialization: the same accumulation restricted to `target_gsp`.
_mo_objectives_species(eco_losses::Vector{PU.SiteLoss}, eco_species_ids::Vector{Vector{Int}}, lp::PU.LossParams, scales::PU.LossScales, dual_b, target_gsp::Int)::Vector{FloatType} =
  _mo_accumulate(eco_losses, eco_species_ids, lp, scales, dual_b, target_gsp)

# Scalar aggregate paired with the MO objective vector (for the representative / archive eviction). With
# cell_norm the per-cell scales/rank live only in _mo_objectives, so the aggregate = Σ of the objectives;
# otherwise fall back to the established get_total_loss on the (dual-combined) SiteLoss.
@inline _mo_aggregate(objs::Vector{FloatType}, run_result, lp::PU.LossParams)::Float64 =
  lp.cell_norm ? Float64(sum(objs)) : convert(Float64, PU.get_total_loss(run_result, lp))

# Validation aggregate from a fit_params result tuple (run_result, cached, eco_losses), scored with the VAL reference's scales
# (`scales`/`val_dual_b.scales`; RANKW is the train one inside them), so val stays comparable to the train aggregate.
@inline _mo_val_loss(val_result, eco_species_ids, lp::PU.LossParams, scales::PU.LossScales, dual_b=nothing)::Float64 =
  _mo_aggregate(_mo_objectives(val_result[3], eco_species_ids, lp, scales, dual_b), val_result[1], lp)

# Per-generation losses/metrics write, OFFLOADED to the writer thread so the search never blocks on I/O.
# `payload` (a NamedTuple, only on a new best) is inserted into losses.duckdb in ONE transaction — a single
# fsync instead of ~1+n_eco+n_eco·n_species autocommits (the cluster-vs-laptop slowdown). `metrics_line` is
# written every generation. Drained/flushed on normal exit AND interrupt via stop_writer's finally.
struct MOLossesJob
  db::Any
  io::IO
  metrics_line::String
  payload::Any            # NamedTuple(iter, ns, no, total, arch, blob, eco[...]) on new-best; else nothing
  val_rows::Vector{NTuple{4,Float64}}   # (A_W_train,A_AGB_train,A_W,A_AGB) scored this gen → mirrored into losses.duckdb.val_cache
end

function _mo_write_losses!(job::MOLossesJob)
  p = job.payload
  if p !== nothing
    try
      DuckDB.execute(job.db, "BEGIN TRANSACTION")
      DuckDB.execute(job.db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?, ?)", [p.iter, p.ns, p.no, p.total, p.arch, p.blob])
      for er in p.eco
        DuckDB.execute(job.db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [p.iter, er.name, er.ns, er.no, er.tot])
        for sr in er.sp
          DuckDB.execute(job.db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [p.iter, er.name, sr.s, sr.w, sr.a])
        end
      end
      DuckDB.execute(job.db, "COMMIT")
    catch e
      try; DuckDB.execute(job.db, "ROLLBACK"); catch; end
      @error "mo_writer: losses insert failed" exception = (e, catch_backtrace())
    end
  end
  if !isempty(job.val_rows)              # durable mirror of cv_val_cache into losses.duckdb (survives resumes / a lost CSV)
    try
      DuckDB.execute(job.db, "CREATE TABLE IF NOT EXISTS val_cache (A_W_train DOUBLE, A_AGB_train DOUBLE, A_W DOUBLE, A_AGB DOUBLE, PRIMARY KEY (A_W_train, A_AGB_train))")
      DuckDB.execute(job.db, "BEGIN TRANSACTION")
      for v in job.val_rows
        DuckDB.execute(job.db, "INSERT INTO val_cache VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING", [v[1], v[2], v[3], v[4]])
      end
      DuckDB.execute(job.db, "COMMIT")
    catch e
      try; DuckDB.execute(job.db, "ROLLBACK"); catch; end
      @error "mo_writer: val_cache insert failed" exception = (e, catch_backtrace())
    end
  end
  try; println(job.io, job.metrics_line); flush(job.io); catch e; @error "mo_writer: metrics write failed" exception = (e, catch_backtrace()); end
end

# Background writer for MOLBSA: checkpoints the full state (archive included) and
# the representative params, and generates plots, off the search thread. Mirrors
# start_writer but reads `state.representative` instead of `state.best`.
function start_mo_writer(::Type{State}, output_dir::AbstractString; buffer_size::Int=8) where {State}
  ch = Channel{Union{WriterJob{State},MOLossesJob,Symbol}}(buffer_size)
  task = Threads.@spawn begin
    try
      for job in ch
        job === STOP && break
        job isa MOLossesJob && (_mo_write_losses!(job); continue)   # batched losses + metrics, off the search thread
        state = job.state
        if job.save_ckpt
          try
            @info "Checkpoint (archive changed): agg=$(state.representative.fx.aggregate)" iter = state.i archive = length(state.archive)
            mkpath(output_dir)
            JLD2.save_object(joinpath(output_dir, "search_state@$(state.i).jld2"), state)
            PU.save_json(joinpath(output_dir, "best_params@$(state.i).json"), state.representative.x)
            JLD2.save_object(joinpath(output_dir, "best_params@$(state.i).jld2"), state.representative.x)
          catch e
            @error "mo_writer: save failed" iter = state.i exception = (e, catch_backtrace())
          end
          if !isnothing(job.emp_sample) && !isnothing(job.sim_sample)
            try
              generate_plots(job.emp_sample, job.sim_sample, "training_$(state.i)", state.representative.fx.aggregate, output_dir)
            catch e
              @error "mo_writer: generate_plots (train) failed" iter = state.i exception = (e, catch_backtrace())
            end
          end
          if !isnothing(job.emp_sample_val) && !isnothing(job.sim_sample_val)
            try
              generate_plots(job.emp_sample_val, job.sim_sample_val, "validation_$(state.i)", state.representative.fx.aggregate, output_dir)
            catch e
              @error "mo_writer: generate_plots (val) failed" iter = state.i exception = (e, catch_backtrace())
            end
          end
        else
          @info ("Best@$(state.best_iteration): agg=$(state.representative.fx.aggregate), archive=$(length(state.archive)), Avg diff: $(state.diff_avg), Temp: $(state.t), Prob: $(state.prob_avg)")
        end
      end
    catch e
      @error "mo_writer: fatal" exception = (e, catch_backtrace())
      rethrow()
    end
  end
  return ch, task
end
