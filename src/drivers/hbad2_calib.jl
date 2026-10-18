# Search driver of the HBAD2 calibration (docs/hbad2_calibration.md). `succession_plugin: hbad2` with a `hbad2_calib:` block calibrates the plugin on the remeasurement-interval
# product instead of running the Biomass-Succession style fit_params machinery: the objective is `HBAD2Calib.scalar_objective` (Sim A joint NLL, or the augmented Chebyshev
# scalarisation of Sim A and Sim B), the engine `Search.PlainCMA` (the prototype's CMA-ES) for `search_mode: cmaes`, and the life cycle (background checkpoint writer,
# losses.duckdb, interrupt handling, exit checkpoint, resume) is the shared `run_search_loop`.

import Printf: @sprintf
const HC = HBAD2Calib

"A path of a config value: absolute, or relative to the working directory when it exists there, else relative to the repository root."
function _repo_path(p::AbstractString)
  isempty(p) && return ""
  (isabspath(p) || ispath(p)) && return String(p)
  q = joinpath(pkgdir(@__MODULE__), p)
  return ispath(q) ? q : String(p)
end

"True when `cfg` asks for an HBAD2 calibration (`succession_plugin: hbad2` and a `hbad2_calib:` block)."
is_hbad2_calibration(cfg) = lowercase(cfg["succession_plugin"]) in ("hbad2", "hbad-v2") && isgiven(cfg, "hbad2_calib")

"""
    hbad2_calib_problem(cfg; output_dir) -> HBAD2Calib.CalibProblem

The calibration problem of a (validated) config: split (from `hbad2_calib.split_csv`, else drawn by plots from `val_frac` / `test_frac` / `split_seed`; written to `<output_dir>/split_plots.csv`),
interval product (Arrow cache), data sets and parameter layout. `lambda` selects the scalarisation weight (a joint run with several weights is one search per weight, see `run_hbad2_calibration`).
"""
function hbad2_calib_problem(cfg; output_dir::AbstractString, lambda::Real=first(cfg["hbad2_calib"]["lambda"]), sets::AbstractVector{Symbol}=Symbol[])
  hc = cfg["hbad2_calib"]
  db = cfg["cohorts_db_path"]
  states = hc["states"]
  split = if isempty(hc["split_csv"])
    cfg["val_frac"] + cfg["test_frac"] > 0 || error("hbad2_calib: no split_csv and val_frac + test_frac = 0: nothing to validate on; give a split file or set val_frac")
    Data.draw_plot_split(Data.list_hbad2_plots(db; states=states, table=hc["trees_table"]); val_frac=cfg["val_frac"], test_frac=cfg["test_frac"], seed=cfg["split_seed"])
  else
    Data.read_split_plots(_repo_path(hc["split_csv"]))
  end
  mkpath(output_dir)
  CSV.write(joinpath(output_dir, "split_plots.csv"), split)
  sw = HC.CalibSwitches(; form=hc["form"], size_split=hc["size_split"], site=hc["site"], vigour=hc["vigour"], sigv_max=hc["sigv_max"], birth_w=hc["birth_w"],
                        min_sp_trees=hc["min_sp_trees"], sizecls=hc["sizecls"], bounds=hc["bounds"])
  cache = isempty(hc["intervals_cache"]) ? joinpath(output_dir, "hbad2_intervals") : hc["intervals_cache"]
  objective = Symbol(hc["objective"])
  sets_ = isempty(sets) ? (hc["test_set"] ? [:train, :val, :test] : [:train, :val]) : sets
  has_val = any(==("val"), split.set)
  has_val || (sets_ = filter(!=(:val), sets_)); any(==("test"), split.set) || (sets_ = filter(!=(:test), sets_))
  return HC.build_calib_problem(; db_path=db, split=split, intervals_cache=cache, sw=sw, table=hc["trees_table"], states=states, rebuild=hc["rebuild_intervals"],
    longevity_csv=_repo_path(hc["longevity_csv"]), recruitment_csv=_repo_path(hc["recruitment_csv"]), priors_csv=_repo_path(hc["priors_csv"]),
    start_theta=_repo_path(hc["start_theta"]), start_recr=_repo_path(hc["start_recr"]), recr=Symbol(hc["recruitment"]), freeze=hc["freeze"], objective=objective,
    fb=Symbol(hc["fb"]), sets=sets_, lambda=lambda, rho=hc["rho"], ideal=hc["ideal"], nadir=hc["nadir"], simb_sigma=hc["simb_sigma"])
end

_fmt_scores(a) = @sprintf("total %.1f (mort %.1f D %.1f H %.1f sigD %.4f sigH %.4f)", a.total, a.mort, a.D, a.H, a.sigD, a.sigH)

"Log the TRAIN / VAL(/TEST) values of both objectives at the search vector `θ` (fA always; fB for a joint problem)."
function hbad2_report(prob::HC.CalibProblem, θ::AbstractVector, label::AbstractString; io=stdout)
  for st in sort(collect(keys(prob.A)); by=s -> (s == :train ? 0 : s == :val ? 1 : 2))
    a = HC.simA_scores(prob, θ, st)
    line = @sprintf("%-6s %-5s fA %s", label, st, _fmt_scores(a))
    if haskey(prob.B, st)
      b = HC.simB_scores(prob, θ, st)
      line *= @sprintf(" | fB(%s) %.1f [counts %.1f size %.1f species %.1f H %.1f plotstand %.1f dropped %d]", prob.fb, HC.fb_value(b, prob.fb), b.counts, b.size, b.species, b.H, b.plotstand, b.dropped)
    end
    println(io, line)
  end
  return nothing
end

"""
    hbad2_cma_search(prob; output_dir, trials, sigma0, lambda, seed, stall, ftol, ckpt_every, resume_from=nothing, label="") -> PlainCMA.PlainCMASearch

CMA-ES (the prototype's, `Search.PlainCMA`) on the searched coordinates of `prob` through `run_search_loop`: new best / every `ckpt_every` generations the writer saves
`search_state@<gen>.jld2` + `best_params@<gen>.{json,jld2}` and the plugin-ready `best/` parameter directory is rewritten; `losses.duckdb` gets a `total_loss` row per new best; the exit
checkpoint is `search_state@<gen>.jld2` + `search_state_latest.jld2`. `resume_from` = a `search_state@N.jld2` (or the latest link): the run continues exactly as the uninterrupted one
(the evaluation budget `trials` may be raised).
"""
function hbad2_cma_search(prob::HC.CalibProblem; output_dir::AbstractString, trials::Int, sigma0::Real, lambda::Union{Nothing,Int}, seed::Int, stall::Int, ftol::Real,
                          ckpt_every::Int, resume_from::Union{Nothing,AbstractString}=nothing, label::AbstractString="", log_every::Int=25)
  mkpath(output_dir)
  L = prob.layout
  f = z -> HC.scalar_objective(prob, z)
  search = if resume_from === nothing
    x0 = HC.restrict(L, prob.θ0)
    f0 = f(x0)
    lam = lambda === nothing ? 4 + floor(Int, 3log(length(x0))) : lambda
    PlainCMA.PlainCMASearch(PlainCMA.PlainCMAState(x0, f0, sigma0; lambda=lam, maxevals=trials, rng=Random.Xoshiro(seed), ftol=ftol, stall=stall))
  else
    @info "Resuming from $resume_from"
    s = PlainCMA.load_state(PlainCMA.PlainCMASearch, resume_from)
    s.st.maxevals = trials
    s
  end
  st = search.st
  @info "HBAD2 calibration" label dims = length(st.m) lambda = st.lambda evals = st.evals trials objective = String(prob.objective) searched = length(L.free_idx) frozen = length(L.frozen)
  if resume_from === nothing
    hbad2_report(prob, prob.θ0, "START")
    prob.objective == :joint && @info @sprintf("scalarisation: ideal fA %.1f fB %.1f | nadir fA %.1f fB %.1f | lambda %.2f rho %.2f", prob.cheb.ideal..., prob.cheb.nadir..., prob.cheb.lambda, prob.cheb.rho)
  end
  loop, losses_db = scalar_loop_open(st; output_dir)
  writer_ch = loop.writer_ch
  ntr = prob.A[:train]; nsites = ntr.nsub_sites; nobs = sum(ntr.n1)
  empty_df = DataFrame()
  write_best() = HC.write_params(joinpath(output_dir, "best"), prob, HC.expand(L, prob.θ0, st.best.x); note="$(label) gen $(st.i) fx $(st.best.fx)")
  run_search_loop(loop; get_state=() -> st, save_exit=() -> PlainCMA.save_state(search, output_dir)) do
    while !PlainCMA.finished(search)
      xs = PlainCMA.ask(search)
      fx = Vector{Float64}(undef, length(xs))
      for k in eachindex(xs)         # SERIAL: the site loops of one evaluation are threaded
        fx[k] = f(xs[k])
      end
      improved = PlainCMA.tell!(search, fx)
      if improved
        iter = st.i
        let buf = IOBuffer()
          Serialization.serialize(buf, st.best.x)
          DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?)", [iter, nsites, nobs, st.best.fx, take!(buf)])
        end
        write_best()
      end
      periodic = ckpt_every > 0 && st.i % ckpt_every == 0
      (improved || periodic) && put!(writer_ch, WriterJob(true, deepcopy(st), empty_df, empty_df, nothing, nothing, nothing, nothing))
      st.i % log_every == 0 && @info @sprintf("CMA gen %d evals %d fbest %.4f fgen %.4f sigma %.3g cond %.2g", st.i, st.evals, st.best.fx, minimum(st.fx), st.sigma, (maximum(st.Dg) / minimum(st.Dg))^2)
    end
    @info "search ended" reason = PlainCMA.stop_reason(st) gens = st.i evals = st.evals
  end
  write_best()
  θ = HC.expand(L, prob.θ0, st.best.x)
  hbad2_report(prob, θ, "FIT")
  return search
end

"""
    hbad2_igel_search(prob; output_dir, trials, mu, sigma0, box, ckpt_every, seed, archive_cap, resume_from=nothing, label="") -> IgelMOCMAES.IgelSearch

Two-objective calibration (`search_mode: igelmo`): the population-based MO-CMA-ES (`Search.IgelMOCMAES`, mu independent (1+1)-CMA-ES individuals, Pareto archive) over (fA, fB), both on
TRAIN and normalised by the problem's ideal / nadir ((f - ideal) / (nadir - ideal), so the engine's Float32 objective vector keeps full resolution). The engine works in the box
`[z0 - box, z0 + box]` around the start's searched coordinates (u in [0,1]); the initial population is the start plus (mu - 1) Gaussian perturbations of width `sigma0` (u-space), all
evaluated. Same life cycle as the CMA-ES driver (`run_search_loop`, background writer, `search_state@<gen>.jld2` + `search_state_latest.jld2`, resume with the evaluation budget
`trials`); `metrics.csv` has one line per generation, `front.csv` the final archive with its TRAIN and VAL values.
"""
function hbad2_igel_search(prob::HC.CalibProblem; output_dir::AbstractString, trials::Int, mu::Int, sigma0::Real, box::Real, ckpt_every::Int, seed::Int, archive_cap::Int,
                           resume_from::Union{Nothing,AbstractString}=nothing, label::AbstractString="")
  prob.objective == :joint || error("hbad2 igelmo needs hbad2_calib.objective: joint (two objectives)")
  mkpath(output_dir)
  L = prob.layout; z0 = HC.restrict(L, prob.θ0); lo = z0 .- box; width = 2box
  to_z(u) = lo .+ width .* u
  to_u(z) = clamp.((z .- lo) ./ width, 0.0, 1.0)
  c = prob.cheb
  norm_obj(a, b) = ((a - c.ideal[1]) / (c.nadir[1] - c.ideal[1]), (b - c.ideal[2]) / (c.nadir[2] - c.ideal[2]))
  function evaluate(u)
    θ = HC.expand(L, prob.θ0, to_z(u))
    a, b = HC.objectives(prob, θ, :train)
    x, y = norm_obj(a, b)
    return MOLBSA.MOFitness(Float32[x, y], HC.chebyshev(c, a, b)), θ
  end
  rng = Random.Xoshiro(seed)
  isearch = if resume_from === nothing
    us = [to_u(z0)]
    for _ in 2:mu; push!(us, clamp.(us[1] .+ sigma0 .* randn(rng, length(z0)), 0.0, 1.0)); end
    ev = [evaluate(u) for u in us]
    cands = [MOLBSA.MOCandidate(ev[k][2], ev[k][1]) for k in eachindex(us)]
    st = IgelMOCMAES.IgelState(us, cands, rng; sigma0=sigma0, archive_cap=archive_cap, max_iter=typemax(Int))
    st.n_evals = mu
    IgelMOCMAES.IgelSearch(st, IgelMOCMAES.IgelOptions())
  else
    @info "Resuming from $resume_from"
    IgelMOCMAES.load_state(IgelMOCMAES.IgelSearch, resume_from; dir=output_dir)
  end
  st = IgelMOCMAES.state(isearch)
  @info "HBAD2 igelmo calibration" label dims = length(z0) mu archive_cap evals = st.n_evals trials box sigma0
  hbad2_report(prob, prob.θ0, "START")
  if st.n_evals >= trials
    IgelMOCMAES.save_state(isearch, output_dir)
    return isearch
  end
  ch, task = start_mo_writer(typeof(st), output_dir)
  lfile, ldb = open_losses_db(output_dir)
  loop = SearchLoop(String(output_dir), ch, task, lfile, ldb, nothing)
  metrics = open(joinpath(output_dir, "metrics.csv"), resume_from === nothing ? "w" : "a")
  resume_from === nothing && println(metrics, "iteration,evals,archive_size,best_scalar,min_fA_norm,min_fB_norm")
  empty_df = DataFrame(); last_sig = length(st.archive)
  run_search_loop(loop; get_state=() -> st, save_exit=() -> IgelMOCMAES.save_state(isearch, output_dir), close_extra=() -> close(metrics)) do
    while st.n_evals + mu <= trials
      offs = IgelMOCMAES.ask(isearch)
      fxs = Vector{MOLBSA.MOFitness}(undef, mu); ths = Vector{Vector{Float64}}(undef, mu)
      for k in 1:mu; fxs[k], ths[k] = evaluate(offs[k]); end      # SERIAL: the site loops of one evaluation are threaded
      is_new_best = IgelMOCMAES.tell!(isearch, fxs, ths)
      st.n_evals += mu
      sig = (length(st.archive), sum(m.fx.aggregate for m in st.archive))
      objs = [m.fx.objectives for m in st.archive]
      println(metrics, string(st.i, ",", st.n_evals, ",", length(st.archive), ",", st.representative.fx.aggregate, ",", minimum(o[1] for o in objs), ",", minimum(o[2] for o in objs))); flush(metrics)
      if is_new_best
        let buf = IOBuffer()
          Serialization.serialize(buf, st.representative.x)
          DuckDB.execute(ldb, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?, ?)", [st.i, prob.A[:train].nsub_sites, sum(prob.A[:train].n1), st.representative.fx.aggregate, length(st.archive), take!(buf)])
        end
        HC.write_params(joinpath(output_dir, "best"), prob, st.representative.x; note="$(label) gen $(st.i) aggregate $(st.representative.fx.aggregate)")
      end
      (is_new_best || (ckpt_every > 0 && st.i % ckpt_every == 0)) && put!(ch, WriterJob(true, deepcopy(st), empty_df, empty_df, nothing, nothing, nothing, nothing))
      st.i % 10 == 0 && @info @sprintf("igelmo gen %d evals %d archive %d scalar %.4f", st.i, st.n_evals, length(st.archive), st.representative.fx.aggregate)
    end
  end
  # the archive with TRAIN and VAL values of both objectives
  rows = DataFrame(index=Int[], trainA=Float64[], trainB=Float64[], valA=Float64[], valB=Float64[], scalar=Float64[])
  for (i, m) in enumerate(sort(st.archive; by=m -> m.fx.objectives[1]))
    a, b = HC.objectives(prob, m.x, :train)
    va, vb = haskey(prob.A, :val) ? HC.objectives(prob, m.x, :val) : (NaN, NaN)
    push!(rows, (i, a, b, va, vb, m.fx.aggregate))
  end
  CSV.write(joinpath(output_dir, "front.csv"), rows)
  HC.write_params(joinpath(output_dir, "best"), prob, st.representative.x; note="$(label) representative (minimum Chebyshev scalar)")
  hbad2_report(prob, st.representative.x, "FIT")
  return isearch
end

"""
    run_hbad2_calibration(cfg) -> NamedTuple

Entry of a calibration run (`run_from_yaml` for `succession_plugin: hbad2` + `hbad2_calib:`): builds the problem, runs the search per scalarisation weight and writes `front.csv`
(joint runs). `search_mode`: `cmaes` (the prototype's CMA-ES on the scalar objective).
"""
function run_hbad2_calibration(cfg)
  hc = cfg["hbad2_calib"]
  mode = cfg["search_mode"]
  mode in ("cmaes", "igelmo") || error("hbad2 calibration supports search_mode cmaes (the prototype's CMA-ES) and igelmo (two objectives) (got $(repr(mode)))")
  cfg["n_folds"] > 1 && error("hbad2 calibration does not support n_folds > 1 (use the split file)")
  out = cfg["output_dir"]; mkpath(out)
  LinearAlgebra.BLAS.set_num_threads(1)
  mode == "igelmo" && hc["objective"] != "joint" && error("hbad2 igelmo needs hbad2_calib.objective: joint")
  lams = hc["objective"] == "joint" && mode == "cmaes" ? hc["lambda"] : [first(hc["lambda"])]
  results = []
  front = DataFrame(lambda=Float64[], trainA=Float64[], trainB=Float64[], valA=Float64[], valB=Float64[])
  prob = nothing
  for lam in lams
    dir = length(lams) == 1 ? out : joinpath(out, @sprintf("lam_%.2f", lam))
    prob = prob === nothing ? hbad2_calib_problem(cfg; output_dir=out, lambda=lam) : HC.with_lambda(prob, lam)
    if mode == "igelmo"
      s = hbad2_igel_search(prob; output_dir=dir, trials=cfg["trials"], mu=cfg["igel_mu"], sigma0=cfg["igel_sigma0"], box=hc["box"], ckpt_every=hc["ckpt_every"], seed=hc["cma_seed"],
                            archive_cap=cfg["cmaes_archive_cap"], resume_from=cfg["resume_from"], label=@sprintf("lambda %.2f", lam))
      θ = s.st.representative.x
      a, b = HC.objectives(prob, θ, :train)
      va, vb = haskey(prob.A, :val) ? HC.objectives(prob, θ, :val) : (NaN, NaN)
      push!(front, (lam, a, b, va, vb)); CSV.write(joinpath(out, "front_representative.csv"), front)
      push!(results, (lambda=lam, state=s.st, theta=θ))
      continue
    end
    s = hbad2_cma_search(prob; output_dir=dir, trials=cfg["trials"], sigma0=cfg["cmaes_sigma0"], lambda=cfg["cmaes_lambda"], seed=hc["cma_seed"], stall=hc["cma_stall"], ftol=hc["cma_ftol"],
                         ckpt_every=hc["ckpt_every"], resume_from=cfg["resume_from"], label=@sprintf("lambda %.2f", lam))
    θ = HC.expand(prob.layout, prob.θ0, s.st.best.x)
    a, b = HC.objectives(prob, θ, :train)
    va, vb = haskey(prob.A, :val) ? HC.objectives(prob, θ, :val) : (NaN, NaN)
    push!(front, (lam, a, b, va, vb)); CSV.write(joinpath(out, "front.csv"), front)
    push!(results, (lambda=lam, state=s.st, theta=θ))
  end
  return (problem=prob, results=results, front=front)
end
