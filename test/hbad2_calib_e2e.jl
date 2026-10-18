# End-to-end test of the HBAD2 calibration on a SUBSET of the real plots (every 15th plot of the prototype's split): Sim A and joint searches of ~200 evaluations complete, write
# checkpoints / the plugin-ready parameter directory / losses.duckdb, improve on the start, and resume identically (a run of N evaluations checkpointed and resumed for M more ends in exactly
# the state of an uninterrupted N+M run); thread-independence of the objective (threaded vs serial site loops give identical numbers); allocation-free kernel entry points; the igelmo variant.
#   ./julia_gdal.sh --project=. --threads=2 test/hbad2_calib_e2e.jl      (subprocess test; needs the golden data PAN_DATA and the prototype's split file)
using Test, Random, Pan, DataFrames, CSV, DuckDB
const DATA = get(ENV, "PAN_DATA", "/workspace/landis_region4/data_r4")
const SPLITCSV = get(ENV, "HBAD2_CALIB_SPLIT", "/workspace/landis_region4/runs_local/gaflst_split/split_plots.csv")
if !isfile(joinpath(DATA, "slim_r4.duckdb")) || !isfile(SPLITCSV)
  @info "hbad2_calib_e2e: data ($DATA, $SPLITCSV) not available - skipped"; exit(0)
end
ENV["PAN_DATA"] = DATA
const HC = Pan.HBAD2Calib
tmp = mktempdir()
sp = Pan.Data.read_split_plots(SPLITCSV)
sub = sp[1:15:end, :]
CSV.write(joinpath(tmp, "split.csv"), sub)
@info "subset" plots = nrow(sub) train = count(==("train"), sub.set) val = count(==("val"), sub.set)

hcbase(extra=Dict{String,Any}()) = merge(Dict{String,Any}("split_csv" => joinpath(tmp, "split.csv"), "intervals_cache" => joinpath(tmp, "cache"), "sigv_max" => 0.7), extra)
function run_cfg(out, hc; trials, resume=nothing, mode="cmaes", extra=Dict{String,Any}())
  ov = Dict{String,Any}("output_dir" => out, "trials" => trials, "search_mode" => mode, "succession_plugin" => "hbad2", "cohorts_db_path" => joinpath(DATA, "slim_r4.duckdb"),
                        "cmaes_lambda" => 8, "cmaes_sigma0" => 0.05, "hbad2_calib" => hc)
  resume === nothing || (ov["resume_from"] = resume)
  Pan.run_from_yaml(joinpath(dirname(@__DIR__), "configs", "region4", "hbad2_calib_simA.yml"); overrides=merge(ov, extra))
end
load(dir) = Pan.JLD2.load_object(joinpath(dir, "search_state_latest.jld2"))
sig(st) = (st.gen, st.evals, st.sigma, st.best.fx, st.best.x, st.m, st.C, st.pc, st.ps, st.hist, rand(copy(st.rng), 4))

@testset "hbad2 calibration end to end" begin
  @testset "Sim A: 200 evaluations, checkpoint, resume == uninterrupted" begin
    hc = hcbase(Dict{String,Any}("objective" => "sim_a", "ckpt_every" => 5, "min_sp_trees" => 600))
    full = joinpath(tmp, "a_full"); part = joinpath(tmp, "a_part")
    run_cfg(full, hc; trials=200)
    run_cfg(part, hc; trials=100)
    a = load(full); mid = load(part)
    @test a.evals <= 200 && a.evals > 190 && a.gen == (a.evals - 1) ÷ 8 && mid.evals <= 100 && mid.evals > 90
    @test isfile(joinpath(part, "search_state@$(mid.gen).jld2")) && isfile(joinpath(part, "best", "params_species.csv")) && isfile(joinpath(full, "split_plots.csv"))
    run_cfg(part, hc; trials=200, resume=joinpath(part, "search_state_latest.jld2"))
    b = load(part)
    @test b.gen == a.gen && b.evals == a.evals && isequal(sig(a), sig(b))
    # improvement on the start, decoded files, losses.duckdb
    prob = Pan.hbad2_calib_problem(Pan.load_config(joinpath(dirname(@__DIR__), "configs", "region4", "hbad2_calib_simA.yml");
      overrides=Dict{String,Any}("output_dir" => joinpath(tmp, "a_prob"), "cohorts_db_path" => joinpath(DATA, "slim_r4.duckdb"), "hbad2_calib" => hc)); output_dir=joinpath(tmp, "a_prob"))
    f0 = HC.scalar_objective(prob, HC.restrict(prob.layout, prob.θ0))
    @test a.best.fx < f0 && a.hist[end] == a.best.fx && issorted(a.hist; rev=true)
    @test HC.scalar_objective(prob, a.best.x) == a.best.fx
    θ = HC.expand(prob.layout, prob.θ0, a.best.x)
    ctx = Pan.Plugins.HBAD2Plugin.HBAD2Context(prob.keys, [collect(1:length(prob.keys))]; params_dir=joinpath(full, "best"), verbose=false)
    P, G, _ = HC.decode_theta(θ, length(prob.keys), prob.sw, prob.longevity)
    @test ctx.sp == P && ctx.glob == G && ctx.keys == prob.keys                     # the exported directory IS the decoded parameter set
    db = DuckDB.DB(joinpath(full, "losses.duckdb")); con = DuckDB.connect(db)
    n = only(DataFrame(DuckDB.execute(con, "SELECT count(*) AS n FROM total_loss")).n); @test n >= 1
    DuckDB.close(db)
    # thread independence of the objective: threaded vs serial site loops, bit-identical
    v1 = HC.scalar_objective(prob, a.best.x)
    for (_, A) in prob.A; A.soa.parallel = false; end
    v2 = HC.scalar_objective(prob, a.best.x)
    @test v1 == v2
    sA = HC.simA_scores(prob, θ, :val)
    for (_, A) in prob.A; A.soa.parallel = true; end
    @test HC.simA_scores(prob, θ, :val) == sA
    # allocation-free kernel entry points
    A = prob.A[:train]; ep, _ = HC.model_params(prob, θ)
    Pan.Plugins.HBAD2Plugin.run_interval!(A.soa, 1, ep, A.nyr[1], A.frac[1])
    @test (@allocated Pan.Plugins.HBAD2Plugin.run_interval!(A.soa, 1, ep, A.nyr[1], A.frac[1])) == 0
  end

  @testset "trials: 1 only evaluates the start" begin
    hc = hcbase(Dict{String,Any}("objective" => "sim_a", "min_sp_trees" => 600))
    out = joinpath(tmp, "a_one")
    run_cfg(out, hc; trials=1)
    st = load(out); @test st.evals == 1 && st.gen == 0 && isfile(joinpath(out, "best", "theta.csv"))
  end

  @testset "joint (kappa, frozen mortality): 200 evaluations, resume == uninterrupted" begin
    hc = hcbase(Dict{String,Any}("objective" => "joint", "fb" => "margins", "lambda" => [0.75], "recruitment" => "kappa", "freeze" => ["mbg", "mD", "msup", "G:mD2", "G:mSD", "G:pM"],
                                 "ckpt_every" => 5, "min_sp_trees" => 600, "ideal" => [-20000.0, 100.0], "nadir" => [-10000.0, 2000.0]))
    full = joinpath(tmp, "j_full"); part = joinpath(tmp, "j_part")
    run_cfg(full, hc; trials=200)
    run_cfg(part, hc; trials=100)
    mid = load(part)
    run_cfg(part, hc; trials=200, resume=joinpath(part, "search_state_latest.jld2"))
    a = load(full); b = load(part)
    @test a.evals > 190 && b.gen == a.gen && isequal(sig(a), sig(b))
    @test isfile(joinpath(full, "front.csv")) && isfile(joinpath(full, "best", "recr.csv"))
    fr = CSV.read(joinpath(full, "front.csv"), DataFrame); @test nrow(fr) == 1 && fr.lambda == [0.75] && all(isfinite, [fr.trainA[1], fr.trainB[1]])
    # the frozen coordinates stayed at their start; kappa is one extra coordinate
    prob = Pan.hbad2_calib_problem(Pan.load_config(joinpath(dirname(@__DIR__), "configs", "region4", "hbad2_calib_simA.yml");
      overrides=Dict{String,Any}("output_dir" => joinpath(tmp, "j_prob"), "cohorts_db_path" => joinpath(DATA, "slim_r4.duckdb"), "hbad2_calib" => hc)); output_dir=joinpath(tmp, "j_prob"))
    L = prob.layout; @test L.recr == :kappa && length(L.frozen) == 3 * length(prob.keys) + 3 && L.nrec == 1
    @test length(a.best.x) == length(L.free_idx) && length(L.free_idx) == length(L) - length(L.frozen)
    θ = HC.expand(L, prob.θ0, a.best.x); @test all(θ[L.frozen] .== prob.θ0[L.frozen])
    @test HC.scalar_objective(prob, a.best.x) == a.best.fx
    # threaded == serial for the joint objective too
    v1 = HC.objectives(prob, θ, :val)
    for (_, A) in prob.A; A.soa.parallel = false; end; for (_, B) in prob.B; B.soa.parallel = false; end
    @test HC.objectives(prob, θ, :val) == v1
  end

  @testset "joint: lambda list, plot-level sites option, igelmo" begin
    hc = hcbase(Dict{String,Any}("objective" => "joint", "lambda" => [1.0, 0.0], "min_sp_trees" => 600, "ideal" => [-20000.0, 100.0], "nadir" => [-10000.0, 2000.0], "ckpt_every" => 0))
    out = joinpath(tmp, "j_two")
    run_cfg(out, hc; trials=60)
    @test isdir(joinpath(out, "lam_1.00")) && isdir(joinpath(out, "lam_0.00")) && nrow(CSV.read(joinpath(out, "front.csv"), DataFrame)) == 2
    # igelmo over (fA, fB): 3 generations, then resume to 5 == uninterrupted
    hci = hcbase(Dict{String,Any}("objective" => "joint", "min_sp_trees" => 600, "ideal" => [-20000.0, 100.0], "nadir" => [-10000.0, 2000.0], "ckpt_every" => 1, "box" => 0.3))
    ex = Dict{String,Any}("igel_mu" => 4, "igel_sigma0" => 0.05)
    run_cfg(joinpath(tmp, "i_full"), hci; trials=24, mode="igelmo", extra=ex)
    run_cfg(joinpath(tmp, "i_part"), hci; trials=16, mode="igelmo", extra=ex)
    mid = Pan.JLD2.load_object(joinpath(tmp, "i_part", "search_state_latest.jld2")); @test mid.i == 3
    run_cfg(joinpath(tmp, "i_part"), hci; trials=24, mode="igelmo", extra=ex, resume=joinpath(tmp, "i_part", "search_state_latest.jld2"))
    a = Pan.JLD2.load_object(joinpath(tmp, "i_full", "search_state_latest.jld2")); b = Pan.JLD2.load_object(joinpath(tmp, "i_part", "search_state_latest.jld2"))
    @test a.i == 5 && b.i == 5
    sg(st) = (st.i, st.n_evals, [(ind.x, ind.sigma, ind.C, ind.fx.objectives) for ind in st.pop], [(m.fx.objectives, m.fx.aggregate) for m in st.archive], rand(copy(st.rng), 4))
    @test isequal(sg(a), sg(b))
    @test isfile(joinpath(tmp, "i_full", "front.csv")) && isfile(joinpath(tmp, "i_full", "best", "params_species.csv"))
  end
end
