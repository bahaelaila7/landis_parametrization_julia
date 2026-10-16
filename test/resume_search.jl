# Step 5a: checkpoint/resume round trip of the two engines that implement the search interface (igelmo, gdigel).
# A golden-size run of N evaluations, checkpointed and resumed for M more, must end in exactly the state of an uninterrupted N+M run.
#   ./julia_gdal.sh --project=. --threads=2 test/resume_search.jl      (subprocess test; needs the golden data, PAN_DATA)
using Test, Random, Pan, YAML
const DATA = get(ENV, "PAN_DATA", "/workspace/landis_region4/data_r4")
if !isdir(DATA)
  @info "resume_search: golden data $DATA not available — skipped"; exit(0)
end
ENV["PAN_DATA"] = DATA; ENV["PAN_PARALLEL"] = "soa"
include(joinpath(@__DIR__, "golden", "scenarios.jl"))
const BASE = joinpath(dirname(@__DIR__), SCENARIOS["bs_igelmo"][1])

run_cfg(over) = Pan.run_from_yaml(BASE; overrides=merge(common(), over))
fold_dir(d) = isdir(joinpath(d, "fold_1")) ? joinpath(d, "fold_1") : d
latest(dir, pat) = (fs = filter(f -> occursin(pat, f), readdir(dir)); joinpath(dir, last(sort(fs; by=f -> parse(Int, match(r"@(\d+)", f)[1])))))
rngsig(r) = rand(copy(r), 4)

# ---- comparable signatures of the final engine states ----
function sig_igel(st)
  (i=st.i, n_evals=st.n_evals, rep=st.representative.fx.aggregate, rng=rngsig(st.rng), sob=(st._sobol.n, copy(st._sobol.x)),
   pop=[(ind.x, ind.sigma, ind.p_succ, ind.C, ind.p_c, ind.fx.objectives, ind.fx.aggregate, ind.mature_at) for ind in st.pop],
   arch=[(m.fx.objectives, m.fx.aggregate) for m in st.archive])
end
function sig_gd(s)
  st = s.st
  (gen=st.gen, n_evals=st.n_evals, rng=rngsig(st.rng), next_id=st.next_id, aref=st.aref, wref=st.wref,
   lin=[(L.id, L.x, L.f, L.tot, [(B.sigma, B.p_succ, B.C, B.p_c) for B in L.bs], L.born, L.best_tot, L.last_improve, L.lam) for L in st.lineages],
   sol=[(x.lineage, x.gen, x.x, x.tot) for x in st.solutions],
   nd=[(e.A, e.W, e.u) for e in s.nd], arch=[(e.gen, e.lineage, e.u) for e in s.archive])
end

@testset "resume round trip" begin
  tmp = mktempdir()
  @testset "igelmo (μ=4): 3 + 2 generations == 5 generations" begin
    base = Dict{String,Any}("search_mode" => "igelmo")
    full = joinpath(tmp, "igel_full"); part = joinpath(tmp, "igel_part")
    run_cfg(merge(base, Dict{String,Any}("output_dir" => full, "trials" => 24)))
    run_cfg(merge(base, Dict{String,Any}("output_dir" => part, "trials" => 16)))
    ck = joinpath(fold_dir(part), "search_state_latest.jld2")
    @test isfile(ck)
    mid = Pan.JLD2.load_object(ck); @test mid.i == 3
    run_cfg(merge(base, Dict{String,Any}("output_dir" => part, "trials" => 24, "resume_from" => ck)))
    a = Pan.JLD2.load_object(joinpath(fold_dir(full), "search_state_latest.jld2")); b = Pan.JLD2.load_object(joinpath(fold_dir(part), "search_state_latest.jld2"))
    @test a.i == 5 && b.i == 5
    @test isequal(sig_igel(a), sig_igel(b))
  end
  @testset "gdigel mo: resume == uninterrupted" begin
    full = joinpath(tmp, "gd_full"); part = joinpath(tmp, "gd_part")
    cfg = gd(objective="mo")
    run_cfg(merge(cfg, Dict{String,Any}("output_dir" => full, "trials" => 30)))
    run_cfg(merge(cfg, Dict{String,Any}("output_dir" => part, "trials" => 14)))
    ck = latest(fold_dir(part), r"^gd_state@\d+\.jld2$")
    mid = Pan.Search.GDIgel.load_state(Pan.Search.GDIgel.GDSearch, ck); @test 14 <= mid.st.n_evals < 30
    run_cfg(merge(cfg, Dict{String,Any}("output_dir" => part, "trials" => 30, "resume_from" => ck)))
    a = Pan.Search.GDIgel.load_state(Pan.Search.GDIgel.GDSearch, latest(fold_dir(full), r"^gd_state@\d+\.jld2$"))
    b = Pan.Search.GDIgel.load_state(Pan.Search.GDIgel.GDSearch, latest(fold_dir(part), r"^gd_state@\d+\.jld2$"))
    @test a.st.gen == b.st.gen > mid.st.gen
    @test isequal(sig_gd(a), sig_gd(b))
  end
end
