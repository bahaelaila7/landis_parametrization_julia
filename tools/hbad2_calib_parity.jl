# HBAD2 CALIBRATION PARITY GATE: Pan's Sim A and Sim B objectives against the PROTOTYPE harness (tools/hbad2/{common,simB_lik}.jl of branch hbad-v2, read-only, run in this process),
# for the prototype parameter vectors fit_v2j (joint fit j15: theta.csv + recr.csv, sigma_v <= 0.7) and fit_v2fv5 (Sim A fit, sigma_v <= 0.5, Poisson recruitment), on TRAIN and VAL.
#   PAN_DATA=/workspace/landis_region4/data_r4 ./julia_gdal.sh --project=. --threads=2 tools/hbad2_calib_parity.jl [--proto /workspace/landis_hbad2] [--split <split_plots.csv>] [--cache <dir>]
# Pan side: the interval product rebuilt from the database (Data.build_hbad2_intervals), HBAD2Calib data + objectives running the plugin kernel. Prototype side: its own Arrow files and functions.
# Every term is printed side by side (Pan / prototype / relative difference); the logged numbers of the prototype runs are printed for reference. Exit code 0 iff the maximum
# relative difference <= 1e-9 (PARITY PASS).
using Pan, DataFrames, CSV, Printf, Random, Statistics

function argval(name, default); i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i+1]; end
const PROTO = argval("--proto", "/workspace/landis_hbad2")
const SPLIT = argval("--split", "/workspace/landis_region4/runs_local/gaflst_split/split_plots.csv")
const DB = argval("--db", joinpath(get(ENV, "PAN_DATA", "/workspace/landis_region4/data_r4"), "slim_r4.duckdb"))
const CACHE = argval("--cache", mktempdir())
const LONGEVITY = argval("--longevity", "/workspace/species_longevity_ref.csv")
const TOL = 1e-9
const R2 = joinpath(PROTO, "results", "hbad2")
const HC = Pan.HBAD2Calib

# the prototype harness reads its structural switches from the environment at include time (fit meta.txt); one module per sigma_v bound
function load_proto(sigv::String)
  ENV["HBAD2_FORM"] = "2"; ENV["HBAD2_SIZE_SPLIT"] = "5,12.7,25,40"; ENV["HBAD2_SITE"] = "subplot"; ENV["HBAD2_VIGOUR"] = "1"; ENV["HBAD2_SIGV_MAX"] = sigv
  name = Symbol("Proto", replace(sigv, "." => ""))
  Core.eval(Main, :(module $name
    using Arrow, DataFrames, CSV, Statistics, Printf, Dates, Random, LinearAlgebra
  end))
  m = getfield(Main, name)
  Base.include(m, joinpath(PROTO, "tools", "hbad2", "common.jl"))
  Base.include(m, joinpath(PROTO, "tools", "hbad2", "simB_lik.jl"))
  return m
end

rel(a, b) = a == b ? 0.0 : abs(a - b) / max(abs(b), 1e-300)
rows = Tuple{String,Float64,Float64,Float64}[]
CASE = ""
function cmp!(label, pan, proto)
  label = CASE * ": " * label
  r = rel(Float64(pan), Float64(proto)); push!(rows, (label, Float64(pan), Float64(proto), r))
  @printf("  %-34s Pan %20.10f  proto %20.10f  rel %.2e\n", label, pan, proto, r)
end

println("== data")
split = Pan.Data.read_split_plots(SPLIT)
t0 = time(); x = Pan.Data.load_hbad2_intervals(CACHE, DB, split; rebuild=true); @printf("Pan interval product built in %.0f s\n", time() - t0)
println(Pan.Data.compare_interval_products(x, Pan.Data.read_hbad2_intervals(R2)))

println("== prototype modules (sigma_v bound 0.7 for fit_v2j, 0.5 for fit_v2fv5)")
P7 = load_proto("0.7"); P5 = load_proto("0.5")
tr7 = Base.invokelatest(P7.load_trees); keys7 = sort(unique(tr7.key)); nsp = length(keys7)
tr5 = Base.invokelatest(P5.load_trees); keys5 = sort(unique(tr5.key)); keys7 == keys5 || error("species keys differ between the prototype modules")
lon = CSV.read(LONGEVITY, DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr7.sym[tr7.key .== k])]) for k in keys7]

println("== Pan problem")
sw7 = HC.CalibSwitches(form=2, size_split=[5, 12.7, 25, 40], site="subplot", vigour=true, sigv_max=0.7)
sw5 = HC.CalibSwitches(form=2, size_split=[5, 12.7, 25, 40], site="subplot", vigour=true, sigv_max=0.5)
v2j_theta = joinpath(R2, "fit_v2j", "theta.csv"); v2j_recr = joinpath(R2, "fit_v2j", "recr.csv"); fv5_theta = joinpath(R2, "fit_v2fv5", "theta.csv")
prob = HC.build_calib_problem(; db_path=DB, split=split, intervals_cache=CACHE, sw=sw7, longevity_csv=LONGEVITY, recruitment_csv=joinpath(R2, "recruitment_coefs_subplot.csv"),
                              priors_csv="", start_theta=v2j_theta, recr=:free, objective=:joint, fb=:margins, sets=[:train, :val])
prob.keys == keys7 || error("species keys differ: Pan $(prob.keys) prototype $(keys7)")
maximum(abs.(prob.longevity .- longev)) == 0 || error("longevity differs")
prob5 = HC.CalibProblem(sw5, prob.keys, prob.longevity, prob.layout, prob.recr_data, prob.A, prob.B, prob.fb, prob.objective, prob.θ0, prob.cheb, prob.pc)

# the prototype's recruitment: the free vector of recr.csv (v2j: data fit x kappa) or the Poisson fit recruitment_coefs_subplot.csv (fv5)
rcd = Dict(String(r.key) => r for r in eachrow(CSV.read(joinpath(R2, "recruitment_coefs_subplot.csv"), DataFrame)))
recr_poisson = reduce(vcat, [[rcd[k].rho0, rcd[k].rhoS, rcd[k].rhoL] for k in keys7])
zj = Float64.(vec(CSV.read(v2j_recr, DataFrame).theta))
cases = [("v2j (j15)", P7, prob, HC.read_theta_csv(v2j_theta, nsp), zj), ("v2fv5", P5, prob5, HC.read_theta_csv(fv5_theta, nsp), recr_poisson)]

for (name, PM, pr, θ, z) in cases
  global CASE = name
  println("\n==== $name")
  inv(f, args...) = Base.invokelatest(f, args...)
  P, G = inv(PM.decode, θ, nsp, longev)
  R = [PM.RecP(clamp(z[3k-2], -12, 2), clamp(z[3k-1], -1, 4), clamp(z[3k], -10, 2), 2.54, 3.0, 5.0) for k in 1:nsp]
  σw_proto = inv(() -> PM.SIGMA_W[])
  θfull = vcat(θ, z)
  for set in ("train", "val")
    println("-- Sim A $set")
    C = inv(PM.build_cohorts, tr7, set, keys7)
    inv(PM.simulate!, C.S, P, G)
    ra = inv(PM.nll_parts, inv(PM.cohort_view, C), C.obs)
    pa = HC.simA_scores(pr, θfull, Symbol(set))
    for f in (:total, :mort, :D, :H, :sigD, :sigH); cmp!("simA $set $f", getproperty(pa, f), getproperty(ra, f)); end
    cmp!("simA $set ncoh", pa.ncoh, ra.ncoh); cmp!("simA $set ntrees", pa.ntrees, ra.ntrees)
    for sg in (σw_proto, 0.083)
      ep, _ = HC.model_params(pr, θfull); HC.simulate_simA!(pr.A[Symbol(set)], ep)
      cmp!(@sprintf("simA %s size_nll sigma_w=%.3f", set, sg), HC.size_nll(pr.A[Symbol(set)], sg), inv(PM.size_nll, C, sg))
    end
    println("-- Sim B $set")
    B = inv(PM.build_simB, set, keys7)
    rb = inv(PM.simB_nll!, B, P, G, R)
    pb = HC.simB_scores(pr, θfull, Symbol(set))
    for f in (:total, :counts, :H, :size, :species, :plotstand, :plotstand_trees); cmp!("simB $set $f", getproperty(pb, f), getproperty(rb, f)); end
    cmp!("simB $set margins (fB)", HC.fb_value(pb, :margins), rb.size + rb.species + rb.H)
    cmp!("simB $set dropped", pb.dropped, rb.dropped)
    @printf("  sites Pan %d / proto %d, plots %d / %d\n", length(pr.B[Symbol(set)].live0), length(B.live0), pr.B[Symbol(set)].nplots, maximum(B.plot_of))
  end
end

println("\n== logged numbers of the prototype runs (results/hbad2/fit_v2fv5/log.txt, joint_j15/log.txt), Pan values above")
println("  fit_v2fv5: train total -273425.4 (mort 31941.9 D -164983.2 H -140384.2 sigD 0.1117 sigH 0.1548) | val total -272655.4 (mort 32960.9 D -163997.3 H -141619.0 sigD 0.1146 sigH 0.1541)")
println("  joint_j15 (v2j): lambda=0.75 train fA -270648.8 fB 16462.8 | VAL fA -269744.7 fB 18317.9")
d = Dict(r[1] => r[2] for r in rows)
@printf("  Pan fv5:  train total %.1f (mort %.1f D %.1f H %.1f sigD %.4f sigH %.4f) | val total %.1f (mort %.1f D %.1f H %.1f sigD %.4f sigH %.4f)\n",
        [d["v2fv5: simA train $f"] for f in (:total, :mort, :D, :H, :sigD, :sigH)]..., [d["v2fv5: simA val $f"] for f in (:total, :mort, :D, :H, :sigD, :sigH)]...)
@printf("  Pan v2j:  train fA %.1f fB %.1f | VAL fA %.1f fB %.1f\n", d["v2j (j15): simA train total"], d["v2j (j15): simB train margins (fB)"], d["v2j (j15): simA val total"], d["v2j (j15): simB val margins (fB)"])
println("\n== SUMMARY")
mx = maximum(r[4] for r in rows)
@printf("terms compared: %d, max relative difference %.3e (tolerance %.0e)\n", length(rows), mx, TOL)
worst = sort(rows; by=r -> -r[4])[1:min(5, length(rows))]
for r in worst; @printf("  worst: %-34s %.3e\n", r[1], r[4]); end
println(mx <= TOL ? "CALIB PARITY PASS (max relative difference <= $(TOL))" : "CALIB PARITY FAIL")
exit(mx <= TOL ? 0 : 1)
