# Evaluate a fitted HBAD v2 parameter vector on train/val(/test): joint NLL parts, per-species diagnostics and residuals by class
# (overtopping, stand RD, D, age, planted, L3). Test is reported only with HBAD2_TEST=1 (information; val decides).
#   ./julia_gdal.sh --project=. --threads=4 tools/hbad2/eval_fit.jl <tag>
include(joinpath(@__DIR__, "common.jl"))
tag = ARGS[1]; D_ = joinpath(R2DIR, "fit_" * tag)
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
θ = read_theta(joinpath(R2DIR, "fit_" * tag, "theta.csv"), nsp); P, G = decode(θ, nsp, longev)
sets = get(ENV, "HBAD2_TEST", "0") == "1" ? ("train", "val", "test") : ("train", "val")
for set in sets
  C = build_cohorts(tr, set, keys_)
  # O and RD at the START (one snapshot) for residual classes: run the kernel 0 years
  S0 = Sites(C.S.off, C.S.sp, C.S.N0, C.S.D0, C.S.H0, C.S.A0, zeros(Int, length(C.S.nyr)), fill(1e-9, length(C.S.nyr)))
  simulate!(S0, P, G); O0 = copy(S0.O); RD0 = zeros(length(O0))
  for s in 1:length(C.S.off)-1
    lo, hi = C.S.off[s], C.S.off[s+1] - 1; tot = sum(S0.S[lo:hi]); sm = sum(S0.S[i] * P[C.S.sp[i]].sdimax for i in lo:hi) / tot
    RD0[lo:hi] .= tot / sm
  end
  simulate!(C.S, P, G); p = nll_parts(C.S, C.obs)
  @printf("EVAL %-5s total %.1f (mort %.1f D %.1f H %.1f σD %.4f σH %.4f)\n", set, p.total, p.mort, p.D, p.H, p.sigD, p.sigH)
  set == "train" && continue
  diagnostics(C.S, C.obs, keys_)
  o = C.obs; ok = o.n1 .> 0
  df = DataFrame(O=O0, RD=RD0, D=C.S.D0, A=C.S.A0, pl=C.coh.planted, l3=C.coh.l3, n1=o.n1, s=o.s,
                 ps=C.S.N ./ C.S.N0, eD=log.(C.S.D ./ o.D2o), eH=log.(max.(C.S.H, 0.1) ./ o.H2o))[ok, :]
  cls(x, br) = [string("≤", br[findfirst(>=(v), br)]) for v in x]
  for (nm, col, br) in (("overtop O", :O, [0.1, 0.2, 0.35, 0.5, 0.75, 1e9]), ("stand RD", :RD, [0.2, 0.4, 0.6, 0.8, 1.0, 1e9]),
                         ("D cm", :D, [5.0, 12.7, 20, 30, 45, 1e9]), ("age", :A, [10.0, 20, 35, 50, 80, 1e9]))
    df.c = cls(df[!, col], br)
    t = combine(groupby(df, :c), nrow => :coh, :n1 => sum => :trees, [:n1, :s] => ((n, s) -> sum(n .- s)) => :dead_obs,
                [:n1, :ps] => ((n, p) -> sum(n .* (1 .- p))) => :dead_sim,
                [:s, :eD] => ((s, e) -> sum(s[s.>0] .* e[s.>0]) / sum(s)) => :mean_lnD_resid, [:s, :eH] => ((s, e) -> sum(s[s.>0] .* e[s.>0]) / sum(s)) => :mean_lnH_resid)
    sort!(t, :c, by=x -> parse(Float64, x[4:end])); println("-- by ", nm); show(stdout, t; allrows=true, summary=false, eltypes=false); println()
  end
  sm = df[df.D .< 12.7, :]; sm.c = cls(sm.O, [0.1, 0.2, 0.35, 0.5, 0.75, 1e9])
  t = combine(groupby(sm, :c), nrow => :coh, :n1 => sum => :trees, [:n1, :s] => ((n, s) -> sum(n .- s)) => :dead_obs,
              [:n1, :ps] => ((n, p) -> sum(n .* (1 .- p))) => :dead_sim, [:s, :eD] => ((s, e) -> sum(s[s.>0] .* e[s.>0]) / sum(s)) => :lnD_resid)
  sort!(t, :c, by=x -> parse(Float64, x[4:end])); println("-- SMALL trees (D < 12.7 cm) by overtopping O"); show(stdout, t; allrows=true, summary=false, eltypes=false); println()
  for col in (:pl, :l3)
    t = combine(groupby(df, col), nrow => :coh, [:n1, :s] => ((n, s) -> sum(n .- s)) => :dead_obs, [:n1, :ps] => ((n, p) -> sum(n .* (1 .- p))) => :dead_sim,
                [:s, :eD] => ((s, e) -> sum(s[s.>0] .* e[s.>0]) / sum(s)) => :lnD_resid, [:s, :eH] => ((s, e) -> sum(s[s.>0] .* e[s.>0]) / sum(s)) => :lnH_resid)
    println("-- by ", col); show(stdout, t; allrows=true, summary=false, eltypes=false); println()
  end
end
