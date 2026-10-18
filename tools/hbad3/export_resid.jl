# HBAD3 design study, step S2: per-cohort predictions of a fitted v2 model (default v2j) on train + val, written with site/interval ids
# so the residuals can be related to climate (scripts/hbad3/climate_signal.py). Cohort-level (vigour sub-cohorts aggregated).
#   HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot HBAD2_VIGOUR=1 HBAD2_SIGV_MAX=0.7 ./julia_gdal.sh --project=. --threads=6 tools/hbad3/export_resid.jl [tag]
include(joinpath(@__DIR__, "..", "hbad2", "common.jl"))
tag = length(ARGS) >= 1 ? ARGS[1] : "v2j"
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
check_meta(joinpath(R2DIR, "fit_" * tag)); θ = read_theta(joinpath(R2DIR, "fit_" * tag, "theta.csv"), nsp); P, G = decode(θ, nsp, longev)
out = DataFrame()
for set in ("train", "val")
  C = build_cohorts(tr, set, keys_)
  S0 = Sites(C.S.off, C.S.sp, C.S.N0, C.S.D0, C.S.H0, C.S.A0, zeros(Int, length(C.S.nyr)), fill(1e-9, length(C.S.nyr)), C.S.planted, C.S.zq)
  simulate!(S0, P, G); O0 = copy(S0.O); RD0 = zeros(length(O0))
  for s in 1:length(C.S.off)-1
    lo, hi = C.S.off[s], C.S.off[s+1] - 1; tot = sum(S0.S[lo:hi]); sm = sum(S0.S[i] * P[C.S.sp[i]].sdimax for i in lo:hi) / tot
    RD0[lo:hi] .= tot / sm
  end
  simulate!(C.S, P, G); V = cohort_view(C); o = C.obs
  fo = [searchsortedfirst(C.parent, p) for p in 1:length(o.n1)]
  df = DataFrame(set=set, site=C.coh.site, iid=C.coh.site .÷ 10, key=C.coh.key, planted=C.coh.planted, dt=o.dt, n1=o.n1, s=o.s,
                 N1=V.N0, D1=V.D0, H1=V.H0, A1=C.S.A0[fo], O=O0[fo], RD=RD0[fo], ps=V.N ./ V.N0, D2s=V.D, H2s=V.H, D2o=o.D2o, H2o=o.H2o)
  append!(out, df)
end
mkpath(joinpath(R2DIR, "..", "hbad3")); CSV.write(joinpath(R2DIR, "..", "hbad3", "resid_" * tag * ".csv"), out)
println("wrote ", nrow(out), " cohorts")
