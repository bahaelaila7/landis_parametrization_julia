# HBAD3 design study, step S3: which per-species v2 parameters carry information? Sim A fit (TRAIN, val reported) where each per-species
# parameter is either species-specific ("sp"), one global value ("g"), or a linear function of standardised species TRAITS
# ("ST", "WD+ST", …: intercept + slopes, in the encoded coordinate of the parameter). Traits per key (tree-weighted over member species):
#   ST  shade-tolerance class 1–5 (FIA crown class, data_r4/shadetol_all_species.csv)     WD  wood specific gravity (REF_SPECIES)
#   SW  softwood indicator   HX  ln 99th-percentile observed height (train)   DX  ln 99th-percentile observed diameter   LG  longevity
#   HBAD3_TIE="gOD:ST,gRD:ST,gOH:ST,gRH:ST"  (unlisted parameters stay "sp")     HBAD3_TIE_DEFAULT=sp|g
# Start = least-squares projection of a full fit (default fit_v2fv5) onto the tied design, then CMA-ES on the reduced coordinates.
#   HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot HBAD2_VIGOUR=1 HBAD2_SIGV_MAX=0.5 ./julia_gdal.sh --project=. --threads=4 tools/hbad3/fit_tied.jl <tag> [start fit dir]
include(joinpath(@__DIR__, "..", "hbad2", "common.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
using DuckDB
tag = ARGS[1]; START = length(ARGS) >= 2 ? ARGS[2] : joinpath(R2DIR, "fit_v2fv5")
OUTD = joinpath(R2DIR, "..", "hbad3", "tied_" * tag); mkpath(OUTD); write_meta(OUTD)
logio = open(joinpath(OUTD, "log.txt"), "w"); lg(s) = (println(s); println(logio, s); flush(logio); flush(stdout))
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
# ---- traits per key
st = CSV.read("/workspace/landis_region4/data_r4/shadetol_all_species.csv", DataFrame); std_ = Dict(String(r.sym) => Float64(r.shade_class) for r in eachrow(st))
db = DBInterface.connect(DuckDB.DB, get(ENV, "PAN_HBAD_DB", normpath(joinpath(@__DIR__, "..", "..", "..", "landis_region4", "data_r4", "fia_truth_r4.duckdb"))); readonly=true)
rs = DataFrame(DBInterface.execute(db, "SELECT SPECIES_SYMBOL s, CAST(WOOD_SPGR_GREENVOL_DRYWT AS DOUBLE) wd, SFTWD_HRDWD sw FROM REF_SPECIES ORDER BY SPECIES_SYMBOL"))
wdd = Dict(String(r.s) => r.wd for r in eachrow(rs) if !ismissing(r.wd))
trn = tr[tr.set .== "train", :]
function trait(k)
  t = trn[trn.key .== k, :]; c = combine(groupby(t, :sym), nrow => :n)
  wavg(d, dflt) = (v = [get(d, s, dflt) for s in c.sym]; sum(v .* c.n) / sum(c.n))
  (ST=wavg(std_, 3.0), WD=wavg(wdd, 0.5), SW=mean(t.grp .== "S"), HX=log(quantile(t.ht_m, 0.99)), DX=log(quantile(t.dia_cm, 0.99)), LG=longev[findfirst(==(k), keys_)])
end
TR_ = [trait(k) for k in keys_]
TD = DataFrame(key=keys_, ST=[t.ST for t in TR_], WD=[t.WD for t in TR_], SW=[t.SW for t in TR_], HX=[t.HX for t in TR_], DX=[t.DX for t in TR_], LG=[t.LG for t in TR_])
lg("traits per key:\n" * string(TD)); CSV.write(joinpath(OUTD, "traits.csv"), TD)
Z = Dict(nm => (x = Float64.(TD[!, nm]); s = std(x); s > 0 ? (x .- mean(x)) ./ s : zero(x)) for nm in ("ST", "WD", "SW", "HX", "DX", "LG"))
# ---- tie spec → design matrix per FREE parameter
dflt = get(ENV, "HBAD3_TIE_DEFAULT", "sp")
spec = Dict(Symbol(p[1]) => String(p[2]) for p in (split(x, ":") for x in filter(!isempty, strip.(split(get(ENV, "HBAD3_TIE", ""), ",")))))
for k in keys(spec); k in FREE || error("unknown parameter $k"); end
function design(nm)
  m = get(spec, nm, dflt)
  m == "sp" && return Matrix{Float64}(I, nsp, nsp)
  m == "g" && return ones(nsp, 1)
  hcat(ones(nsp), [Z[t] for t in split(m, "+")]...)
end
X = [design(nm) for nm in FREE]; q = [size(x, 2) for x in X]; nred = sum(q) + NGLOB
lg(@sprintf("tie: %s | coordinates: per-species %d + global %d = %d (full %d)", join(["$(nm):$(get(spec, nm, dflt))" for nm in FREE], " "), sum(q), NGLOB, nred, nsp * NF + NGLOB))
off = cumsum(vcat(0, q))
function expand(z)
  θ = zeros(nsp * NF + NGLOB)
  for (j, nm) in enumerate(FREE)
    v = X[j] * z[off[j]+1:off[j+1]]
    for s in 1:nsp; θ[(s-1)*NF+j] = v[s]; end
  end
  θ[nsp*NF+1:end] = z[sum(q)+1:end]; θ
end
θs = read_theta(joinpath(START, "theta.csv"), nsp)
z0 = zeros(nred)
for (j, nm) in enumerate(FREE)
  y = [θs[(s-1)*NF+j] for s in 1:nsp]; z0[off[j]+1:off[j+1]] = X[j] \ y
end
z0[sum(q)+1:end] = θs[nsp*NF+1:end]
TRc = build_cohorts(tr, "train", keys_); VAc = build_cohorts(tr, "val", keys_)
function obj(θ, D); P, G = decode(θ, nsp, longev); simulate!(D.S, P, G); nll_parts(cohort_view(D), D.obs); end
pf = obj(θs, TRc); pfv = obj(θs, VAc)
lg(@sprintf("FULL start fit  train %.1f | val %.1f (mort %.1f D %.1f H %.1f)", pf.total, pfv.total, pfv.mort, pfv.D, pfv.H))
p0 = obj(expand(z0), TRc); p0v = obj(expand(z0), VAc)
lg(@sprintf("PROJECTED start train %.1f | val %.1f (mort %.1f D %.1f H %.1f)", p0.total, p0v.total, p0v.mort, p0v.D, p0v.H))
maxe = parse(Int, get(ENV, "HBAD2_EVALS", "6000"))
res = cmaes(z -> obj(expand(z), TRc).total, z0, parse(Float64, get(ENV, "HBAD2_SIGMA", "0.15")); λ=parse(Int, get(ENV, "HBAD2_LAMBDA", "24")), maxevals=maxe, logf=lg, log_every=25,
            callback=(g, b, fb) -> (g % 50 == 0 && CSV.write(joinpath(OUTD, "theta_reduced.csv"), DataFrame(theta=b))))
CSV.write(joinpath(OUTD, "theta_reduced.csv"), DataFrame(theta=res.x)); CSV.write(joinpath(OUTD, "theta.csv"), DataFrame(theta=expand(res.x)))
pt = obj(expand(res.x), TRc); pv = obj(expand(res.x), VAc)
k = nred; ntr = pt.ntrees
lg(@sprintf("FIT   train %.1f (mort %.1f D %.1f H %.1f) | VAL %.1f (mort %.1f D %.1f H %.1f σD %.4f σH %.4f) | coords %d | train BIC-ish %.1f | evals %d",
            pt.total, pt.mort, pt.D, pt.H, pv.total, pv.mort, pv.D, pv.H, pv.sigD, pv.sigH, k, 2pt.total + k * log(ntr), res.evals))
P, G = decode(expand(res.x), nsp, longev)
pdf = DataFrame([nm => [getfield(P[s], nm) for s in 1:nsp] for nm in HBAD2.SP_PARAM_NAMES]); insertcols!(pdf, 1, :key => keys_)
CSV.write(joinpath(OUTD, "params.csv"), pdf); lg(string(pdf)); lg(@sprintf("global mD2 %.3f pH %.3f pD %.3f pM %.3f mSD %.3f σv %.3f", G.mD2, G.pH, G.pD, G.pM, G.mSD, G.σv))
