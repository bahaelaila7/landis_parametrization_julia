# HBAD3 Sim A fit (TRAIN; VAL reported for selection) by CMA-ES. Per-species parameters (FREE3) are tied by HBAD3_TIE as in fit_tied.jl:
#   "sp" (species-specific), "g" (one global value), "T1+T2" (intercept + standardised trait slopes; traits ST WD SW HX DX LG).
# Default = the lean design of docs/hbad3_design.md §4. Globals GN3 are free unless fixed by HBAD3_GFIX="ω:0,γ:1" (decoded values).
# HBAD3_WATER=0 sets the site water index W ≡ 1 (no water modifier; ω then has no effect and is fixed at 0).
# Start: HBAD3_START=<fit3 dir> (its sv_full.csv / gv_full.csv, projected onto the tie design) or, by default, v2fv5 mapped to HBAD3.
#   HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot HBAD2_VIGOUR=1 HBAD2_SIGV_MAX=0.5 ./julia_gdal.sh --project=. --threads=4 tools/hbad3/fit3.jl <tag>
include(joinpath(@__DIR__, "common3.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
tag = ARGS[1]; OUTD = joinpath(R3DIR, "fit3_" * tag); mkpath(OUTD); write_meta(OUTD)
logio = open(joinpath(OUTD, "log.txt"), "w"); lg(s) = (println(s); println(logio, s); flush(logio); flush(stdout))
const DEFAULT_TIE = "eD:sp,cD:g,Dmax:g,lnK:ST,gR:g,mbg:WD+ST,mD:g,ms:ST,sdimax:g"
tiestr = get(ENV, "HBAD3_TIE", DEFAULT_TIE); WATER = get(ENV, "HBAD3_WATER", "1") == "1"
open(io -> (println(io, "tie=", tiestr); println(io, "water=", WATER ? 1 : 0); println(io, "sform=", SFORM); println(io, "gfix=", get(ENV, "HBAD3_GFIX", ""))), joinpath(OUTD, "meta.txt"), "a")
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read("/workspace/species_longevity_ref.csv", DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
TD = key_traits(tr, keys_, longev); CSV.write(joinpath(OUTD, "traits.csv"), TD)
hd = get(ENV, "HBAD3_HD", "csv") == "data" ? fit_hd(tr, keys_) : hd_table(keys_)       # data: fitted to this run's train trees (other regions)
CSV.write(joinpath(OUTD, "hd_allometry.csv"), DataFrame(key=keys_, a=first.(hd), b=getindex.(hd, 2), c=last.(hd)))
lg("region " * get(ENV, "HBAD2_REGION", "(none)") * " data " * DATA_DIR * " | keys " * join(keys_, ","))
spec = Dict(Symbol(p[1]) => String(p[2]) for p in (split(x, ":") for x in filter(!isempty, strip.(split(tiestr, ",")))))
for k in keys(spec); k in FREE3 || error("unknown per-species parameter $k"); end
X = [design_matrix(get(spec, nm, "sp"), TD) for nm in FREE3]; q = [size(x, 2) for x in X]; offq = cumsum(vcat(0, q))
gfix = Dict{Symbol,Float64}()
for x in filter(!isempty, strip.(split(get(ENV, "HBAD3_GFIX", ""), ",")))
  a, b = split(x, ":"); s = Symbol(a); s in GN3 || error("unknown global $a"); gfix[s] = parse(Float64, b)
end
WATER || (gfix[:ω] = 0.0)
SFORM == 4 || (gfix[:km] = 1.0)          # km acts only in the diagnostic decoupled stress form
for nm in (:mW, :τg, :τm); haskey(gfix, nm) || (gfix[nm] = 0.0); end   # climate terms: fitted on a climate gradient (tools/hbad3/fit_climate.jl), 0 here
gfree = [j for j in 1:length(GN3) if !haskey(gfix, GN3[j])]
nred = sum(q) + length(gfree)
lg(@sprintf("tie %s | water %s | fixed globals %s | coordinates: per-species %d + global %d = %d", tiestr, WATER, string(gfix), sum(q), length(gfree), nred))
# ---- start (encoded per-species table sv0 nsp × 9, encoded globals gv0)
function start_from_v2()
  n2 = 12;                       # start values only (no structural meta check: the run may use another sigv_max / region)
  θ = read_theta(joinpath(R2DIR, "fit_v2fv5", "theta.csv"), n2); P2, G2 = decode(θ, n2, fill(275.0, n2))
  keys2 = sort(String.(CSV.read(joinpath(R2DIR, "fit_v2j", "params_species.csv"), DataFrame).key))
  sv = zeros(nsp, length(FREE3))
  for s in 1:nsp
    # a key of another region without a GA+FL fit starts from the GA+FL group key of its softwood / hardwood class
    k = keys_[s]; j = findfirst(==(k), keys2); j === nothing && (j = findfirst(==("_" * (TD.SW[s] > 0.5 ? "S" : "H")), keys2))
    p = P2[j]
    v = Dict(:eD => log(p.rD), :cD => p.cD, :Dmax => p.Dmax, :lnK => 0.0, :gR => p.gRD, :mbg => p.mbg, :mD => p.mD, :ms => p.msup, :sdimax => p.sdimax)
    for (j, nm) in enumerate(FREE3); sv[s, j] = enc3(nm, dec3(nm, enc3(nm, v[nm]))); end
  end
  g = Dict(:k => 3.0, :η => 0.15, :λ => 0.015, :μ => 0.5, :γ => (SFORM == 2 ? 8.0 : SFORM == 3 ? 3.0 : 1.0), :mD2 => G2.mD2, :mSD => G2.mSD, :pH => G2.pH, :pD => G2.pD, :pM => G2.pM, :σv => G2.σv, :ω => 1.0, :km => 3.0, :mW => 0.0, :τg => 0.0, :τm => 0.0)
  sv, [genc3(nm, g[nm]) for nm in GN3]
end
st = get(ENV, "HBAD3_START", "")
sv0, gv0 = isempty(st) ? start_from_v2() : (Matrix(CSV.read(joinpath(st, "sv_full.csv"), DataFrame)[:, 2:end]), vec(CSV.read(joinpath(st, "gv_full.csv"), DataFrame).value))
length(gv0) < length(GN3) && (gv0 = vcat(gv0, [genc3(GN3[j], GDEFAULT3[GN3[j]]) for j in length(gv0)+1:length(GN3)]))   # older fits
for (s, v) in gfix; gv0[findfirst(==(s), GN3)] = genc3(s, v); end
z0 = zeros(nred)
const CONST3 = Dict(j => enc3(FREE3[j], parse(Float64, get(spec, FREE3[j], "sp")[2:end])) for j in 1:length(FREE3) if startswith(get(spec, FREE3[j], "sp"), "="))
for j in 1:length(FREE3); q[j] > 0 && (z0[offq[j]+1:offq[j+1]] = X[j] \ sv0[:, j]); end
z0[sum(q)+1:end] = gv0[gfree]
function expand(z)
  sv = zeros(nsp, length(FREE3))
  for j in 1:length(FREE3); sv[:, j] = haskey(CONST3, j) ? fill(CONST3[j], nsp) : X[j] * z[offq[j]+1:offq[j+1]]; end
  gv = copy(gv0); gv[gfree] = z[sum(q)+1:end]; sv, gv
end
wd = water_by_iid()
TRc = build3(tr, "train", keys_, wd); VAc = build3(tr, "val", keys_, wd)
WATER || (set_water!(TRc, 1.0); set_water!(VAc, 1.0))
function obj(z, C); sv, gv = expand(z); P, G = decode3(sv, gv, hd, longev); eval3(C, P, G); end
for _ in 1:2; obj(z0, TRc); end
t0 = time(); for _ in 1:5; obj(z0, TRc); end; lg(@sprintf("eval time train %.1f ms", (time() - t0) * 200))
p0 = obj(z0, TRc); p0v = obj(z0, VAc)
lg(@sprintf("START train %.1f (mort %.1f D %.1f H %.1f) | val %.1f (mort %.1f D %.1f H %.1f)", p0.total, p0.mort, p0.D, p0.H, p0v.total, p0v.mort, p0v.D, p0v.H))
lg("reference v2fv5 (175 coordinates): train -273425.4 | val -272655.4 (mort 32960.9 D -163997.3 H -141619.0)")
# null model (as tools/hbad2/fit_simA.jl): no growth, per-key constant annual survival (MLE on train) — the scale of the gain
function null_parts(C, surv)
  nm = 0.0; sD = 0.0; sH = 0.0; c = 0; co = C.coh
  for i in 1:nrow(co)
    n1 = co.n1[i]; n1 == 0 && continue; s = co.s[i]; p = clamp(surv[co.spi[i]]^co.dt[i], 1e-9, 1 - 1e-9)
    nm -= s * log(p) + (n1 - s) * log(1 - p)
    s > 0 && (sD += s * log(co.D1[i] / co.D2o[i])^2; sH += s * log(co.H1[i] / co.H2o[i])^2; c += 1)
  end
  nm + 0.5c * log(sD / c) + 0.5c * log(sH / c)
end
surv = [begin m = TRc.coh[TRc.coh.spi .== s, :]; (sum(m.s) / sum(m.n1))^(1 / mean(m.dt)) end for s in 1:nsp]
NULLT = null_parts(TRc, surv); NULLV = null_parts(VAc, surv)
lg(@sprintf("NULL  train %.1f | val %.1f", NULLT, NULLV))
maxe = parse(Int, get(ENV, "HBAD2_EVALS", "6000"))
function save(z)
  sv, gv = expand(z)
  CSV.write(joinpath(OUTD, "theta_reduced.csv"), DataFrame(theta=z))
  CSV.write(joinpath(OUTD, "sv_full.csv"), hcat(DataFrame(key=keys_), DataFrame(sv, collect(String.(FREE3)))))
  CSV.write(joinpath(OUTD, "gv_full.csv"), DataFrame(name=collect(String.(GN3)), value=gv))
end
res = cmaes(z -> obj(z, TRc).total, z0, parse(Float64, get(ENV, "HBAD2_SIGMA", "0.15")); λ=parse(Int, get(ENV, "HBAD2_LAMBDA", "24")), maxevals=maxe, logf=lg, log_every=25,
            callback=(g, b, fb) -> (g % 50 == 0 && save(b)))
save(res.x)
pt = obj(res.x, TRc); pv = obj(res.x, VAc)
lg(@sprintf("FIT   train %.1f (mort %.1f D %.1f H %.1f) | VAL %.1f (mort %.1f D %.1f H %.1f σD %.4f σH %.4f) | coords %d | train BIC %.1f | evals %d",
            pt.total, pt.mort, pt.D, pt.H, pv.total, pv.mort, pv.D, pv.H, pv.sigD, pv.sigH, nred, 2pt.total + nred * log(pt.ntrees), res.evals))
lg(@sprintf("GAIN over null: train %.1f (%.1f %% of |null|) | val %.1f; val gain per tracked tree %.4f", NULLT - pt.total, 100 * (NULLT - pt.total) / abs(NULLT), NULLV - pv.total, (NULLV - pv.total) / pv.ntrees))
sv, gv = expand(res.x); P, G = decode3(sv, gv, hd, longev)
# coordinates at their bounds (a sign of mis-specification / non-identifiability)
atb = String[]
for (j, nm) in enumerate(FREE3), s in 1:nsp
  haskey(CONST3, j) && continue
  lo, hi = BOUNDS3[nm]; v = getfield(P[s], nm); (isapprox(v, lo; rtol=1e-6) || isapprox(v, hi; rtol=1e-6)) && push!(atb, "$(keys_[s]):$(nm)")
end
for nm in GN3
  haskey(gfix, nm) && continue; lo, hi = GBOUNDS3[nm]; v = getfield(G, nm); (isapprox(v, lo; rtol=1e-6) || isapprox(v, hi; rtol=1e-6)) && push!(atb, "G:$(nm)")
end
lg("AT BOUNDS: " * (isempty(atb) ? "none" : join(unique(atb), " ")))
pdf = DataFrame([nm => [getfield(P[s], nm) for s in 1:nsp] for nm in HBAD3.SP3_NAMES]); insertcols!(pdf, 1, :key => keys_)
CSV.write(joinpath(OUTD, "params.csv"), pdf); lg(string(pdf)); lg(string(G))
r = vcat(export_resid3(TRc, P, G, "train"), export_resid3(VAc, P, G, "val")); CSV.write(joinpath(R3DIR, "resid_fit3_" * tag * ".csv"), r)
lg("wrote resid_fit3_$(tag).csv")
