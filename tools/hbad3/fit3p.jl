# HBAD3P (process-grounded HBAD3) Sim A fit on one region by CMA-ES: per species key only ln ε (growth efficiency); every other parameter global; species
# traits Hmax (99.5th percentile height of the key's TRAIN trees), shade class ST (FIA crown class table), LONGEVITY (table). Water W = period AET/PET,
# Wn = its plot normal (scripts/hbad3/water_balance.py). Null baseline, gain, coordinates at bounds and the fitted globals are logged.
#   HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot HBAD2_VIGOUR=1 HBAD2_SIGV_MAX=0.7 HBAD2_DATA_DIR=results/hbad2_r4 HBAD2_REGION=<reg|L3:x.y.z>
#   HBAD3_WATER_CSV=results/hbad3/interval_water_r4.csv [HBAD3P_FIX="ω:0,mW:0"] ./julia_gdal.sh --project=. --threads=4 tools/hbad3/fit3p.jl <tag>
include(joinpath(@__DIR__, "common3.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD3", "HBAD3P.jl")); import .HBAD3P
include(joinpath(@__DIR__, "lib3p.jl"))
# HBAD3P_FAREA (default 1): stand densities per ha of the FORESTED subplot (lib3p.jl); recorded in meta.txt, read by every later stage
const FAREA = get(ENV, "HBAD3P_FAREA", "1") == "1"
tag = ARGS[1]; OUTD = joinpath(R3DIR, "fit3p_" * tag); mkpath(OUTD); write_meta(OUTD); open(io -> println(io, "farea=", FAREA ? 1 : 0), joinpath(OUTD, "meta.txt"), "a")
logio = open(joinpath(OUTD, "log.txt"), "w"); lg(s) = (println(s); println(logio, s); flush(logio); flush(stdout))
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
lon = CSV.read(LONGEVITY_CSV, DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
longev = [median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
lg_defaults = sort(unique([s for s in tr.sym if !haskey(lond, s)]))   # species without a LONGEVITY entry (300 yr used)
TD = key_traits(tr, keys_, longev)
Hmax = [quantile(tr.ht_m[(tr.key .== k) .& (tr.set .== "train")], 0.995) for k in keys_]
lg("region " * get(ENV, "HBAD2_REGION", "(none)") * " | keys " * join(keys_, ",") * " | Hmax " * join(round.(Hmax; digits=1), ","))
lg("species without LONGEVITY (300 yr used): " * (isempty(lg_defaults) ? "none" : join(lg_defaults, ",")))
# ---- globals: name, start, lo, hi, log-encoded
# HBAD3P_COUPLED=1: the survival light coefficient am is tied to the growth one (am = a, the original coupled form)
const COUPLED = get(ENV, "HBAD3P_COUPLED", "0") == "1"
const GSPEC = [(:b, 1.88, 1.0, 3.0, false), (:a, 2.6, 0.1, 10.0, true), (:gR, -0.1, -2.0, 1.0, false), (:θ, 8.0, 1.0, 40.0, true), (:ω, 0.2, 0.0, 5.0, false),
               (:s, 1.0, 0.2, 3.0, true), (:c, 0.85, 0.3, 2.0, true), (:η, 0.0, -1.0, 2.0, false), (:λ, 0.026, 1e-3, 0.3, true), (:μ, 0.3, -0.5, 2.0, false), (:kh, 6.0, 0.2, 30.0, true),
               (:m0, 1.2, 0.01, 10.0, true), (:ms, 0.035, 1e-4, 1.0, true), (:σST, -0.2, -2.0, 2.0, false), (:β, 2.6, 0.1, 20.0, true), (:ζ, -0.65, -3.0, 2.0, false),
               (:mW, 0.3, -2.0, 3.0, false), (:pD, 0.1, -1.0, 1.0, false), (:pH, 0.3, -1.0, 1.0, false), (:pM, -0.2, -2.0, 2.0, false), (:σv, 0.5, 0.0, SIGV_MAX, false), (:am, 2.6, 0.1, 20.0, true), (:ρ, 1e-4, 1e-7, 0.05, true), (:f0, 0.05, 0.0, 0.6, false),
               (:φ, 0.5, 0.0, 2.0, false), (:crhi, 0.5, 0.2, 0.95, false), (:crlo, 0.33, 0.05, 0.8, false), (:kc, 3.0, 0.1, 30.0, true), (:τr, 0.1, 1e-3, 1.0, true), (:τe, 0.01, 1e-4, 0.5, true)]
gfix = Dict{Symbol,Float64}()
for x in filter(!isempty, strip.(split(get(ENV, "HBAD3P_FIX", ""), ",")))
  a_, b_ = split(x, ":"); gfix[Symbol(a_)] = parse(Float64, b_)
end
COUPLED && (gfix[:am] = NaN)
gfree = [j for j in eachindex(GSPEC) if !haskey(gfix, GSPEC[j][1])]
genc(j, v) = GSPEC[j][5] ? log(v) : v
function gdec(j, z); lo, hi = GSPEC[j][3], GSPEC[j][4]; clamp(GSPEC[j][5] ? exp(z) : (GSPEC[j][1] == :σv ? abs(z) : z), lo, hi); end
gv0 = [genc(j, (v = get(gfix, GSPEC[j][1], GSPEC[j][2]); isnan(v) ? GSPEC[j][2] : v)) for j in eachindex(GSPEC)]
# ---- data
wp = water_pairs()
IFOR = FAREA ? interval_forest(DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))), forest_table()) : nothing
TRc = sima_sites(tr, "train", keys_, wp, IFOR); VAc = sima_sites(tr, "val", keys_, wp, IFOR)
# ---- start ε per key: open-grown-like cohorts' volume increment at the start globals (median ratio)
function eps_start()
  co = TRc.coh; o = TRc.obs; r = [Float64[] for _ in 1:nsp]
  for i in 1:nrow(co)
    o.s[i] > 0 || continue; dD = (o.D2o[i] - co.D1[i]) / o.dt[i]; dD > 0 || continue
    push!(r[co.spi[i]], dD * 2 * co.D1[i] * max(co.H1[i], 1.37) / ((co.D1[i] / 20)^1.88 * 122.98 * 0.5))
  end
  [log(median(x)) for x in r]
end
z0 = vcat(eps_start(), gv0[gfree])
znames = vcat(["ε:" * k for k in keys_], [String(GSPEC[j][1]) for j in gfree])   # coordinate names, saved with z.csv
# HBAD3P_START=<fit3p dir>: warm start from its z.csv (same region/keys), coordinates matched BY NAME; a coordinate absent from the start fit keeps its
# start value (am: the start fit's a). Older z.csv without names: positional, accepted only for the full layout of that time (nsp + 30 or nsp + 24).
if haskey(ENV, "HBAD3P_START")
  zt = CSV.read(joinpath(ENV["HBAD3P_START"], "z.csv"), DataFrame)
  if hasproperty(zt, :name)
    zd = Dict(String(n) => v for (n, v) in zip(zt.name, zt.z))
  else
    nz = nrow(zt); nfull = nsp + length(GSPEC)
    nz in (nfull, nfull - 6) || error("HBAD3P_START: unnamed z.csv of length $(nz) (layout ambiguous: expected $(nfull) or $(nfull - 6)); refit or add names")
    zd = Dict(n => v for (n, v) in zip(vcat(["ε:" * k for k in keys_], [String(GSPEC[j][1]) for j in 1:nz-nsp]), zt.z))
  end
  !haskey(zd, "am") && haskey(zd, "a") && (zd["am"] = zd["a"])
  miss = [n for n in znames if !haskey(zd, n)]; isempty(miss) || lg("HBAD3P_START: not in the start fit (start values): " * join(miss, ","))
  z0 = [get(zd, n, z0[i]) for (i, n) in enumerate(znames)]
end
function params(z)
  gv = copy(gv0); gv[gfree] = z[nsp+1:end]; g = Dict(GSPEC[j][1] => gdec(j, gv[j]) for j in eachindex(GSPEC))
  P = [HBAD3P.SpPP(clamp(z[k], -8.0, 6.0), Hmax[k], TD.ST[k], longev[k]) for k in 1:nsp]
  G = HBAD3P.GlobPP(g[:b], g[:a], g[:gR], g[:θ], g[:ω], g[:s], g[:c], g[:η], g[:λ], g[:μ], g[:kh], g[:m0], g[:ms], g[:σST], g[:β], g[:ζ], g[:mW], 10.0,
                    g[:pD], g[:pH], g[:pM], g[:σv], 1500.0, COUPLED ? g[:a] : g[:am], g[:ρ], g[:f0], g[:φ], g[:crhi], g[:crlo], g[:kc], g[:τr], g[:τe])
  P, G
end
function obj(z, C); P, G = params(z); HBAD3P.simulate!(C.S, P, G); nll_parts(cohort_view(C), C.obs); end
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
for _ in 1:2; obj(z0, TRc); end
t0 = time(); for _ in 1:5; obj(z0, TRc); end
p0 = obj(z0, TRc); p0v = obj(z0, VAc)
lg(@sprintf("coords %d (per key %d + global %d; fixed %s) | eval %.0f ms | NULL train %.1f val %.1f | START train %.1f val %.1f", length(z0), nsp, length(gfree), string(gfix), (time() - t0) * 200, NULLT, NULLV, p0.total, p0v.total))
function save(z)
  P, G = params(z)
  CSV.write(joinpath(OUTD, "species.csv"), DataFrame(key=keys_, eps=exp.(z[1:nsp]), Hmax=Hmax, ST=TD.ST, longevity=longev))
  CSV.write(joinpath(OUTD, "globals.csv"), DataFrame(name=String.(collect(fieldnames(HBAD3P.GlobPP))), value=[getfield(G, f) for f in fieldnames(HBAD3P.GlobPP)]))
  CSV.write(joinpath(OUTD, "z.csv"), DataFrame(name=znames, z=z))
end
res = cmaes(z -> obj(z, TRc).total, z0, parse(Float64, get(ENV, "HBAD2_SIGMA", "0.15")); λ=parse(Int, get(ENV, "HBAD2_LAMBDA", "24")), maxevals=parse(Int, get(ENV, "HBAD2_EVALS", "6000")),
            logf=lg, log_every=25, callback=(g, b, fb) -> (g % 50 == 0 && save(b)))
save(res.x)
pt = obj(res.x, TRc); pv = obj(res.x, VAc)
lg(@sprintf("FIT   train %.1f (mort %.1f D %.1f H %.1f) | VAL %.1f (mort %.1f D %.1f H %.1f σD %.4f σH %.4f) | evals %d", pt.total, pt.mort, pt.D, pt.H, pv.total, pv.mort, pv.D, pv.H, pv.sigD, pv.sigH, res.evals))
lg(@sprintf("GAIN over null: train %.1f | val %.1f; val gain per tracked tree %.4f", NULLT - pt.total, NULLV - pv.total, (NULLV - pv.total) / pv.ntrees))
if any(==("test"), tr.set)   # nested CV: the outer test fold, scored once after the fit (report only)
  TEc = sima_sites(tr, "test", keys_, wp, IFOR); NULLE = null_parts(TEc, surv); pe = obj(res.x, TEc)
  lg(@sprintf("TEST  %.1f (mort %.1f D %.1f H %.1f) | NULL %.1f | gain %.1f; test gain per tracked tree %.4f", pe.total, pe.mort, pe.D, pe.H, NULLE, NULLE - pe.total, (NULLE - pe.total) / pe.ntrees))
end
P, G = params(res.x); lg(string(G))
atb = [String(GSPEC[j][1]) for j in gfree if (v = getfield(G, GSPEC[j][1]); isapprox(v, GSPEC[j][3]; atol=1e-9, rtol=1e-6) || isapprox(v, GSPEC[j][4]; atol=1e-9, rtol=1e-6))]
append!(atb, [keys_[k] * ":ε" for k in 1:nsp if res.x[k] <= -7.999 || res.x[k] >= 5.999])   # ε clamp in params(): [−8, 6]
lg("AT BOUNDS: " * (isempty(atb) ? "none" : join(atb, " ")))
