# HBAD4P harness (docs/hbad4p_design.md): plot stand tables of the natural remeasurement intervals of the interval product (HBAD2_DATA_DIR:
# intervals.arrow with the nested-CV `set`, interval_trees.arrow, interval_ingrowth.arrow) → HBAD4P.Sites4 (v1 state) + the v2 stand table (observation),
# and the stand-table negative log-likelihood. Natural = the HBAD3P filter (no treatment / disturbance / harvest code, dt > 2); region HBAD4_REGION (GAFL).
# Every quantity estimated from data (keys, traits, the sapling prior, the observation spreads) comes from the TRAIN set.
using Arrow, DataFrames, CSV, Statistics, Printf, Dates, Random
import Distributions; const loggamma = Distributions.SpecialFunctions.loggamma   # (SpecialFunctions is not a direct dependency of the project)
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD4", "HBAD4P.jl")); import .HBAD4P
const NCLS = HBAD4P.NCLS; const NL = HBAD4P.NL
const WSROOT = get(ENV, "WORKSPACE", normpath(joinpath(@__DIR__, "..", "..", "..")))
const DATA_DIR = get(ENV, "HBAD2_DATA_DIR", joinpath(@__DIR__, "..", "..", "results", "hbad2_r4"))
const H3DATA = get(ENV, "HBAD4_H3DATA", joinpath(WSROOT, "landis_hbad3", "results", "hbad3"))       # HBAD3 climate / water / forest tables (R4)
const WATER_CSV = get(ENV, "HBAD3_WATER_CSV", joinpath(H3DATA, "interval_water_r4.csv"))
const CLIM_CSV = get(ENV, "HBAD3_CLIM_CSV", joinpath(H3DATA, "interval_climate_r4.csv"))
const FOREST_CSV = get(ENV, "HBAD3_FOREST_CSV", joinpath(H3DATA, "subplot_forest_r4.csv"))
const CO2_CSV = get(ENV, "HBAD4_CO2_CSV", joinpath(WSROOT, "landis_region4", "data_r4", "co2_annual.csv"))
const SHADETOL_CSV = get(ENV, "HBAD3_SHADETOL_CSV", joinpath(WSROOT, "landis_region4", "data_r4", "shadetol_all_species.csv"))
const LONGEVITY_CSV = get(ENV, "HBAD3_LONGEVITY_CSV", joinpath(WSROOT, "species_longevity_ref.csv"))
const REGION = get(ENV, "HBAD4_REGION", "GAFL")
const MIN_SP_TREES = parse(Int, get(ENV, "HBAD2_MIN_SP_TREES", "3000"))   # own key with ≥ this many TRAIN trees, else _S / _H (as HBAD3P)
const LANEMODE = Symbol(get(ENV, "HBAD4P_TRIPLE", "sym"))                 # vigour lanes :sym (0.5 / 0.25 / 0.25) | :fvs (FVS SN 0.60 / 0.25 / 0.15)
const FMIN = 0.25                                                         # smallest effective forested fraction (as HBAD3P)
# tally areas of the FIA design from its expansion factors (TPA_UNADJ = 1 / sampled acres): subplot trees (D ≥ 12.7 cm) 6.018046 / ac, microplot
# saplings 74.965282 / ac → the 4-subplot / 4-microplot areas in ha (1 tree = 1 / area trees per ha, as the w_ha of the product)
const AREA_SUB = 1 / (6.018046 * 2.4710538); const AREA_MIC = 1 / (74.965282 * 2.4710538)
area_class(j) = j >= 3 ? AREA_SUB : AREA_MIC
ismicro(w_ha) = w_ha > 50                                                 # tally of a tree from its weight (microplot 185.2 / ha, subplot 14.87 / ha)

inregion(st) = REGION == "GAFL" ? st in (12, 13) : REGION == "R4" ? true : error("HBAD4_REGION $REGION: GAFL | R4")
plotf(v) = (x = [isnan(a) ? 1.0 : a for a in v]; m = isempty(x) ? 1.0 : mean(x); m <= 0 ? 1.0 : max(m, FMIN))   # forested fraction of a plot

"Load the natural intervals of the region with their trees and ingrowth; species keys from the TRAIN trees (the HBAD3P rule)."
function load4(dir=DATA_DIR)
  iv = DataFrame(Arrow.Table(joinpath(dir, "intervals.arrow")))
  iv = iv[.!(iv.trt .| iv.dst .| iv.harv) .& (iv.dt .> 2) .& inregion.(iv.st), :]
  iv.set = coalesce.(iv.set, ""); ok = Set(iv.iid)
  tr = DataFrame(Arrow.Table(joinpath(dir, "interval_trees.arrow"))); tr = tr[in.(tr.iid, Ref(ok)) .& (tr.dia_cm .> 0), :]
  ig = DataFrame(Arrow.Table(joinpath(dir, "interval_ingrowth.arrow"))); ig = ig[in.(ig.iid, Ref(ok)) .& (ig.dia_cm .> 0), :]
  sset = Dict(r.iid => r.set for r in eachrow(iv)); tr.set = getindex.(Ref(sset), tr.iid)
  ktr = tr[(tr.set .== "train") .& (tr.age .> 0) .& (tr.ht_m .> 0), :]
  cnt = combine(groupby(ktr, :sym), nrow => :n); big = Set(cnt.sym[cnt.n .>= MIN_SP_TREES])
  keyof(s, g) = s in big ? String(s) : "_" * String(g)
  tr.key = keyof.(tr.sym, tr.grp); ig.key = keyof.(ig.sym, ig.grp)
  keys_ = sort(unique(vcat(tr.key, ig.key)))
  (iv=iv, tr=tr, ig=ig, keys=keys_, big=big)
end

"""Traits per key from the TRAIN trees: Hmax (99.5th pct height), shade class and LONGEVITY (tree-weighted over member species), Topt and Tref.
Topt = the key's range temperature optimum: the tree-weighted mean annual temperature NORMAL of its intervals — a named species over the whole product
(Region 4: every interval outside the region + the region's train intervals); a group key (_H / _S) over the region's train intervals only (its members
elsewhere are other species). Tref = the tree-weighted mean PERIOD temperature of its train intervals (f_T = 1 there). Temperatures are EPA L4 means."""
function traits4(L, clim)
  st = CSV.read(SHADETOL_CSV, DataFrame); std_ = Dict(String(r.sym) => Float64(r.shade_class) for r in eachrow(st))
  lon = CSV.read(LONGEVITY_CSV, DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
  trn = L.tr[L.tr.set .== "train", :]
  tn = Dict(r.iid => r.tmean_n for r in eachrow(clim)); tiv = Dict(r.iid => r.tmean_iv for r in eachrow(clim))
  ivall = DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))); okT = Set(ivall.iid[.!inregion.(ivall.st) .| (coalesce.(ivall.set, "") .== "train")])
  tall = DataFrame(Arrow.Table(joinpath(DATA_DIR, "interval_trees.arrow")))
  tall = tall[in.(tall.iid, Ref(okT)) .& haskey.(Ref(tn), tall.iid) .& in.(tall.sym, Ref(L.big)), :]
  map(L.keys) do k
    t = trn[trn.key .== k, :]; c = combine(groupby(t, :sym), nrow => :n)
    wavg(d, dflt) = (v = [get(d, s, dflt) for s in c.sym]; sum(v .* c.n) / sum(c.n))
    hs = t.ht_m[t.ht_m .> 0]; tt = t[haskey.(Ref(tn), t.iid), :]
    topt = startswith(k, "_") ? mean(getindex.(Ref(tn), tt.iid)) : mean(getindex.(Ref(tn), tall.iid[tall.sym .== k]))
    (key=k, Hmax=quantile(hs, 0.995), ST=wavg(std_, 3.0), longevity=median([get(lond, s, 300.0) for s in c.sym]), Topt=topt,
     Tref=mean(getindex.(Ref(tiv), tt.iid)), ntrain=nrow(t))
  end |> DataFrame
end

"Plot forested fractions of every interval of a set: (f1 subplot, fm1 microplot at v1; fs2, fm2 at v2)."
function fractions4(iv, FT)
  f1 = zeros(nrow(iv)); fm1 = similar(f1); fs2 = similar(f1); fm2 = similar(f1)
  for (s, r) in enumerate(eachrow(iv))
    a1 = [get(FT, (r.st, r.u, r.c, r.p, r.d1, sb), (NaN, NaN)) for sb in 1:4]; a2 = [get(FT, (r.st, r.u, r.c, r.p, r.d2, sb), (NaN, NaN)) for sb in 1:4]
    f1[s] = plotf(first.(a1)); fm1[s] = plotf(last.(a1)); fs2[s] = plotf(first.(a2)); fm2[s] = plotf(last.(a2))
  end
  f1, fm1, fs2, fm2
end
"v1 trees of a set that enter the state (alive at v1; outcome not missing / harvest — those have no v2 record)."
v1trees(L, iids) = L.tr[in.(L.tr.iid, Ref(iids)) .& .!in.(L.tr.out, Ref(("missing", "harvest"))), :]

"""Empirical-Bayes prior of the v1 SAPLING layer (classes 1–2), fitted on the TRAIN intervals. The microplots sample 8 % of the subplot area, so a v1 count is a
noisy sample of the stand's sapling density (46 % of the ingrowth ≥ 12.7 cm comes from keys with no v1 microplot sapling): per group g = (key, class,
adult: a tree of the key ≥ 12.7 cm on the plot at v1) n ~ NegBin(mean m_g e, size α_g), e = microplot area × forested fraction; m_g = Σn / Σe, α_g by
profile likelihood. Posterior mean density N̂ = (n + α) / (e + α / m). Also the train mean QMD / height / age of a key's saplings per class (the attributes
of prior saplings of an unobserved cell)."""
function sapling_prior(L, keys_, FT)
  iv = L.iv[L.iv.set .== "train", :]; sort!(iv, :iid); nk = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
  _, fm1, _, _ = fractions4(iv, FT); sidx = Dict(r.iid => s for (s, r) in enumerate(eachrow(iv)))
  t1 = v1trees(L, keys(sidx)); ns = nrow(iv)
  n = zeros(Int, nk, 2, ns); adult = falses(nk, ns); sD2 = zeros(nk, 2); sH = zeros(nk, 2); sA = zeros(nk, 2); cH = zeros(nk, 2); cA = zeros(nk, 2); c = zeros(nk, 2)
  for r in eachrow(t1)
    s = sidx[r.iid]; k = kix[r.key]; r.dia_cm >= 12.7 && (adult[k, s] = true)
    j = HBAD4P.dclass(r.dia_cm); (j in (1, 2) && ismicro(r.w_ha)) || continue
    n[k, j, s] += 1; c[k, j] += 1; sD2[k, j] += r.dia_cm^2
    r.ht_m > 0 && (sH[k, j] += r.ht_m; cH[k, j] += 1); r.age > 0 && (sA[k, j] += r.age; cA[k, j] += 1)
  end
  e = AREA_MIC .* fm1
  m = zeros(nk, 2, 2); α = fill(Inf, nk, 2, 2)
  for k in 1:nk, j in 1:2, a in (false, true)
    idx = [s for s in 1:ns if adult[k, s] == a]; isempty(idx) && continue
    y = n[k, j, idx]; x = e[idx]; mg = sum(y) / sum(x); m[k, j, a+1] = mg
    mg > 0 || continue
    ll(lα) = (r = exp(lα); sum(loggamma(yi + r) - loggamma(r) + r * log(r / (r + mg * xi)) + yi * log(mg * xi / (r + mg * xi)) for (yi, xi) in zip(y, x)))
    lo, hi = -8.0, 8.0                                                     # golden-section on log α
    for _ in 1:80
      a1 = hi - 0.618 * (hi - lo); a2 = lo + 0.618 * (hi - lo); ll(a1) > ll(a2) ? (hi = a2) : (lo = a1)
    end
    α[k, j, a+1] = exp((lo + hi) / 2)
  end
  pooled(x, cnt, k, j) = cnt[k, j] > 0 ? x[k, j] / cnt[k, j] : sum(x[:, j]) / max(sum(cnt[:, j]), 1)
  (m=m, α=α, D=[sqrt(pooled(sD2, c, k, j)) for k in 1:nk, j in 1:2], H=[pooled(sH, cH, k, j) for k in 1:nk, j in 1:2],
   A=[pooled(sA, cA, k, j) for k in 1:nk, j in 1:2])
end
@inline eb_density(n, e, m, α) = m > 0 ? (isinf(α) ? m : (n + α) / (e + α / m)) : (e > 0 ? n / e : 0.0)

"""Sites (v1 plot stand tables) and observations (v2 stand tables) of one set. Per site every key × class × lane is allocated (HBAD4P layout). v1 trees →
(key, class): N = Σ w / f (f = the forested fraction of the tree's tally: subplot or microplot), B = Σ w π/4 (D/100)² / f, H = Σ w H / Σ w (trees with a
height), A = Σ w A / Σ w (aged trees; else the key's TRAIN median age); the sapling classes 1–2 take the empirical-Bayes density of `eb` (QMD, H, A of the
observed saplings, or the key's train means when none was tallied); split over the lanes by their weights. Observation per (site, key, class): tally count n,
Σ D², Σ log D, Σ (log D)², Σ H, Σ log H, Σ (log H)² (n_H trees with a height) of the v2 trees (survivors with their v2 diameter + ingrowth), the forested fraction
of each v2 tally (subplot for D ≥ 12.7 cm, microplot below)."""
function build4(L, set::AbstractString, keys_, CL, FT, eb)
  iv = L.iv[L.iv.set .== set, :]; sort!(iv, :iid); ns = nrow(iv); nk = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
  lz, lw = HBAD4P.lanes(LANEMODE)
  ncell = nk * NCLS * NL; off = [1 + (s - 1) * ncell for s in 1:ns+1]; ntot = ns * ncell
  sp = Int32[k for _ in 1:ns for k in 1:nk for _ in 1:NCLS for _ in 1:NL]
  cl = Int8[j for _ in 1:ns for _ in 1:nk for j in 1:NCLS for _ in 1:NL]; ln = Int8[v for _ in 1:ns for _ in 1:nk for _ in 1:NCLS for v in 1:NL]
  N0 = zeros(ntot); B0 = zeros(ntot); H0 = zeros(ntot); A0 = zeros(ntot)
  nyr = floor.(Int, iv.dt); frac = iv.dt .- nyr
  sidx = Dict(r.iid => s for (s, r) in enumerate(eachrow(iv)))
  trn = L.tr[L.tr.set .== "train", :]; medage = Dict(k => median(trn.age[(trn.key .== k) .& (trn.age .> 0)]) for k in keys_)
  f1, fm1, fs2, fm2 = fractions4(iv, FT)
  t1 = v1trees(L, keys(sidx))
  acc = Dict{Tuple{Int,Int,Int},NTuple{7,Float64}}()     # (s, k, j) → (Σw, ΣwB, ΣwH, Σw|H, ΣwA, Σw|A, n tally trees)
  seed0 = falses(nk, ns)
  for r in eachrow(t1)
    j = HBAD4P.dclass(r.dia_cm); j == 0 && continue
    s = sidx[r.iid]; k = kix[r.key]; r.dia_cm >= 12.7 && (seed0[k, s] = true)
    w = r.w_ha / (ismicro(r.w_ha) ? fm1[s] : f1[s]); a = get(acc, (s, k, j), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    hok = r.ht_m > 0; aok = r.age > 0
    acc[(s, k, j)] = (a[1] + w, a[2] + w * π / 4 * (r.dia_cm / 100)^2, a[3] + (hok ? w * r.ht_m : 0.0), a[4] + (hok ? w : 0.0), a[5] + (aok ? w * r.age : 0.0),
                      a[6] + (aok ? w : 0.0), a[7] + 1)
  end
  nohgt = 0
  setcell!(s, k, j, N, B, H, A) = for v in 1:NL
    c = off[s] + ((k - 1) * NCLS + (j - 1)) * NL + (v - 1); N0[c] = N * lw[v]; B0[c] = B * lw[v]; H0[c] = H; A0[c] = A
  end
  for ((s, k, j), a) in acc
    j <= 2 && continue                                                     # (sapling classes: below)
    H = a[4] > 0 ? a[3] / a[4] : NaN; isnan(H) && (nohgt += 1; H = 1.37 + 0.5 * HBAD4P.CLS_LO[j])   # (no height in the cell: a crude placeholder, counted)
    setcell!(s, k, j, a[1], a[2], H, a[6] > 0 ? a[5] / a[6] : medage[keys_[k]])
  end
  neb = 0.0; nobs_sap = 0.0
  for s in 1:ns, k in 1:nk, j in 1:2                                       # sapling layer: empirical-Bayes density
    a = get(acc, (s, k, j), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)); n_ = a[7]; e = AREA_MIC * fm1[s]
    Nh = eb_density(n_, e, eb.m[k, j, seed0[k, s]+1], eb.α[k, j, seed0[k, s]+1]); Nh > 0 || continue
    neb += Nh; nobs_sap += a[1]
    if n_ > 0
      setcell!(s, k, j, Nh, a[2] * Nh / a[1], a[4] > 0 ? a[3] / a[4] : eb.H[k, j], a[6] > 0 ? a[5] / a[6] : eb.A[k, j])
    else
      setcell!(s, k, j, Nh, HBAD4P.ba(Nh, eb.D[k, j]), eb.H[k, j], eb.A[k, j])
    end
  end
  # v2 observation: survivors (v2 diameter / height) + ingrowth, each in the class of its v2 diameter (its v2 tally: FIA re-tallies a tree that shrank below
  # 5 in as a microplot sapling)
  On = zeros(Int32, nk, NCLS, ns); OD2 = zeros(nk, NCLS, ns); OlD = zeros(nk, NCLS, ns); OlD2 = zeros(nk, NCLS, ns)
  OH = zeros(nk, NCLS, ns); OlH = zeros(nk, NCLS, ns); OlH2 = zeros(nk, NCLS, ns); OnH = zeros(Int32, nk, NCLS, ns)
  function obs_add!(s, k, D, H)
    j = HBAD4P.dclass(D); j == 0 && return
    On[k, j, s] += 1; OD2[k, j, s] += D^2; OlD[k, j, s] += log(D); OlD2[k, j, s] += log(D)^2
    H > 0 && (OH[k, j, s] += H; OlH[k, j, s] += log(H); OlH2[k, j, s] += log(H)^2; OnH[k, j, s] += 1)
  end
  for r in eachrow(t1); r.out == "alive" && obs_add!(sidx[r.iid], kix[r.key], r.dia2_cm, r.ht2_m); end
  for r in eachrow(L.ig); haskey(sidx, r.iid) && obs_add!(sidx[r.iid], kix[r.key], r.dia_cm, r.ht_m); end
  co2 = Dict(Int(r.year) => Float64(r.co2_ppm) for r in eachrow(CSV.read(CO2_CSV, DataFrame)))
  Cs = [mean(co2[y] for y in year(r.d1):max(year(r.d2) - 1, year(r.d1))) for r in eachrow(iv)]
  getw(d, i, dflt) = (v = get(d, i, dflt); isnan(v) ? dflt : v)
  Wv = [getw(CL.W, i, 1.0) for i in iv.iid]; Wnv = [getw(CL.Wn, i, 1.0) for i in iv.iid]; Tv = [getw(CL.T, i, NaN) for i in iv.iid]
  nmiss = (W=count(i -> isnan(get(CL.W, i, NaN)), iv.iid), Wn=count(i -> isnan(get(CL.Wn, i, NaN)), iv.iid), T=count(isnan, Tv))
  S = HBAD4P.Sites4(off, nk, sp, cl, ln, N0, B0, H0, A0, copy(N0), copy(B0), copy(H0), copy(A0), nyr, frac, Vector{Bool}(iv.planted), Wv, Wnv, Tv, Cs, seed0, lz, lw)
  (S=S, iid=iv.iid, n=On, D2=OD2, lD=OlD, lD2=OlD2, Hs=OH, lH=OlH, lH2=OlH2, nH=OnH, fs2=fs2, fm2=fm2, f1=f1, fm1=fm1, nohgt=nohgt, nmiss=nmiss,
   ntrees=sum(On), iv=iv, sap=(eb=neb / ns, obs=nobs_sap / ns))
end
"Pooled within-cell sd of log D and log H per class (cells with ≥ 2 trees) of a built set — the sampling spread of a cell's mean (TRAIN values are used)."
function within_sd(D)
  sd(s1, s2, n) = (ss = 0.0; df = 0.0; for i in eachindex(n); n[i] >= 2 && (ss += s2[i] - s1[i]^2 / n[i]; df += n[i] - 1); end; df > 0 ? sqrt(ss / df) : 0.1)
  [sd(view(D.lD, :, j, :), view(D.lD2, :, j, :), view(D.n, :, j, :)) for j in 1:NCLS], [sd(view(D.lH, :, j, :), view(D.lH2, :, j, :), view(D.nH, :, j, :)) for j in 1:NCLS]
end

"Climate dictionaries by iid: W = period AET/PET, Wn = its normal, T = period mean annual temperature, the climate table (for Topt / Tref)."
function climate4()
  w = CSV.read(WATER_CSV, DataFrame); c = CSV.read(CLIM_CSV, DataFrame; select=[:iid, :tmean_iv, :tmean_n])
  (W=Dict(Int(r.iid) => Float64(r.aetpet_iv) for r in eachrow(w)), Wn=Dict(Int(r.iid) => Float64(r.aetpet_n) for r in eachrow(w)),
   T=Dict(Int(r.iid) => Float64(r.tmean_iv) for r in eachrow(c)), clim=c)
end
"(st, u, c, p, date, sb) => (f_subp, f_micr) (HBAD3 subplot forest table; NaN = status unknown)."
function forest4(path=FOREST_CSV)
  f = CSV.read(path, DataFrame; types=Dict(:md => Date))
  Dict((Int(r.st), Int(r.u), Int(r.c), Int(r.p), r.md, Int(r.sb)) => (Float64(coalesce(r.f_subp, NaN)), Float64(coalesce(r.f_micr, NaN))) for r in eachrow(f))
end

# ---------------------------------------------------------------- likelihood
"""Observation parameters: negative-binomial size r for microplot (classes 1–2) and subplot (≥ 3) tallies; background densities bgM / bgS [trees / forested ha]
(immigration, misidentification, recording error — the tally of a cell the model leaves empty; replaces an arbitrary floor); model-error spreads σD, σH of
log QMD / log mean height; sD, sH the per-class within-cell sd of log D / log H (data constants, TRAIN)."""
struct ObsP; rM::Float64; rS::Float64; bgM::Float64; bgS::Float64; σD::Float64; σH::Float64; sD::Vector{Float64}; sH::Vector{Float64}; end
@inline nb_logpmf(n, μ, r) = loggamma(n + r) - loggamma(r) - loggamma(n + 1.0) + r * log(r / (r + μ)) + (n > 0 ? n * log(μ / (r + μ)) : 0.0)
"""Stand-table NLL of the simulated state (S.N/B/H after simulate!) against the v2 tables: Σ over sites, keys, classes of − log NegBin(n | μ = (N + bg) f area, r)
+ size and height terms for the occupied cells: log QMD_obs − log Dq ~ N(0, sD_j² / n + σD²), log mean H_obs − log H ~ N(0, sH_j² / n_H + σH²). Returns parts."""
function nll4(D, Ob::ObsP)
  S = D.S; nk = S.nk; ns = HBAD4P.nsites(S)
  pc = zeros(ns); pd = zeros(ns); ph = zeros(ns)
  Threads.@threads :static for s in 1:ns
    c_ = 0.0; d_ = 0.0; h_ = 0.0
    @inbounds for k in 1:nk, j in 1:NCLS
      i0 = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL
      Np = S.N[i0] + S.N[i0+1] + S.N[i0+2]; n = D.n[k, j, s]
      μ = (Np + (j >= 3 ? Ob.bgS : Ob.bgM)) * (j >= 3 ? D.fs2[s] : D.fm2[s]) * area_class(j)
      c_ -= nb_logpmf(Float64(n), μ, j >= 3 ? Ob.rS : Ob.rM)
      if n > 0 && Np > 1e-9
        Bp = S.B[i0] + S.B[i0+1] + S.B[i0+2]; dq = HBAD4P.qmd(Np, Bp); qo = sqrt(D.D2[k, j, s] / n)
        v = Ob.sD[j]^2 / n + Ob.σD^2; d_ += 0.5 * (log(qo / dq))^2 / v + 0.5 * log(2π * v)
        if D.nH[k, j, s] > 0
          Hp = (S.N[i0] * S.H[i0] + S.N[i0+1] * S.H[i0+1] + S.N[i0+2] * S.H[i0+2]) / Np; ho = D.Hs[k, j, s] / D.nH[k, j, s]
          vh = Ob.sH[j]^2 / D.nH[k, j, s] + Ob.σH^2; h_ += 0.5 * (log(ho / Hp))^2 / vh + 0.5 * log(2π * vh)
        end
      end
    end
    pc[s] = c_; pd[s] = d_; ph[s] = h_
  end
  (total=sum(pc) + sum(pd) + sum(ph), count=sum(pc), size=sum(pd), height=sum(ph))
end
"Persistence null: the v1 table (with the same empirical-Bayes sapling layer) carried to v2 unchanged; the same observation model (its own fitted ObsP)."
function null_state!(D)
  S = D.S; copyto!(S.N, S.N0); copyto!(S.B, S.B0); copyto!(S.H, S.H0); copyto!(S.A, S.A0); D
end

# ---------------------------------------------------------------- cache (load + traits + build take minutes; the fit / eval tools reuse them)
using Serialization
"Fingerprint of everything a build depends on: the builder and kernel code, the switches, and the input files (path, size, mtime)."
function build_key(dir)
  h = hash(read(joinpath(@__DIR__, "data4.jl"))); h = hash(read(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD4", "HBAD4P.jl")), h)
  h = hash((REGION, MIN_SP_TREES, LANEMODE, FMIN), h)
  for f in (joinpath(dir, "intervals.arrow"), joinpath(dir, "interval_trees.arrow"), joinpath(dir, "interval_ingrowth.arrow"), WATER_CSV, CLIM_CSV, FOREST_CSV,
            CO2_CSV, SHADETOL_CSV, LONGEVITY_CSV)
    st = stat(realpath(f)); h = hash((abspath(f), st.size, st.mtime), h)
  end
  string(h; base=16)
end
"Built sets of a product directory (train first: the sapling prior and the within-cell spreads come from it), cached under HBAD4_CACHE by build_key."
function cached_sets(sets::Vector{String}; dir=DATA_DIR)
  cd_ = get(ENV, "HBAD4_CACHE", joinpath(@__DIR__, "..", "..", "results", "hbad4", "cache")); mkpath(cd_)
  f = joinpath(cd_, "sets_$(join(sets, "_"))_$(build_key(dir)).jls")
  isfile(f) && return deserialize(f)
  L = load4(dir); CL = climate4(); FT = forest4(); TR = traits4(L, CL.clim); eb = sapling_prior(L, L.keys, FT)
  B = Dict(s => build4(L, s, L.keys, CL, FT, eb) for s in sets if any(==(s), L.iv.set))
  sD, sH = within_sd(B["train"])
  trn = L.tr[(L.tr.set .== "train") .& .!ismissing.(L.tr.cr), :]; crt = Float64.(trn.cr[trn.cr .> 0]) ./ 100   # train v1 crown ratios (the HBAD3P warm-start rescaling)
  out = (keys=L.keys, TR=TR, eb=eb, sD=sD, sH=sH, lanemode=LANEMODE, cr_train=crt, sets=B)
  serialize(f * ".part", out); mv(f * ".part", f; force=true)
  out
end
