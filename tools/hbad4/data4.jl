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
"""Forested fraction of a plot's tally (4 subplots or 4 microplots; NaN = status unknown → counted as forested). 0 when every known one is non-forest (audit r4
M1: was 1.0 — an unobserved tally scored as a full plot): zero exposure, no count term for that tally, the v1 sapling layer takes its prior mean. Otherwise ≥ FMIN."""
plotf(v) = (x = [isnan(a) ? 1.0 : a for a in v]; m = isempty(x) ? 1.0 : mean(x); m <= 0 ? 0.0 : max(m, FMIN))

"""Load the natural intervals of the region with their trees and ingrowth; species keys from the TRAIN trees (the HBAD3P rule). HBAD4_SPLIT_CSV (statecd,
unitcd,countycd,plot,set): the sets by plot (the campaign's role split folds/split_<role>.csv: train = the role's 3 folds — the product's `set` moved the
species-balanced train cap's surplus to val, audit r2 I5); plots absent from it get no set. Intervals whose v2 visit has NO forested subplot (every subplot's
forested fraction known and 0: a land-use change inside a "natural" interval — FIA tallies no tree there) are dropped (audit r3 I6; logged): they would be
scored against an empty tally on a full plot."""
const SPLIT_CSV = get(ENV, "HBAD4_SPLIT_CSV", "")
noforest_v2(r, FT) = all(sb -> get(FT, (r.st, r.u, r.c, r.p, r.d2, sb), (NaN, NaN))[1] == 0, 1:4)
noforest_v1(r, FT) = all(sb -> get(FT, (r.st, r.u, r.c, r.p, r.d1, sb), (NaN, NaN))[1] == 0, 1:4)
function load4(dir=DATA_DIR; FT=forest4(), quiet=false)
  iv = DataFrame(Arrow.Table(joinpath(dir, "intervals.arrow")))
  iv = iv[.!(iv.trt .| iv.dst .| iv.harv) .& (iv.dt .> 2) .& inregion.(iv.st), :]
  iv.set = coalesce.(iv.set, "")
  if !isempty(SPLIT_CSV)
    sp = CSV.read(SPLIT_CSV, DataFrame); sd = Dict((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)) => String(r.set) for r in eachrow(sp))
    iv.set = [get(sd, (r.st, r.u, r.c, r.p), "") for r in eachrow(iv)]
  end
  nf = [noforest_v2(r, FT) for r in eachrow(iv)]
  quiet || println("load4: dropped $(count(nf)) intervals whose v2 visit has no forested subplot (" *
                   join(["$(isempty(x) ? "no set" : x) $(count(nf .& (iv.set .== x)))" for x in sort(unique(iv.set[nf]))], ", ") * ")")
  # audit r4 M1 (decision H10): an interval whose v1 visit has no forested subplot has no observed v1 stand (non-forest → forest): dropped likewise
  nf1 = [!nf[i] && noforest_v1(r, FT) for (i, r) in enumerate(eachrow(iv))]
  quiet || println("load4: dropped $(count(nf1)) intervals whose v1 visit has no forested subplot (" *
                   join(["$(isempty(x) ? "no set" : x) $(count(nf1 .& (iv.set .== x)))" for x in sort(unique(iv.set[nf1]))], ", ") * ")")
  iv = iv[.!(nf .| nf1), :]
  ok = Set(iv.iid)
  tr = DataFrame(Arrow.Table(joinpath(dir, "interval_trees.arrow"))); tr = tr[in.(tr.iid, Ref(ok)) .& (tr.dia_cm .> 0), :]
  ig = DataFrame(Arrow.Table(joinpath(dir, "interval_ingrowth.arrow"))); ig = ig[in.(ig.iid, Ref(ok)) .& (ig.dia_cm .> 0), :]
  sset = Dict(r.iid => r.set for r in eachrow(iv)); tr.set = getindex.(Ref(sset), tr.iid)
  ktr = tr[(tr.set .== "train") .& (tr.age .> 0) .& (tr.ht_m .> 0), :]
  cnt = combine(groupby(ktr, :sym), nrow => :n); big = Set(cnt.sym[cnt.n .>= MIN_SP_TREES])
  keyof(s, g) = s in big ? String(s) : "_" * String(g)
  tr.key = keyof.(tr.sym, tr.grp); ig.key = keyof.(ig.sym, ig.grp)
  keys_ = sort(unique(vcat(tr.key, ig.key)))
  (iv=iv, tr=tr, ig=ig, keys=keys_, big=big, ndrop=count(nf), ndrop1=count(nf1))
end

"""Traits per key from the TRAIN trees: Hmax (99.5th pct height), shade class and LONGEVITY (tree-weighted over member species), and Tref = the tree-weighted
mean PERIOD temperature (the variable the kernel's T is) of its train intervals: f_T = exp(βT (T − Tref)) is 1 there. Temperatures are EPA L4 means.
grp = the key's softwood / hardwood group (S / H; the HBAD3P group key _S / _H a new own key falls back to in the warm start)."""
function traits4(L, clim; dir=DATA_DIR)
  st = CSV.read(SHADETOL_CSV, DataFrame); std_ = Dict(String(r.sym) => Float64(r.shade_class) for r in eachrow(st))
  lon = CSV.read(LONGEVITY_CSV, DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
  trn = L.tr[L.tr.set .== "train", :]
  tiv = Dict(r.iid => r.tmean_iv for r in eachrow(clim))
  map(L.keys) do k
    t = trn[trn.key .== k, :]; c = combine(groupby(t, :sym), nrow => :n)
    wavg(d, dflt) = (v = [get(d, s, dflt) for s in c.sym]; sum(v .* c.n) / sum(c.n))
    hs = t.ht_m[t.ht_m .> 0]; tt = t[haskey.(Ref(tiv), t.iid), :]
    gr = k[1] == '_' ? k[2:end] : (g = unique(String.(L.tr.grp[L.tr.sym .== k])); length(g) == 1 ? g[1] : error("key $k: groups $g"))
    (key=k, Hmax=quantile(hs, 0.995), ST=wavg(std_, 3.0), longevity=wavg(lond, 300.0), Tref=mean(getindex.(Ref(tiv), tt.iid)), ntrain=nrow(t), grp=gr)
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
adult: a tree of the key ≥ 12.7 cm on the plot at v1) n ~ NegBin(mean m_g e, size α_g), e = microplot area × forested fraction; (m_g, α_g) jointly by maximum
likelihood (α by profile likelihood over (α, m), nb_joint; audit r3: m = Σn / Σe was the Poisson MLE — `gain` = the summed log-likelihood gain over it). Posterior mean density N̂ = (n + α) / (e + α / m). Also the train mean QMD / height / age of a key's saplings per class (the attributes
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
  m = zeros(nk, 2, 2); α = fill(Inf, nk, 2, 2); llp = zeros(nk, 2, 2); llj = zeros(nk, 2, 2)
  for k in 1:nk, j in 1:2, a in (false, true)
    idx = [s for s in 1:ns if adult[k, s] == a && e[s] > 0]; isempty(idx) && continue        # (unobserved microplots, fm1 = 0, are no zeros: r4 M1)
    y = n[k, j, idx]; x = e[idx]; mp = sum(y) / sum(x)
    mp > 0 || continue
    lα_, lm_, l_ = nb_joint(y, x); m[k, j, a+1] = exp(lm_); α[k, j, a+1] = exp(lα_); llj[k, j, a+1] = l_
    llp[k, j, a+1] = nb_joint(y, x; m=mp)[3]
  end
  pooled(x, cnt, k, j) = cnt[k, j] > 0 ? x[k, j] / cnt[k, j] : sum(x[:, j]) / max(sum(cnt[:, j]), 1)
  (m=m, α=α, D=[sqrt(pooled(sD2, c, k, j)) for k in 1:nk, j in 1:2], H=[pooled(sH, cH, k, j) for k in 1:nk, j in 1:2],
   A=[pooled(sA, cA, k, j) for k in 1:nk, j in 1:2], gain=sum(llj .- llp))
end
"""Negative-binomial (mean m x_i, size α) log-likelihood of counts y with exposures x, maximised jointly over (ln α, ln m) as a profile likelihood of α: for
each α the MLE of m solves the score Σ y_i / m = Σ (y_i + α) x_i / (α + m x_i) (bisection on ln m), ln α by golden section on [−8, 8]. m = given: the
profile over α at that fixed mean (the round-2 rule, m = Σy / Σx). Returns (ln α, ln m, log-likelihood)."""
function nb_joint(y, x; m=nothing)
  cnt = Dict{Tuple{Float64,Float64},Int}(); for (yi, xi) in zip(y, x); cnt[(Float64(yi), xi)] = get(cnt, (Float64(yi), xi), 0) + 1; end   # (distinct (y, x) pairs, weighted)
  U = collect(keys(cnt)); w = [cnt[u] for u in U]; yy = first.(U); xx = last.(U)
  ll(r, mg) = sum(wi * (loggamma(yi + r) - loggamma(r) + r * log(r / (r + mg * xi)) + yi * log(mg * xi / (r + mg * xi))) for (wi, yi, xi) in zip(w, yy, xx))
  m0 = sum(y) / sum(x)
  function mhat(r)
    m === nothing || return m
    ml, mh = log(m0) - 8, log(m0) + 8
    for _ in 1:80
      mid = (ml + mh) / 2; mm = exp(mid); sc = sum(wi * (yi / mm - (yi + r) * xi / (r + mm * xi)) for (wi, yi, xi) in zip(w, yy, xx))
      sc > 0 ? (ml = mid) : (mh = mid)
    end
    exp((ml + mh) / 2)
  end
  prof(lα) = (r = exp(lα); ll(r, mhat(r)))
  lo, hi = -8.0, 8.0
  for _ in 1:80
    a1 = hi - 0.618 * (hi - lo); a2 = lo + 0.618 * (hi - lo); prof(a1) > prof(a2) ? (hi = a2) : (lo = a1)
  end
  lα = (lo + hi) / 2; mm = mhat(exp(lα))
  (lα, log(mm), ll(exp(lα), mm))
end
@inline eb_density(n, e, m, α) = m > 0 ? (isinf(α) ? m : (n + α) / (e + α / m)) : (e > 0 ? n / e : 0.0)

"""Sites (v1 plot stand tables) and observations (v2 stand tables) of one set. Per site every key × class × lane has its cell totals; the initial atoms are
stored for the occupied cells only (HBAD4P.make_sites4: compact layout). v1 trees →
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
  H0 = zeros(ntot); A0 = zeros(ntot); at0 = Pair{Int,Vector{Tuple{Float64,Float64}}}[]   # (cell => its initial atoms: the occupied cells only)
  nyr = floor.(Int, iv.dt); frac = iv.dt .- nyr
  sidx = Dict(r.iid => s for (s, r) in enumerate(eachrow(iv)))
  trn = L.tr[L.tr.set .== "train", :]; medage = Dict(k => median(trn.age[(trn.key .== k) .& (trn.age .> 0)]) for k in keys_)
  f1, fm1, fs2, fm2 = fractions4(iv, FT)
  t1 = v1trees(L, keys(sidx))
  acc = Dict{Tuple{Int,Int,Int},NTuple{7,Float64}}()     # (s, k, j) → (Σw, ΣwB, ΣwH, Σw|H, ΣwA, Σw|A, n tally trees)
  atoms = Dict{Tuple{Int,Int,Int},Vector{Tuple{Float64,Float64}}}()   # (s, k, j) → the cell's trees (w, D)
  seed0 = falses(nk, ns); nzt = 0
  for r in eachrow(t1)
    j = HBAD4P.dclass(r.dia_cm); j == 0 && continue
    s = sidx[r.iid]; k = kix[r.key]; ft = ismicro(r.w_ha) ? fm1[s] : f1[s]; ft > 0 || (nzt += 1; continue)   # (a tree on a tally with no forested area: none expected)
    r.dia_cm >= 12.7 && (seed0[k, s] = true)
    w = r.w_ha / ft; a = get(acc, (s, k, j), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    hok = r.ht_m > 0; aok = r.age > 0; push!(get!(atoms, (s, k, j), Tuple{Float64,Float64}[]), (w, r.dia_cm))
    acc[(s, k, j)] = (a[1] + w, a[2] + w * π / 4 * (r.dia_cm / 100)^2, a[3] + (hok ? w * r.ht_m : 0.0), a[4] + (hok ? w : 0.0), a[5] + (aok ? w * r.age : 0.0),
                      a[6] + (aok ? w : 0.0), a[7] + 1)
  end
  nohgt = 0
  # a cell's initial atoms: its trees (w, D) scaled by fN (the sapling layer's empirical-Bayes factor), merged down to NATOM (N, N·D² conserved), per lane × w_v
  bn = zeros(1024); bd = zeros(1024)
  function setcell!(s, k, j, trees, fN, H, A)
    c0 = 0
    for (w, d) in trees; c0 = HBAD4P.buf_add!(bn, bd, 0, c0, 1024, w * fN, d); end
    c0 = HBAD4P.compact!(bn, bd, 0, c0)
    for v in 1:NL
      c = off[s] + ((k - 1) * NCLS + (j - 1)) * NL + (v - 1)
      push!(at0, c => [(bn[m] * lw[v], bd[m]) for m in 1:c0]); H0[c] = H; A0[c] = A
    end
  end
  for ((s, k, j), a) in acc
    j <= 2 && continue                                                     # (sapling classes: below)
    H = a[4] > 0 ? a[3] / a[4] : NaN; isnan(H) && (nohgt += 1; H = 1.37 + 0.5 * HBAD4P.CLS_LO[j])   # (no height in the cell: a crude placeholder, counted)
    setcell!(s, k, j, atoms[(s, k, j)], 1.0, H, a[6] > 0 ? a[5] / a[6] : medage[keys_[k]])
  end
  neb = 0.0; nobs_sap = 0.0
  for s in 1:ns, k in 1:nk, j in 1:2                                       # sapling layer: empirical-Bayes density
    a = get(acc, (s, k, j), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)); n_ = a[7]; e = AREA_MIC * fm1[s]
    Nh = eb_density(n_, e, eb.m[k, j, seed0[k, s]+1], eb.α[k, j, seed0[k, s]+1]); Nh > 0 || continue
    neb += Nh; nobs_sap += a[1]
    if n_ > 0
      setcell!(s, k, j, atoms[(s, k, j)], Nh / a[1], a[4] > 0 ? a[3] / a[4] : eb.H[k, j], a[6] > 0 ? a[5] / a[6] : eb.A[k, j])
    else
      setcell!(s, k, j, [(Nh, eb.D[k, j])], 1.0, eb.H[k, j], eb.A[k, j])
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
  S = HBAD4P.make_sites4(off, nk, sp, cl, ln, at0, H0, A0, nyr, frac, Vector{Bool}(iv.planted), Wv, Wnv, Tv, Cs, seed0, lz, lw)
  (S=S, iid=iv.iid, n=On, D2=OD2, lD=OlD, lD2=OlD2, Hs=OH, lH=OlH, lH2=OlH2, nH=OnH, fs2=fs2, fm2=fm2, f1=f1, fm1=fm1, nohgt=nohgt, nmiss=nmiss, nzt=nzt,
   ntrees=sum(On), iv=iv, sap=(eb=neb / ns, obs=nobs_sap / ns))
end
"""Pooled within-cell sd of log D and log H per class (cells with ≥ 2 trees) of a built set — the sampling spread of a cell's mean (TRAIN values are used).
Classes with < 30 degrees of freedom: log D the uniform-spread value of the class, log H pooled over the neighbouring classes (audit r3: was 0.1)."""
function within_sd(D)
  # (< 30 degrees of freedom: the physical value of a uniform spread over the class, w / (√12 · class midpoint) in log D — the top classes hold few cells)
  phys(j) = (c = j < NCLS ? (HBAD4P.CLS_LO[j] + HBAD4P.CLS_HI[j]) / 2 : HBAD4P.CLS_LO[NCLS] + 2.54; 5.08 / (sqrt(12) * c))
  sd(s1, s2, n, dflt) = (ss = 0.0; df = 0.0; for i in eachindex(n); n[i] >= 2 && (ss += s2[i] - s1[i]^2 / n[i]; df += n[i] - 1); end; df >= 30 ? sqrt(ss / df) : dflt)
  sDv = [sd(view(D.lD, :, j, :), view(D.lD2, :, j, :), view(D.n, :, j, :), phys(j)) for j in 1:NCLS]
  # log H: a class with < 30 df pools the neighbouring classes (window j ± w, w = 1, 2, … until ≥ 30 df; the spread of log H changes slowly with D)
  function sdpool(j)
    for w in 0:NCLS
      ss = 0.0; df = 0.0
      for t in max(1, j - w):min(NCLS, j + w), i in eachindex(view(D.nH, :, t, :))
        nn = view(D.nH, :, t, :)[i]; nn >= 2 || continue; ss += view(D.lH2, :, t, :)[i] - view(D.lH, :, t, :)[i]^2 / nn; df += nn - 1
      end
      df >= 30 && return sqrt(ss / df)
    end
    0.1
  end
  sDv, [sdpool(j) for j in 1:NCLS]
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
"""Background densities of the size / height terms (audit r4 I1; data constants from the TRAIN v2 table, like sD / sH): what the observed mean of a cell looks
like when its trees are not the model's (immigration, misidentification, recording error — the count model's background bg). Per class j: log QMD_obs ~
N(mD_j, sDb_j²) over all train cells with ≥ 1 tree; per key × class: log mean H_obs ~ N(mH_kj, sHb_j²) (key mean over ≥ 10 train cells, else the class mean;
sHb_j the pooled sd around the key means)."""
struct ObsBG; mD::Vector{Float64}; sDb::Vector{Float64}; mH::Matrix{Float64}; sHb::Vector{Float64}; end
function obs_background(D)
  nk = size(D.n, 1); mD = zeros(NCLS); sDb = zeros(NCLS); mH = zeros(nk, NCLS); sHb = zeros(NCLS)
  for j in 1:NCLS
    x = [log(sqrt(D.D2[k, j, s] / D.n[k, j, s])) for k in 1:nk, s in axes(D.n, 3) if D.n[k, j, s] > 0]
    mD[j] = mean(x); sDb[j] = length(x) >= 3 ? std(x) : 0.1
    hk = [[log(D.Hs[k, j, s] / D.nH[k, j, s]) for s in axes(D.n, 3) if D.nH[k, j, s] > 0] for k in 1:nk]; allh = reduce(vcat, hk)
    mc = isempty(allh) ? log(1.37 + 0.5 * HBAD4P.CLS_LO[j]) : mean(allh); ss = 0.0; df = 0
    for k in 1:nk
      mH[k, j] = length(hk[k]) >= 10 ? mean(hk[k]) : mc
      for h in hk[k]; ss += (h - mH[k, j])^2; df += 1; end
    end
    sHb[j] = df >= 3 ? sqrt(ss / df) : 0.3
  end
  ObsBG(mD, sDb, mH, sHb)
end
"""Observation parameters: negative-binomial size r for microplot (classes 1–2) and subplot (≥ 3) tallies; background densities bgM / bgS [trees / forested ha]
(immigration, misidentification, recording error — the tally of a cell the model leaves empty; replaces an arbitrary floor); model-error spreads σD, σH of
log QMD / log mean height; sD, sH the per-class within-cell sd of log D / log H (data constants, TRAIN); BG the background densities of the size / height
terms (ObsBG, TRAIN). The 8-argument form takes BG from the last cached_sets (OBS_BG)."""
struct ObsP; rM::Float64; rS::Float64; bgM::Float64; bgS::Float64; σD::Float64; σH::Float64; sD::Vector{Float64}; sH::Vector{Float64}; BG::ObsBG; end
const OBS_BG = Ref{Union{Nothing,ObsBG}}(nothing)
ObsP(rM, rS, bgM, bgS, σD, σH, sD, sH) = (OBS_BG[] === nothing && error("ObsP: no background densities (load cached_sets first or pass BG)");
                                           ObsP(rM, rS, bgM, bgS, σD, σH, sD, sH, OBS_BG[]))
@inline nb_logpmf(n, μ, r) = loggamma(n + r) - loggamma(r) - loggamma(n + 1.0) + r * log(r / (r + μ)) + (n > 0 ? n * log(μ / (r + μ)) : 0.0)
"log(w e^{la} + (1 − w) e^{lb}) for log densities la, lb (w ∈ [0, 1])."
@inline function logmix(w, la, lb)
  w <= 0 && return lb; w >= 1 && return la
  a = log(w) + la; b = log1p(-w) + lb; m = max(a, b); m + log(exp(a - m) + exp(b - m))
end
@inline lnorm(x, μ, v) = -0.5 * (x - μ)^2 / v - 0.5 * log(2π * v)
"""Stand-table NLL of a state's OBSERVATION-SPACE prediction (S.PN / PB / PH per (site, key, class): after simulate! the atoms blurred by the FIA DBH error,
HBAD4P.finish_site!; for the nulls the cell totals, HBAD4P.obs_cells!) against the v2 tables. Per (site, key, class) of a tally with forested area (f > 0; an
unobserved tally, f = 0, scores nothing — audit r4 M1):
  count  − log NegBin(n | μ = (N + bg) f area, r);
  size   (every cell with n ≥ 1 tree) − log[ w N(log QMD_obs; log Dq, sD_j²/n + σD²) + (1 − w) N(log QMD_obs; mD_j, sDb_j²) ];
  height (every cell with n_H ≥ 1)     − log[ w N(log H_obs; log H, sH_j²/n_H + σH²) + (1 − w) N(log H_obs; mH_kj, sHb_j²) ];
  w = N / (N + bg): under the count model a tallied tree of the cell is the model's with probability N / (N + bg) — the background density bg (the same
  fitted bgM / bgS) is the pseudo-count of the background trees (audit r4 I1, decision H7: the hard gate N > 1e-9 switched −1.5 … +13-nat terms on and off
  and gave each model / null its own scored set; now every model and null scores the same cells). Returns parts."""
function nll4(D, Ob::ObsP)
  S = D.S; nk = S.nk; ns = HBAD4P.nsites(S); BG = Ob.BG
  pc = zeros(ns); pd = zeros(ns); ph = zeros(ns)
  Threads.@threads :static for s in 1:ns
    c_ = 0.0; d_ = 0.0; h_ = 0.0
    @inbounds for k in 1:nk, j in 1:NCLS
      x = ((s - 1) * nk + (k - 1)) * NCLS + j
      f = j >= 3 ? D.fs2[s] : D.fm2[s]; f > 0 || continue
      Np = S.PN[x]; n = D.n[k, j, s]; bg = j >= 3 ? Ob.bgS : Ob.bgM
      c_ -= nb_logpmf(Float64(n), (Np + bg) * f * area_class(j), j >= 3 ? Ob.rS : Ob.rM)
      n > 0 || continue
      w = Np / (Np + bg); qo = log(sqrt(D.D2[k, j, s] / n))
      la = Np > 0 ? lnorm(qo, log(HBAD4P.qmd(Np, S.PB[x])), Ob.sD[j]^2 / n + Ob.σD^2) : -Inf
      d_ -= logmix(w, la, lnorm(qo, BG.mD[j], BG.sDb[j]^2))
      if D.nH[k, j, s] > 0
        ho = log(D.Hs[k, j, s] / D.nH[k, j, s])
        lh = Np > 0 ? lnorm(ho, log(S.PH[x] / Np), Ob.sH[j]^2 / D.nH[k, j, s] + Ob.σH^2) : -Inf
        h_ -= logmix(w, lh, lnorm(ho, BG.mH[k, j], BG.sHb[j]^2))
      end
    end
    pc[s] = c_; pd[s] = d_; ph[s] = h_
  end
  (total=sum(pc) + sum(pd) + sum(ph), count=sum(pc), size=sum(pd), height=sum(ph))
end
"""Persistence null: the v1 table (with the same empirical-Bayes sapling layer) carried to v2 unchanged, scored on its cell totals (it is a table in observation
space already: no DBH blur); the same observation model (its own fitted ObsP). Not the bar HBAD4P must beat — that is the transition-matrix null (tm_state!)."""
function null_state!(D)
  S = D.S; copyto!(S.N, S.N0); copyto!(S.B, S.B0); copyto!(S.H, S.H0); copyto!(S.A, S.A0); HBAD4P.obs_cells!(S); D
end
"""Train-estimated TRANSITION-MATRIX null of the stand table (audit r2 I3 / r3 I3: the baseline HBAD4P must beat on val counts; the auditor's verify4p
tm_state!): from the TRAIN v1 → v2 tree records per key × v1 class × dt tercile, the fraction of the v1 density (w / f) alive at v2 in each v2 class and its
basal area; ingrowth density (v2 ingrowth w / f) per key × class × v1 seed presence per year of interval; heights = the train mean v2 height of the key × class.
A table in observation space (no blur)."""
function tm_null(L, Dtr, keys_)
  t1 = v1trees(L, Set(Dtr.iid)); sidx = Dict(iid => s for (s, iid) in enumerate(Dtr.iid)); kix = Dict(k => i for (i, k) in enumerate(keys_)); dtv = Dtr.iv.dt
  nk = length(keys_); qs = quantile(dtv, [1 / 3, 2 / 3])
  Tn = zeros(nk, NCLS, 3, NCLS); Tb = zeros(nk, NCLS, 3, NCLS); W0 = zeros(nk, NCLS, 3); Hs = zeros(nk, NCLS); Hc = zeros(nk, NCLS)
  bin(dt) = dt <= qs[1] ? 1 : dt <= qs[2] ? 2 : 3
  for r in eachrow(t1)
    j1 = HBAD4P.dclass(r.dia_cm); j1 == 0 && continue; s = sidx[r.iid]; k = kix[r.key]; ft = ismicro(r.w_ha) ? Dtr.fm1[s] : Dtr.f1[s]; ft > 0 || continue
    w = r.w_ha / ft; b = bin(dtv[s])
    W0[k, j1, b] += w
    if r.out == "alive"
      j2 = HBAD4P.dclass(r.dia2_cm); j2 == 0 && continue; Tn[k, j1, b, j2] += w; Tb[k, j1, b, j2] += w * π / 4e4 * r.dia2_cm^2
      r.ht2_m > 0 && (Hs[k, j2] += r.ht2_m; Hc[k, j2] += 1)
    end
  end
  Ig = zeros(nk, NCLS, 2); Igb = zeros(nk, NCLS, 2); Dsum = zeros(2, nk); S = Dtr.S
  for s in 1:length(Dtr.iid), k in 1:nk; Dsum[S.seed0[k, s]+1, k] += dtv[s]; end
  for r in eachrow(L.ig[in.(L.ig.iid, Ref(keys(sidx))), :])
    j = HBAD4P.dclass(r.dia_cm); j == 0 && continue; s = sidx[r.iid]; k = kix[r.key]; a = S.seed0[k, s] + 1; ft = ismicro(r.w_ha) ? Dtr.fm2[s] : Dtr.fs2[s]
    ft > 0 || continue; w = r.w_ha / ft
    Ig[k, j, a] += w; Igb[k, j, a] += w * π / 4e4 * r.dia_cm^2; r.ht_m > 0 && (Hs[k, j] += r.ht_m; Hc[k, j] += 1)
  end
  (Tn=Tn, Tb=Tb, W0=W0, Ig=Ig, Igb=Igb, Dsum=Dsum, H=Hs ./ max.(Hc, 1), qs=qs)
end
"Write the transition-matrix null's v2 state of set D (lane 1 of every key × class) and its observation-space values."
function tm_state!(D, TM)
  S = D.S; fill!(S.N, 0.0); fill!(S.B, 0.0); fill!(S.H, 0.0)
  bin(dt) = dt <= TM.qs[1] ? 1 : dt <= TM.qs[2] ? 2 : 3
  for s in 1:HBAD4P.nsites(S)
    b = bin(D.iv.dt[s])
    for k in 1:S.nk
      n2 = zeros(NCLS); b2 = zeros(NCLS)
      for j1 in 1:NCLS
        i = HBAD4P.cell(S, s, k, j1, 1); N0 = S.N0[i] + S.N0[i+1] + S.N0[i+2]; (N0 > 0 && TM.W0[k, j1, b] > 0) || continue
        for j2 in 1:NCLS; n2[j2] += N0 * TM.Tn[k, j1, b, j2] / TM.W0[k, j1, b]; b2[j2] += N0 * TM.Tb[k, j1, b, j2] / TM.W0[k, j1, b]; end
      end
      a = S.seed0[k, s] + 1; rate = TM.Dsum[a, k] > 0 ? D.iv.dt[s] / TM.Dsum[a, k] : 0.0
      for j in 1:NCLS; n2[j] += TM.Ig[k, j, a] * rate; b2[j] += TM.Igb[k, j, a] * rate; end
      for j in 1:NCLS; i = HBAD4P.cell(S, s, k, j, 1); S.N[i] = n2[j]; S.B[i] = b2[j]; S.H[i] = TM.H[k, j] > 0 ? TM.H[k, j] : 1.37 + 0.5 * HBAD4P.CLS_LO[j]; end
    end
  end
  HBAD4P.obs_cells!(S); D
end

# ---------------------------------------------------------------- cache (load + traits + build take minutes; the fit / eval tools reuse them)
using Serialization
"Fingerprint of everything a build depends on: the builder and kernel code, the switches, and the input files (path, size, mtime)."
function build_key(dir)
  # (the files are hashed as Strings: Base.hash of a byte VECTOR samples its elements — round 4 found 56 % of one-byte edits of data4.jl left that hash unchanged)
  h = hash(read(joinpath(@__DIR__, "data4.jl"), String)); h = hash(read(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD4", "HBAD4P.jl"), String), h)
  h = hash((REGION, MIN_SP_TREES, LANEMODE, FMIN, HBAD4P.NATOM), h)
  isempty(SPLIT_CSV) || (st = stat(SPLIT_CSV); h = hash((abspath(SPLIT_CSV), st.size, st.mtime), h))
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
  isfile(f) && (out = deserialize(f); OBS_BG[] = out.bgobs; return out)
  FT = forest4(); L = load4(dir; FT=FT); CL = climate4(); TR = traits4(L, CL.clim; dir=dir); eb = sapling_prior(L, L.keys, FT)
  B = Dict(s => build4(L, s, L.keys, CL, FT, eb) for s in sets if any(==(s), L.iv.set))
  sD, sH = within_sd(B["train"]); bgo = obs_background(B["train"])
  # the HBAD3P warm start (params4.jl warm_start4), per train v1 tree: its subplot, key, D, H, crown ratio, and HBAD3P's competition unit — subplot densities
  # × 4 per FORESTED subplot ha (÷ a = max(f_subplot(v1, else v2), FMIN), 1 if unknown / 0: HBAD3P's area_of; audit r3 I2e) — as relative density SDI / 1500
  ivd = Dict(r.iid => r for r in eachrow(L.iv))
  function asub(iid, sb)
    r = ivd[iid]; fa = get(FT, (r.st, r.u, r.c, r.p, r.d1, sb), (NaN, NaN))[1]; isnan(fa) && (fa = get(FT, (r.st, r.u, r.c, r.p, r.d2, sb), (NaN, NaN))[1])
    (isnan(fa) || fa <= 0) ? 1.0 : max(fa, FMIN)
  end
  trn = L.tr[(L.tr.set .== "train") .& .!in.(L.tr.out, Ref(("missing", "harvest"))), :]
  wst = DataFrame(iid=trn.iid, sb=trn.sb, key=trn.key, w=trn.w_ha, dia=trn.dia_cm, ht=trn.ht_m, a=asub.(trn.iid, trn.sb),
                  cr=[ismissing(c) || c <= 0 ? 0.4 : c / 100 for c in trn.cr])
  sdi_sub = combine(groupby(DataFrame(iid=wst.iid, sb=wst.sb, x=4 .* wst.w .* (wst.dia ./ 25.4) .^ 1.605 ./ wst.a), [:iid, :sb]), :x => sum => :sdi)
  rdd = Dict((r.iid, r.sb) => r.sdi / 1500 for r in eachrow(sdi_sub)); wst.rd = [rdd[(i, b)] for (i, b) in zip(wst.iid, wst.sb)]
  TM = haskey(B, "train") ? tm_null(L, B["train"], L.keys) : nothing
  # the TRAIN sapling ingrowth density per key (v2 ingrowth D < 12.7 cm on the microplots, w / f_micr(v2)): a split key's share of its group's fecundity
  # (params4.jl warm_start4, audit r4 I2)
  Dtr = B["train"]; sx = Dict(iid => s for (s, iid) in enumerate(Dtr.iid)); kx = Dict(k => i for (i, k) in enumerate(L.keys)); igk = zeros(length(L.keys))
  for r in eachrow(L.ig)
    s = get(sx, r.iid, 0); (s > 0 && r.dia_cm < 12.7 && Dtr.fm2[s] > 0) || continue; igk[kx[r.key]] += r.w_ha / Dtr.fm2[s]
  end
  out = (keys=L.keys, TR=TR, eb=eb, sD=sD, sH=sH, bgobs=bgo, lanemode=LANEMODE, warm=wst, split=SPLIT_CSV, sets=B, TM=TM, ndrop=L.ndrop, ndrop1=L.ndrop1, igk=igk)
  serialize(f * ".part", out); mv(f * ".part", f; force=true); OBS_BG[] = bgo
  out
end
