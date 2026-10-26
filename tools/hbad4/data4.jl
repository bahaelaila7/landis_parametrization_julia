# HBAD4P harness (docs/hbad4p_design.md): plot stand tables of the natural remeasurement intervals of the interval product (HBAD2_DATA_DIR:
# intervals.arrow with the nested-CV `set`, interval_trees.arrow, interval_ingrowth.arrow) → HBAD4P.Sites4 (v1 state) + the v2 stand table (observation),
# and the stand-table negative log-likelihood. Natural = the HBAD3P filter (no treatment / disturbance / harvest code, dt > 2); region HBAD4_REGION (GAFL).
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
const FMIN = 0.25                                                         # smallest effective forested fraction (as HBAD3P)
const AREA_SUB = 4 * 4046.8564224 / 24 / 1e4                              # 4 subplots of 1/24 ac [ha]: trees D ≥ 12.7 cm
const AREA_MIC = 4 * 4046.8564224 / 300 / 1e4                             # 4 microplots of 1/300 ac [ha]: saplings 2.54 ≤ D < 12.7 cm
area_class(j) = j >= 3 ? AREA_SUB : AREA_MIC

inregion(st) = REGION == "GAFL" ? st in (12, 13) : REGION == "R4" ? true : error("HBAD4_REGION $REGION: GAFL | R4")
plotf(v) = (x = [isnan(a) ? 1.0 : a for a in v]; m = isempty(x) ? 1.0 : mean(x); m <= 0 ? 1.0 : max(m, FMIN))   # forested fraction of a plot

"Load the natural intervals of the region with their trees and ingrowth; species keys from the TRAIN trees (the HBAD3P rule)."
function load4(dir=DATA_DIR)
  iv = DataFrame(Arrow.Table(joinpath(dir, "intervals.arrow")))
  iv = iv[.!(iv.trt .| iv.dst .| iv.harv) .& (iv.dt .> 2) .& inregion.(iv.st), :]
  iv.set = coalesce.(iv.set, ""); ok = Set(iv.iid)
  tr = DataFrame(Arrow.Table(joinpath(dir, "interval_trees.arrow"))); tr = tr[in.(tr.iid, Ref(ok)) .& (tr.dia_cm .> 0), :]
  ig = DataFrame(Arrow.Table(joinpath(dir, "interval_ingrowth.arrow"))); ig = ig[in.(ig.iid, Ref(ok)) .& (ig.dia_cm .> 0), :]
  sset = Dict(r.iid => r.set for r in eachrow(iv))
  ktr = tr[(getindex.(Ref(sset), tr.iid) .== "train") .& (tr.age .> 0) .& (tr.ht_m .> 0), :]
  cnt = combine(groupby(ktr, :sym), nrow => :n); big = Set(cnt.sym[cnt.n .>= MIN_SP_TREES])
  keyof(s, g) = s in big ? String(s) : "_" * String(g)
  tr.key = keyof.(tr.sym, tr.grp); ig.key = keyof.(ig.sym, ig.grp)
  keys_ = sort(unique(vcat(tr.key, ig.key)))
  (iv=iv, tr=tr, ig=ig, keys=keys_)
end

"Traits per key: Hmax (99.5th pct height of its TRAIN trees), shade class and LONGEVITY (tree-weighted over member species), Topt (tree-weighted mean annual
temperature normal of its intervals: TRAIN in the region, every interval outside it)."
function traits4(L, clim)
  st = CSV.read(SHADETOL_CSV, DataFrame); std_ = Dict(String(r.sym) => Float64(r.shade_class) for r in eachrow(st))
  lon = CSV.read(LONGEVITY_CSV, DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
  sset = Dict(r.iid => r.set for r in eachrow(L.iv))
  trn = L.tr[getindex.(Ref(sset), L.tr.iid) .== "train", :]
  tn = Dict(r.iid => r.tmean_n for r in eachrow(clim))
  # Topt over the whole product (outside the region every interval; inside it the train intervals): read the full interval list
  ivall = DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))); okT = Set(ivall.iid[.!inregion.(ivall.st) .| (coalesce.(ivall.set, "") .== "train")])
  tall = DataFrame(Arrow.Table(joinpath(DATA_DIR, "interval_trees.arrow")))
  big = Set(k for k in L.keys if !startswith(k, "_"))
  tall = tall[in.(tall.iid, Ref(okT)) .& haskey.(Ref(tn), tall.iid), :]
  tall.key = [s in big ? String(s) : "_" * String(g) for (s, g) in zip(tall.sym, tall.grp)]
  map(L.keys) do k
    t = trn[trn.key .== k, :]; c = combine(groupby(t, :sym), nrow => :n)
    wavg(d, dflt) = (v = [get(d, s, dflt) for s in c.sym]; sum(v .* c.n) / sum(c.n))
    hs = t.ht_m[t.ht_m .> 0]; ta = tall[tall.key .== k, :]
    (key=k, Hmax=quantile(hs, 0.995), ST=wavg(std_, 3.0), longevity=median([get(lond, s, 300.0) for s in c.sym for _ in 1:1]),
     Topt=mean(getindex.(Ref(tn), ta.iid)), ntrain=nrow(t))
  end |> DataFrame
end

"""Sites (v1 plot stand tables) and observations (v2 stand tables) of one set. Per site: every key × class × lane allocated (HBAD4P layout); v1 trees
(out ≠ missing / harvest) → (key, class): N = Σw / f1, B = Σ w π/4 (D/100)² / f1, H = Σ w H / Σ w (trees with a height), A = Σ w A / Σ w (aged trees; else
the key's median age), split over the lanes by their weights. Observation per (site, key, class): tally count n, Σ D² and Σ H (n_H trees with a height) of the
v2 trees (survivors with their v2 diameter + ingrowth), and the forested fraction of the v2 tally (subplot for D ≥ 12.7 cm, microplot below)."""
function build4(L, set::AbstractString, keys_, W, Wn, T, CO2, FT; lanemode::Symbol=:sym)
  iv = L.iv[L.iv.set .== set, :]; sort!(iv, :iid); ns = nrow(iv); nk = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
  _, lw = HBAD4P.lanes(lanemode)
  ncell = nk * NCLS * NL; off = [1 + (s - 1) * ncell for s in 1:ns+1]
  n = ns * ncell
  sp = Int32[k for _ in 1:ns for k in 1:nk for _ in 1:NCLS for _ in 1:NL]
  cl = Int8[j for _ in 1:ns for _ in 1:nk for j in 1:NCLS for _ in 1:NL]; ln = Int8[v for _ in 1:ns for _ in 1:nk for _ in 1:NCLS for v in 1:NL]
  N0 = zeros(n); B0 = zeros(n); H0 = zeros(n); A0 = zeros(n)
  nyr = floor.(Int, iv.dt); frac = iv.dt .- nyr
  sidx = Dict(r.iid => s for (s, r) in enumerate(eachrow(iv)))
  medage = Dict(k => median(L.tr.age[(L.tr.key .== k) .& (L.tr.age .> 0)]) for k in keys_)
  On = zeros(Int32, nk, NCLS, ns); OD2 = zeros(nk, NCLS, ns); OH = zeros(nk, NCLS, ns); OnH = zeros(Int32, nk, NCLS, ns)
  f1 = zeros(ns); fm1 = zeros(ns); fs2 = zeros(ns); fm2 = zeros(ns)
  for (s, r) in enumerate(eachrow(iv))
    a1 = [get(FT, (r.st, r.u, r.c, r.p, r.d1, sb), (NaN, NaN)) for sb in 1:4]; a2 = [get(FT, (r.st, r.u, r.c, r.p, r.d2, sb), (NaN, NaN)) for sb in 1:4]
    f1[s] = plotf(first.(a1)); fm1[s] = plotf(last.(a1)); fs2[s] = plotf(first.(a2)); fm2[s] = plotf(last.(a2))
  end
  t1 = L.tr[in.(L.tr.iid, Ref(keys(sidx))) .& .!in.(L.tr.out, Ref(("missing", "harvest"))), :]
  acc = Dict{Tuple{Int,Int,Int},NTuple{6,Float64}}()     # (s, k, j) → (Σw, ΣwB, ΣwH, Σw|H, ΣwA, Σw|A)
  for r in eachrow(t1)
    j = HBAD4P.dclass(r.dia_cm); j == 0 && continue
    s = sidx[r.iid]; k = kix[r.key]; w = r.w_ha / (r.dia_cm >= 12.7 ? f1[s] : fm1[s]); a = get(acc, (s, k, j), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))   # tally: subplot / microplot
    hok = r.ht_m > 0; aok = r.age > 0
    acc[(s, k, j)] = (a[1] + w, a[2] + w * π / 4 * (r.dia_cm / 100)^2, a[3] + (hok ? w * r.ht_m : 0.0), a[4] + (hok ? w : 0.0), a[5] + (aok ? w * r.age : 0.0), a[6] + (aok ? w : 0.0))
  end
  nohgt = 0
  for ((s, k, j), a) in acc
    H = a[4] > 0 ? a[3] / a[4] : NaN; isnan(H) && (nohgt += 1; H = 1.37 + 0.5 * HBAD4P.CLS_LO[j])   # (no height in the cell: a crude placeholder, counted)
    A = a[6] > 0 ? a[5] / a[6] : medage[keys_[k]]
    for v in 1:NL
      c = off[s] + ((k - 1) * NCLS + (j - 1)) * NL + (v - 1)
      N0[c] = a[1] * lw[v]; B0[c] = a[2] * lw[v]; H0[c] = H; A0[c] = A
    end
  end
  # v2 observation: survivors (v2 diameter / height) + ingrowth
  obs_add!(s, k, D, H) = (j = HBAD4P.dclass(D); j == 0 && return; On[k, j, s] += 1; OD2[k, j, s] += D^2; H > 0 && (OH[k, j, s] += H; OnH[k, j, s] += 1))
  # a subplot-tally tree (D ≥ 12.7 cm at v1) recorded below 12.7 cm at v2 (measurement noise) stays in the subplot tally (class 3 at 12.7 cm): scored on the
  # microplot area it would count 12.5 x
  nshrink = 0
  for r in eachrow(t1)
    r.out == "alive" || continue
    d2 = r.dia2_cm; (r.dia_cm >= 12.7 && d2 < 12.7) && (nshrink += 1; d2 = 12.7)
    obs_add!(sidx[r.iid], kix[r.key], d2, r.ht2_m)
  end
  for r in eachrow(L.ig); haskey(sidx, r.iid) && obs_add!(sidx[r.iid], kix[r.key], r.dia_cm, r.ht_m); end
  co2 = Dict(Int(r.year) => Float64(r.co2_ppm) for r in eachrow(CSV.read(CO2_CSV, DataFrame)))
  Cs = [mean(co2[y] for y in year(r.d1):max(year(r.d2) - 1, year(r.d1))) for r in eachrow(iv)]
  getw(d, i, dflt) = (v = get(d, i, dflt); isnan(v) ? dflt : v)
  Wv = [getw(W, i, 1.0) for i in iv.iid]; Wnv = [getw(Wn, i, 1.0) for i in iv.iid]; Tv = [getw(T, i, NaN) for i in iv.iid]
  nmiss = (W=count(i -> isnan(get(W, i, NaN)), iv.iid), Wn=count(i -> isnan(get(Wn, i, NaN)), iv.iid), T=count(isnan, Tv))
  S = HBAD4P.Sites4(off, nk, sp, cl, ln, N0, B0, H0, A0, copy(N0), copy(B0), copy(H0), copy(A0), nyr, frac, Vector{Bool}(iv.planted), Wv, Wnv, Tv, Cs)
  (S=S, iid=iv.iid, n=On, D2=OD2, Hs=OH, nH=OnH, fs2=fs2, fm2=fm2, f1=f1, fm1=fm1, nohgt=nohgt, nshrink=nshrink, nmiss=nmiss, ntrees=sum(On), iv=iv)
end

"Climate dictionaries by iid: W = period AET/PET, Wn = its normal, T = period mean annual temperature, the climate table (for Topt)."
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
"Observation parameters: negative-binomial size r for microplot (classes 1–2) and subplot (≥ 3) tallies; σD, σH of log QMD / log mean height per tree."
struct ObsP; rM::Float64; rS::Float64; σD::Float64; σH::Float64; end
@inline function nb_logpmf(n, μ, r)
  μ = max(μ, 1e-12)
  loggamma(n + r) - loggamma(r) - loggamma(n + 1.0) + r * log(r / (r + μ)) + n * log(μ / (r + μ))
end
"""Stand-table NLL of the simulated state (S.N/B/H after simulate!) against the v2 tables: Σ over sites, keys, classes of − log NegBin(n | μ = N f area, r)
+ size and height terms for the occupied cells (log QMD_obs − log Dq ~ N(0, σD² / n); log mean H_obs − log H ~ N(0, σH² / n_H)). Returns parts."""
function nll4(D, Ob::ObsP)
  S = D.S; nk = S.nk; ns = HBAD4P.nsites(S)
  pc = zeros(ns); pd = zeros(ns); ph = zeros(ns)
  Threads.@threads :static for s in 1:ns
    c_ = 0.0; d_ = 0.0; h_ = 0.0
    @inbounds for k in 1:nk, j in 1:NCLS
      i0 = S.off[s] + ((k - 1) * NCLS + (j - 1)) * NL
      Np = S.N[i0] + S.N[i0+1] + S.N[i0+2]; n = D.n[k, j, s]
      μ = Np * (j >= 3 ? D.fs2[s] : D.fm2[s]) * area_class(j)
      c_ -= nb_logpmf(Float64(n), μ, j >= 3 ? Ob.rS : Ob.rM)
      if n > 0 && Np > 1e-12
        Bp = S.B[i0] + S.B[i0+1] + S.B[i0+2]; dq = HBAD4P.qmd(Np, Bp); qo = sqrt(D.D2[k, j, s] / n)
        v = Ob.σD^2 / n; d_ += 0.5 * (log(qo / dq))^2 / v + 0.5 * log(2π * v)
        if D.nH[k, j, s] > 0
          Hp = (S.N[i0] * S.H[i0] + S.N[i0+1] * S.H[i0+1] + S.N[i0+2] * S.H[i0+2]) / Np; ho = D.Hs[k, j, s] / D.nH[k, j, s]
          vh = Ob.σH^2 / D.nH[k, j, s]; h_ += 0.5 * (log(ho / Hp))^2 / vh + 0.5 * log(2π * vh)
        end
      end
    end
    pc[s] = c_; pd[s] = d_; ph[s] = h_
  end
  (total=sum(pc) + sum(pd) + sum(ph), count=sum(pc), size=sum(pd), height=sum(ph))
end
"Persistence null: the v1 table carried to v2 unchanged (simulate nothing); same observation model."
function null_state!(D)
  S = D.S; copyto!(S.N, S.N0); copyto!(S.B, S.B0); copyto!(S.H, S.H0); copyto!(S.A, S.A0); D
end

# ---------------------------------------------------------------- cache (load + traits + build take minutes; the fit / eval tools reuse them)
using Serialization
"Built sets of a product directory, cached under HBAD4_CACHE (default <product>/hbad4_cache) by the set names and the lane mode."
function cached_sets(sets::Vector{String}; lanemode::Symbol=:sym, dir=DATA_DIR)
  code = string(hash(read(joinpath(@__DIR__, "data4.jl"))); base=16)                  # the builder's code version
  cd_ = get(ENV, "HBAD4_CACHE", joinpath(@__DIR__, "..", "..", "results", "hbad4", "cache")); mkpath(cd_)
  f = joinpath(cd_, "sets_$(string(hash(abspath(dir)); base=16))_$(join(sets, "_"))_$(lanemode)_$(REGION)_$(code).jls")
  isfile(f) && return deserialize(f)
  L = load4(dir); CL = climate4(); FT = forest4(); TR = traits4(L, CL.clim)
  out = (keys=L.keys, TR=TR, sets=Dict(s => build4(L, s, L.keys, CL.W, CL.Wn, CL.T, nothing, FT; lanemode=lanemode) for s in sets if any(==(s), L.iv.set)))
  serialize(f * ".part", out); mv(f * ".part", f; force=true)
  out
end
