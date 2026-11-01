# HBAD3 Sim A harness: the v2 harness (tools/hbad2/common.jl: tracked-tree cohorts, vigour sub-cohorts, likelihood) driving the HBAD3 kernel.
# Shared by tools/hbad3/fit3.jl and the HBAD3 evaluation tools. Requires the v2 env switches HBAD2_FORM=2 HBAD2_SITE=subplot HBAD2_VIGOUR=1
# HBAD2_SIZE_SPLIT=5,12.7,25,40 (+ HBAD2_SIGV_MAX).
include(joinpath(@__DIR__, "..", "hbad2", "common.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD3", "HBAD3.jl")); import .HBAD3
using DuckDB
const SFORM = parse(Int, get(ENV, "HBAD3_SFORM", "1"))    # growth-stress hazard form (HBAD3.stress)
const R3DIR = get(ENV, "HBAD3_R3DIR", joinpath(@__DIR__, "..", "..", "results", "hbad3"))   # HBAD3_R3DIR: fit3p_<tag> outputs elsewhere (nested CV)
# external inputs (read-only), under the workspace root (env WORKSPACE, default: the parent folder of this repo) — tools/hbad3/pipeline3p.env
const WSROOT = get(ENV, "WORKSPACE", normpath(joinpath(@__DIR__, "..", "..", "..")))
const SLIM_DB = get(ENV, "HBAD3_SLIM_DB", joinpath(WSROOT, "landis_region4", "data_r4", "fia_truth_r4.duckdb"))
const SHADETOL_CSV = get(ENV, "HBAD3_SHADETOL_CSV", joinpath(WSROOT, "landis_region4", "data_r4", "shadetol_all_species.csv"))
const LONGEVITY_CSV = get(ENV, "HBAD3_LONGEVITY_CSV", joinpath(WSROOT, "species_longevity_ref.csv"))
SITE_SUB && VIGOUR || error("HBAD3 harness expects HBAD2_SITE=subplot and HBAD2_VIGOUR=1")

# ---------------------------------------------------------------- per-species parameter space
const FREE3 = (:eD, :cD, :Dmax, :lnK, :gR, :mbg, :mD, :ms, :sdimax)
const LOG3 = Set([:Dmax, :ms, :sdimax])
const BOUNDS3 = Dict(:eD => (-6.0, 2.0), :cD => (-1.5, 3.0), :Dmax => (5.0, 300.0), :lnK => (-6.0, 4.0), :gR => (-3.0, 3.0),
                     :mbg => (-9.0, -1.0), :mD => (-4.0, 4.0), :ms => (1e-4, 2.0), :sdimax => (250.0, 3000.0))
const GN3 = (:k, :η, :λ, :μ, :γ, :mD2, :mSD, :pH, :pD, :pM, :σv, :ω, :km, :mW, :τg, :τm)
const GDEFAULT3 = Dict(:km => 3.0, :mW => 0.0, :τg => 0.0, :τm => 0.0)   # values of globals absent from older fits
const GLOG3 = Set([:k, :λ, :γ, :km])
const GBOUNDS3 = Dict(:k => (0.05, 50.0), :η => (-1.0, 3.0), :λ => (1e-4, 0.5), :μ => (-0.5, 3.0), :γ => (0.2, 60.0), :mD2 => (-3.0, 3.0),
                      :mSD => (-4.0, 2.0), :pH => (-1.0, 1.0), :pD => (-1.0, 1.0), :pM => (-2.0, 2.0), :σv => (0.0, SIGV_MAX), :ω => (0.0, 10.0), :km => (0.05, 50.0), :mW => (-3.0, 3.0), :τg => (-3.0, 3.0), :τm => (-3.0, 3.0))
enc3(nm, x) = nm in LOG3 ? log(x) : x
function dec3(nm, z); lo, hi = BOUNDS3[nm]; clamp(nm in LOG3 ? exp(z) : z, lo, hi); end
genc3(nm, x) = nm in GLOG3 ? log(x) : x
function gdec3(nm, z); lo, hi = GBOUNDS3[nm]; clamp(nm in GLOG3 ? exp(z) : (nm == :σv ? abs(z) : z), lo, hi); end

# ---------------------------------------------------------------- traits per key (tree-weighted over member species, TRAIN trees)
function key_traits(tr, keys_, longev)
  st = CSV.read(SHADETOL_CSV, DataFrame); std_ = Dict(String(r.sym) => Float64(r.shade_class) for r in eachrow(st))
  db = DBInterface.connect(DuckDB.DB, SLIM_DB; readonly=true)
  rs = DataFrame(DBInterface.execute(db, "SELECT SPECIES_SYMBOL s, CAST(WOOD_SPGR_GREENVOL_DRYWT AS DOUBLE) wd FROM REF_SPECIES ORDER BY SPECIES_SYMBOL"))
  DBInterface.close!(db)
  wdd = Dict(String(r.s) => r.wd for r in eachrow(rs) if !ismissing(r.wd))
  trn = tr[tr.set .== "train", :]
  miss = sort(setdiff(unique(trn.sym), keys(std_))); isempty(miss) || @info "species without a shade class in $(SHADETOL_CSV) (class 3 used)" miss
  rows = map(enumerate(keys_)) do (i, k)
    t = trn[trn.key .== k, :]; c = combine(groupby(t, :sym), nrow => :n)
    wavg(d, dflt) = (v = [get(d, s, dflt) for s in c.sym]; sum(v .* c.n) / sum(c.n))
    (key=k, ST=wavg(std_, 3.0), WD=wavg(wdd, 0.5), SW=mean(t.grp .== "S"), HX=log(quantile(t.ht_m, 0.99)), DX=log(quantile(t.dia_cm, 0.99)), LG=longev[i])
  end
  DataFrame(rows)
end
zscore(x) = (s = std(x); s > 0 ? (x .- mean(x)) ./ s : zero(x))

"Design matrix of a per-species parameter: \"sp\" (identity), \"g\" (one value), \"T1+T2\" (intercept + standardised traits), or \"=v\" (constant v, no coordinate)."
function design_matrix(mode::AbstractString, TD::DataFrame)
  nsp = nrow(TD)
  startswith(mode, "=") && return zeros(nsp, 0)
  mode == "sp" && return Matrix{Float64}(I, nsp, nsp)
  mode == "g" && return ones(nsp, 1)
  hcat(ones(nsp), [zscore(Float64.(TD[!, Symbol(t)])) for t in split(mode, "+")]...)
end

# ---------------------------------------------------------------- data: HBAD3 sites from the v2 cohort builder + water index per site
function water_by_iid(path=get(ENV, "HBAD3_WATER_CSV", joinpath(R3DIR, "interval_water.csv")))
  w = CSV.read(path, DataFrame)
  Dict(Int(r.iid) => Float64(r.aetpet_iv) for r in eachrow(w))
end
"Annual mean temperature normal per interval (interval_climate[_r4].csv, column tmean_n)."
function temp_by_iid(path=get(ENV, "HBAD3_CLIM_CSV", joinpath(R3DIR, "interval_climate.csv")))
  c = CSV.read(path, DataFrame; select=[:iid, :tmean_n])
  Dict(Int(r.iid) => Float64(r.tmean_n) for r in eachrow(c))
end
function build3(tr, set, keys_, wd, td=nothing)
  C = build_cohorts(tr, set, keys_); S = C.S; ns = length(S.off) - 1
  W = [wd[C.coh.site[C.parent[S.off[s]]] ÷ 10] for s in 1:ns]
  T = td === nothing ? fill(17.0, ns) : [td[C.coh.site[C.parent[S.off[s]]] ÷ 10] for s in 1:ns]
  S3 = HBAD3.Sites3(S.off, S.sp, S.N0, S.D0, S.H0, S.A0, S.nyr, S.frac, S.planted, S.zq, W, T)
  (S=S3, coh=C.coh, obs=C.obs, parent=C.parent, agg=C.agg)
end
"Mean of W over sites, for a W override (water switched off / constant)."
set_water!(C, w::Real) = (fill!(C.S.W, w); C)

# ---------------------------------------------------------------- decode
function hd_table(keys_)
  h = CSV.read(joinpath(R3DIR, "hd_allometry.csv"), DataFrame); d = Dict(String(r.key) => (r.a, r.b, r.c) for r in eachrow(h))
  [d[k] for k in keys_]
end
"H–D allometry per key fitted to the TRAIN trees (Chapman–Richards on ln H, log parameters, damped Gauss–Newton; as scripts/hbad3/hd_allocation.py)."
function fit_hd(tr, keys_)
  map(keys_) do k
    t = tr[(tr.key .== k) .& (tr.set .== "train"), :]; D = Float64.(t.dia_cm); y = log.(Float64.(t.ht_m))
    f(p) = log.(1.37 .+ exp(p[1]) .* (1 .- exp.(-exp(p[2]) .* D)) .^ exp(p[3])) .- y
    p = [log(30.0), log(0.03), 0.0]; μ = 1e-3; r = f(p); c = sum(abs2, r)
    for _ in 1:200
      J = hcat([(f(p .+ 1e-6 .* (1:3 .== j)) .- r) ./ 1e-6 for j in 1:3]...)
      δ = -((J' * J + μ * I) \ (J' * r)); pn = p .+ δ; rn = f(pn); cn = sum(abs2, rn)
      if cn < c; p, r, μ = pn, rn, μ / 3; abs(c - cn) < 1e-10 * c && (c = cn; break); c = cn; else; μ *= 5; end
    end
    (exp(p[1]), exp(p[2]), exp(p[3]))
  end
end
function decode3(sv::AbstractMatrix, gv::AbstractVector, hd, longev)   # sv: nsp × length(FREE3) encoded; gv: encoded globals
  nsp = size(sv, 1)
  P = [HBAD3.SpP3([dec3(FREE3[j], sv[s, j]) for j in 1:length(FREE3)]..., hd[s]..., longev[s]) for s in 1:nsp]
  g = Dict(GN3[j] => gdec3(GN3[j], gv[j]) for j in 1:length(GN3))
  P, HBAD3.GlobP3(g[:k], g[:η], g[:λ], g[:μ], g[:γ], g[:mD2], g[:mSD], 10.0, g[:pH], g[:pD], g[:pM], g[:σv], g[:ω], SFORM, g[:km], g[:mW], g[:τg], g[:τm])
end
function eval3(C, P, G); HBAD3.simulate!(C.S, P, G); nll_parts(cohort_view(C), C.obs); end

"Per-cohort predictions (cohort level) with start-of-interval O and RD — the table the scripts/hbad3 diagnostics read (resid_<tag>.csv)."
function export_resid3(C, P, G, set)
  S = C.S
  S0 = HBAD3.Sites3(S.off, S.sp, S.N0, S.D0, S.H0, S.A0, zeros(Int, length(S.nyr)), fill(1e-9, length(S.nyr)), S.planted, S.zq, S.W, S.T)
  HBAD3.simulate!(S0, P, G); O0 = copy(S0.O); RD0 = zeros(length(O0))
  for s in 1:length(S.off)-1
    lo, hi = S.off[s], S.off[s+1] - 1; tot = sum(S0.S[lo:hi]); sm = sum(S0.S[i] * P[S.sp[i]].sdimax for i in lo:hi) / tot
    RD0[lo:hi] .= tot / sm
  end
  HBAD3.simulate!(S, P, G); V = cohort_view(C); o = C.obs
  fo = [searchsortedfirst(C.parent, p) for p in 1:length(o.n1)]
  DataFrame(set=set, site=C.coh.site, iid=C.coh.site .÷ 10, key=C.coh.key, planted=C.coh.planted, dt=o.dt, n1=o.n1, s=o.s,
            N1=V.N0, D1=V.D0, H1=V.H0, A1=S.A0[fo], O=O0[fo], RD=RD0[fo], ps=V.N ./ V.N0, D2s=V.D, H2s=V.H, D2o=o.D2o, H2o=o.H2o)
end
