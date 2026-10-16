# HBAD v2 Sim A harness: tracked-tree cohorts of undisturbed remeasurement intervals → HBAD2.Sites; joint likelihood.
using Arrow, DataFrames, CSV, Statistics, Printf, Dates, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD2", "HBAD2.jl")); using .HBAD2
const R2DIR = joinpath(@__DIR__, "..", "..", "results", "hbad2")
const BIRTH_W = parse(Int, get(ENV, "HBAD2_BIRTH_W", "10"))      # cohort = (plot, species key, birth-year class of this width)
const MIN_SP_TREES = parse(Int, get(ENV, "HBAD2_MIN_SP_TREES", "3000"))
# optional split of a birth-year class by initial diameter class (cm breaks), e.g. "5,12.7,25,40": sub-cohorts of one age class that
# differ in size are kept apart instead of being lumped into one quadratic mean (mean-field threshold artefact, docs/hbad_v2.md §5)
const BOUNDS_OVERRIDE = Dict{Symbol,Tuple{Float64,Float64}}()
const SIZE_BREAKS = (b = get(ENV, "HBAD2_SIZE_SPLIT", ""); isempty(b) ? Float64[] : parse.(Float64, split(b, ",")))
dclass(D) = searchsortedfirst(SIZE_BREAKS, D)
const FORM = parse(Int, get(ENV, "HBAD2_FORM", "1"))          # potential-growth form (HBAD2.GlobP.form)
if FORM == 2   # power-exponential: rX = increment at the reference size [cm/yr | m/yr], cX = size exponent, Xmax = decay scale
  merge!(BOUNDS_OVERRIDE, Dict(:rD => (0.01, 3.0), :cD => (-1.5, 3.0), :Dmax => (5.0, 300.0), :rH => (0.005, 3.0), :cH => (-1.5, 3.0), :Hmax => (2.0, 200.0)))
end

"Tracked trees of undisturbed intervals with their species key (own key if ≥ MIN_SP_TREES train trees, else _S/_H)."
function load_trees()
  iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); tr = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow")))
  ok = Set(iv.iid[.!(iv.trt .| iv.dst .| iv.harv) .& (iv.dt .> 2)])
  ivd = Dict(r.iid => r for r in eachrow(iv))
  tr = filter(r -> r.iid in ok && r.age > 0 && r.ht_m > 0 && r.dia_cm > 0, tr)
  tr.set = [coalesce(ivd[i].set, "") for i in tr.iid]; tr.dt = [ivd[i].dt for i in tr.iid]; tr.y1 = [year(ivd[i].d1) for i in tr.iid]
  tr.l3 = [coalesce(ivd[i].l3, "") for i in tr.iid]; tr.planted = [ivd[i].planted for i in tr.iid]
  cnt = combine(groupby(filter(r -> r.set == "train", tr), :sym), nrow => :n); big = Set(cnt.sym[cnt.n .>= MIN_SP_TREES])
  tr.key = [s in big ? s : "_" * g for (s, g) in zip(tr.sym, tr.grp)]
  tr
end

"Cohorts of one set: initial state from all live trees at v1, tracked survivors at v2 (v1 expansion weights → N only decreases)."
function build_cohorts(tr::DataFrame, set::AbstractString, keys_::Vector{String})
  kix = Dict(k => i for (i, k) in enumerate(keys_))
  d = filter(r -> r.set == set, tr)
  d.bc = floor.(Int, (d.y1 .- d.age) ./ BIRTH_W) .* 100 .+ dclass.(d.dia_cm)
  d.alive = d.out .== "alive"; d.counted = d.out .!= "missing"
  g = combine(groupby(d, [:iid, :key, :bc]),
    :w_ha => sum => :N1, [:w_ha, :dia_cm] => ((w, x) -> sqrt(sum(w .* x .^ 2) / sum(w))) => :D1,
    [:w_ha, :ht_m] => ((w, x) -> sum(w .* x) / sum(w)) => :H1, [:w_ha, :age] => ((w, x) -> sum(w .* x) / sum(w)) => :A1,
    :counted => sum => :n1, :alive => sum => :s,
    [:w_ha, :alive] => ((w, a) -> sum(w[a])) => :N2o,
    [:w_ha, :dia2_cm, :alive] => ((w, x, a) -> any(a) ? sqrt(sum(w[a] .* x[a] .^ 2) / sum(w[a])) : NaN) => :D2o,
    [:w_ha, :ht2_m, :alive] => ((w, x, a) -> any(a) ? sum(w[a] .* x[a]) / sum(w[a]) : NaN) => :H2o,
    :dt => first => :dt, :l3 => first => :l3, :planted => first => :planted)
  sort!(g, [:iid, :key, :bc]); g.spi = [kix[k] for k in g.key]
  sites = unique(g.iid); off = Vector{Int}(undef, length(sites) + 1); off[1] = 1
  ix = 1
  for (s, i) in enumerate(sites)
    while ix <= nrow(g) && g.iid[ix] == i; ix += 1; end
    off[s+1] = ix
  end
  dts = [g.dt[off[s]] for s in eachindex(sites)]; pls = [Bool(g.planted[off[s]]) for s in eachindex(sites)]
  S = Sites(off, g.spi, Float64.(g.N1), Float64.(g.D1), Float64.(g.H1), Float64.(g.A1), floor.(Int, dts), dts .- floor.(dts), pls)
  obs = (n1=Int.(g.n1), s=Int.(g.s), D2o=Float64.(g.D2o), H2o=Float64.(g.H2o), dt=Float64.(g.dt))
  (S=S, coh=g, obs=obs)
end

"Joint negative log-likelihood (per tree-observation units): binomial survival + Gaussian ln D2, ln H2 of survivors (σ profiled)."
function nll_parts(S::Sites, coh::NamedTuple)
  nm = 0.0; sD = 0.0; sH = 0.0; cD = 0; nobs_m = 0
  @inbounds for i in eachindex(S.N)
    n1 = coh.n1[i]; n1 == 0 && continue
    s = coh.s[i]; p = clamp(S.N[i] / S.N0[i], 1e-9, 1 - 1e-9)
    nm -= s * log(p) + (n1 - s) * log(1 - p); nobs_m += n1
    if s > 0
      eD = log(S.D[i] / coh.D2o[i]); eH = log(max(S.H[i], 0.1) / coh.H2o[i])
      sD += s * eD^2; sH += s * eH^2; cD += 1
    end
  end
  nD = 0.5 * cD * log(sD / cD); nH = 0.5 * cD * log(sH / cD)
  (total=nm + nD + nH, mort=nm, D=nD, H=nH, sigD=sqrt(sD / cD), sigH=sqrt(sH / cD), ncoh=cD, ntrees=nobs_m)
end

# ---------------- parameter vector ⇄ species parameters
# per species (14 free, log-space where positive): Hmax rH cH Dmax rD cD gOD gRD gOH gRH mbg mD msup sdimax ; global: mD2 pH pD pM
const FREE = (:Hmax, :rH, :cH, :Dmax, :rD, :cD, :gOD, :gRD, :gOH, :gRH, :mbg, :mD, :msup, :sdimax)
const NGLOB = 5
const LOGP = Set([:Hmax, :rH, :cH, :Dmax, :rD, :cD, :msup, :sdimax])
const BOUNDS = Dict(:Hmax => (8.0, 70.0), :rH => (1e-3, 0.5), :cH => (0.3, 4.0), :Dmax => (15.0, 250.0), :rD => (1e-3, 0.5), :cD => (0.3, 4.0),
                    :gOD => (-1.0, 8.0), :gRD => (-3.0, 3.0), :gOH => (-1.0, 8.0), :gRH => (-3.0, 3.0), :mbg => (-9.0, -1.0), :mD => (-4.0, 4.0),
                    :msup => (1e-4, 1.0), :sdimax => (250.0, 2000.0))
const NF = length(FREE)
for (k, v) in BOUNDS_OVERRIDE; BOUNDS[k] = v; end
FORM == 2 && (delete!(LOGP, :cD); delete!(LOGP, :cH))
enc(nm, x) = nm in LOGP ? log(x) : x
function dec(nm, z)
  lo, hi = BOUNDS[nm]; clamp(nm in LOGP ? exp(z) : z, lo, hi)
end
function decode(θ::AbstractVector, nsp::Int, longevity::Vector{Float64})
  P = Vector{SpP}(undef, nsp)
  for s in 1:nsp
    v = [dec(FREE[k], θ[(s-1)*NF+k]) for k in 1:NF]
    P[s] = SpP(v..., longevity[s])
  end
  g = θ[nsp*NF+1:end]
  P, HBAD2.GlobP(clamp(g[1], -3, 3), 10.0, clamp(g[2], -1, 1), clamp(g[3], -1, 1), clamp(g[4], -2, 2), FORM, clamp(g[5], -4, 2))
end

"Per-species diagnostics: observed vs predicted deaths (tree counts), mean annual ΔD and ΔH of survivors (obs vs sim), RD at start."
function diagnostics(S::Sites, obs::NamedTuple, keys_::Vector{String}; io=stdout)
  nsp = length(keys_)
  od = zeros(nsp); pd = zeros(nsp); n = zeros(nsp); dDo = zeros(nsp); dDs = zeros(nsp); dHo = zeros(nsp); dHs = zeros(nsp); ns = zeros(nsp)
  for i in eachindex(S.N)
    k = S.sp[i]; n1 = obs.n1[i]; n1 == 0 && continue; s = obs.s[i]
    n[k] += n1; od[k] += n1 - s; pd[k] += n1 * (1 - S.N[i] / S.N0[i])
    if s > 0
      ns[k] += s; dDo[k] += s * (obs.D2o[i] - S.D0[i]) / obs.dt[i]; dDs[k] += s * (S.D[i] - S.D0[i]) / obs.dt[i]
      dHo[k] += s * (obs.H2o[i] - S.H0[i]) / obs.dt[i]; dHs[k] += s * (S.H[i] - S.H0[i]) / obs.dt[i]
    end
  end
  @printf(io, "%-7s %7s %8s %8s %7s %7s %7s %7s\n", "key", "trees", "dead_obs", "dead_sim", "dD_obs", "dD_sim", "dH_obs", "dH_sim")
  for k in 1:nsp
    @printf(io, "%-7s %7d %8.0f %8.0f %7.3f %7.3f %7.3f %7.3f\n", keys_[k], n[k], od[k], pd[k], dDo[k] / ns[k], dDs[k] / ns[k], dHo[k] / ns[k], dHs[k] / ns[k])
  end
  @printf(io, "%-7s %7d %8.0f %8.0f %7.3f %7.3f %7.3f %7.3f\n", "ALL", sum(n), sum(od), sum(pd), sum(dDo) / sum(ns), sum(dDs) / sum(ns), sum(dHo) / sum(ns), sum(dHs) / sum(ns))
end

"Read a θ vector; a v2a-layout vector (13/species + one global size slope) is expanded to the current layout."
function read_theta(path, nsp)
  θi = vec(CSV.read(path, DataFrame).theta)
  length(θi) == nsp * NF + NGLOB && return θi
  length(θi) == nsp * NF + 4 && return vcat(θi, 0.0)           # before mSD (v2b/v2c)
  length(θi) == nsp * 13 + 1 || error("θ length $(length(θi)) not recognised")
  θ = zeros(nsp * NF + NGLOB)
  for s in 1:nsp
    o = (s - 1) * 13   # v2a layout; n = (s - 1) * NF
    θ[n+1:n+11] = θi[o+1:o+11]; θ[n+12] = θi[end]; θ[n+13] = θi[o+12]; θ[n+14] = θi[o+13]
  end
  θ
end

"Each fit directory records the structural switches it was fitted with; every tool checks them against the current env (HBAD2_FORM, HBAD2_SIZE_SPLIT)."
function write_meta(dir); open(joinpath(dir, "meta.txt"), "w") do io; println(io, "form=", FORM); println(io, "size_split=", join(SIZE_BREAKS, ",")); end; end
function check_meta(dir)
  f = joinpath(dir, "meta.txt"); isfile(f) || error("$f missing: cannot verify growth form / size split of this fit")
  m = Dict(split(l, "=")[1] => (length(split(l, "=")) > 1 ? split(l, "=")[2] : "") for l in readlines(f))
  parse(Int, m["form"]) == FORM || error("fit $(dir) has form=$(m["form"]) but HBAD2_FORM=$(FORM)")
  m["size_split"] == join(SIZE_BREAKS, ",") || @warn "fit $(dir) was fitted with size_split=$(m["size_split"]); evaluating with $(join(SIZE_BREAKS, ","))"
end
