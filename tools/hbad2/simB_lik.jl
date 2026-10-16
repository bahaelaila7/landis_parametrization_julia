# HBAD v2 Sim B stand-table likelihood (for calibration): free runs from the first visit with recruitment; at every later visit the
# expected number of TALLIED trees per (species key, size class) — model N through the lognormal within-cohort D operator, divided by
# the FIA expansion of that class (microplot saplings 185.2 /ha, subplot trees 14.87 /ha) — is scored against the observed tally
# with a Poisson NLL (stand table = growth, mortality, recruitment and composition at once; no biomass, nothing tracked).
# Library: include after common.jl.  build_simB(set, keys_, ...) → B;  simB_nll!(B, P, G, R) → NamedTuple
using Distributions: Normal, cdf
const SB_BREAKS = [2.54, 5.0, 12.7, 25.0, 40.0]                       # classes [2.54,5) [5,12.7) [12.7,25) [25,40) [40,∞)
const SB_EXPAND = [185.2, 185.2, 14.8709, 14.8709, 14.8709]            # trees/ha represented by one tallied tree
const SB_NC = length(SB_BREAKS)

struct SimBData
  F::FreeSites
  N0::Vector{Float64}; D0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}; sp0::Vector{Int32}; live0::Vector{Int}
  years::Vector{Int}
  vis_year::Vector{Vector{Int}}           # per site: run years of the scored visits
  obs::Array{Float64,4}                   # site × visit × species × class: observed tallied tree counts
  lam::Array{Float64,4}                   # same shape: expected counts (filled by the run)
  hl_obs::Matrix{Float64}; hl_sim::Matrix{Float64}   # Lorey height per site × visit (NaN = none)
  σ::Float64
end

function build_simB(set::AbstractString, keys_::Vector{String}; σ=0.16, cap_extra=120, natural_only=true)
  iv = DataFrame(Arrow.Table(joinpath(R2DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_trees.arrow")))
  ig = DataFrame(Arrow.Table(joinpath(R2DIR, "interval_ingrowth.arrow")))
  kix = Dict(k => i for (i, k) in enumerate(keys_)); big = Set(k for k in keys_ if k[1] != '_'); keyof(s, g) = s in big ? s : "_" * g
  iv.pk = [(a, b, c, d) for (a, b, c, d) in zip(iv.st, iv.u, iv.c, iv.p)]
  pl = combine(groupby(iv, :pk), :set => first => :set, [:trt, :dst, :harv, :planted] => ((a, b, c, d) -> !any(a .| b .| c .| (natural_only ? d : falses(length(d))))) => :clean, nrow => :nint)
  pl = pl[(coalesce.(pl.set, "") .== set) .& pl.clean .& (pl.nint .>= 2), :]
  raw.key = [keyof(s, g) for (s, g) in zip(raw.sym, raw.grp)]; ig.key = [keyof(s, g) for (s, g) in zip(ig.sym, ig.grp)]
  rbyi = groupby(raw, :iid); igbyi = groupby(ig, :iid); ivbypk = groupby(iv, :pk)
  cls(D) = searchsortedlast(SB_BREAKS, D)
  sites = []
  for r in eachrow(pl)
    ivs = sort(DataFrame(ivbypk[(r.pk,)]), :k); all(i -> haskey(rbyi, (i,)), ivs.iid) || continue
    y0 = ivs.d1[1]
    for sbsel in (SITE_SUB ? (1:4) : (0:0))                     # subplot sites (empty ones included), or the whole plot (0)
      f(t) = sbsel == 0 ? t : t[t.sb .== sbsel, :]
      vis = Tuple{Int,Vector{String},Vector{Float64},Vector{Float64},Vector{Float64}}[]
      for (j, ivr) in enumerate(eachrow(ivs))
        t = f(DataFrame(rbyi[(ivr.iid,)]))
        j > 1 && push!(vis, (round(Int, Dates.value(ivr.d1 - y0) / 365.25), t.key, t.dia_cm, t.ht_m, t.w_ha))
        if j == nrow(ivs)
          a = t[t.out .== "alive", :]; g = haskey(igbyi, (ivr.iid,)) ? f(DataFrame(igbyi[(ivr.iid,)])) : DataFrame(key=String[], dia_cm=Float64[], ht_m=Float64[], w_ha=Float64[])
          push!(vis, (round(Int, Dates.value(ivr.d2 - y0) / 365.25), vcat(a.key, g.key), vcat(a.dia2_cm, g.dia_cm), vcat(a.ht2_m, g.ht_m), vcat(a.w2_ha, g.w_ha)))
        end
      end
      push!(sites, (init=f(DataFrame(rbyi[(ivs.iid[1],)])), vis=vis))
    end
  end
  ns = length(sites); nv = maximum(length(s.vis) for s in sites); nsp = length(keys_)
  off = [1]; spv = Int[]; N0 = Float64[]; D0 = Float64[]; H0 = Float64[]; A0 = Float64[]; live = Int[]
  obs = zeros(ns, nv, nsp, SB_NC); hlo = fill(NaN, ns, nv); vy = Vector{Vector{Int}}()
  for (s, st) in enumerate(sites)
    t = st.init; t.bc = floor.(Int, -t.age ./ BIRTH_W) .* 100 .+ dclass.(t.dia_cm)
    g = nrow(t) == 0 ? DataFrame(key=String[], N=Float64[], D=Float64[], H=Float64[], A=Float64[]) :
      combine(groupby(t, [:key, :bc]), :w_ha => (w -> sum(w) * SITE_MULT) => :N, [:w_ha, :dia_cm] => ((w, x) -> sqrt(sum(w .* x .^ 2) / sum(w))) => :D,
                [:w_ha, :ht_m] => ((w, x) -> sum(w .* x) / sum(w)) => :H, [:w_ha, :age] => ((w, x) -> sum(w .* x) / sum(w)) => :A)
    for r in eachrow(g); push!(spv, kix[r.key]); push!(N0, r.N); push!(D0, r.D); push!(H0, r.H); push!(A0, max(r.A, 1.0)); end
    push!(live, nrow(g))
    for _ in 1:cap_extra; push!(spv, 1); push!(N0, 0.0); push!(D0, 2.54); push!(H0, 3.0); push!(A0, 5.0); end
    push!(off, length(spv) + 1)
    push!(vy, [v[1] for v in st.vis])
    for (k, (y, key, D, H, w)) in enumerate(st.vis)
      for i in eachindex(D); c = cls(D[i]); c >= 1 && (obs[s, k, kix[key[i]], c] += 1); end      # tallied trees (each record = one tree)
      ba = (pi / 4) .* (D ./ 100) .^ 2 .* w; sum(ba) > 0 && (hlo[s, k] = sum(ba .* H) / sum(ba))
    end
  end
  S = Sites(off, spv, copy(N0), copy(D0), copy(H0), copy(A0), zeros(Int, ns), zeros(ns), fill(false, ns))
  F = FreeSites(S, copy(live), nsp)
  SimBData(F, N0, D0, H0, A0, Int32.(spv), live, [maximum(v) for v in vy], vy, obs, similar(obs), hlo, fill(NaN, ns, nv), σ)
end

function _reset!(B::SimBData)
  S = B.F.S
  copyto!(S.N, B.N0); copyto!(S.D, B.D0); copyto!(S.H, B.H0); copyto!(S.A, B.A0); copyto!(S.sp, B.sp0); copyto!(S.N0, B.N0)
  copyto!(B.F.live, B.live0); fill!(B.F.open_slot, 0); fill!(B.F.open_year, 0); fill!(B.F.dropped, 0); fill!(B.lam, 0.0); fill!(B.hl_sim, NaN)
end

@inline Φ(z) = 0.5 * erfc_(-z / sqrt(2))
@inline erfc_(x) = (t = 1 / (1 + 0.5 * abs(x)); y = t * exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
                    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))); x >= 0 ? y : 2 - y)

function simB_nll!(B::SimBData, P, G, R)
  _reset!(B); F = B.F; S = F.S; σ = B.σ; lb = log.(SB_BREAKS)
  snap = function (s, y)
    k = findfirst(==(y), B.vis_year[s]); k === nothing && return
    lo = S.off[s]; hi = lo + F.live[s] - 1; bas = 0.0; bah = 0.0
    for i in lo:hi
      N = S.N[i]; N <= 0 && continue
      μ = log(S.D[i]) - σ^2; sp = S.sp[i]; prev = Φ((lb[1] - μ) / σ)
      for c in 1:SB_NC
        nxt = c < SB_NC ? Φ((lb[c+1] - μ) / σ) : 1.0
        B.lam[s, k, sp, c] += N * (nxt - prev) / (SB_EXPAND[c] * SITE_MULT); prev = nxt
      end
      ba = N * S.D[i]^2; bas += ba; bah += ba * S.H[i]
    end
    B.hl_sim[s, k] = bas > 0 ? bah / bas : NaN
  end
  run_free!(F, P, G, R, B.years, snap)
  nll = 0.0
  @inbounds for ix in eachindex(B.obs)
    o = B.obs[ix]; λ = max(B.lam[ix], 1e-6)
    # only cells of visits that exist (obs or lam > 0 there); padded visits have lam = 0 and obs = 0 → contribute λ = 1e-6 ≈ 0
    nll += λ - o * log(λ) + (o > 0 ? o * log(o) - o : 0.0)            # Poisson deviance/2 (0 for a perfect fit)
  end
  e = [log(B.hl_sim[i] / B.hl_obs[i]) for i in eachindex(B.hl_obs) if isfinite(B.hl_obs[i]) && isfinite(B.hl_sim[i])]
  nH = 0.5 * length(e) * log(sum(abs2, e) / length(e))
  (total=nll + nH, counts=nll, H=nH, dropped=sum(F.dropped))
end
