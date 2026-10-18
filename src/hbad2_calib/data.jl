# HBAD2 calibration data: the Sim A tracked-tree cohorts (prototype tools/hbad2/common.jl `load_trees`, `build_cohorts`) and the Sim B free-run sites with their observed
# stand tables (tools/hbad2/simB_lik.jl `build_simB`), built from the interval product (Data.HBAD2Intervals) and loaded as HBAD2 plugin SoAs.

const SZ_EDGES = [0.0, 5.0, 12.7, 25.0, 40.0]      # Sim A size-class term: classes [<5) [5,12.7) [12.7,25) [25,40) [40,inf)
const SB_BREAKS = [2.54, 5.0, 12.7, 25.0, 40.0]    # Sim B tally classes [2.54,5) [5,12.7) [12.7,25) [25,40) [40,inf)
const SB_EXPAND = [185.2, 185.2, 14.8709, 14.8709, 14.8709]   # trees/ha represented by one tallied tree (microplot saplings, subplot trees)
const SB_NC = length(SB_BREAKS)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# tracked trees
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
    load_tracked_trees(x::Data.HBAD2Intervals, sw::CalibSwitches) -> DataFrame

The tracked trees of the undisturbed intervals (no treatment / disturbance / harvest flag, dt > 2 yr; age, height, diameter > 0) with their interval attributes
(`set dt y1 l3 planted`), species key (own symbol if >= `sw.min_sp_trees` TRAIN trees, else `_S` / `_H` by softwood / hardwood), site id (`iid*10 + SUBP` for subplot sites)
and the densities `w_ha`, `w2_ha` multiplied by the site multiplier.
"""
function load_tracked_trees(x::Data.HBAD2Intervals, sw::CalibSwitches)::DataFrame
  iv = x.intervals; tr = x.trees
  ok = Set(iv.iid[.!(iv.trt .| iv.dst .| iv.harv) .& (iv.dt .> 2)])
  ivd = Dict(r.iid => r for r in eachrow(iv))
  tr = filter(r -> r.iid in ok && r.age > 0 && r.ht_m > 0 && r.dia_cm > 0, tr)
  tr.set = [coalesce(ivd[i].set, "") for i in tr.iid]; tr.dt = [ivd[i].dt for i in tr.iid]; tr.y1 = [Dates.year(ivd[i].d1) for i in tr.iid]
  tr.l3 = [coalesce(ivd[i].l3, "") for i in tr.iid]; tr.planted = [ivd[i].planted for i in tr.iid]
  cnt = combine(groupby(filter(r -> r.set == "train", tr), :sym), nrow => :n); big = Set(cnt.sym[cnt.n .>= sw.min_sp_trees])
  tr.key = [s in big ? s : "_" * g for (s, g) in zip(tr.sym, tr.grp)]
  tr.site = sw.site_sub ? tr.iid .* 10 .+ tr.sb : copy(tr.iid)
  tr.w_ha = tr.w_ha .* sw.site_mult; tr.w2_ha = tr.w2_ha .* sw.site_mult
  return tr
end

"""
    species_longevity(csv, tr, keys) -> Vector{Float64}

LONGEVITY (yr) of every species key: the median over the symbols of the key's trees of the table `csv` (columns `species_symbol`, `longevity_yr`; 300 for a symbol without a row).
"""
function species_longevity(csv::AbstractString, tr::DataFrame, keys_::Vector{String})
  isfile(csv) || error("species longevity table not found: $csv")
  lon = CSV.read(csv, DataFrame); lond = Dict(String(r.species_symbol) => Float64(r.longevity_yr) for r in eachrow(lon))
  return [Statistics.median([get(lond, s, 300.0) for s in unique(tr.sym[tr.key .== k])]) for k in keys_]
end

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sim A
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
    SimAData

One set (train / val / test) of Sim A: independent interval sites on the plugin SoA `soa` (state arrays `c_tph c_dq c_h c_age c_zq`; `q` sub-cohorts per observed cohort = 3 with vigour),
the initial state `N0 D0 H0 A0` of every sub-cohort, per-site whole years `nyr` and fractional remainder `frac`, and the per-cohort observations (`n1` tracked trees, `s` survivors,
`D2o`, `H2o` survivor means at v2, `dt`, `sc` survivor size-class counts). Working buffers `aggN aggD aggH` hold the cohort view (sub-cohorts aggregated back to the observed cohort).
"""
struct SimAData{S<:PanCore.AnySoA}
  soa::S
  q::Int
  nco::Int
  coh::DataFrame
  n1::Vector{Int}; s::Vector{Int}; D2o::Vector{Float64}; H2o::Vector{Float64}; dt::Vector{Float64}; sc::Matrix{Int}
  N0::Vector{Float64}; D0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}; sp0::Vector{UIntType}
  Nc0::Vector{Float64}; Dc0::Vector{Float64}; Hc0::Vector{Float64}      # per observed cohort: initial N, D, H (the view's fallback when a cohort dies out)
  nyr::Vector{Int}; frac::Vector{Float64}
  aggN::Vector{Float64}; aggD::Vector{Float64}; aggH::Vector{Float64}
  nsub_sites::Int
end

"Number of interval sites, observed cohorts and sub-cohorts of a Sim A set."
Base.show(io::IO, A::SimAData) = print(io, "SimAData(", A.nsub_sites, " sites, ", A.nco, " cohorts, ", length(A.N0), " sub-cohorts)")

"""
    build_simA(tr, set, keys, sw, pc; rng) -> SimAData

Cohorts of `set` from the tracked trees `tr` (see [`load_tracked_trees`](@ref)): initial state from all live trees at v1 (cohort = site x key x birth class x initial size class), tracked
survivors at v2 (v1 expansion weights, so N only decreases). The SoA is built for the plugin config `pc` (species list = `keys`).
"""
function build_simA(tr::DataFrame, set::AbstractString, keys_::Vector{String}, sw::CalibSwitches, pc; rng::Random.AbstractRNG=Random.Xoshiro(1))
  kix = Dict(k => i for (i, k) in enumerate(keys_))
  d = filter(r -> r.set == set, tr)
  d.bc = floor.(Int, (d.y1 .- d.age) ./ sw.birth_w) .* 100 .+ dclass.(Ref(sw), d.dia_cm)
  d.alive = d.out .== "alive"; d.counted = d.out .!= "missing"
  g = combine(groupby(d, [:site, :key, :bc]),
    :w_ha => sum => :N1, [:w_ha, :dia_cm] => ((w, x) -> sqrt(sum(w .* x .^ 2) / sum(w))) => :D1,
    [:w_ha, :ht_m] => ((w, x) -> sum(w .* x) / sum(w)) => :H1, [:w_ha, :age] => ((w, x) -> sum(w .* x) / sum(w)) => :A1,
    :counted => sum => :n1, :alive => sum => :s,
    [:w_ha, :alive] => ((w, a) -> sum(w[a])) => :N2o,
    [:w_ha, :dia2_cm, :alive] => ((w, x, a) -> any(a) ? sqrt(sum(w[a] .* x[a] .^ 2) / sum(w[a])) : NaN) => :D2o,
    [:w_ha, :ht2_m, :alive] => ((w, x, a) -> any(a) ? sum(w[a] .* x[a]) / sum(w[a]) : NaN) => :H2o,
    :dt => first => :dt, :l3 => first => :l3, :planted => first => :planted)
  sort!(g, [:site, :key, :bc]); g.spi = [kix[k] for k in g.key]
  sites = unique(g.site); off = Vector{Int}(undef, length(sites) + 1); off[1] = 1
  ix = 1
  for (s, i) in enumerate(sites)
    while ix <= nrow(g) && g.site[ix] == i; ix += 1; end
    off[s+1] = ix
  end
  dts = [g.dt[off[s]] for s in eachindex(sites)]; pls = [Bool(g.planted[off[s]]) for s in eachindex(sites)]
  a = d[d.alive, :]; a.c2 = [searchsortedlast(SZ_EDGES, x) for x in a.dia2_cm]
  ci = Dict((r.site, r.key, r.bc) => i for (i, r) in enumerate(eachrow(g)))
  sc = zeros(Int, nrow(g), length(SZ_EDGES))
  for r in eachrow(a); c = clamp(r.c2, 1, length(SZ_EDGES)); sc[ci[(r.site, r.key, r.bc)], c] += 1; end
  nc = nrow(g)
  q = sw.vigour ? 3 : 1
  rep(x) = repeat(x; inner=q)
  zq = sw.vigour ? repeat(collect(HBAD2Plugin.VIG_Z), nc) : zeros(nc)
  wq = sw.vigour ? repeat(collect(HBAD2Plugin.VIG_W), nc) : ones(nc)
  N0 = rep(Float64.(g.N1)) .* wq; D0 = rep(Float64.(g.D1)); H0 = rep(Float64.(g.H1)); A0 = rep(Float64.(g.A1)); spi = rep(g.spi)
  offq = (off .- 1) .* q .+ 1
  nyr = floor.(Int, dts); frac = dts .- floor.(dts)
  soa = Pan.make_sites_hbad2_arrays(pc, offq, spi, A0, N0, D0, H0, zq; rng=rng, no_establishment=true, planted=pls)
  # the SoA keeps the initial state as the reset source: copies of the state arrays in CSR order
  return SimAData(soa, q, nc, g, Int.(g.n1), Int.(g.s), Float64.(g.D2o), Float64.(g.H2o), Float64.(g.dt), sc,
                  N0, D0, H0, A0, UIntType.(spi), Float64.(g.N1), Float64.(g.D1), Float64.(g.H1), nyr, frac,
                  zeros(nc), zeros(nc), zeros(nc), length(sites))
end

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sim B
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
    SimBData

One set of Sim B: free runs (recruitment on) of the 4 subplot sites of every clean natural plot with >= 2 intervals, from the first visit; the stand tables observed at every later
visit (`obs[site, visit, key, class]` tallied trees, `hl_obs[site, visit]` Lorey height), the visit run years `vyear[site, visit]` (0 = no visit), `years[site]` run length, the expected
tally buffer `lam` and `hl_sim`. `soa` = plugin SoA with `extra` free cohort slots per site; `N0 .. zq0` hold the live initial cohorts in CSR order (`off0`).
"""
struct SimBData{S<:PanCore.AnySoA}
  soa::S
  nsp::Int
  off0::Vector{Int}
  live0::Vector{Int}
  sp0::Vector{UIntType}; N0::Vector{Float64}; D0::Vector{Float64}; H0::Vector{Float64}; A0::Vector{Float64}; zq0::Vector{Float64}
  years::Vector{Int}
  vyear::Matrix{Int}
  obs::Array{Float64,4}
  lam::Array{Float64,4}
  hl_obs::Matrix{Float64}; hl_sim::Matrix{Float64}
  sigma::Float64
  plot_of::Vector{Int}
  dropped::Vector{Int}
  nplots::Int
end
Base.show(io::IO, B::SimBData) = print(io, "SimBData(", length(B.live0), " sites, ", B.nplots, " plots, ", sum(B.live0), " initial cohorts)")

"""
    build_simB(x, set, keys, sw, pc; sigma=0.16, cap_extra=(vigour ? 330 : 120), natural_only=true, rng) -> SimBData

Clean natural plots of `set` with >= 2 intervals (no treatment / disturbance / harvest flag in any interval; `natural_only` also excludes planted stands). Per plot, the first visit's live
trees give the initial cohorts of each of the 4 subplot sites (empty ones included; one site for the whole plot with `site: plot`); every later visit contributes the observed tally per
(key, size class) and the Lorey height; the last visit is the tracked survivors (v2 weights) plus the first-tallied ingrowth.
"""
function build_simB(x::Data.HBAD2Intervals, set::AbstractString, keys_::Vector{String}, sw::CalibSwitches, pc; sigma::Real=0.16,
                    cap_extra::Integer=(sw.vigour ? 330 : 120), natural_only::Bool=true, rng::Random.AbstractRNG=Random.Xoshiro(1))
  iv = copy(x.intervals); raw = copy(x.trees); ig = copy(x.ingrowth)
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
    for sbsel in (sw.site_sub ? (1:4) : (0:0))
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
      push!(sites, (init=f(DataFrame(rbyi[(ivs.iid[1],)])), vis=vis, plot=rownumber(r)))
    end
  end
  ns = length(sites); nv = maximum(length(s.vis) for s in sites); nsp = length(keys_)
  off = [1]; spv = Int[]; N0 = Float64[]; D0 = Float64[]; H0 = Float64[]; A0 = Float64[]; live = Int[]; zq = Float64[]
  QZ = sw.vigour ? collect(HBAD2Plugin.VIG_Z) : [0.0]; QW = sw.vigour ? collect(HBAD2Plugin.VIG_W) : [1.0]
  obs = zeros(ns, nv, nsp, SB_NC); hlo = fill(NaN, ns, nv); vy = zeros(Int, ns, nv); years = zeros(Int, ns)
  for (s, st) in enumerate(sites)
    t = st.init; t.bc = floor.(Int, -t.age ./ sw.birth_w) .* 100 .+ dclass.(Ref(sw), t.dia_cm)
    g = nrow(t) == 0 ? DataFrame(key=String[], N=Float64[], D=Float64[], H=Float64[], A=Float64[]) :
      combine(groupby(t, [:key, :bc]), :w_ha => (w -> sum(w) * sw.site_mult) => :N, [:w_ha, :dia_cm] => ((w, x) -> sqrt(sum(w .* x .^ 2) / sum(w))) => :D,
                [:w_ha, :ht_m] => ((w, x) -> sum(w .* x) / sum(w)) => :H, [:w_ha, :age] => ((w, x) -> sum(w .* x) / sum(w)) => :A)
    for r in eachrow(g), (z, w) in zip(QZ, QW); push!(spv, kix[r.key]); push!(N0, r.N * w); push!(D0, r.D); push!(H0, r.H); push!(A0, max(r.A, 1.0)); push!(zq, z); end
    push!(live, nrow(g) * length(QZ))
    push!(off, length(spv) + 1)
    for (k, v) in enumerate(st.vis); vy[s, k] = v[1]; end
    years[s] = maximum(v[1] for v in st.vis)
    for (k, (y, key, D, H, w)) in enumerate(st.vis)
      for i in eachindex(D); c = cls(D[i]); c >= 1 && (obs[s, k, kix[key[i]], c] += 1); end
      ba = (pi / 4) .* (D ./ 100) .^ 2 .* w; sum(ba) > 0 && (hlo[s, k] = sum(ba .* H) / sum(ba))
    end
  end
  zqs = zq
  soa = Pan.make_sites_hbad2_arrays(pc, off, spv, A0, N0, D0, H0, zqs; rng=rng, no_establishment=false, extra_capacity=cap_extra)
  pl_of = [st.plot for st in sites]; upl = unique(pl_of); pl_of = [findfirst(==(x), upl) for x in pl_of]
  return SimBData(soa, nsp, off, live, UIntType.(spv), N0, D0, H0, A0, zqs, years, vy, obs, similar(obs), hlo, fill(NaN, ns, nv), Float64(sigma), pl_of, zeros(Int, ns), length(upl))
end
