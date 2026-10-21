# HBAD3P pipeline library (docs/hbad3p_pipeline.md), shared by the stage scripts fit3p / recruit3p / freerun3p / crossing3p and verify3p (include after common3.jl and HBAD3P.jl):
#   * forested area of subplot sites (scripts/hbad3/subplot_forest.py): FIA tallies trees only on forest conditions, so a subplot's tree list covers the
#     forested fraction f of its area. Density convention of a fit (meta `farea`): 1 = per ha of the FORESTED subplot (N / a, a = max(f, FMIN)), 0 = per ha
#     of the whole subplot (the older fits). Plot values convert back with × a (farea = 1); recruitment (per forested ha) is scaled × a when farea = 0.
#   * subplot cohorts of a tree list (the cohort keys of build_cohorts: key × birth class (y1 − age) / BIRTH_W × diameter class) × vigour nodes;
#   * recruitment fitted CONSISTENTLY with the free-run mechanism. Observed Y = saplings (D < 12.7) first tallied at v2 on the forested microplot (area
#     f_micr(v2)). In a free run a recruit enters at the end of each year (step, then recruit; weight Δ of a final partial year) and then lives under the
#     kernel's growth / mortality. Hence
#       E[Y] = e^{lF_k} f_micr2 Σ_y Δ_y e^{−λ0 e^{λ1 (ST_k − 3)} RD_y} (seed_ky ? 1 : ε0) s_ky
#     with RD_y / seed_ky the stand at entry and s_ky the fraction of a recruit of key k entering in year y that is alive with D < 12.7 at v2, from PROBE
#     recruits (density PN, negligible) simulated with the fitted kernel inside the observed v1 stand. (Before: E[Y] = e^{lF} dt e^{−λ RD1} …, i.e. the
#     survivors at v2 were taken as the entry rate and the free run killed them a second time; and empty subplots entered only when they had ingrowth.)
const FMIN = 0.25           # smallest effective forested fraction (a forest sliver is not a closed stand at 1 / f its tree density)
const PN = 1e-6             # probe recruit density [trees/ha]
"(st, u, c, p, date, sb) => (f_subp, f_micr); NaN = status unknown."
function forest_table(path=get(ENV, "HBAD3_FOREST_CSV", joinpath(R3DIR, "subplot_forest_r4.csv")))
  f = CSV.read(path, DataFrame; types=Dict(:md => Date))
  Dict((Int(r.st), Int(r.u), Int(r.c), Int(r.p), r.md, Int(r.sb)) => (Float64(coalesce(r.f_subp, NaN)), Float64(coalesce(r.f_micr, NaN))) for r in eachrow(f))
end
"Per (iid, sb): forested fraction of the subplot at v1 (v2 when v1 is unknown) and of the microplot at v2 (NaN: unknown / no record)."
function interval_forest(iv::DataFrame, FT)
  out = Dict{Tuple{Int,Int},NTuple{2,Float64}}()
  for r in eachrow(iv), sb in 1:4
    a = get(FT, (r.st, r.u, r.c, r.p, r.d1, sb), (NaN, NaN)); b = get(FT, (r.st, r.u, r.c, r.p, r.d2, sb), (NaN, NaN))
    out[(r.iid, sb)] = (isnan(a[1]) ? b[1] : a[1], b[2])
  end
  out
end
"Effective forested fraction of a STOCKED site (status unknown, or 0 although trees were tallied: 1)."
area_of(f) = (isnan(f) || f <= 0) ? 1.0 : max(f, FMIN)
"Density convention of a fit directory (meta farea; absent = 0, the older fits)."
function meta_farea(dir)
  m = Dict(split(l, "=")[1] => (length(split(l, "=")) > 1 ? split(l, "=")[2] : "") for l in readlines(joinpath(dir, "meta.txt")))
  parse(Int, get(m, "farea", "0")) == 1
end
"Scale a Sim A cohort set to per ha of the forested subplot: N0 of every (sub-)cohort / a(site). Survival ratios are unchanged; competition sees the true density."
function scale_area!(C, IF)
  a = Dict(s => area_of(get(IF, (s ÷ 10, s % 10), (NaN, NaN))[1]) for s in unique(C.coh.site))
  @inbounds for i in eachindex(C.S.N0); C.S.N0[i] /= a[C.coh.site[C.parent[i]]]; end
  C.agg === nothing || (@inbounds for p in eachindex(C.agg.N0); C.agg.N0[p] /= a[C.coh.site[p]]; end)
  C
end

"Cohorts × vigour nodes of one subplot's live tree list t (v1, year y1): the build_cohorts keys, densities ×4 / a [trees/ha]. Returns columns."
function subplot_cohorts(t::AbstractDataFrame, kix, y1::Int, a::Float64)
  sp = Int32[]; N = Float64[]; D = Float64[]; H = Float64[]; A = Float64[]; zq = Float64[]; CR = Float64[]; big = Int8[]
  nrow(t) == 0 && return (sp=sp, N=N, D=D, H=H, A=A, zq=zq, CR=CR, big=big)
  bc = floor.(Int, (y1 .- t.age) ./ BIRTH_W) .* 100 .+ dclass.(t.dia_cm)
  g = combine(groupby(DataFrame(key=t.key, bc=bc, w=t.w_ha, d=t.dia_cm, h=t.ht_m, age=t.age, cr=t.cr), [:key, :bc]),
              :w => (x -> 4sum(x)) => :N, [:w, :d] => ((w, d) -> sqrt(sum(w .* d .^ 2) / sum(w))) => :D, [:w, :h] => ((w, h) -> sum(w .* h) / sum(w)) => :H,
              [:w, :age] => ((w, x) -> sum(w .* x) / sum(w)) => :A,
              [:w, :cr] => ((w, c) -> (m = .!ismissing.(c) .& (coalesce.(c, 0) .> 0); any(m) ? sum(w[m] .* c[m]) / sum(w[m]) / 100 : 0.4)) => :CR)
  for r in eachrow(g), q in eachindex(VQ_Z)
    push!(sp, kix[r.key]); push!(N, r.N * VQ_W[q] / a); push!(D, r.D); push!(H, max(r.H, 1.37)); push!(A, r.A); push!(zq, VQ_Z[q]); push!(CR, r.CR); push!(big, r.D >= 12.7)
  end
  (sp=sp, N=N, D=D, H=H, A=A, zq=zq, CR=CR, big=big)
end

"Probe exposure of one site (concrete type: the recruitment objective loops over these)."
struct Exposure
  RD::Vector{Float64}; wy::Vector{Float64}; seed::BitMatrix; surv::Matrix{Float64}
end
"""Probe recruitment exposure of subplot sites. `sites[s]` = (coh = subplot_cohorts(...), nyr, frac, W, Wn, seedplot::BitVector nsp, planted). For every site:
RD[y] (stand after the step of year y), wy[y] (1 or frac), seed[k, y], surv[k, y] (fraction of a key-k recruit entering in year y alive with D < 12.7 at the end)."""
function probe_exposure(sites, P, G)
  nsp = length(P); nq = length(VQ_Z); ns = length(sites)
  nys = [st.nyr + (st.frac > 0) for st in sites]
  off = [1]; for s in 1:ns; push!(off, off[end] + length(sites[s].coh.N) + nys[s] * nsp * nq); end
  n = off[end] - 1; sp = zeros(Int32, n); N = zeros(n); D = fill(D_REC_P, n); H = fill(1.37, n); A = zeros(n); zq = zeros(n); CR = fill(0.4, n)
  for s in 1:ns
    c = sites[s].coh; r = off[s]:off[s]+length(c.N)-1
    sp[r] .= c.sp; N[r] .= c.N; D[r] .= c.D; H[r] .= c.H; A[r] .= c.A; zq[r] .= c.zq; CR[r] .= c.CR
  end
  S = HBAD3P.SitesP(off, sp, copy(N), D, H, A, zeros(Int, ns), zeros(ns), [st.planted for st in sites], zq, [st.W for st in sites], [st.Wn for st in sites], CR)
  copyto!(S.N, N)
  out = Vector{Exposure}(undef, ns)
  Threads.@threads :static for s in 1:ns
    st = sites[s]; lo = off[s]; nst = length(st.coh.N); live = nst; ny = nys[s]
    RD = zeros(ny); wy = zeros(ny); seed = falses(nsp, ny); pidx = zeros(Int, nsp, ny)
    for y in 1:ny
      Δ = y <= st.nyr ? 1.0 : st.frac
      live > 0 && HBAD3P.step_site!(S, P, G, lo, lo + live - 1, Δ, st.planted, st.W, st.Wn)
      tot = 0.0; for i in lo:lo+live-1; tot += HBAD3P.sdi(S.N[i], S.D[i]); end       # as recruit_siteP! (probes contribute ~PN)
      RD[y] = tot / G.sdimax; wy[y] = Δ
      for k in 1:nsp
        sd = st.seedplot[k]
        for i in lo:lo+nst-1; (S.sp[i] == k && S.D[i] >= HBAD3P.DSEED && S.N[i] > 0) && (sd = true); end   # stand cohorts only: probes are not seed trees
        seed[k, y] = sd
        j = lo + live; pidx[k, y] = j; h0 = HBAD3P.hstar(P[k], G, D_REC_P)
        for q in 1:nq
          jj = j + q - 1; S.sp[jj] = k; S.N[jj] = PN * VQ_W[q]; S.N0[jj] = S.N[jj]; S.D[jj] = D_REC_P; S.H[jj] = h0; S.A[jj] = HBAD3P.A_REC; S.zq[jj] = VQ_Z[q]; S.CR[jj] = G.crhi
        end
        live += nq
      end
    end
    surv = zeros(nsp, ny)
    for y in 1:ny, k in 1:nsp, q in 1:nq
      jj = pidx[k, y] + q - 1; S.D[jj] < 12.7 && (surv[k, y] += S.N[jj] / PN)
    end
    out[s] = Exposure(RD, wy, seed, surv)
  end
  out
end
const D_REC_P = HBAD3P.D_REC

"Expected sapling count of every (site, key) row: e^{lF_k} fm Σ_y wy e^{−λ0 e^{λ1 (ST−3)} RD_y} (seed ? 1 : ε0) surv (see the header)."
function recr_mu!(μ, z, X, ST)
  nsp = length(ST); λ0 = exp(z[nsp+1]); λ1 = z[nsp+2]; e0 = exp(z[nsp+3])
  Threads.@threads :static for j in eachindex(X.Y)
    e = X.ex[X.site[j]]; k = X.K[j]; lk = λ0 * exp(λ1 * (ST[k] - 3)); acc = 0.0
    for y in eachindex(e.RD); acc += e.wy[y] * exp(-lk * e.RD[y]) * (e.seed[k, y] ? 1.0 : e0) * e.surv[k, y]; end
    μ[j] = exp(z[k]) * X.fm[j] * acc
  end
  μ
end
function recr_nll(z, X, ST)
  μ = recr_mu!(X.μ, z, X, ST); ll = 0.0
  @inbounds for j in eachindex(X.Y); ll += X.Y[j] * log(max(μ[j], 1e-300)) - μ[j]; end
  -ll
end

# ---------------------------------------------------------------- shared loaders of the stage scripts
"Period / normal water index per interval (scripts/hbad3/water_balance.py): iid => (aetpet_iv, aetpet_n)."
function water_pairs(path=get(ENV, "HBAD3_WATER_CSV", joinpath(R3DIR, "interval_water.csv")))
  w = CSV.read(path, DataFrame); Dict(Int(r.iid) => (Float64(r.aetpet_iv), Float64(r.aetpet_n)) for r in eachrow(w))
end
"The interval product's three tables with the tracked-tree row filter of load_trees (age, height, diameter > 0) and the species key of the region's keys
(own key if in `keys_`, else _S / _H); weights as stored (per ha of the PLOT, not × SITE_MULT)."
function interval_tables(keys_::Vector{String}; dir=DATA_DIR)
  iv = DataFrame(Arrow.Table(joinpath(dir, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(dir, "interval_trees.arrow")))
  ig = DataFrame(Arrow.Table(joinpath(dir, "interval_ingrowth.arrow")))
  big = Set(k for k in keys_ if k[1] != '_'); keyof(s, g) = s in big ? s : "_" * g
  raw = filter(r -> r.age > 0 && r.ht_m > 0 && r.dia_cm > 0, raw)
  raw.key = [keyof(s, g) for (s, g) in zip(raw.sym, raw.grp)]; ig.key = [keyof(s, g) for (s, g) in zip(ig.sym, ig.grp)]
  setdiff(Set(raw.key), Set(keys_)) == Set() || error("interval_tables: keys outside the fit's keys")
  (iv=iv, raw=raw, ig=ig)
end
"Growth / mortality of a fit3p directory (species.csv, globals.csv; globals absent from older fits take their neutral values, am = a)."
function load_fit3p(FD, keys_)
  spc = CSV.read(joinpath(FD, "species.csv"), DataFrame); String.(spc.key) == keys_ || error("keys of $(FD) differ from this region's keys")
  gd = Dict(Symbol(r.name) => Float64(r.value) for r in eachrow(CSV.read(joinpath(FD, "globals.csv"), DataFrame)))
  dflt(f) = f == :am ? gd[:a] : f in (:ρ, :f0, :φ, :τr, :τe) ? 0.0 : f in (:crhi, :crlo) ? 0.4 : f == :kc ? 1.0 : error("$(FD): global $(f) missing")
  G = HBAD3P.GlobPP((haskey(gd, f) ? gd[f] : dflt(f) for f in fieldnames(HBAD3P.GlobPP))...)
  P = [HBAD3P.SpPP(log(spc.eps[k]), spc.Hmax[k], spc.ST[k], spc.longevity[k]) for k in 1:nrow(spc)]
  (P=P, G=G, ST=Float64.(spc.ST))
end
"Recruitment of a fit directory (recruit3p.jl -> recruitment.csv): species with ln F, and the recruitment globals."
function load_recruit3p(FD, P)
  f = joinpath(FD, "recruitment.csv"); isfile(f) || error("$(f) missing: run the recruitment stage (tools/hbad3/recruit3p.jl) first")
  r = CSV.read(f, DataFrame); lF = r.value[startswith.(r.name, "lnF:")]; length(lF) == length(P) || error("$(f): $(length(lF)) ln F for $(length(P)) keys")
  g(n) = r.value[findfirst(==(n), r.name)]
  ([HBAD3P.SpPP(p.lnε, p.Hmax, p.ST, p.longevity, lF[k]) for (k, p) in enumerate(P)], HBAD3P.RecGP(g("λ0"), g("λ1"), g("ε0")))
end
cut_rd(x) = x < 0.1 ? "a RD<.1" : x < 0.3 ? "b RD.1-.3" : x < 0.6 ? "c RD.3-.6" : "d RD>=.6"

# ---------------------------------------------------------------- free-run stores (freerun3p.jl; verify3p.jl)
const σW = 0.16                                                               # within-cohort lognormal spread at the 12.7 cm threshold
above(D) = 1 - cdf(Normal(), (log(12.7) - (log(D) - σW^2)) / σW)              # D = cohort QMD = lognormal median × e^{σ²}
"Plot values per ha from cohorts (D QMD, H, N per ha of the plot): BA ≥ 2.54 cm, stems ≥ 12.7 (lognormal operator), their QMD and Lorey height."
function standvals(D, H, N)
  ba = 0.0; ba12 = 0.0; n12 = 0.0; s2 = 0.0; lh = 0.0
  for i in eachindex(D)
    D[i] < 2.54 && continue
    b = N[i] * pi / 4 * (D[i] / 100)^2; ba += b
    f = above(D[i]); n12 += f * N[i]; s2 += f * N[i] * D[i]^2; ba12 += f * b; lh += f * b * H[i]
  end
  (BA=ba, N12=n12, QMD=n12 > 0 ? sqrt(s2 / n12) : NaN, LH=ba12 > 0 ? lh / ba12 : NaN)
end
"The same values from tree records (Wt = trees/ha of the plot)."
function obsvals(D, H, Wt)
  ba = 0.0; ba12 = 0.0; n12 = 0.0; s2 = 0.0; lh = 0.0
  for i in eachindex(D)
    (D[i] >= 2.54 && Wt[i] > 0) || continue
    b = Wt[i] * pi / 4 * (D[i] / 100)^2; ba += b
    if D[i] >= 12.7; n12 += Wt[i]; s2 += Wt[i] * D[i]^2; ba12 += b; lh += b * H[i]; end
  end
  (BA=ba, N12=n12, QMD=n12 > 0 ? sqrt(s2 / n12) : NaN, LH=ba12 > 0 ? lh / ba12 : NaN)
end
"""Free-run sites of `plots` (rows st u c p; ivbp: plot => its intervals sorted): 4 subplot sites per plot from the FIRST visit's trees, run length = first to
last visit (whole years + fraction), water = mean period index / plot normal of the first interval, the fit's density convention; observed first / last
visit values and the origin decomposition of the last visit's stems ≥ 12.7 cm."""
function build_free(plots, ivbp, T, IF, wd, keys_, kix, FAREA)
  nsp = length(keys_)
  rawby = Dict(k.iid => DataFrame(g) for (k, g) in pairs(groupby(T.raw, :iid))); igby = Dict(k.iid => DataFrame(g) for (k, g) in pairs(groupby(T.ig, :iid)))
  grp0 = Int8[]; CRv = Float64[]                       # initial group of each slot: 1 cohort QMD >= 12.7 at the first visit, 2 < 12.7, 3 recruit slot
  off = [1]; spv = Int32[]; N0 = Float64[]; D0 = Float64[]; H0 = Float64[]; A0 = Float64[]; zq = Float64[]; live = Int[]; Ws = Float64[]; Wns = Float64[]
  yrs = Int[]; fracs = Float64[]; seedm = falses(nsp, 4nrow(plots)); res = NamedTuple[]; pl_site = Int[]; rmult = Float64[]; outm = Float64[]; nnf = 0; f1v = Float64[]
  for (pi_, r) in enumerate(eachrow(plots))
    ivs = ivbp[(r.st, r.u, r.c, r.p)]; i1 = ivs.iid[1]; iL = ivs.iid[end]
    Ty = Dates.value(ivs.d2[end] - ivs.d1[1]) / 365.25
    t1 = get(rawby, i1, T.raw[1:0, :])
    ws = mean(first.(wd[i] for i in ivs.iid)); wn = wd[i1][2]
    tL = get(rawby, iL, T.raw[1:0, :]); tL = tL[(tL.out .== "alive") .& (tL.dia2_cm .> 0), :]; gL = get(igby, iL, T.ig[1:0, :])
    o1 = obsvals(t1.dia_cm, t1.ht_m, t1.w_ha); oL = obsvals(vcat(tL.dia2_cm, gL.dia_cm), vcat(tL.ht2_m, gL.ht_m), vcat(tL.w2_ha, gL.w_ha))
    # stems >= 12.7 at the last visit by origin: big at the first visit / crossed from < 12.7 / new since the first visit (FIA tree ids)
    d1 = Dict((x.sb, x.tr) => x.dia_cm for x in eachrow(t1)); ob = [0.0, 0.0, 0.0]
    for x in eachrow(tL)
      x.dia2_cm >= 12.7 || continue; d0 = get(d1, (x.sb, x.tr), nothing)
      ob[d0 === nothing ? 3 : d0 >= 12.7 ? 1 : 2] += x.w2_ha
    end
    for x in eachrow(gL); x.dia_cm >= 12.7 && (ob[3] += x.w_ha); end
    push!(res, (st=r.st, u=r.u, c=r.c, p=r.p, T=Ty, obs1=o1, obsL=oL, ob=ob))
    for sb in 1:4
      s = length(off); t = t1[t1.sb .== sb, :]; f1 = IF[(i1, sb)][1]
      a = area_of(f1); nonforest = f1 == 0 && nrow(t) == 0; nnf += nonforest
      for k in 1:nsp; seedm[k, s] = any((t1.key .== keys_[k]) .& (t1.dia_cm .>= 12.7)); end
      c = subplot_cohorts(t, kix, year(ivs.d1[1]), FAREA ? a : 1.0); n = length(c.N)
      append!(grp0, ifelse.(c.big .== 1, Int8(1), Int8(2))); append!(CRv, c.CR); append!(spv, c.sp); append!(N0, c.N); append!(D0, c.D); append!(H0, c.H); append!(A0, c.A); append!(zq, c.zq)
      cap = n + length(VQ_Z) * nsp * (ceil(Int, Ty / HBAD3P.RCLASS) + 1)
      for _ in 1:(cap-n); push!(grp0, 3); push!(CRv, 0.4); push!(spv, 1); push!(N0, 0.0); push!(D0, 2.54); push!(H0, 1.37); push!(A0, 0.0); push!(zq, 0.0); end
      push!(off, off[end] + cap); push!(live, n); push!(Ws, ws); push!(Wns, wn); push!(yrs, floor(Int, Ty)); push!(fracs, Ty - floor(Ty)); push!(pl_site, pi_)
      fr = (isnan(f1) || (f1 == 0 && nrow(t) > 0)) ? 1.0 : f1                    # true forested fraction (recruitment area); a = max(fr, FMIN) for the stand
      push!(rmult, FAREA ? fr / a : fr); push!(outm, FAREA ? a : 1.0); push!(f1v, f1)   # nonforest empty: fr = 0 -> no recruitment
    end
  end
  (off=off, spv=spv, N0=N0, D0=D0, H0=H0, A0=A0, zq=zq, CRv=CRv, grp0=grp0, live=live, Ws=Ws, Wns=Wns, yrs=yrs, fracs=fracs, seedm=seedm[:, 1:length(live)],
   res=res, pl_site=pl_site, rmult=rmult, outm=outm, nnf=nnf, f1=f1v)
end
"A fresh FreeSitesP of a build_free result (the build is not modified: runs can be repeated)."
function free_store(B, nsp)
  ns = length(B.live)
  S = HBAD3P.SitesP(B.off, copy(B.spv), copy(B.N0), copy(B.D0), copy(B.H0), copy(B.A0), zeros(Int, ns), zeros(ns), fill(false, ns), copy(B.zq), B.Ws, B.Wns, copy(B.CRv))
  HBAD3P.FreeSitesP(S, copy(B.live), zeros(Int, nsp, ns), zeros(Int, nsp, ns), zeros(Int, ns), B.seedm, B.rmult)
end
"Simulated vs observed plot values (rows T var obs1 obsL sim): subplot sites per ha (× outm: of the plot when farea) → plot = Σ over the 4 sites / 4."
function plot_values(F, B)
  S = F.S; rows = DataFrame(T=Float64[], var=String[], obs1=Float64[], obsL=Float64[], sim=Float64[])
  for (pi_, r) in enumerate(B.res)
    Dv = Float64[]; Hv = Float64[]; Nv = Float64[]
    for s in findall(==(pi_), B.pl_site), i in B.off[s]:B.off[s]+F.live[s]-1; push!(Dv, S.D[i]); push!(Hv, S.H[i]); push!(Nv, S.N[i] * B.outm[s] / 4); end
    sv = standvals(Dv, Hv, Nv)
    for v in (:BA, :N12, :QMD, :LH); push!(rows, (r.T, String(v), getfield(r.obs1, v), getfield(r.obsL, v), getfield(sv, v))); end
  end
  rows
end

# ---------------------------------------------------------------- Sim A sites (fit3p.jl; verify3p.jl)
"Sim A cohort set of one split set as HBAD3P sites: build_cohorts (tracked trees of clean intervals, vigour sub-cohorts), the density convention (IFOR =
interval_forest(...) for farea = 1, nothing for 0), period / normal water index per site, FIA crown ratio at v1 as the initial CR."
function sima_sites(tr, set, keys_, wp, IFOR)
  C = build_cohorts(tr, set, keys_); IFOR === nothing || scale_area!(C, IFOR); S = C.S; ns = length(S.off) - 1
  iids = [C.coh.site[C.parent[S.off[s]]] ÷ 10 for s in 1:ns]
  SP = HBAD3P.SitesP(S.off, S.sp, S.N0, S.D0, S.H0, S.A0, S.nyr, S.frac, S.planted, S.zq, [wp[i][1] for i in iids], [wp[i][2] for i in iids],
                     [Float64(C.coh.CR1[C.parent[i]]) for i in eachindex(S.sp)])
  (S=SP, coh=C.coh, obs=C.obs, parent=C.parent, agg=C.agg)
end
