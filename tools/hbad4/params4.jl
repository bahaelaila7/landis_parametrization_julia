# HBAD4P parameter space (fit4p / eval4p / verify4p): per key ln ε (growth efficiency) and ln F (fecundity); globals (process + observation model).
# Encoding: log-encoded where marked, clamped to the bounds. Fixed by default (HBAD4P_FIX overrides / adds, e.g. "gR:0.3,wT:Inf"):
#   gR = 0 (HBAD3P's crowding gain e^{gR RD} grew with density, a site-quality proxy, docs/hbad3_design.md §8.3), mA = 10, sdimax = 1500 (as HBAD3P),
#   βC = HBAD4P_BETAC (0.58: FACE, +23 % NPP at 550 vs 370 ppm; not identifiable from 2000–2024 FIA),
#   sdM = SD_DBH = 0.2 cm: the FIA DBH measurement error of the LIKELIHOOD's observation blur (HBAD4P.finish_site!; audit r3 I1b) — the value set for the
#   audit fix, of the order of FIA's measurement-quality tolerance for DBH (±0.1 in per 20 in of diameter, i.e. ±0.1–0.25 cm for 20–50 cm trees). Fixed, not
#   fitted (the dynamics do not see it; it smooths the class-boundary jumps of the predicted table).
# κ (lane mixing) is LINEAR on [0, 1] (clamped), so κ = 0 — fully persistent lanes, the FVS triple and HBAD3P's persistent vigour nodes — is reachable
# (audit r3 I2d; was log-encoded with a 1e-3 floor).
using CSV, DataFrames
const GSPEC4 = [(:b, 1.75, 1.0, 3.0, false), (:a, 2.35, 0.1, 10.0, true), (:gR, 0.0, -2.0, 1.0, false), (:θ, 8.7, 1.0, 40.0, true), (:ω, 0.96, 0.0, 5.0, false),
                (:s, 1.16, 0.2, 3.0, true), (:c, 1.25, 0.3, 2.0, true), (:η, 0.69, -1.0, 2.0, false), (:λ, 0.0023, 1e-4, 0.3, true), (:μ, 1.22, -0.5, 2.0, false),
                (:kh, 5.1, 0.2, 30.0, true), (:m0, 1.11, 0.01, 10.0, true), (:ms, 0.082, 1e-4, 1.0, true), (:σST, 0.28, -2.0, 2.0, false), (:β, 2.95, 0.1, 20.0, true),
                (:ζ, -0.62, -3.0, 2.0, false), (:mW, 0.12, -2.0, 3.0, false), (:mA, 10.0, 1.0, 50.0, true), (:pD, 0.11, -1.0, 1.0, false), (:pH, 0.26, -1.0, 1.0, false),
                (:pM, -0.06, -2.0, 2.0, false), (:σv, 0.36, 0.0, 1.2, false), (:sdimax, 1500.0, 500.0, 3000.0, true), (:am, 0.83, 0.1, 20.0, true),
                (:ρ, 2.6e-4, 0.0, 0.01, false), (:f0, 0.0, 0.0, 0.5, false), (:βT, 0.0, -0.5, 0.5, false), (:βC, 0.58, 0.0, 2.0, false), (:κ, 0.0, 0.0, 1.0, false),
                (:λ0, 3.1, 0.01, 20.0, true), (:λ1, -0.07, -2.0, 2.0, false), (:ε0, 0.3, 1e-3, 1.0, true), (:ωR, 0.0, -3.0, 5.0, false),
                (:rM, 1.0, 0.05, 100.0, true), (:rS, 2.0, 0.05, 100.0, true), (:bgM, 5.0, 1e-3, 500.0, true), (:bgS, 0.3, 1e-4, 50.0, true),
                (:σD, 0.02, 1e-5, 0.5, true), (:σH, 0.05, 1e-5, 1.0, true), (:sdM, 0.2, 0.0, 1.0, false)]
const GN4 = [g[1] for g in GSPEC4]
const EPS_MAX = 4.0                # ln ε bound (e^4 = 55; HBAD3P's fitted ε are 0.8–3.5): beyond it dD would reach tens of cm / yr
const OBS4 = (:rM, :rS, :bgM, :bgS, :σD, :σH)
const SD_DBH = 0.2                 # FIA DBH measurement error sd [cm] of the likelihood's observation blur (fixed; see the header)
const FIX_DEFAULT = Dict(:gR => 0.0, :mA => 10.0, :sdimax => 1500.0, :βC => parse(Float64, get(ENV, "HBAD4P_BETAC", "0.58")), :sdM => SD_DBH)
function gfix4()
  f = copy(FIX_DEFAULT)
  for x in filter(!isempty, strip.(split(get(ENV, "HBAD4P_FIX", ""), ",")))
    a_, b_ = split(x, ":"); f[Symbol(a_)] = parse(Float64, b_)
  end
  for x in filter(!isempty, strip.(split(get(ENV, "HBAD4P_FREE", ""), ",")))   # release a default-fixed global, e.g. HBAD4P_FREE=gR
    delete!(f, Symbol(x))
  end
  f
end
genc4(j, v) = GSPEC4[j][5] ? log(v) : v
function gdec4(j, z); lo, hi = GSPEC4[j][3], GSPEC4[j][4]; clamp(GSPEC4[j][5] ? exp(z) : (GSPEC4[j][1] == :σv ? abs(z) : z), lo, hi); end

"Coordinate layout for keys_ and the fixed globals: names, start vector, decoder z → (P, G, Ob). sD / sH: the TRAIN per-class within-cell spreads (data constants)."
function space4(keys_, TR, sD, sH; start_eps=nothing, start_lnF=nothing, start_glob=Dict{Symbol,Float64}(), fix_extra=Dict{Symbol,Float64}())
  nk = length(keys_); fix = merge(gfix4(), fix_extra)        # fix_extra: globals fixed at a value by the caller (e.g. the κ profile, kappa_profile4.jl)
  gfree = [j for j in eachindex(GSPEC4) if !haskey(fix, GSPEC4[j][1])]
  gval(j) = (nm = GSPEC4[j][1]; haskey(fix, nm) ? fix[nm] : get(start_glob, nm, GSPEC4[j][2]))
  gv0 = [genc4(j, clamp(gval(j), GSPEC4[j][3], GSPEC4[j][4])) for j in eachindex(GSPEC4)]
  le = start_eps === nothing ? fill(log(2.0), nk) : log.(start_eps); lf = start_lnF === nothing ? fill(-3.0, nk) : start_lnF
  z0 = vcat(le, lf, gv0[gfree])
  names = vcat(["ε:" * k for k in keys_], ["lnF:" * k for k in keys_], [String(GSPEC4[j][1]) for j in gfree])
  function decode(z)
    gv = copy(gv0); gv[gfree] = z[2nk+1:end]
    g = Dict(GSPEC4[j][1] => gdec4(j, gv[j]) for j in eachindex(GSPEC4))
    P = [HBAD4P.SpP4(clamp(z[k], -8.0, EPS_MAX), clamp(z[nk+k], -15.0, 5.0), TR.Hmax[k], TR.ST[k], TR.longevity[k], TR.Tref[k]) for k in 1:nk]
    G = HBAD4P.GlobP4(g[:b], g[:a], g[:gR], g[:θ], g[:ω], g[:s], g[:c], g[:η], g[:λ], g[:μ], g[:kh], g[:m0], g[:ms], g[:σST], g[:β], g[:ζ], g[:mW], g[:mA],
                      g[:pD], g[:pH], g[:pM], g[:σv], g[:sdimax], g[:am], g[:ρ], g[:f0], g[:βT], g[:βC], g[:κ], g[:λ0], g[:λ1], g[:ε0], g[:ωR], g[:sdM])
    Ob = ObsP(g[:rM], g[:rS], g[:bgM], g[:bgS], g[:σD], g[:σH], sD, sH)
    P, G, Ob
  end
  (z0=z0, names=names, decode=decode, gfree=gfree, fix=fix, nk=nk)
end
"""Warm start from an HBAD3P fit directory (fit3p_<role>): ε per key, lnF per key (recruitment.csv), the shared globals by name. A key HBAD3P does not have
(a species that became its own key since: HBAD4P's keys come from the role's full train set) inherits the values of its HBAD3P GROUP key _S / _H (`grp`, the
traits' S / H; audit r3 I2a), not the generic defaults. `eps_factor` rescales ε (warm_start4)."""
function hbad3p_start(dir, keys_; eps_factor=1.0, grp=nothing)
  sp = CSV.read(joinpath(dir, "species.csv"), DataFrame); gl = CSV.read(joinpath(dir, "globals.csv"), DataFrame)
  ed = Dict(String(r.key) => Float64(r.eps) for r in eachrow(sp))
  rc = isfile(joinpath(dir, "recruitment.csv")) ? CSV.read(joinpath(dir, "recruitment.csv"), DataFrame) : DataFrame(name=String[], value=Float64[])
  rd = Dict(String(r.name) => Float64(r.value) for r in eachrow(rc))
  g = Dict(Symbol(r.name) => Float64(r.value) for r in eachrow(gl))
  for nm in (:λ0, :λ1, :ε0); haskey(rd, String(nm)) && (g[nm] = rd[String(nm)]); end
  src(i, k) = haskey(ed, k) || grp === nothing ? k : "_" * grp[i]                 # the HBAD3P key whose values a key takes
  ([get(ed, src(i, k), 2.0) * eps_factor for (i, k) in enumerate(keys_)], [get(rd, "lnF:" * src(i, k), -3.0) for (i, k) in enumerate(keys_)], g,
   [src(i, k) for (i, k) in enumerate(keys_)])
end

"""The HBAD3P → HBAD4P warm start (smoke4 / verify4p / the fit tools use this one function). HBAD3P competed per SUBPLOT (cohort densities × 4 per forested
subplot ha) and multiplied growth AND the mortality stress gm by its crowding / crown factors e^{gR RD} (CR/0.4)^φ, which HBAD4P drops; HBAD4P competes per
PLOT. At the TRAIN v1 state (CS.warm: the v1 trees with their subplot, the forested-area division a of HBAD3P, crown ratio; the train store for HBAD4P's cells):
  F    = tree-weighted mean of e^{gR RD_subplot} (CR/0.4)^φ (RD per forested subplot ha, ÷ a: audit r3 I2e)
  Lr   = tree-weighted e^{−a O3} / e^{−a O4}: the light HBAD3P computed per subplot (O3: trees above, ½ equal heights, ½ own, densities 4 w / a) vs HBAD4P's
         plot light of the tree's cell at v1 (audit r3 I2c: 0.583 vs 0.630)
  ε  ← ε_HBAD3P × F × Lr (growth level kept)
  Lmr  = the same ratio with am (the mortality light);  gm_HBAD3P ≈ gm_HBAD4P × F × Lmr
  ms ← ms_HBAD3P × r, r = Σ N fM e^{σST (ST−3)} e^{−β gm F Lmr} (D/20)^ζ / Σ N fM e^{σST (ST−3)} e^{−β gm} (D/20)^ζ over the v1 atoms (N-weighted, per-atom size
       term): the N-weighted hazard at the v1 state equals HBAD3P's (audit r3 I2b: it was 1.29×; the m0 and senescence terms are unchanged)
  κ  = 0: HBAD3P's vigour nodes are fully persistent over an interval and its σv was fitted that way (no HBAD3P parameter maps to a mixing rate; HBAD2's
       remeasurement vigour correlation 0.49 is a different, tree-level quantity), audit r3 I2d.
Returns (P, G, Ob, SP, e3, f3, g3, src, fac) with fac = (F, Lr, Lmr, eps, ms) and src = the HBAD3P key each key's ε / lnF came from."""
function warm_start4(CS, fit3p; quiet=false)
  gl3 = Dict(Symbol(r.name) => Float64(r.value) for r in eachrow(CSV.read(joinpath(fit3p, "globals.csv"), DataFrame)))
  W = CS.warm; F = sum(W.w .* exp.(get(gl3, :gR, 0.0) .* W.rd) .* (W.cr ./ 0.4) .^ get(gl3, :φ, 0.0)) / sum(W.w)
  e3, f3, g3, src = hbad3p_start(fit3p, CS.keys; grp=CS.TR.grp)
  SP0 = space4(CS.keys, CS.TR, CS.sD, CS.sH; start_eps=e3, start_lnF=f3, start_glob=g3); P0, G0, _ = SP0.decode(SP0.z0)
  D = CS.sets["train"]; S = D.S; H4 = HBAD4P; T0 = H4.exact_tabs(length(P0)); X = H4.Scratch(maximum(diff(S.off))); nk = S.nk
  # HBAD4P's v1 plot overtopping per (site, key, class) (lane 1: the lanes share H at v1), and the mortality sums
  O4 = Dict{Tuple{Int,Int,Int},Float64}(); s1 = Ref(0.0); s0 = Ref(0.0)
  hz(gm, p, d) = exp(G0.σST * (p.ST - 3)) * exp(-G0.β * gm) * (d / 20)^G0.ζ
  Fm = Ref(NaN)
  RD4 = zeros(H4.nsites(S))
  for s in 1:H4.nsites(S)
    H4.load_site!(S, X, s); na, RD4[s] = H4.competition!(S, G0, X, s, T0)
    for q in 1:na
      i = X.act[q]; li = i - S.off[s] + 1; S.ln[i] == 1 && (O4[(s, Int(S.sp[i]), Int(S.cl[i]))] = X.O[li])
    end
  end
  # HBAD3P's subplot overtopping per tree (trees above full, equal heights ½, own ½; densities 4 w / a; sdimax 1500)
  sidx = Dict(iid => s for (s, iid) in enumerate(D.iid)); kix = Dict(k => i for (i, k) in enumerate(CS.keys))
  num3 = 0.0; num4 = 0.0; m3 = 0.0; m4 = 0.0; den = 0.0
  W1 = W[HBAD4P.dclass.(W.dia) .>= 1, :]
  for g in groupby(W1, [:iid, :sb])
    gi = parentindices(g)[1]
    ws = Vector{Float64}(W1.w[gi]); ds = Vector{Float64}(W1.dia[gi]); h = Vector{Float64}(W1.ht[gi]); ks = W1.key[gi]; s = sidx[W1.iid[gi[1]]]
    sd = 4 .* ws .* (ds ./ 25.4) .^ 1.605 ./ W1.a[gi]
    for i in eachindex(gi)
      o = 0.5 * sd[i]; for j in eachindex(gi); j == i && continue; h[j] > h[i] && (o += sd[j]); h[j] == h[i] && (o += 0.5 * sd[j]); end
      o4 = get(O4, (s, kix[ks[i]], HBAD4P.dclass(ds[i])), NaN); isnan(o4) && continue
      w = ws[i]; num3 += w * exp(-G0.a * o / 1500); num4 += w * exp(-G0.a * o4); m3 += w * exp(-G0.am * o / 1500); m4 += w * exp(-G0.am * o4); den += w
    end
  end
  Lr = num3 / num4; Lmr = m3 / m4; Fm[] = F * Lmr
  for s in 1:H4.nsites(S)
    H4.load_site!(S, X, s); na, RD = H4.competition!(S, G0, X, s, T0)
    fM = (S.planted[s] ? exp(G0.pM) : 1.0) * exp(G0.mW * (S.Wn[s] - S.W[s]) / 0.1)
    for q in 1:na
      i = X.act[q]; li = i - S.off[s] + 1; p = P0[S.sp[i]]
      gm = exp(-G0.am * X.O[li]) * exp(G0.gR * RD) * S.W[s]^G0.ω * H4.fT_rel(p, G0, S.T[s]) * exp(G0.σv * S.lz[S.ln[i]])
      for m in 1:X.ac[li]
        n = X.an[(li-1)*H4.NATOM+m]; d = X.ad[(li-1)*H4.NATOM+m]
        s1[] += n * fM * hz(gm * Fm[], p, d); s0[] += n * fM * hz(gm, p, d)
      end
    end
  end
  r = s1[] / s0[]; fe = F * Lr
  g3 = copy(g3); g3[:ms] = get(g3, :ms, 0.082) * r; g3[:κ] = 0.0
  e3 = e3 .* fe
  # fecundity of the keys split from a group (audit r4 I2, decision H8): a key HBAD3P did not have takes its group's lnF + ln(its share of the group's TRAIN
  # sapling ingrowth); the group key keeps the remaining share (the group's total recruitment is kept)
  f3 = copy(f3); share = ones(nk)
  for g in unique(src[k] for k in 1:nk if src[k] != CS.keys[k])
    mem = [k for k in 1:nk if src[k] == g]; tot = sum(CS.igk[mem])
    for k in mem; share[k] = max(CS.igk[k] / tot, 1e-6); f3[k] += log(share[k]); end
  end
  # recruitment light: HBAD3P's e^{−λ RD} was evaluated per SUBPLOT (RD of the subplot, × 4 / a), HBAD4P's per plot; at the train v1 state the ratio
  # Rk = Σ_sites (seed ? 1 : ε0) mean_sb e^{−λk RD_sb} / Σ_sites (seed ? 1 : ε0) e^{−λk RD_plot}, λk = λ0 e^{λ1 (ST_k − 3)} (Jensen: convexity, audit r4 I2).
  # Applied to lnF (as the light ratio to ε) when some |Rk − 1| > 5 % (decision H8).
  rdsb = Dict((r_.iid, r_.sb) => r_.rd for r_ in eachrow(W)); Rk = zeros(nk)
  for k in 1:nk
    lk = G0.λ0 * exp(G0.λ1 * (P0[k].ST - 3)); a = 0.0; b = 0.0
    for s in 1:H4.nsites(S)
      ws = S.seed0[k, s] ? 1.0 : G0.ε0; iid = D.iid[s]
      a += ws * sum(exp(-lk * get(rdsb, (iid, sb), 0.0)) for sb in 1:4) / 4; b += ws * exp(-lk * RD4[s])
    end
    Rk[k] = a / b
  end
  rec = maximum(abs.(Rk .- 1)) > 0.05
  rec && (f3 .+= log.(Rk))
  SP = space4(CS.keys, CS.TR, CS.sD, CS.sH; start_eps=e3, start_lnF=f3, start_glob=g3); P, G, Ob = SP.decode(SP.z0)
  quiet || println(@sprintf("warm start from %s: ε × %.4f (crowding / crown F %.4f × subplot / plot light %.4f), ms × %.4f (stress gm × F × %.4f), κ = 0; keys from their HBAD3P group key: %s; recruitment subplot / plot light ratio %.3f–%.3f (%s)",
                            fit3p, fe, F, Lr, r, Lmr, join(["$(k) ← $(s)" for (k, s) in zip(CS.keys, src) if k != s], ", "), minimum(Rk), maximum(Rk),
                            rec ? "applied to lnF" : "within 5 %: not applied"))
  quiet || println("  fecundity shares: ", join(["$(CS.keys[k]) $(round(share[k]; digits=3))" for k in 1:nk if share[k] < 1], ", "))
  (P=P, G=G, Ob=Ob, SP=SP, e3=e3, f3=f3, g3=g3, src=src, fac=(F=F, Lr=Lr, Lmr=Lmr, eps=fe, ms=r, share=share, Rk=Rk, Rapplied=rec))
end

# ---------------------------------------------------------------- parameter files (stage results of a fit; read by kappa_profile4.jl)
"Write a coordinate vector z of the space SP as CSV (name, z, decoded value): the format a stage of a fit hands to the next."
function save_z4(path, SP, z)
  P, G, Ob = SP.decode(z); val(nm) = startswith(nm, "ε:") ? exp(z[findfirst(==(nm), SP.names)]) : startswith(nm, "lnF:") ? z[findfirst(==(nm), SP.names)] :
                                                        (s = Symbol(nm); s in OBS4 ? getfield(Ob, s) : getfield(G, s))
  CSV.write(path, DataFrame(name=SP.names, z=z, value=[val(nm) for nm in SP.names]))
end
"""Start values (start_eps, start_lnF, start_glob) of a parameter set (P, G, Ob) for space4 — e.g. to re-encode a stage-1 optimum in a space with other fixed
globals (the κ profile)."""
function starts4(P, G, Ob)
  g = Dict{Symbol,Float64}(f => getfield(G, f) for f in fieldnames(HBAD4P.GlobP4)); for f in OBS4; g[f] = getfield(Ob, f); end
  (start_eps=[exp(p.lnε) for p in P], start_lnF=[p.lnF for p in P], start_glob=g)
end
"Read a save_z4 file into the space SP (coordinates by name; names absent from the file keep SP.z0)."
function load_z4(path, SP)
  t = CSV.read(path, DataFrame); d = Dict(String(r.name) => Float64(r.z) for r in eachrow(t)); z = copy(SP.z0)
  for (i, nm) in enumerate(SP.names); haskey(d, nm) && (z[i] = d[nm]); end
  z
end
