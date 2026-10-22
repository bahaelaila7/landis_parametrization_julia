# The annual kernel of HBAD3P: a line-by-line port of the reference `step_site!` (growth, height, mortality, crown, self-thinning cap, ageing) and `recruit_siteP!`
# (recruitment into the 1-in tally) -- same equations, same order of operations, Float64 state. The reference stores one `SitesP` for all sites; here the arrays are the SoA's CSR
# arrays and `lo:hi` is the cohort range of one site. Equations (reference header), all rates from the START-of-step state (explicit Euler), Δ = 1 or the final fraction of a year:
#   SDI_i = N_i (D_i/25.4)^1.605; RD = ΣSDI / sdimax; O_i = (Σ_{H_j>H_i} SDI_j + ½ Σ_{j≠i, H_j=H_i} SDI_j + ½ SDI_i) / sdimax
#   cw = e^{gR RD} W^ω e^{σv z} (CR/0.4)^φ;  light L = f0 + (1-f0) e^{-a O};  g = L cw;  hyd = max(0, 1 - min(H/1.1Hmax, 1)^θ)
#   dV = max(e^{lnε} (D/20)^b 122.98 g hyd fD - ρ D² H, 0);  dD = dV Δ / (2 D max(H, 1.37))
#   H*(D) = 1.37 + (Hmax-1.37)(1 - exp(-s D/(Hmax-1.37)))^c;  Lh = e^{-kh O};  dH = [H*(D+dD) - H*(D)](1 + η(1-Lh)) + λΔ(H*(D)(1 + μ(1-Lh)) - H);  H += max(fH dH, 0)
#   gm = e^{-am O} cw;  h = fM [m0/LONGEVITY + ms e^{σST (ST-3)} e^{-β gm} (D/20)^ζ] + e^{mA (A/LONGEVITY - 1)},  fM = (planted ? e^{pM} : 1) e^{mW (Wn - W)/0.1};  N *= e^{-hΔ}
#   CR* = crlo + (crhi-crlo) e^{-kc O};  CR += (CR > CR* ? τr : τe)(CR* - CR)Δ, clamped to [0.05, 0.95]
#   self-thinning after the step, <= 6 passes while ΣSDI > sdimax: remove the excess SDI in proportion to SDI_i (1 - min(1, g_i)) (+1e-9 SDI_i);  A += Δ
#   recruitment (free runs): rate = RSCALE e^{lnF - λ0 e^{λ1 (ST-3)} RD} (seed ? 1 : ε0) rmult Δ at D_REC, A_REC, H*(D_REC), CR = crhi

"Gauss-Hermite (5-point, probabilists') vigour nodes z and population shares w (= tools/hbad2/common.jl VQ_Z / VQ_W for HBAD2_VIG_N=5)."
const VIG_Z = (-2.856970013872806, -1.355626179974266, 0.0, 1.355626179974266, 2.856970013872806)
const VIG_W = (0.011257411327721, 0.222075922005613, 0.533333333333333, 0.222075922005613, 0.011257411327721)

const RSCALE = 185.2          # one FIA microplot sapling in trees/ha
const DSEED = 12.7            # seed trees: D >= 5 in
const RCLASS = 5              # recruits of one species within RCLASS years form one cohort (its N only decreases once the class closes)
const D_REC = 2.54            # recruits enter at 1 in
const A_REC = 5.0
const _PI4E4 = 7.853981633974483e-5     # pi/4/1e4: basal area (m2) of one tree of diameter 1 cm

@inline sdi(N, D) = N * (D / 25.4)^1.605
@inline hstar(p::SpPP, G::GlobPP, D) = (h = p.Hmax - 1.37; 1.37 + h * (1 - exp(-G.s * D / h))^G.c)

"""
    step_cohorts!(sp, N, D, H, A, zq, CR, O, SD, V, P, G, lo, hi, Δ, pl, W, Wn)

One step of length `Δ` (1.0 in Pan) of the cohorts `lo:hi` of ONE site (reference `step_site!`). `sp` = eco-local species index into `P`, `O`/`SD`/`V` = scratch (overtopping, SDI, growth vigour min(g, 1)),
`pl` = planted stand, `W`/`Wn` = period / normal water index of the site. An empty stand (total SDI 0) returns before the ageing, as in the reference.
"""
@inline function step_cohorts!(sp, N, D, H, A, zq, CR, O, SD, V, P::Vector{SpPP}, G::GlobPP, lo::Int, hi::Int, Δ::Float64, pl::Bool, W::Float64, Wn::Float64)
  tot = 0.0
  @inbounds for i in lo:hi; SD[i] = sdi(N[i], D[i]); tot += SD[i]; end
  tot <= 0 && return nothing
  RD = tot / G.sdimax
  @inbounds for i in lo:hi
    o = 0.5 * SD[i]; hi_ = H[i]
    for j in lo:hi
      j == i && continue
      H[j] > hi_ && (o += SD[j]); H[j] == hi_ && (o += 0.5 * SD[j])
    end
    O[i] = o / G.sdimax
  end
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0
  fM = (pl ? exp(G.pM) : 1.0) * exp(G.mW * (Wn - W) / 0.1)
  wω = W^G.ω; cr = exp(G.gR * RD)
  @inbounds for i in lo:hi
    p = P[Int(sp[i])]; Di = D[i]; Hi = H[i]
    fcr = (CR[i] / 0.4)^G.φ
    vgz = exp(G.σv * zq[i]); cw = cr * wω * vgz * fcr
    lt = G.f0 + (1 - G.f0) * exp(-G.a * O[i])
    g = lt * cw; V[i] = min(g, 1.0)
    gm = exp(-G.am * O[i]) * cw
    hyd = max(0.0, 1 - min(Hi / (1.1 * p.Hmax), 1.0)^G.θ)
    dV = max(exp(p.lnε) * (Di / 20)^G.b * 122.98 * g * hyd * fD - G.ρ * Di^2 * Hi, 0.0)
    dD = dV / (2 * Di * max(Hi, 1.37)) * Δ
    Lh = exp(-G.kh * O[i]); hs = hstar(p, G, Di)
    dH = (hstar(p, G, Di + dD) - hs) * (1 + G.η * (1 - Lh)) + G.λ * Δ * (hs * (1 + G.μ * (1 - Lh)) - Hi)
    h = fM * (G.m0 / p.longevity + G.ms * exp(G.σST * (p.ST - 3)) * exp(-G.β * gm) * (Di / 20)^G.ζ) + exp(G.mA * (A[i] / p.longevity - 1))
    N[i] *= exp(-h * Δ); D[i] = Di + dD; H[i] = Hi + max(dH * fH, 0.0)
    crt = G.crlo + (G.crhi - G.crlo) * exp(-G.kc * O[i]); c_ = CR[i]
    CR[i] = clamp(c_ + (c_ > crt ? G.τr : G.τe) * (crt - c_) * Δ, 0.05, 0.95)
  end
  for _ in 1:6
    tot = 0.0; ws = 0.0
    @inbounds for i in lo:hi; SD[i] = sdi(N[i], D[i]); tot += SD[i]; ws += SD[i] * (1 - V[i]) + 1e-9 * SD[i]; end
    ex = tot - G.sdimax
    ex <= 1e-6 * G.sdimax && break
    @inbounds for i in lo:hi
      r = min(SD[i], ex * (SD[i] * (1 - V[i]) + 1e-9 * SD[i]) / ws); SD[i] > 0 && (N[i] *= 1 - r / SD[i])
    end
  end
  @inbounds for i in lo:hi; A[i] += Δ; end
  nothing
end

"""
    recruit_rates!(sp, N, D, ropen, rate, oy, seedplot, estab, P, rec, G, rmult, Δ, lo, hi, so, nsp, yr, nq) -> Int

Phase 1 of the recruitment of one site (reference `recruit_siteP!`, everything it computes BEFORE writing): per species the recruit rate
R = RSCALE e^{lnF - λ0 e^{λ1 (ST-3)} RD} (seed ? 1 : ε0) rmult Δ (trees/ha; RD of the post-step state; seed = the plot-level seed presence `seedplot` OR a live cohort of the species with D >= DSEED),
and whether this year opens a new recruit class (no open class of the species, or the open one is RCLASS years old). Writes the per-species scratch `rate`, `estab`, `oy` (offset `so`); a species with
rate <= 0 does nothing (no class change). Returns the number of new cohort slots needed (nq per new class). The species are independent of each other, so computing all rates first equals the reference's interleaved loop.
"""
@inline function recruit_rates!(sp, N, D, ropen, rate, oy, seedplot, estab, P::Vector{SpPP}, rec::RecGP, G::GlobPP, rmult::Float64, Δ::Float64, lo::Int, hi::Int, so::Int, nsp::Int, yr::Int, nq::Int)::Int
  tot = 0.0
  @inbounds for i in lo:hi; tot += sdi(N[i], D[i]); end
  RD = tot / G.sdimax
  @inbounds for k in 1:nsp
    oy[so+k] = 0; estab[so+k] = false
  end
  @inbounds for i in lo:hi
    k = Int(sp[i])
    ropen[i] > 0 && (oy[so+k] = ropen[i])
  end
  nnew = 0
  @inbounds for k in 1:nsp
    p = P[k]; seed = seedplot[so+k]
    for i in lo:hi; (Int(sp[i]) == k && D[i] >= DSEED && N[i] > 0) && (seed = true); end
    r = RSCALE * exp(p.lnF - rec.λ0 * exp(rec.λ1 * (p.ST - 3)) * RD) * (seed ? 1.0 : rec.ε0) * rmult * Δ
    rate[so+k] = r
    r <= 0 && continue
    open = Int(oy[so+k])
    nc = open == 0 || yr - open >= RCLASS
    estab[so+k] = nc
    nc && (nnew += nq)
  end
  return nnew
end

"""
    recruit_apply_cap!(sp, age, N, D, H, zq, CR, ropen, rate, estab, P, rec, G, o, n, so, nsp, yr, nq, cap) -> (n, dropped)

Phase 2 of the recruitment of one site: for every species with a positive rate either merge this year's recruits into the open class (quadratic-mean D, N-weighted H, densities add) or close the old class
and append a new one (`nq` vigour sub-cohorts at (D_REC, H*(D_REC), A_REC, CR = crhi), N = rate x share). Cohorts `o+1:o+n` are the existing ones; the slots `o+n+1:o+cap` are free. A new class that does
not fit (`n + nq > cap`) is DROPPED (the reference's `FreeSitesP.dropped`): the species' open class is closed and nothing is appended (the next year opens a new one). Returns the new live count and the number of dropped classes.
"""
@inline function recruit_apply_cap!(sp, age, N, D, H, zq, CR, ropen, rate, estab, P::Vector{SpPP}, G::GlobPP, o::Int, n::Int, so::Int, nsp::Int, yr::Int, nq::Int, cap::Int)
  n0 = n; dropped = 0
  @inbounds for k in 1:nsp
    rate[so+k] <= 0 && continue
    h0 = hstar(P[k], G, D_REC)
    if estab[so+k]
      for i in o+1:o+n0                                  # the previous class of this species closes
        Int(sp[i]) == k && (ropen[i] = 0)
      end
      if n + nq <= cap
        for q in 1:nq
          n += 1; jj = o + n; w = nq == 1 ? 1.0 : VIG_W[q]
          sp[jj] = eltype(sp)(k); N[jj] = rate[so+k] * w; D[jj] = D_REC; H[jj] = h0; age[jj] = A_REC; zq[jj] = nq == 1 ? 0.0 : VIG_Z[q]; CR[jj] = G.crhi; ropen[jj] = Int32(yr)
        end
      else
        dropped += 1
      end
      estab[so+k] = false
    else
      for q in 1:nq
        zqq = nq == 1 ? 0.0 : VIG_Z[q]
        jj = 0
        for i in o+1:o+n0
          if Int(sp[i]) == k && ropen[i] > 0 && zq[i] == zqq
            jj = i; break
          end
        end
        jj == 0 && continue            # the open class was removed from outside (harvest / injection): nothing to merge into
        add = rate[so+k] * (nq == 1 ? 1.0 : VIG_W[q]); m0 = N[jj]; m = m0 + add
        D[jj] = sqrt((m0 * D[jj]^2 + add * D_REC^2) / m); H[jj] = (m0 * H[jj] + add * h0) / m; N[jj] = m
      end
    end
  end
  return n, dropped
end

"""Reported allometric AGB (g/m2) of `tph` trees/ha of quadratic mean diameter `dq` (cm): the HBAD v1 cohort mass-QMD relation. Reporting only; never used by the dynamics."""
@inline agb_of(a::Allo, tph::Float64, dq::Float64) = tph / HBADPlugin.HB_ACRE2HA * exp(a.mq_a + a.mq_b * log(dq / 2.54)) * HBADPlugin.HB_LB
