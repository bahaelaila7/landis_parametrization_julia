# The annual kernel of HBAD v2: a line-by-line port of the prototype's `step_site!` (growth, mortality, self-thinning cap, ageing) and
# `recruit_site!` (recruitment into the 1-in tally) -- same equations, same order of operations, Float64 state. The prototype stores one
# `Sites` for all sites; here the arrays are the SoA's CSR arrays and `lo:hi` is the cohort range of one site.

"Gauss-Hermite (3-point) vigour sub-cohorts: quantiles z and population shares."
const VIG_Z = (-sqrt(3.0), 0.0, sqrt(3.0))
const VIG_W = (1 / 6, 2 / 3, 1 / 6)

const RD_REF = 600.0          # SDI scale of the recruitment light index (the scale its coefficients were fitted on)
const RSCALE = 185.2          # one FIA microplot sapling in trees/ha
const DSEED = 12.7            # seed-source trees: D >= 5 in
const RCLASS = 5              # recruits of one species within RCLASS years form one cohort (its N only decreases once the class closes)

const _PI4E4 = 7.853981633974483e-5     # pi/4/1e4: basal area (m2) of one tree of diameter 1 cm (the prototype's literal)

@inline sdi(N, D) = N * (D / 25.4)^1.605
@inline function dpot(X, Xmax, r, c)        # Chapman-Richards increment at the current size X (form 1)
  X >= Xmax && return 0.0
  X <= 0 && return 0.0
  c * r * X * ((Xmax / X)^(1 / c) - 1)
end
# power-exponential potential increment (form 2): r (X/X0)^c exp(-X/Xmax); peak at X = c Xmax; X0 = 20 cm (D) / 15 m (H)
@inline dpow(X, X0, Xmax, r, c) = X <= 0 ? 0.0 : r * exp(c * log(X / X0) - X / Xmax)
@inline potD(G, D, p, f) = G.form == 1 ? dpot(D, p.Dmax, p.rD * f, p.cD) : dpow(D, 20.0, p.Dmax, p.rD * f, p.cD)
@inline potH(G, H, p, f) = G.form == 1 ? dpot(H, p.Hmax, p.rH * f, p.cH) : dpow(H, 15.0, p.Hmax, p.rH * f, p.cH)

"""
    step_cohorts!(sp, N, D, H, A, zq, O, SD, V, P, G, lo, hi, Δ, pl)

One step of length `Δ` (1.0 in Pan) of the cohorts `lo:hi` of ONE site (prototype `step_site!`). `sp` = eco-local species index into `P`,
`O`/`SD`/`V` = scratch (overtopping, SDI, vigour), `pl` = planted stand. 1. snapshot SDI_i, SDImax_site, RD, overtopping O_i; 2. growth
(power-exponential or Chapman-Richards potential x competition x vigour); 3. mortality hazard (background + suppression + age); 4. self-thinning cap
(suppressed first); 5. A += Δ. An empty stand (total SDI 0) returns before the ageing, as in the prototype.
"""
@inline function step_cohorts!(sp, N, D, H, A, zq, O, SD, V, P::Vector{SpP}, G::GlobP, lo::Int, hi::Int, Δ::Float64, pl::Bool)
  tot = 0.0; wmax = 0.0
  @inbounds for i in lo:hi
    SD[i] = sdi(N[i], D[i]); tot += SD[i]; wmax += SD[i] * P[Int(sp[i])].sdimax
  end
  tot <= 0 && return nothing
  smax = wmax / tot; RD = tot / smax
  @inbounds for i in lo:hi                   # overtopping by taller cohorts (+ half of equal-height cohorts and of itself)
    o = 0.5 * SD[i]; hi_ = H[i]
    for j in lo:hi
      j == i && continue
      H[j] > hi_ && (o += SD[j]); H[j] == hi_ && (o += 0.5 * SD[j])
    end
    O[i] = o / smax
  end
  fH = pl ? exp(G.pH) : 1.0; fD = pl ? exp(G.pD) : 1.0; fM = pl ? exp(G.pM) : 1.0
  @inbounds for i in lo:hi
    p = P[Int(sp[i])]; o = O[i]
    vg = exp(G.σv * zq[i])                   # persistent vigour of this sub-cohort (1 for a mean-field cohort)
    v = exp(-p.gOD * o); ve = min(v * vg, 1.0); V[i] = ve
    dD = potD(G, D[i], p, fD) * v * vg * exp(p.gRD * RD)
    dH = potH(G, H[i], p, fH) * vg * exp(-p.gOH * o + p.gRH * RD)
    lD = log(D[i] / 20)
    h = fM * (exp(p.mbg + p.mD * lD + G.mD2 * D[i] / 20) + p.msup * (1 - ve) * exp(G.mSD * lD)) + exp(G.mA * (A[i] / p.longevity - 1))
    N[i] *= exp(-h * Δ)
    D[i] += dD * Δ; H[i] += dH * Δ
  end
  for _ in 1:6                               # self-thinning to RD <= 1 (composition-dependent SDImax -> repeat)
    tot = 0.0; wmax = 0.0; ws = 0.0
    @inbounds for i in lo:hi
      SD[i] = sdi(N[i], D[i]); tot += SD[i]; wmax += SD[i] * P[Int(sp[i])].sdimax; ws += SD[i] * (1 - V[i]) + 1e-9 * SD[i]
    end
    tot <= 0 && break
    smax = wmax / tot; ex = tot - smax
    ex <= 1e-6 * smax && break
    @inbounds for i in lo:hi
      r = min(SD[i], ex * (SD[i] * (1 - V[i]) + 1e-9 * SD[i]) / ws)
      SD[i] > 0 && (N[i] *= 1 - r / SD[i])
    end
  end
  @inbounds for i in lo:hi; A[i] += Δ; end
  nothing
end

"""
    recruit_rates!(sp, N, D, ropen, rate, ba, oy, mature, estab, P_rec, lo, hi, so, nsp, yr, nq) -> Int

Phase 1 of the recruitment of one site (prototype `recruit_site!`, everything it computes BEFORE writing): per species the recruit rate
R = RSCALE exp(rho0 + rhoS ln(1 + BA_sp(D >= DSEED)) + rhoL SDI/RD_REF) (trees/ha this year; SDI of the post-step state), and whether this year opens a new
recruit class (no open class of the species, or the open one is RCLASS years old). Writes the per-species scratch `rate`, `estab`, `mature`
(offset `so`); returns the number of new cohort slots needed (nq per new class). The species rates are independent of each other (recruits
of species k never enter the BA of species k' and the light index uses the state before any recruit), so computing all rates first is identical
to the prototype's interleaved loop.
"""
@inline function recruit_rates!(sp, N, D, ropen, rate, ba, oy, mature, estab, rec::Vector{RecP}, lo::Int, hi::Int, so::Int, nsp::Int, yr::Int, nq::Int)::Int
  tot = 0.0
  @inbounds for i in lo:hi; tot += sdi(N[i], D[i]); end
  RD = tot / RD_REF
  @inbounds for k in 1:nsp
    ba[so+k] = 0.0; oy[so+k] = 0; mature[so+k] = false
  end
  @inbounds for i in lo:hi
    k = Int(sp[i])
    if D[i] >= DSEED
      ba[so+k] += 7.853981633974483e-5 * D[i]^2 * N[i]
      mature[so+k] = true
    end
    ropen[i] > 0 && (oy[so+k] = ropen[i])
  end
  nnew = 0
  @inbounds for k in 1:nsp
    r = rec[k]
    rate[so+k] = RSCALE * exp(r.rho0 + r.rhoS * log1p(ba[so+k]) + r.rhoL * RD) * 1.0
    open = Int(oy[so+k])
    nc = open == 0 || yr - open >= RCLASS
    estab[so+k] = nc
    nc && (nnew += nq)
  end
  return nnew
end

"""
    recruit_apply_cap!(sp, age, N, D, H, zq, ropen, rate, estab, rec, o, n, so, nsp, yr, nq, cap) -> (n, dropped)

Phase 2 of the recruitment of one site: for every species either merge this year's recruits into the open class (quadratic-mean D, N-weighted H,
densities add) or close the old class and append a new one (`nq` sub-cohorts at (D0, H0, A0), N = rate x share). Cohorts `o+1:o+n` are the existing ones;
the slots `o+n+1:o+cap` are free. A new class that does not fit (`n + nq > cap`) is DROPPED (the prototype's `FreeSites.dropped` diagnostic): the species' open class
is closed and nothing is appended (the next year opens a new one). Returns the new live count and the number of dropped classes.
"""
@inline function recruit_apply_cap!(sp, age, N, D, H, zq, ropen, rate, estab, rec::Vector{RecP}, o::Int, n::Int, so::Int, nsp::Int, yr::Int, nq::Int, cap::Int)
  n0 = n; dropped = 0
  @inbounds for k in 1:nsp
    r = rec[k]
    if estab[so+k]
      for i in o+1:o+n0                                  # the previous class of this species closes
        Int(sp[i]) == k && (ropen[i] = 0)
      end
      if n + nq <= cap
        for q in 1:nq
          n += 1; jj = o + n; w = nq == 1 ? 1.0 : VIG_W[q]
          sp[jj] = eltype(sp)(k); N[jj] = rate[so+k] * w; D[jj] = r.D0; H[jj] = r.H0; age[jj] = r.A0; zq[jj] = nq == 1 ? 0.0 : VIG_Z[q]; ropen[jj] = Int32(yr)
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
        D[jj] = sqrt((m0 * D[jj]^2 + add * r.D0^2) / m); H[jj] = (m0 * H[jj] + add * r.H0) / m; N[jj] = m
      end
    end
  end
  return n, dropped
end

"""
    recruit_apply!(sp, age, N, D, H, zq, ropen, rate, estab, rec, o, n, so, nsp, yr, nq) -> Int

[`recruit_apply_cap!`](@ref) without a capacity limit (the caller grew the CSR for every class that opens). Returns the new live count.
"""
@inline recruit_apply!(sp, age, N, D, H, zq, ropen, rate, estab, rec::Vector{RecP}, o::Int, n::Int, so::Int, nsp::Int, yr::Int, nq::Int)::Int =
  recruit_apply_cap!(sp, age, N, D, H, zq, ropen, rate, estab, rec, o, n, so, nsp, yr, nq, typemax(Int))[1]

"""Reported allometric AGB (g/m2) of `tph` trees/ha of quadratic mean diameter `dq` (cm): the HBAD v1 cohort mass-QMD relation. Reporting only; never used by the dynamics."""
@inline agb_of(a::Allo, tph::Float64, dq::Float64) = tph / HBADPlugin.HB_ACRE2HA * exp(a.mq_a + a.mq_b * log(dq / 2.54)) * HBADPlugin.HB_LB
