# HBAD2 calibration objectives: Sim A (tracked-tree cohorts of undisturbed remeasurement intervals; joint negative log-likelihood of binomial survival and Gaussian ln D, ln H of the
# survivors) and Sim B (free runs with recruitment from the first visit of clean natural plots; Poisson stand-table deviance + Lorey-height Gaussian). Port of the prototype
# tools/hbad2/common.jl (`nll_parts`, `size_nll`, `cohort_view`) and tools/hbad2/simB_lik.jl (`simB_nll!`) running the Pan plugin kernel on SoA arrays
# (HBAD2Plugin.run_interval! / site_year_free!). The site loops are threaded over sites (`soa.parallel`), every site touches only its own slice; the sums over cohorts / cells are
# sequential in a fixed order, so the numbers do not depend on the thread count.

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sim A
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

"Reset the working state of a Sim A set to its initial cohorts and integrate every interval site (whole years + fractional step) under the parameters `ep`."
function simulate_simA!(A::SimAData, ep::HBAD2Plugin.HBAD2EcoParams)
  soa = A.soa; csr = soa.csr
  length(csr.c_tph) == length(A.N0) || error("SimAData: SoA capacity differs from the initial state")
  copyto!(csr.c_tph, A.N0); copyto!(csr.c_dq, A.D0); copyto!(csr.c_h, A.H0); copyto!(csr.c_age, A.A0)
  nyr = A.nyr; frac = A.frac
  PanCore.@maybe_threads soa.parallel for i in 1:soa.n
    @inbounds HBAD2Plugin.run_interval!(soa, i, ep, nyr[i], frac[i])
  end
  return A
end

"""
Cohort-level state after a run (prototype `cohort_view`): with vigour the `q` sub-cohorts of each observed cohort are aggregated back (sum of N, quadratic-mean D, N-weighted H;
a cohort that died out keeps its initial D, H); without vigour the sub-cohort IS the cohort. Returns `(N, D, H)` vectors (views of the SoA arrays or the work buffers).
"""
function cohort_view!(A::SimAData)
  csr = A.soa.csr
  A.q == 1 && return csr.c_tph, csr.c_dq, csr.c_h
  q = A.q; N = csr.c_tph; D = csr.c_dq; H = csr.c_h
  aN = A.aggN; aD = A.aggD; aH = A.aggH
  fill!(aN, 0.0); fill!(aD, 0.0); fill!(aH, 0.0)
  @inbounds for i in eachindex(N)
    p = (i - 1) ÷ q + 1; n = N[i]
    aN[p] += n; aD[p] += n * D[i]^2; aH[p] += n * H[i]
  end
  @inbounds for p in eachindex(aN)
    n = aN[p]; aD[p] = n > 0 ? sqrt(aD[p] / n) : A.Dc0[p]; aH[p] = n > 0 ? aH[p] / n : A.Hc0[p]
  end
  return aN, aD, aH
end

"""
    simA_parts(A) -> NamedTuple

Joint negative log-likelihood of the simulated state of `A` (after [`simulate_simA!`](@ref)): binomial survival of the tracked tree counts (p = N2/N1) plus Gaussian ln D2 and
ln H2 of the survivors (weight = survivors, sigma profiled out). `(total, mort, D, H, sigD, sigH, ncoh, ntrees)`.
"""
function simA_parts(A::SimAData)
  N, D, H = cohort_view!(A)
  N0 = A.q == 1 ? A.N0 : A.Nc0
  nm = 0.0; sD = 0.0; sH = 0.0; cD = 0; nobs_m = 0
  @inbounds for i in eachindex(N)
    n1 = A.n1[i]; n1 == 0 && continue
    s = A.s[i]; p = clamp(N[i] / N0[i], 1e-9, 1 - 1e-9)
    nm -= s * log(p) + (n1 - s) * log(1 - p); nobs_m += n1
    if s > 0
      eD = log(D[i] / A.D2o[i]); eH = log(max(H[i], 0.1) / A.H2o[i])
      sD += s * eD^2; sH += s * eH^2; cD += 1
    end
  end
  nD = 0.5 * cD * log(sD / cD); nH = 0.5 * cD * log(sH / cD)
  return (total=nm + nD + nH, mort=nm, D=nD, H=nH, sigD=sqrt(sD / cD), sigH=sqrt(sH / cD), ncoh=cD, ntrees=nobs_m)
end

"""
    size_nll(A, sigma_w) -> Float64

Size-class negative log-likelihood of the survivors (multinomial per cohort over the classes `SZ_EDGES`; predicted through the sub-cohorts with a within-sub-cohort lognormal
spread `sigma_w` around each sub-cohort quadratic mean diameter). Used when `sizecls` is on.
"""
function size_nll(A::SimAData, σw::Float64)
  csr = A.soa.csr; Nv = csr.c_tph; Dv = csr.c_dq; q = A.q
  sc = A.sc; nc = size(sc, 1); K = length(SZ_EDGES)
  pr = zeros(nc, K); tot = zeros(nc)
  lb = [log(e) for e in SZ_EDGES[2:end]]
  nrm = Distributions.Normal()
  @inbounds for i in eachindex(Nv)
    p = (i - 1) ÷ q + 1; n = Nv[i]; n <= 0 && continue
    μ = log(Dv[i]) - σw^2; prev = 0.0
    for k in 1:K
      cum = k < K ? Distributions.cdf(nrm, (lb[k] - μ) / σw) : 1.0
      pr[p, k] += n * (cum - prev); prev = cum
    end
    tot[p] += n
  end
  nll = 0.0
  @inbounds for p in 1:nc
    tot[p] > 0 || continue
    for k in 1:K; c = sc[p, k]; c == 0 && continue; nll -= c * log(max(pr[p, k] / tot[p], 1e-12)); end
  end
  return nll
end

"""
    eval_simA!(A, ep, sw, sigma_w) -> NamedTuple

Simulate and score a Sim A set: [`simA_parts`](@ref) (and, with `sw.sizecls`, the size-class term added to `total` and reported as `size`).
"""
function eval_simA!(A::SimAData, ep::HBAD2Plugin.HBAD2EcoParams, sw::CalibSwitches, σw::Float64)
  simulate_simA!(A, ep)
  r = simA_parts(A)
  sw.sizecls || return r
  sz = size_nll(A, σw)
  return merge(r, (total=r.total + sz, size=sz))
end

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Sim B
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

@inline erfc_(x) = (t = 1 / (1 + 0.5 * abs(x)); y = t * exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
                    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))); x >= 0 ? y : 2 - y)
"Standard normal cdf by the Numerical Recipes erfc approximation (the prototype's `Φ`; kept for exact numbers)."
@inline Φ(z) = 0.5 * erfc_(-z / sqrt(2))

const SB_LB = log.(SB_BREAKS)

"Expected tallied trees per (key, class) and the Lorey height of site `s` at its visit `k` (run year `y`): cohorts through the lognormal observation operator with spread `σ`."
@inline function snapshot_site!(B::SimBData, s::Int, k::Int, site_mult::Float64)
  soa = B.soa; csr = soa.csr
  lo = Int(soa.refs.cohort[s]) - 1; n = Int(soa.scalar.live[s]); σ = B.sigma
  bas = 0.0; bah = 0.0
  @inbounds for i in lo+1:lo+n
    N = csr.c_tph[i]; N <= 0 && continue
    Dq = csr.c_dq[i]
    μ = log(Dq) - σ^2; sp = Int(csr.c_species[i]); prev = Φ((SB_LB[1] - μ) / σ)
    for c in 1:SB_NC
      nxt = c < SB_NC ? Φ((SB_LB[c+1] - μ) / σ) : 1.0
      B.lam[s, k, sp, c] += N * (nxt - prev) / (SB_EXPAND[c] * site_mult); prev = nxt
    end
    ba = N * Dq^2; bas += ba; bah += ba * csr.c_h[i]
  end
  B.hl_sim[s, k] = bas > 0 ? bah / bas : NaN
  return nothing
end

"Reset site `s` of a Sim B set to its initial live cohorts (clock 0, no open recruit class)."
@inline function reset_site!(B::SimBData, s::Int)
  soa = B.soa; csr = soa.csr; sc = soa.scalar
  o = Int(soa.refs.cohort[s]) - 1; o0 = B.off0[s] - 1; n = B.live0[s]
  @inbounds for k in 1:n
    csr.c_species[o+k] = B.sp0[o0+k]; csr.c_tph[o+k] = B.N0[o0+k]; csr.c_dq[o+k] = B.D0[o0+k]; csr.c_h[o+k] = B.H0[o0+k]
    csr.c_age[o+k] = B.A0[o0+k]; csr.c_zq[o+k] = B.zq0[o0+k]; csr.c_ropen[o+k] = Int32(0)
  end
  sc.live[s] = n; sc.ryear[s] = Int32(0)
  return nothing
end

"""
    simulate_simB!(B, ep, nq, site_mult)

Free runs (annual kernel step + recruitment) of every site of `B` for its `years[s]` years; at each visit year the expected tally and Lorey height are recorded (`B.lam`, `B.hl_sim`).
"""
function simulate_simB!(B::SimBData, ep::HBAD2Plugin.HBAD2EcoParams, nq::Int, site_mult::Float64)
  fill!(B.lam, 0.0); fill!(B.hl_sim, NaN); fill!(B.dropped, 0)
  soa = B.soa; nv = size(B.vyear, 2)
  PanCore.@maybe_threads soa.parallel for s in 1:soa.n
    reset_site!(B, s)
    dropped = 0
    for y in 1:B.years[s]
      dropped += HBAD2Plugin.site_year_free!(soa, s, ep, nq)
      k = 0
      for j in 1:nv
        if B.vyear[s, j] == y; k = j; break; end
      end
      k == 0 || snapshot_site!(B, s, k, site_mult)
    end
    B.dropped[s] = dropped
  end
  return B
end

"""
    simB_parts(B) -> NamedTuple

Sim B likelihood terms of the simulated tallies `B.lam` against the observed `B.obs`: `counts` (Poisson deviance/2 per site x visit x key x class), `H` (Gaussian ln Lorey height with
sigma profiled), `total = counts + H`, `size` (species-pooled size-class totals per plot and visit), `species` (per plot and visit, saplings 2.54-12.7 cm and trees >= 12.7 cm of each key),
`plotstand` (key x class per plot and visit; `plotstand_trees` its classes >= 12.7 cm), `dropped` recruit classes.
"""
function simB_parts(B::SimBData)
  nll = 0.0
  @inbounds for ix in eachindex(B.obs)
    o = B.obs[ix]; λ = max(B.lam[ix], 1e-6)
    nll += λ - o * log(λ) + (o > 0 ? o * log(o) - o : 0.0)
  end
  e = [log(B.hl_sim[i] / B.hl_obs[i]) for i in eachindex(B.hl_obs) if isfinite(B.hl_obs[i]) && isfinite(B.hl_sim[i])]
  nH = 0.5 * length(e) * log(sum(abs2, e) / length(e))
  npl = B.nplots; nv = size(B.obs, 2)
  To = zeros(npl, nv, SB_NC); Tl = zeros(npl, nv, SB_NC)
  @inbounds for s in axes(B.obs, 1), k in 1:nv, sp in axes(B.obs, 3), c in 1:SB_NC
    p = B.plot_of[s]; To[p, k, c] += B.obs[s, k, sp, c]; Tl[p, k, c] += B.lam[s, k, sp, c]
  end
  nsz = 0.0
  @inbounds for ix in eachindex(To)
    o = To[ix]; λ = Tl[ix]; (o == 0 && λ == 0) && continue; λ = max(λ, 1e-6)
    nsz += λ - o * log(λ) + (o > 0 ? o * log(o) - o : 0.0)
  end
  nspp = size(B.obs, 3)
  So = zeros(npl, nv, nspp, 2); Sl = zeros(npl, nv, nspp, 2)
  @inbounds for s in axes(B.obs, 1), k in 1:nv, sp in 1:nspp, c in 1:SB_NC
    p = B.plot_of[s]; q = c <= 2 ? 1 : 2; So[p, k, sp, q] += B.obs[s, k, sp, c]; Sl[p, k, sp, q] += B.lam[s, k, sp, c]
  end
  nspc = 0.0
  @inbounds for ix in eachindex(So)
    o = So[ix]; λ = Sl[ix]; (o == 0 && λ == 0) && continue; λ = max(λ, 1e-6)
    nspc += λ - o * log(λ) + (o > 0 ? o * log(o) - o : 0.0)
  end
  Po = zeros(npl, nv, nspp, SB_NC); Pl = zeros(npl, nv, nspp, SB_NC)
  @inbounds for s in axes(B.obs, 1), k in 1:nv, sp in 1:nspp, c in 1:SB_NC
    p = B.plot_of[s]; Po[p, k, sp, c] += B.obs[s, k, sp, c]; Pl[p, k, sp, c] += B.lam[s, k, sp, c]
  end
  npst = 0.0; npst_t = 0.0
  @inbounds for ix in CartesianIndices(Po)
    o = Po[ix]; λ = Pl[ix]; (o == 0 && λ == 0) && continue; λ = max(λ, 1e-6)
    d = λ - o * log(λ) + (o > 0 ? o * log(o) - o : 0.0)
    npst += d; ix[4] >= 3 && (npst_t += d)
  end
  return (total=nll + nH, counts=nll, H=nH, size=nsz, species=nspc, plotstand=npst, plotstand_trees=npst_t, dropped=sum(B.dropped))
end

"Simulate and score a Sim B set under the parameters `ep` (`nq` sub-cohorts per recruit class, `site_mult` the density multiplier of the site type)."
function eval_simB!(B::SimBData, ep::HBAD2Plugin.HBAD2EcoParams, sw::CalibSwitches)
  simulate_simB!(B, ep, sw.vigour ? 3 : 1, sw.site_mult)
  return simB_parts(B)
end

"The scalar Sim B objective of a mode (prototype `HBAD2_FB`): stand | size | margins | plotstand | both."
function fb_value(r, mode::Symbol)
  mode == :size && return r.size + r.H
  mode == :margins && return r.size + r.species + r.H
  mode == :plotstand && return r.plotstand + r.H
  mode == :both && return r.total + r.size
  mode == :stand && return r.total
  error("hbad2_calib: unknown Sim B objective $mode (stand | size | margins | plotstand | both)")
end
