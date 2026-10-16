# HBAD v1 kernel (docs/hbad_plugin.md §3, fixed FIA-regression priors, G_MULT = M_MULT = 1, SDImax = species p95) on the SAME
# tracked-cohort Sim A sites as v2, initialised directly from the observed (N, D, H) — scored with the same joint likelihood.
#   ./julia_gdal.sh --project=. --threads=4 tools/hbad2/v1_baseline.jl
include(joinpath(@__DIR__, "common.jl"))
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
pri = CSV.read(joinpath(@__DIR__, "..", "..", "results", "hbad", "priors.csv"), DataFrame); pk = Dict(String(r.key) => r for r in eachrow(pri))
grpof = Dict(k => (k[1] == '_' ? k[2:2] : first(tr.grp[tr.key .== k])) for k in keys_)
blk(k, col) = (r = get(pk, k, nothing); r !== nothing && !ismissing(r[col]) ? r : pk["_" * grpof[k]])
PR = [(g=blk(k, :g_b0), m=blk(k, :m_c0), h=blk(k, :hd_a), sd=blk(k, :sdimax_p95_all).sdimax_p95_all) for k in keys_]
CR(r, D) = r.hd_a * (1 - exp(-r.hd_b * D))^r.hd_c
function v1_sim!(S::Sites)
  Threads.@threads for s in 1:length(S.off)-1
    lo = S.off[s]; hi = S.off[s+1] - 1; n = hi - lo + 1
    for i in lo:hi; S.N[i] = S.N0[i]; S.D[i] = S.D0[i]; S.H[i] = S.H0[i]; S.A[i] = S.A0[i]; end
    steps = vcat(fill(1.0, S.nyr[s]), S.frac[s] > 0 ? [S.frac[s]] : Float64[])
    bal = zeros(n)
    for Δ in steps
      ba(i) = pi / 4 * (S.D[i] / 100)^2 * S.N[i]
      sdi_(i) = S.N[i] * (S.D[i] / 25.4)^1.605
      tot = sum(sdi_(i) for i in lo:hi); bat = sum(ba(i) for i in lo:hi)
      smax = sum(ba(i) * PR[S.sp[i]].sd for i in lo:hi) / bat; RD = tot / smax; RDc = min(RD, 1.5)
      for (k, i) in enumerate(lo:hi); bal[k] = sum(ba(j) for j in lo:hi if S.D[j] > S.D[i]; init=0.0); end
      for (k, i) in enumerate(lo:hi)
        p = PR[S.sp[i]]; g = p.g; m = p.m; D = S.D[i]; Dc = max(D, 2.54); age = S.A[i] + Δ
        dD = clamp(exp(g.g_b0 + g.g_b1_lnD * log(Dc / 20) + g.g_b2_D * Dc / 20 + g.g_b3_lnage * log(age / 30) - g.g_k_bal * bal[k] / 10), 0, 3)
        hz = min(exp(m.m_c0 + m.m_c1_lnD * log(Dc / 20) + m.m_c2_D * Dc / 20 + m.m_c3_RD * RDc + m.m_c4_BAL * bal[k] / 10 + m.m_c5_RDxlnD * RDc * log(Dc / 20)), 0.5)
        D2 = D + dD * Δ; c0 = CR(p.h, D)
        S.H[i] = c0 < 0.1 ? 1.3 + CR(p.h, D2) : 1.3 + (S.H[i] - 1.3) * CR(p.h, D2) / c0
        S.D[i] = D2; S.N[i] *= exp(-hz * Δ); S.A[i] = age
      end
      for _ in 1:(get(ENV, "V1_NOTHIN", "0") == "1" ? 0 : 8)                                       # self-thinning, smallest dq first
        tot = sum(sdi_(i) for i in lo:hi); bat = sum(ba(i) for i in lo:hi); bat <= 0 && break
        smax = sum(ba(i) * PR[S.sp[i]].sd for i in lo:hi) / bat; ex = tot - smax; ex <= 1e-3 * smax && break
        for i in sort(collect(lo:hi), by=j -> S.D[j])
          ex <= 0 && break; si = sdi_(i); r = min(si, ex); S.N[i] *= si > 0 ? 1 - r / si : 1; ex -= r
        end
      end
    end
  end
  S
end
for set in ("train", "val", "test")
  C = build_cohorts(tr, set, keys_); v1_sim!(C.S); p = nll_parts(C.S, C.obs)
  @printf("V1 %-5s total %.1f (mort %.1f D %.1f H %.1f σD %.4f σH %.4f) cohorts %d\n", set, p.total, p.mort, p.D, p.H, p.sigD, p.sigH, p.ncoh)
  set == "val" && diagnostics(C.S, C.obs, keys_)
end
