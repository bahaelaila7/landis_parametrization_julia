# Unit tests for the multi-objective gdigel objectives (docs/mo_gdigel.md).
#   source wt_env.sh && ./julia_gdal.sh --project=. --threads=2 test/mo_gdigel_unit.jl
using Pan, Random, Statistics
const P = Pan; const PU = P.PU; const GD = P.GDIgel
nfail = Ref(0); npass = Ref(0)
macro check(cond, msg)
  quote
    if $(esc(cond)); npass[] += 1; else nfail[] += 1; println("  FAIL: ", $(esc(msg))); end
  end
end

# ---- a_g, m_g: synthetic index with 2 groups, 40 pairs each (1 record per pair) ----
G = 2; npair = 80
key2rec = Dict{NTuple{4,Int},Int}(); rec_pair = Int[]; rec_obs = Float64[]; pair_group = Int[]
rng = Random.Xoshiro(3); obs_v = 50 .+ 4000 .* rand(rng, npair)
for p in 1:npair
  push!(pair_group, p <= 40 ? 1 : 2); push!(rec_pair, p); push!(rec_obs, obs_v[p]); key2rec[(p, 10, 1, 30)] = p
end
ix = P.GDGroupIndex(key2rec, rec_pair, rec_obs, pair_group, zeros(Int, npair), G, ["g1", "g2"], Dict{Tuple{Int,Int},Int}())
cache(simf) = [(UInt32(p), 10, UInt32(1), UInt32(30), Float32(simf(p))) for p in 1:npair]
# Float32 storage of the cached biomass limits the exactness; compare with a tolerance
r0 = P._gd_losses(ix, cache(p -> obs_v[p]))
@check all(abs.(r0.a) .< 1e-5) "a_g = 0 when sim == obs"
@check all(abs.(r0.f) .< 1e-8) "bias loss f_g = 0 when sim == obs"
δ = 0.3
rp = P._gd_losses(ix, cache(p -> expm1(log1p(obs_v[p]) + δ)))
rm = P._gd_losses(ix, cache(p -> expm1(log1p(obs_v[p]) - δ)))
@check all(isapprox.(rp.a, δ; atol=1e-4)) "a_g = |bias| for a constant log-bias (+δ): $(rp.a)"
@check all(isapprox.(rm.a, rp.a; atol=1e-4)) "a_g unchanged when the sign of a constant log-bias flips"
@check all(isapprox.(rp.m, -rm.m; atol=1e-4)) && all(rp.m .> 0.29) && all(rm.m .< -0.29) "signed m_g flips with the bias sign"
# mixed-sign errors: signed mean ~ 0 but a_g large (the property that distinguishes accuracy from the TOST statistic)
rmix = P._gd_losses(ix, cache(p -> expm1(log1p(obs_v[p]) + (iseven(p) ? δ : -δ))))
@check all(abs.(rmix.m) .< 1e-3) && all(isapprox.(rmix.a, δ; atol=1e-4)) "alternating-sign errors: m_g ≈ 0 but a_g = δ"

# ---- w_g: Wasserstein-1 on binned age-class CDFs via the main-loss machinery ----
lp = PU.LossParams(age_bins=PU.AgeBins(bins_idx=[10, 20, 30, 40, 50, 60, 80, 100, 120, 150], last_bin_open=true), smoothing_weights=P.FloatType[1.0], lambda=P.FloatType(1.0))
function cdf_of(agevec)   # biomass-by-age vector → normalised binned CDF (as calculate_species_loss! does)
  a = zeros(P.FloatType, 200); for (age, b) in agevec; a[age] += b; end
  PU.smoothen_bin_cdf(a; w=lp.smoothing_weights, age_bins=lp.age_bins)
end
bw = lp.age_bins.bin_widths
ref = cdf_of([(15, 100.0), (35, 300.0), (75, 200.0)])
same = cdf_of([(15, 100.0), (35, 300.0), (75, 200.0)])
shifted = cdf_of([(35, 100.0), (55, 300.0), (115, 200.0)])
w_same = PU._w1_sum(ref, same, bw, 1, 1); w_shift = PU._w1_sum(ref, shifted, bw, 1, 1); w_absent = PU._w1_sum(ref, nothing, bw, 1, 1)
@check w_same == 0 "W1 = 0 when the sim age structure equals the reference"
@check w_shift > 0 && w_absent > 0 "W1 > 0 when the age distribution shifts / the species is absent ($w_shift, $w_absent)"

# group aggregation (ratio of sums over the group's cells; non-CELL_NORM: / visits)
PU.CELL_NORM[] = false
num = [[0.0, 2.0], [4.0]]; den = [[2.0, 2.0], [4.0]]
w = P._gd_group_w(num, den, [[(1, 1), (1, 2)], [(2, 1)], Tuple{Int,Int}[]])
@check isapprox(w[1], 0.5) && isapprox(w[2], 1.0) && isnan(w[3]) "group W = Σnum/Σden over the group's cells (NaN without visits): $w"
@check all(P._gd_group_w([[0.0, 0.0], [0.0]], den, [[(1, 1), (1, 2)], [(2, 1)]]) .== 0.0) "group W = 0 when every cell's W1 is 0"

# ---- GDIgel scalarization / block score / weights ----
@check GD.mo_lambdas(5) == [0.0, 0.25, 0.5, 0.75, 1.0] && GD.mo_lambdas(1) == [0.5] "uniform λ grid"
blocks = [GD.BlockDef([1, 2], [1]), GD.BlockDef([3], [2])]
st = GD.GDState(blocks, 3; obj=:mo, G=2, aref=2.0, wref=4.0, lams=GD.mo_lambdas(3))
f = [1.0, 3.0, 2.0, 2.0]    # a = (1, 3), w = (2, 2)
@check GD.lineage_total(st, 1.0, f) == 4.0 / 2.0 && GD.lineage_total(st, 0.0, f) == 4.0 / 4.0 "F at λ=1 is A/Aref, at λ=0 is W/Wref"
@check GD.lineage_total(st, 0.5, f) == max(0.5 * 4 / 2, 0.5 * 4 / 4) "Chebyshev F = max(...)"
@check GD.block_score(st, 0.5, f, [1]) == max(0.5 * 1 / 2, 0.5 * 2 / 4) "block score uses only the block's groups"
GD.add_lineage!(st, [0.1, 0.2, 0.3], f); GD.add_lineage!(st, [0.1, 0.2, 0.3], f); GD.add_lineage!(st, [0.1, 0.2, 0.3], f); GD.add_lineage!(st, [0.1, 0.2, 0.3], f)
@check [L.lam for L in st.lineages] == [0.0, 0.5, 1.0, 0.0] "lineages beyond k cycle the λ grid"
sb = GD.GDState(blocks, 3)   # default objective: bias — unchanged semantics
@check sb.obj === :bias && GD.lineage_total(sb, 0.5, [1.0, NaN, 2.0]) == 3.0 && GD.block_score(sb, 0.5, [1.0, NaN, 2.0], [1, 2]) == 1.0 "bias mode: Σ f_g"

# ---- non-dominated archive crowding ----
nd = [(A=Float64(i), W=Float64(10 - i), ) for i in 0:10]
k = P._nd_most_crowded(nd, 10.0, 10.0)
@check 1 < k < length(nd) "most-crowded removal never drops an extreme"

println("mo_gdigel_unit: $(npass[]) passed, $(nfail[]) failed")
exit(nfail[] == 0 ? 0 : 1)
