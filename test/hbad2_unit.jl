# HBAD v2 kernel unit tests: ./julia_gdal.sh --project=. --threads=2 test/hbad2_unit.jl  (also run with --threads=1)
using Test, Random
include(joinpath(@__DIR__, "..", "src", "plugins", "HBAD2", "HBAD2.jl")); using .HBAD2
sp(; kw...) = SpP((get(kw, nm, d) for (nm, d) in zip(HBAD2.SP_PARAM_NAMES, (35.0, 0.03, 1.2, 70.0, 0.02, 1.3, 1.5, 0.0, 0.7, 0.3, -5.0, -0.3, 0.03, 600.0, 300.0)))...)
P = [sp(), sp(Hmax=28.0, rH=0.02, gOD=0.6, sdimax=900.0), sp(Hmax=40.0, rD=0.03, cD=2.0)]
G = HBAD2.GlobP(0.2, 10.0, 0.05, 0.05, -0.2)
function mk(nsite, rng; ncoh=6)
  off = [1]; spv = Int[]; N = Float64[]; D = Float64[]; H = Float64[]; A = Float64[]
  for s in 1:nsite
    for c in 1:rand(rng, 1:ncoh)
      push!(spv, rand(rng, 1:3)); push!(N, rand(rng) * 800 + 5); d = 2.54 + rand(rng) * 50; push!(D, d); push!(H, 1.3 + 30 * (1 - exp(-0.04d))); push!(A, 5 + rand(rng) * 80)
    end
    push!(off, length(spv) + 1)
  end
  Sites(off, spv, N, D, H, A, rand(rng, 3:8, nsite), rand(rng, nsite), rand(rng, Bool, nsite))
end
@testset "HBAD2 kernel" begin
  S = mk(500, Xoshiro(1)); simulate!(S, P, G)
  @test all(S.N .<= S.N0 .* (1 + 1e-12))                     # TPA only decreases
  @test all(S.D .>= S.D0) && all(S.H .>= S.H0)               # sizes never shrink (no negative increments)
  @test all(isfinite, S.N) && all(isfinite, S.D) && all(isfinite, S.H)
  @test all(S.A .≈ S.A0 .+ [S.nyr[s] + S.frac[s] for s in 1:length(S.off)-1 for _ in S.off[s]:S.off[s+1]-1])
  # self-thinning cap: final RD ≤ 1
  rd = map(1:length(S.off)-1) do s
    lo, hi = S.off[s], S.off[s+1] - 1; t = sum(S.N[i] * (S.D[i] / 25.4)^1.605 for i in lo:hi)
    t / (sum(S.N[i] * (S.D[i] / 25.4)^1.605 * P[S.sp[i]].sdimax for i in lo:hi) / t)
  end
  @test maximum(rd) <= 1 + 1e-4
  # determinism / thread independence: rerun gives the identical state; serial run equals threaded run
  N1 = copy(S.N); D1 = copy(S.D); simulate!(S, P, G); @test S.N == N1 && S.D == D1
  S2 = mk(500, Xoshiro(1))
  for s in 1:length(S2.off)-1
    lo, hi = S2.off[s], S2.off[s+1] - 1
    for _ in 1:S2.nyr[s]; HBAD2.step_site!(S2, P, G, lo, hi, 1.0, S2.planted[s]); end
    S2.frac[s] > 0 && HBAD2.step_site!(S2, P, G, lo, hi, S2.frac[s], S2.planted[s])
  end
  @test S2.N == N1 && S2.D == D1 && S2.H == S.H
  # allocation-free annual step
  alloc_step(S, P, G, lo, hi) = (HBAD2.step_site!(S, P, G, lo, hi, 1.0, false); @allocated HBAD2.step_site!(S, P, G, lo, hi, 1.0, false))
  @test alloc_step(S, P, G, S.off[3], S.off[4] - 1) == 0
  simulate!(S, P, G); a = @allocated simulate!(S, P, G); @test a < 20_000          # only the task spawn of @threads
  # free growth of an open-grown single cohort follows the Chapman–Richards curve (no competition, tiny density)
  p0 = sp(gOD=0.0, gOH=0.0, gRD=0.0, gRH=0.0, mbg=-20.0, msup=0.0)
  H0 = 35.0 * (1 - exp(-0.03 * 10))^1.2; S1 = Sites([1, 2], [1], [1e-6], [70.0 * (1 - exp(-0.02 * 10))^1.3], [H0], [10.0], [20], [0.0], [false])
  simulate!(S1, [p0], HBAD2.GlobP(0.0, 10.0))
  @test isapprox(S1.H[1], 35.0 * (1 - exp(-0.03 * 30))^1.2; rtol=0.02)               # Euler steps: within 2 % of the analytic curve
  @test isapprox(S1.D[1], 70.0 * (1 - exp(-0.02 * 30))^1.3; rtol=0.02)
  # fractional step: 2 × half-year ≈ one full year (first order)
  Sa = Sites([1, 2], [1], [500.0], [20.0], [15.0], [20.0], [1], [0.0], [false]); Sb = Sites([1, 2], [1], [500.0], [20.0], [15.0], [20.0], [0], [0.5], [false])
  simulate!(Sa, P, G); simulate!(Sb, P, G); @test isapprox(Sb.D[1] - 20.0, 0.5 * (Sa.D[1] - 20.0); rtol=1e-9)
  # suppression: an overtopped cohort grows less in D and dies more than the same cohort alone
  Sx = Sites([1, 3], [1, 1], [300.0, 300.0], [40.0, 15.0], [30.0, 12.0], [60.0, 30.0], [5], [0.0], [false])
  Sy = Sites([1, 2], [1], [300.0], [15.0], [12.0], [30.0], [5], [0.0], [false])
  simulate!(Sx, P, G); simulate!(Sy, P, G)
  @test Sx.D[2] - 15 < Sy.D[1] - 15 && Sx.N[2] < Sy.N[1]
end
