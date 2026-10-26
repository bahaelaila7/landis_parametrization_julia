# Closure checks of the HBAD4P kernel (truncated-exponential two-moment upgrowth): closed-form partial moments vs grid integration, the inversion
# q2 -> a -> q2, conservation of N and B, mover QMDs inside their classes, and a Monte-Carlo replay of the closure (sample trees, grow, bin).
#   ./julia_gdal.sh --project=. tools/hbad4/test_closure4.jl      (exit 1 on a failure)
include(joinpath(@__DIR__, "..", "..", "src", "plugins", "HBAD4", "HBAD4P.jl")); import .HBAD4P as H; using Random, Statistics
fails = 0; ck(nm, ok, d="") = (println(ok ? "PASS  " : "FAIL  ", nm, "  ", d); ok || (global fails += 1))
worst = 0.0
for a in (-150.0, -30.0, -3.0, -0.5, -0.004, 0.0, 0.003, 0.7, 4.0, 40.0, 160.0)
  g = range(0, 1; length=400_001); pdf = abs(a) < 1e-12 ? ones(length(g)) : exp.(a .* (g .- (a > 0 ? 1 : 0))); w = pdf ./ sum(pdf)
  for ξ in (0.1, 0.5, 0.93)
    F, M1, M2 = H.texp_cum(a, ξ); m = g .<= ξ
    global worst = max(worst, abs(F - sum(w[m])), abs(M1 - sum(w[m] .* g[m])), abs(M2 - sum(w[m] .* g[m] .^ 2)))
  end
end
ck("closed-form partial moments = grid integration (|a| ≤ 160)", worst < 2e-4, "max abs err $(round(worst; sigdigits=3)) (grid step 2.5e-6)")
err = 0.0
for j in 1:15, r in (0.01, 0.1, 0.3, 0.5, 0.7, 0.95, 0.995)
  q = H.CLS_LO[j] + r * H.CLS_W; a = H.texp_a(j, q^2); global err = max(err, abs(H.texp_r(a, H.CLS_LO[j] / H.CLS_W) - r))
end
ck("closure inversion r -> a -> r over every class, r ∈ [0.01, 0.995]", err < 1e-4, "max $(round(err; sigdigits=3))")
rng = Xoshiro(1)
for (j, r, dD) in ((3, 0.15, 0.4), (3, 0.85, 0.4), (1, 0.02, 0.3), (5, 0.5, 7.0), (14, 0.6, 9.0), (2, 0.97, 0.16), (8, 0.5, 0.0001))
  n = 100.0; q = H.CLS_LO[j] + r * H.CLS_W; b = H.ba(n, q); a = H.texp_a(j, q^2)
  out = Dict{Int,Tuple{Float64,Float64}}(); Btot = H.upgrowth!((t, Nt, Bt, Dt) -> (out[t] = get(out, t, (0.0, 0.0)) .+ (Nt, Bt)), j, n, b, dD)
  u = rand(rng, 1_000_000); ξ = abs(a) < 1e-9 ? u : (a > 0 ? 1 .+ log.(u .* (1 - exp(-a)) .+ exp(-a)) ./ a : log.(1 .+ u .* (exp(a) - 1)) ./ a)
  D = H.CLS_LO[j] .+ H.CLS_W .* ξ .+ dD; cl = H.dclass.(D)
  Nsum = sum(first.(values(out))); Bsum = sum(last.(values(out)))
  inb = all(t -> t == j || (H.CLS_LO[t] - 1e-9 <= H.qmd(out[t]...) <= H.CLS_HI[t] + 1e-9), keys(out))
  mcN = maximum(abs(out[t][1] - n * mean(cl .== t)) for t in keys(out)); mcB = maximum(abs(out[t][2] - H.ba(1.0, 1.0) * n * mean((cl .== t) .* D .^ 2)) for t in keys(out))
  ck("upgrowth j=$j r=$r dD=$dD: N, B conserved; movers inside their class; = Monte Carlo of the closure", abs(Nsum - n) < 1e-9 && abs(Bsum - Btot) < 1e-9 && inb && mcN < 0.15,
     "targets $(sort(collect(keys(out)))) | max |N − MC| $(round(mcN; digits=3)) (MC sd ≈ 0.01–0.05)")
end
for mode in (:sym, :fvs)
  z, w = H.lanes(mode); ck("lanes $mode: weights 1, mean 0, variance 1", abs(sum(w) - 1) < 1e-12 && abs(sum(w .* z)) < 1e-9 && abs(sum(w .* z .^ 2) - 1) < 1e-9, "$(round.(z; digits=4))")
end
ck("dclass at exact bounds (12.7 → 3, 78.74 → 16, 2.54 → 1, 2.53 → 0)", H.dclass(12.7) == 3 && H.dclass(78.74) == 16 && H.dclass(2.54) == 1 && H.dclass(2.53) == 0)
exit(fails > 0 ? 1 : 0)
