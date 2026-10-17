# HBAD2 PARITY GATE: the Pan plugin (src/plugins/HBAD2) against the PROTOTYPE kernel of branch hbad-v2 (read-only), cohort by cohort, year by year.
#   PAN_DATA=/workspace/landis_region4/data_r4 ./julia_gdal.sh --project=. --threads=2 tools/hbad2_parity.jl [--proto /workspace/landis_hbad2] [--years 25] [--nsimA 200] [--db <slim_r4.duckdb>]
# Runs (state compared per site, per cohort, after EVERY year; max relative difference reported separately for N, D, H, A):
#   B  : Sim B VAL sites of the prototype (build_simB: clean natural plots, 4 subplot sites each, vigour sub-cohorts, recruitment), `years` years: prototype run_free! vs Pan on the same initial arrays
#   B' : the same, with the Pan initial state built by Data.load_tree_cohorts from the database (matched to the prototype's cohort order); also the loader's initial state vs the prototype's
#   A  : Sim A style (no recruitment): `nsimA` VAL sites of the tracked-tree cohorts (planted stands included), whole annual steps of the prototype step_site! vs Pan
# Nothing under --proto is modified (it is only include()d / its arrow files read).
using Pan, Random, DataFrames, Dates, SHA, Printf, Arrow
function argval(name, default); i = findfirst(==(name), ARGS); i === nothing ? default : ARGS[i+1]; end
const PROTO = argval("--proto", "/workspace/landis_hbad2")
const YEARS = parse(Int, argval("--years", "25"))
const NSIMA = parse(Int, argval("--nsimA", "200"))
const DB = argval("--db", joinpath(get(ENV, "PAN_DATA", "/workspace/landis_region4/data_r4"), "slim_r4.duckdb"))
const REPO = normpath(joinpath(@__DIR__, ".."))

# the prototype harness reads its structural switches from the environment (fit v2g: meta.txt)
ENV["HBAD2_FORM"] = "2"; ENV["HBAD2_SIZE_SPLIT"] = "5,12.7,25,40"; ENV["HBAD2_SITE"] = "subplot"; ENV["HBAD2_VIGOUR"] = "1"; ENV["HBAD2_SIGV_MAX"] = "0.7"
module ProtoTools
  using Arrow, DataFrames, CSV, Statistics, Printf, Dates, Random, LinearAlgebra
  proto_dir = Main.PROTO
  include(joinpath(proto_dir, "tools", "hbad2", "common.jl"))
  include(joinpath(proto_dir, "tools", "hbad2", "simB_lik.jl"))
end
const PT = ProtoTools
const PR = PT.HBAD2
include(joinpath(REPO, "test", "hbad2_parity_lib.jl"))

println("== files")
sha(f) = bytes2hex(open(sha256, f))
pf = joinpath(PROTO, "results", "hbad2", "fit_v2g"); mine = HB2.DEFAULT_PARAMS_DIR
for f in ("params_species.csv", "params_global.csv", "params_recruitment.csv", "meta.txt")
  @printf("%-24s prototype %s  pan %s  %s\n", f, sha(joinpath(pf, f))[1:16], sha(joinpath(mine, f))[1:16], sha(joinpath(pf, f)) == sha(joinpath(mine, f)) ? "IDENTICAL" : "DIFFERENT")
end
@printf("kernel: prototype HBAD2.jl %s  vendored test/ref_hbad2_proto.jl %s\n", sha(joinpath(PROTO, "src", "plugins", "HBAD2", "HBAD2.jl"))[1:16], sha(joinpath(REPO, "test", "ref_hbad2_proto.jl"))[1:16])

P, G, R, keys_ = proto_params(PR, pf)
nsp = length(keys_)
ctx = HB2.HBAD2Context(keys_, [collect(1:nsp)]; params_dir=mine, verbose=false)
pcfg = Pan.PluginConfig(ctx)
println("keys: ", join(keys_, ","), "  | threads=", Threads.nthreads())

# ------------------------------------------------------------------------------------------------------------------ Sim B
println("\n== Sim B (VAL, recruitment, $(YEARS) years)")
t0 = time(); B = PT.build_simB("val", keys_); S = B.F.S
initB = (off=S.off, sp=Int.(B.sp0), N0=B.N0, D0=B.D0, H0=B.H0, A0=B.A0, zq0=B.zq0, live0=B.live0, planted=fill(false, length(B.live0)))
ns = length(B.live0)
@printf("sites %d, initial cohorts %d (live), capacity per site %d, build %.1fs\n", ns, sum(B.live0), S.off[2] - S.off[1], time() - t0)
t0 = time(); recP, dropped = proto_free(PR, initB, P, G, R, YEARS); @printf("prototype run %.1fs (dropped recruits %d)\n", time() - t0, dropped)
t0 = time(); soaB = pan_soa(pcfg, initB; no_establishment=false); recB = pan_run(pcfg, soaB, YEARS); @printf("Pan run %.1fs\n", time() - t0)
cB = compare_rec(recP, recB)
println("B  prototype vs Pan (same initial arrays): ", cB)
nfin = sum(length(recP[YEARS][s][1]) for s in 1:ns)
@printf("   final cohorts per site: mean %.1f max %d (initial live mean %.1f)\n", nfin / ns, maximum(length(recP[YEARS][s][1]) for s in 1:ns), sum(B.live0) / ns)

# ---- loader: reconstruct the visits of the prototype's sites (same selection code as build_simB)
iv = DataFrame(Arrow.Table(joinpath(PT.R2DIR, "intervals.arrow"))); raw = DataFrame(Arrow.Table(joinpath(PT.R2DIR, "interval_trees.arrow")))
iv.pk = [(a, b, c, d) for (a, b, c, d) in zip(iv.st, iv.u, iv.c, iv.p)]
pl = combine(groupby(iv, :pk), :set => first => :set, [:trt, :dst, :harv, :planted] => ((a, b, c, d) -> !any(a .| b .| c .| d)) => :clean, nrow => :nint)
pl = pl[(coalesce.(pl.set, "") .== "val") .& pl.clean .& (pl.nint .>= 2), :]
rbyi = groupby(raw, :iid); ivbypk = groupby(iv, :pk)
psite_visit = DataFrame(statecd=Int[], unitcd=Int[], countycd=Int[], plot=Int[], measdate=Date[]); psite_sb = Int[]
for r in eachrow(pl)
  ivs = sort(DataFrame(ivbypk[(r.pk,)]), :k); all(i -> haskey(rbyi, (i,)), ivs.iid) || continue
  for sb in 1:4
    push!(psite_visit, (r.pk[1], r.pk[2], r.pk[3], r.pk[4], Date(ivs.d1[1]))); push!(psite_sb, sb)
  end
end
@assert nrow(psite_visit) == ns "reconstructed $(nrow(psite_visit)) sites, build_simB has $ns"
t0 = time(); tc = Pan.Data.load_tree_cohorts(DB, unique(psite_visit), keys_; birth_w=PT.BIRTH_W, size_breaks=PT.SIZE_BREAKS, vigour=true)
@printf("loader: %d visits -> %d sites, %d cohorts (vigour-expanded), %d trees, %.1fs\n", nrow(tc.visits), length(tc), length(tc.N), tc.n_trees, time() - t0)
vkey(v) = (Int(v.statecd), Int(v.unitcd), Int(v.countycd), Int(v.plot), Date(v.measdate))
lsite = Dict((vkey(tc.visits[tc.site_visit[s], :]), tc.site_subp[s]) => s for s in 1:length(tc))
# match the loader's cohorts to the prototype's (per site: same species + vigour, nearest state) and put the loader's values in the prototype's order
maxd = zeros(4); unmatched = 0; nmatch = 0; sitemis = 0
N1 = copy(B.N0); D1 = copy(B.D0); H1 = copy(B.H0); A1 = copy(B.A0)         # the loader's values at the prototype's positions
for s in 1:ns
  ls = lsite[(vkey(psite_visit[s, :]), psite_sb[s])]
  plo = S.off[s]; phi = plo + B.live0[s] - 1; llo = tc.off[ls]; lhi = tc.off[ls+1] - 1
  global sitemis, unmatched, nmatch, maxd
  if phi - plo != lhi - llo; sitemis += 1; continue; end
  used = falses(lhi - llo + 1)
  for i in plo:phi
    best = 0; bd = Inf
    for (q, c) in enumerate(llo:lhi)
      (used[q] || Int(tc.key[c]) != B.sp0[i] || tc.zq[c] != B.zq0[i]) && continue
      d = max(_rel(tc.N[c], B.N0[i]), _rel(tc.D[c], B.D0[i]), _rel(tc.H[c], B.H0[i]), _rel(tc.A[c], B.A0[i]))
      d < bd && (bd = d; best = q)
    end
    if best == 0; unmatched += 1; continue; end
    used[best] = true; c = llo + best - 1; nmatch += 1
    maxd .= max.(maxd, (_rel(tc.N[c], B.N0[i]), _rel(tc.D[c], B.D0[i]), _rel(tc.H[c], B.H0[i]), _rel(tc.A[c], B.A0[i])))
    N1[i] = tc.N[c]; D1[i] = tc.D[c]; H1[i] = tc.H[c]; A1[i] = tc.A[c]
  end
end
println("loader vs prototype build_simB initial state: sites with a different cohort count $sitemis; cohorts matched $nmatch, unmatched $unmatched; ",
        @sprintf("max rel diff N %.3g D %.3g H %.3g A %.3g", maxd...))
initB2 = merge(initB, (N0=N1, D0=D1, H0=H1, A0=A1))
cB2 = compare_rec(recP, pan_run(pcfg, pan_soa(pcfg, initB2; no_establishment=false), YEARS))
println("B' prototype vs Pan from the loader's initial state: ", cB2)

# ------------------------------------------------------------------------------------------------------------------ Sim A
println("\n== Sim A style (no recruitment; $(NSIMA) VAL sites of the tracked-tree cohorts)")
t0 = time(); tr = PT.load_trees(); @assert sort(unique(tr.key)) == keys_
C = PT.build_cohorts(tr, "val", keys_); SA = C.S; nA = PT.HBAD2.nsites(SA)
@printf("Sim A val: %d sites, %d cohorts (vigour-expanded), build %.1fs\n", nA, length(SA.sp), time() - t0)
pick = unique(round.(Int, range(1, nA; length=NSIMA)))
offA = [1]; spA = Int[]; NA = Float64[]; DA = Float64[]; HA = Float64[]; AA = Float64[]; zA = Float64[]; liveA = Int[]
for s in pick
  r = SA.off[s]:SA.off[s+1]-1
  append!(spA, SA.sp[r]); append!(NA, SA.N0[r]); append!(DA, SA.D0[r]); append!(HA, SA.H0[r]); append!(AA, SA.A0[r]); append!(zA, SA.zq[r]); push!(liveA, length(r)); push!(offA, length(spA) + 1)
end
initA = (off=offA, sp=spA, N0=NA, D0=DA, H0=HA, A0=AA, zq0=zA, live0=liveA, planted=SA.planted[pick])
nyr = SA.nyr[pick]
@printf("picked %d sites: %d cohorts, %d planted stands, years/site %d..%d (whole years; the prototype's fractional last step is not part of Pan's annual step)\n", length(pick), length(spA), count(initA.planted), minimum(nyr), maximum(nyr))
recPA = proto_simA(PR, initA, P, G, nyr)
recBA = pan_run(pcfg, pan_soa(pcfg, initA; no_establishment=true), maximum(nyr); nyr=nyr)
cA = compare_rec(recPA, recBA; nyr=nyr)
println("A  prototype vs Pan: ", cA)
# the prototype's own simulate! (whole years, frac = 0) must equal the per-year loop used above
SAc = PR.Sites(copy(initA.off), Int32.(initA.sp), copy(initA.N0), copy(initA.D0), copy(initA.H0), copy(initA.A0), copy(nyr), zeros(length(pick)), copy(initA.planted), copy(initA.zq0))
PR.simulate!(SAc, P, G)
println("prototype simulate! == per-year step_site! loop: ", all(s -> SAc.N[initA.off[s]:initA.off[s]+initA.live0[s]-1] == recPA[nyr[s]][s][2], eachindex(pick)))

println("\n== SUMMARY")
@printf("B  (Sim B, %d sites, %d years, %d cohort-states): max rel diff = %.3g (N %.3g D %.3g H %.3g A %.3g), structural mismatches %d\n", ns, YEARS, cB.cohorts, cB.maxrel, cB.maxrel_N, cB.maxrel_D, cB.maxrel_H, cB.maxrel_A, cB.mismatches)
@printf("B' (from loader state): max rel diff = %.3g, structural mismatches %d; loader initial state max rel diff %.3g\n", cB2.maxrel, cB2.mismatches, maximum(maxd))
@printf("A  (Sim A, %d sites, %d site-years, %d cohort-states): max rel diff = %.3g (N %.3g D %.3g H %.3g A %.3g), structural mismatches %d\n", length(pick), cA.site_years, cA.cohorts, cA.maxrel, cA.maxrel_N, cA.maxrel_D, cA.maxrel_H, cA.maxrel_A, cA.mismatches)
ok = cB.maxrel <= 1e-12 && cA.maxrel <= 1e-12 && cB.mismatches == 0 && cA.mismatches == 0
println(ok ? "PARITY PASS (max relative difference <= 1e-12)" : "PARITY FAIL")
exit(ok ? 0 : 1)
