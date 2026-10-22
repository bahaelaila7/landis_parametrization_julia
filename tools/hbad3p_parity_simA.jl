# End-to-end gate of the HBAD3P plugin on the prototype's Sim A VAL sites (GA+FL): the plugin kernel, fed the SAME site arrays the prototype harness builds (tools/hbad2/common.jl build_cohorts +
# tools/hbad3/lib3p.jl sima_sites: 14,637 subplot VAL sites, farea = 1, FIA crown ratio, period / normal water index), reproduces the prototype's validation negative log-likelihood (log.txt FIT line of
# fit3p_P3_GAFL: VAL -272207.9) with the exported parameters results/hbad3p/P3_GAFL. Also compares plugin vs reference kernel state (N, D, H, CR) site by site and the TRAIN NLL.
#   ./julia_gdal.sh --project=. --threads=2 tools/hbad3p_parity_simA.jl [<prototype root, default ../landis_hbad3>] [<params dir, default results/hbad3p/P3_GAFL>]
# The prototype environment of tools/hbad3/pipeline3p.env with HBAD2_REGION=GAFL is set here (the harness reads it at include time).
const PROTO = normpath(isempty(ARGS) ? joinpath(@__DIR__, "..", "..", "landis_hbad3") : ARGS[1])
const PDIR = normpath(length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "results", "hbad3p", "P3_GAFL"))
isdir(PROTO) || (println("prototype tree $PROTO not found: nothing to compare"); exit(2))
for (k, v) in ("HBAD2_FORM" => "2", "HBAD2_SIZE_SPLIT" => "5,12.7,25,40", "HBAD2_SITE" => "subplot", "HBAD2_VIGOUR" => "1", "HBAD2_SIGV_MAX" => "1.2", "HBAD2_VIG_N" => "5",
               "HBAD2_REGION" => "GAFL", "HBAD2_DATA_DIR" => joinpath(PROTO, "results", "hbad2_r4"), "HBAD3_WATER_CSV" => joinpath(PROTO, "results", "hbad3", "interval_water_r4.csv"),
               "HBAD3_FOREST_CSV" => joinpath(PROTO, "results", "hbad3", "subplot_forest_r4.csv"), "HBAD3P_FAREA" => "1")
  ENV[k] = v
end
import Pan
using Printf, DataFrames, CSV, Arrow
include(joinpath(PROTO, "tools", "hbad3", "common3.jl"))
include(joinpath(PROTO, "src", "plugins", "HBAD3", "HBAD3P.jl")); import .HBAD3P
include(joinpath(PROTO, "tools", "hbad3", "lib3p.jl"))
const H3 = Pan.Plugins.HBAD3PPlugin
const FD = joinpath(PROTO, "results", "hbad3", "fit3p_P3_GAFL")
NFAIL = Ref(0)
check(id, ok, msg) = (println(rpad(ok ? "PASS" : "FAIL", 5), rpad(id, 6), msg); ok || (NFAIL[] += 1); ok)

tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_)
println("region GAFL | keys ", join(keys_, ","), " | tracked trees ", nrow(tr), " | threads ", Threads.nthreads())
T = interval_tables(keys_); wp = water_pairs()
IFOR = interval_forest(T.iv, forest_table())          # farea = 1
M = load_fit3p(FD, keys_)                              # the prototype's own parameters (reference kernel)

# plugin parameters from the EXPORTED directory (not from the fit dir)
pcfg = Pan.PluginConfig(H3.HBAD3PContext(keys_, [collect(1:nsp)]; params_dir=PDIR, verbose=false))
ep = H3.identity_eco_params(pcfg.ctx)
check("P1", H3.identity_eco_params(pcfg.ctx).sp == [H3.SpPP(p.lnε, p.Hmax, p.ST, p.longevity, p.lnF) for p in ep.sp] && length(ep.sp) == nsp &&
            all(k -> ep.sp[k].lnε == M.P[k].lnε && ep.sp[k].Hmax == M.P[k].Hmax && ep.sp[k].ST == M.P[k].ST && ep.sp[k].longevity == M.P[k].longevity, 1:nsp) &&
            all(f -> getfield(ep.G, f) == getfield(M.G, f), fieldnames(H3.GlobPP)),
      "exported parameters equal the prototype's load_fit3p species traits and all 32 globals exactly")

"Run the plugin kernel over the sites of `C` (prototype sima_sites result); returns the SoA."
function plugin_run!(C)
  S = C.S; ns = length(S.off) - 1
  soa = Pan.make_sites_hbad3p_arrays(pcfg, S.off, S.sp, S.A0, S.N0, S.D0, S.H0, S.zq, S.CR0; rng=Pan.RNGType(1), no_establishment=true, planted=S.planted, W=S.W, Wn=S.Wn)
  soa.parallel = false
  Threads.@threads :static for i in 1:ns
    H3.run_interval!(soa, i, ep, S.nyr[i], S.frac[i])
  end
  soa
end
"Copy the plugin's final state into the prototype's arrays (same cohort order) so the prototype likelihood can score it."
function to_proto!(S, soa)
  for s in 1:length(S.off)-1
    o = Int(soa.refs.cohort[s]) - 1
    for (q, i) in enumerate(S.off[s]:S.off[s+1]-1)
      S.N[i] = soa.csr.c_tph[o+q]; S.D[i] = soa.csr.c_dq[o+q]; S.H[i] = soa.csr.c_h[o+q]; S.CR[i] = soa.csr.c_cr[o+q]; S.A[i] = soa.csr.c_age[o+q]
    end
  end
  S
end

lg = read(joinpath(FD, "log.txt"), String); mt = match(r"FIT   train (-?[0-9.]+) .*VAL (-?[0-9.]+) \(mort (-?[0-9.]+) D (-?[0-9.]+) H (-?[0-9.]+)", lg)
mt === nothing && error("no FIT line in $(FD)/log.txt")
logtrain, logval = parse(Float64, mt[1]), parse(Float64, mt[2])
for (set, logv) in (("val", logval), ("train", logtrain))
  C = sima_sites(tr, set, keys_, wp, IFOR); S = C.S
  ns = length(S.off) - 1
  # reference kernel on a copy of the arrays
  CR_ = sima_sites(tr, set, keys_, wp, IFOR); HBAD3P.simulate!(CR_.S, M.P, M.G); nref = nll_parts(cohort_view(CR_), CR_.obs)
  soa = plugin_run!(C); to_proto!(C.S, soa); npl = nll_parts(cohort_view(C), C.obs)
  dN = maximum(abs.(C.S.N .- CR_.S.N) ./ max.(CR_.S.N, 1e-300)); dD = maximum(abs.(C.S.D .- CR_.S.D) ./ CR_.S.D); dH = maximum(abs.(C.S.H .- CR_.S.H) ./ CR_.S.H); dC = maximum(abs.(C.S.CR .- CR_.S.CR))
  @printf("      %s: %d sites, %d sub-cohorts | log FIT %.1f | reference NLL %.6f (mort %.3f D %.3f H %.3f) | plugin NLL %.6f (mort %.3f D %.3f H %.3f)\n",
          set, ns, length(S.sp), logv, nref.total, nref.mort, nref.D, nref.H, npl.total, npl.mort, npl.D, npl.H)
  check("E-$(set)1", abs(npl.total - logv) <= 1e-6 * abs(logv), @sprintf("plugin %s NLL %.4f vs logged FIT %.1f: relative difference %.3e (<= 1e-6)", set, npl.total, logv, abs(npl.total - logv) / abs(logv)))
  check("E-$(set)2", abs(npl.total - nref.total) <= 1e-12 * abs(nref.total), @sprintf("plugin %s NLL = reference kernel NLL: relative difference %.3e (<= 1e-12)", set, abs(npl.total - nref.total) / abs(nref.total)))
  check("E-$(set)3", max(dN, dD, dH) <= 1e-12 && dC <= 1e-12, @sprintf("plugin vs reference final state over %d sub-cohorts: max rel diff N %.2e D %.2e H %.2e, max abs diff CR %.2e", length(S.sp), dN, dD, dH, dC))
end
println(NFAIL[] == 0 ? "PARITY PASS" : "PARITY FAIL ($(NFAIL[]) checks)")
exit(NFAIL[])
