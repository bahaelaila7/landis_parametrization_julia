# HBAD3 transfer test: a model fitted on GA+FL (v2 fit_<tag> or HBAD3 fit3_<tag>) scored WITHOUT refitting on Region 4 plots outside GA+FL
# (states ≠ 12, 13; R4 split set `val`, never used by any fit; R4 test stays untouched). The model keeps its GA+FL species keys (other species → _H/_S by
# softwood/hardwood) and its key longevity; HBAD3 models get the R4 water index. Sim A likelihood parts per EPA L3 and overall (σ profiled per subset).
#   HBAD2_DATA_DIR=results/hbad2_r4 HBAD3_WATER_CSV=results/hbad3/interval_water_r4.csv HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot HBAD2_VIGOUR=1
#   HBAD2_SIGV_MAX=<fit's> [HBAD3_SFORM=<fit's>] ./julia_gdal.sh --project=. --threads=4 tools/hbad3/transfer_eval.jl v2:v2fv5 | h3:H3b  [set=val]
include(joinpath(@__DIR__, "common3.jl"))
kind, tag = split(ARGS[1], ":"); SET = length(ARGS) >= 2 ? ARGS[2] : "val"
pv = CSV.read(joinpath(R2DIR, "fit_v2j", "params_species.csv"), DataFrame)        # GA+FL keys and key longevity (identical for every GA+FL fit)
keys_ = sort(String.(pv.key)); lvd = Dict(String(r.key) => Float64(r.longevity) for r in eachrow(pv)); longev = [lvd[k] for k in keys_]
kset = Set(keys_)
tr = load_trees()
iv = DataFrame(Arrow.Table(joinpath(DATA_DIR, "intervals.arrow"))); stof = Dict(r.iid => r.st for r in eachrow(iv))
tr = tr[[stof[i] ∉ (12, 13) for i in tr.iid], :]
tr.key = [s in kset ? s : "_" * g for (s, g) in zip(tr.sym, tr.grp)]
println("transfer set: states ≠ 12,13, R4 split '$(SET)': ", count(==(SET), tr.set), " tracked trees; model $(kind):$(tag)")
"v2 per-cohort predictions (as tools/hbad3/export_resid.jl)."
function export_resid_v2(C, P, G)
  S0 = Sites(C.S.off, C.S.sp, C.S.N0, C.S.D0, C.S.H0, C.S.A0, zeros(Int, length(C.S.nyr)), fill(1e-9, length(C.S.nyr)), C.S.planted, C.S.zq)
  simulate!(S0, P, G); O0 = copy(S0.O); RD0 = zeros(length(O0))
  for s in 1:length(C.S.off)-1
    lo, hi = C.S.off[s], C.S.off[s+1] - 1; tot = sum(S0.S[lo:hi]); sm = sum(S0.S[i] * P[C.S.sp[i]].sdimax for i in lo:hi) / tot
    RD0[lo:hi] .= tot / sm
  end
  simulate!(C.S, P, G); V = cohort_view(C); o = C.obs; fo = [searchsortedfirst(C.parent, p) for p in 1:length(o.n1)]
  DataFrame(set=SET, site=C.coh.site, iid=C.coh.site .÷ 10, key=C.coh.key, planted=C.coh.planted, dt=o.dt, n1=o.n1, s=o.s, N1=V.N0, D1=V.D0, H1=V.H0, A1=C.S.A0[fo],
            O=O0[fo], RD=RD0[fo], ps=V.N ./ V.N0, D2s=V.D, H2s=V.H, D2o=o.D2o, H2o=o.H2o)
end
if kind == "v2"
  check_meta(joinpath(R2DIR, "fit_" * tag)); θ = read_theta(joinpath(R2DIR, "fit_" * tag, "theta.csv"), length(keys_)); P, G = decode(θ, length(keys_), longev)
  run(C) = (simulate!(C.S, P, G); nll_parts(cohort_view(C), C.obs))
  resid(C) = export_resid_v2(C, P, G)
  build(t) = build_cohorts(t, SET, keys_)
else
  D3 = joinpath(R3DIR, "fit3_" * tag)
  m = Dict(split(l, "=")[1] => split(l, "=")[2] for l in eachline(joinpath(D3, "meta.txt")) if occursin("=", l))
  parse(Int, get(m, "sform", "1")) == SFORM || error("fit $(tag) has sform=$(get(m, "sform", "1")) but HBAD3_SFORM=$(SFORM)")
  sv = Matrix(CSV.read(joinpath(D3, "sv_full.csv"), DataFrame)[:, 2:end]); gv = vec(CSV.read(joinpath(D3, "gv_full.csv"), DataFrame).value)
  length(gv) < length(GN3) && (gv = vcat(gv, [genc3(GN3[j], GDEFAULT3[GN3[j]]) for j in length(gv)+1:length(GN3)]))
  P, G = decode3(sv, gv, hd_table(keys_), longev)
  wd = water_by_iid(); water = get(m, "water", "1") == "1"
  run(C) = eval3(C, P, G)
  resid(C) = export_resid3(C, P, G, SET)
  build(t) = (C = build3(t, SET, keys_, wd); water || set_water!(C, 1.0); C)
end
out = DataFrame(model=String[], l3=String[], ncoh=Int[], ntrees=Int[], total=Float64[], mort=Float64[], D=Float64[], H=Float64[], sigD=Float64[], sigH=Float64[])
for l in vcat("ALL", sort(unique(tr.l3[tr.set .== SET])))
  t = l == "ALL" ? tr : tr[tr.l3 .== l, :]
  count(==(SET), t.set) < 500 && continue
  r = run(build(t))
  push!(out, (ARGS[1], l, r.ncoh, r.ntrees, r.total, r.mort, r.D, r.H, r.sigD, r.sigH))
  @printf("%-10s %-7s cohorts %6d trees %7d | total %10.1f mort %9.1f D %10.1f H %10.1f σD %.4f σH %.4f\n", ARGS[1], l, r.ncoh, r.ntrees, r.total, r.mort, r.D, r.H, r.sigD, r.sigH)
end
# per-cohort predictions of the whole transfer set (for the climate-class diagnostics: scripts/hbad3/transfer_climate.py)
CSV.write(joinpath(R3DIR, "resid_transfer_" * replace(ARGS[1], ":" => "_") * ".csv"), resid(build(tr)))
mkpath(joinpath(R3DIR, "transfer")); CSV.write(joinpath(R3DIR, "transfer", replace(ARGS[1], ":" => "_") * "_" * SET * ".csv"), out)
