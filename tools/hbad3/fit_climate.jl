# HBAD3 viability test of the climate pathway: start from a GA+FL fit (fit3_<tag>), keep every parameter, and fit ONLY 4 coordinates on Region 4 plots
# OUTSIDE GA+FL (R4 split train, states ≠ 12, 13):
#   mode climate: ω (growth × W^ω), mW (drought mortality), τg / τm (temperature on growth / mortality)        — climate structure
#   mode control: level shifts of eD (all species), mbg (all species), ln ms, ln k                              — a regional re-levelling, no climate
# Scores before/after on R4 val (outside GA+FL) and back on GA+FL val. Writes resid_transfer_h3_<tag>_<mode>.csv for scripts/hbad3/transfer_climate.py.
#   HBAD2_FORM=2 HBAD2_SIZE_SPLIT=5,12.7,25,40 HBAD2_SITE=subplot HBAD2_VIGOUR=1 HBAD2_SIGV_MAX=0.5 HBAD3_SFORM=<fit's> ./julia_gdal.sh --project=. --threads=2 tools/hbad3/fit_climate.jl <tag> climate|control
include(joinpath(@__DIR__, "common3.jl")); include(joinpath(@__DIR__, "..", "hbad2", "cmaes.jl"))
tag, mode = ARGS[1], ARGS[2]; mode in ("climate", "control", "both", "anomboth") || error("mode climate|control|both|anomboth")
#   mode anomboth: climate driven by ANOMALIES only (site W := interval AET/PET ÷ the plot's 1991-2020 normal; T neutral) — ω, mW — plus the control shifts (6)
const ANOM = mode == "anomboth"
R4D = joinpath(R2DIR, "..", "hbad2_r4")
OUTD = joinpath(R3DIR, "climfit_" * tag * "_" * mode); mkpath(OUTD)
logio = open(joinpath(OUTD, "log.txt"), "w"); lg(s) = (println(s); println(logio, s); flush(logio); flush(stdout))
D3 = joinpath(R3DIR, "fit3_" * tag)
m = Dict(split(l, "=")[1] => split(l, "=")[2] for l in eachline(joinpath(D3, "meta.txt")) if occursin("=", l))
parse(Int, get(m, "sform", "1")) == SFORM || error("fit $(tag) has sform=$(get(m, "sform", "1")) but HBAD3_SFORM=$(SFORM)")
pv = CSV.read(joinpath(R2DIR, "fit_v2j", "params_species.csv"), DataFrame)
keys_ = sort(String.(pv.key)); lvd = Dict(String(r.key) => Float64(r.longevity) for r in eachrow(pv)); longev = [lvd[k] for k in keys_]; kset = Set(keys_)
sv0 = Matrix(CSV.read(joinpath(D3, "sv_full.csv"), DataFrame)[:, 2:end]); gv0 = vec(CSV.read(joinpath(D3, "gv_full.csv"), DataFrame).value)
length(gv0) < length(GN3) && (gv0 = vcat(gv0, [genc3(GN3[j], GDEFAULT3[GN3[j]]) for j in length(gv0)+1:length(GN3)]))
hd = hd_table(keys_)
# R4 outside GA+FL
tr4 = load_trees(R4D); iv4 = DataFrame(Arrow.Table(joinpath(R4D, "intervals.arrow"))); stof = Dict(r.iid => r.st for r in eachrow(iv4))
tr4 = tr4[[stof[i] ∉ (12, 13) for i in tr4.iid], :]; tr4.key = [s in kset ? s : "_" * g for (s, g) in zip(tr4.sym, tr4.grp)]
function water_rel(path)                # interval AET/PET relative to the plot normal
  w = CSV.read(path, DataFrame); Dict(Int(r.iid) => Float64(r.aetpet_iv / r.aetpet_n) for r in eachrow(w))
end
w4 = ANOM ? water_rel(joinpath(R3DIR, "interval_water_r4.csv")) : water_by_iid(joinpath(R3DIR, "interval_water_r4.csv"))
t4 = ANOM ? nothing : temp_by_iid(joinpath(R3DIR, "interval_climate_r4.csv"))
TR4 = build3(tr4, "train", keys_, w4, t4); VA4 = build3(tr4, "val", keys_, w4, t4)
# GA+FL val (its own product and climate)
trg = load_trees(R2DIR); sort(unique(trg.key)) == keys_ || error("GA+FL keys differ")
VAG = ANOM ? build3(trg, "val", keys_, water_rel(joinpath(R3DIR, "interval_water.csv"))) :
      build3(trg, "val", keys_, water_by_iid(joinpath(R3DIR, "interval_water.csv")), temp_by_iid(joinpath(R3DIR, "interval_climate.csv")))
gi(nm) = findfirst(==(nm), GN3); fi(nm) = findfirst(==(nm), FREE3)
function params(z)
  sv = copy(sv0); gv = copy(gv0)
  if mode in ("climate", "both")
    gv[gi(:ω)] = z[1]; gv[gi(:mW)] = z[2]; gv[gi(:τg)] = z[3]; gv[gi(:τm)] = z[4]
  end
  ANOM && (gv[gi(:ω)] = z[1]; gv[gi(:mW)] = z[2])
  if mode in ("control", "both", "anomboth")
    o = mode == "both" ? 4 : ANOM ? 2 : 0
    sv[:, fi(:eD)] .+= z[o+1]; sv[:, fi(:mbg)] .+= z[o+2]; sv[:, fi(:ms)] .+= z[o+3]; gv[gi(:k)] += z[o+4]
  end
  decode3(sv, gv, hd, longev)
end
z0 = mode == "climate" ? [gv0[gi(:ω)], 0.0, 0.0, 0.0] : mode == "control" ? zeros(4) : ANOM ? [gv0[gi(:ω)], 0.0, 0.0, 0.0, 0.0, 0.0] : [gv0[gi(:ω)], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#   mode both: the climate coordinates + the control shifts (8)
function sc(z, C); P, G = params(z); eval3(C, P, G); end
lg(@sprintf("climate-pathway test %s / %s: R4-outside train %d trees, val %d trees; GA+FL val", tag, mode, sum(TR4.obs.n1), sum(VA4.obs.n1)))
b = (sc(z0, TR4), sc(z0, VA4), sc(z0, VAG))
lg(@sprintf("BEFORE R4 train %.1f | R4 val %.1f (mort %.1f D %.1f H %.1f) | GA+FL val %.1f (mort %.1f D %.1f H %.1f)", b[1].total, b[2].total, b[2].mort, b[2].D, b[2].H, b[3].total, b[3].mort, b[3].D, b[3].H))
res = cmaes(z -> sc(z, TR4).total, z0, 0.1; λ=12, maxevals=parse(Int, get(ENV, "HBAD2_EVALS", "480")), logf=lg, log_every=5)
a = (sc(res.x, TR4), sc(res.x, VA4), sc(res.x, VAG))
lg(@sprintf("AFTER  R4 train %.1f | R4 val %.1f (mort %.1f D %.1f H %.1f) | GA+FL val %.1f (mort %.1f D %.1f H %.1f) | z %s",
            a[1].total, a[2].total, a[2].mort, a[2].D, a[2].H, a[3].total, a[3].mort, a[3].D, a[3].H, string(round.(res.x; digits=4))))
lg(@sprintf("GAIN   R4 val %.1f (mort %.1f D %.1f H %.1f) | GA+FL val %.1f", a[2].total - b[2].total, a[2].mort - b[2].mort, a[2].D - b[2].D, a[2].H - b[2].H, a[3].total - b[3].total))
P, G = params(res.x); lg(string(G))
CSV.write(joinpath(R3DIR, "resid_transfer_h3_" * tag * "_" * mode * ".csv"), export_resid3(VA4, P, G, "val"))
