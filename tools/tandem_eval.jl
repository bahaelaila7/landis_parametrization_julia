# Held-out evaluation of an in-tandem (gdigel objective tandem) archive: for a subset of the ND front, Sim A group TOST (paired, as
# tools/tost_common.jl) and Sim B plot-cluster bootstrap of log Σsim/Σobs per group + total-AGB bias per FIA cycle — on VAL (decides)
# and TEST (information only). Sim B is built exactly as the run builds it: Sim B table on Sim A's val/test plots, relabelled into
# Sim A's layout, coarse tier-4 bins, simB_* settings of the config.
#   ./julia_gdal.sh --project=. --threads=8 tools/tandem_eval.jl <tandem config.yml> <run dir> <out.csv> [n_points=15] [b_reps=4]
import YAML
cfg = YAML.load_file(ARGS[1]); rundir, outcsv = ARGS[2], ARGS[3]
NPTS = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 15; BREPS = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 4
SPLIT = get(ENV, "TANDEM_SPLIT", "val+test"); MARGINS = [0.05, 0.1, 0.2]
using Printf
include(joinpath(@__DIR__, "tost_common.jl"))   # Sim A contexts (CTXS for val, test), _paired_one, group_stats, boot_ratio, grp, label
const FT = P.FloatType
gdo = get(cfg, "gdigel", Dict{String,Any}())
arch = CSV.read(joinpath(rundir, "mo_archive.csv"), DF.DataFrame); DF.sort!(arch, :sumA)
# front subset: both extremes, the knee, and evenly spaced points along the local axis
pick = unique(vcat(1, DF.nrow(arch), findall(r -> occursin("knee", string(coalesce(r.role, ""))), eachrow(arch)),
                   round.(Int, range(1, DF.nrow(arch); length=min(NPTS, DF.nrow(arch))))))
sort!(pick)
P.SIMB_SPINUP[] = Bool(get(cfg, "simB_spinup", true)); P.SIMB_DISTURB_ONLY[] = Bool(get(cfg, "simB_disturb_only", false))
P.T4_PER_PLOT_MEAN[] = Bool(get(cfg, "simB_per_plot_mean", false)); P.T4_MEAN_OVER_REPS[] = Bool(get(cfg, "simB_rep_mean", false)); P.TIER_B[] = Int(get(cfg, "tier_b", 4))
P.ACTIVE_SUCCESSION[] = P.succession_plugin_for(JLD2.load_object(joinpath(rundir, "mo_archive", arch.file[1])))
bins_b = Int.(get(gdo, "tandem_b_bins", [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]))
lpB = PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins_b, last_bin_open=true), smoothing_weights=FT[one(FT)])
dbB = String(get(gdo, "tandem_b_db", "/workspace/landis_region4/data_r4/slim_r4.duckdb")); tbB = String(get(gdo, "tandem_b_table", "curated_cohorts_landis_simB"))
cyc_years = Float64(g("cycle_years", 8))
BCTX = Dict{String,Any}()
for nm in SPLITS
  spB = P._tandem_b_splots(dbB, tbB, _split_df(nm), species_list, eco_species_ids, D.LAST_SPECIES_MAP[])
  BCTX[nm] = (sp=spB, ctx=P._tandem_b_context(spB, eco_list, eco_species_ids, lpB, cyc_years, P.RNGType(7)))
end
# Sim B pairs per (plot, cycle, eco, species) over age bins ≥ SIMB_MIN_BIN, sim = mean of BREPS free runs (fixed seeds)
function bpairs(p, nm, offset)
  B = BCTX[nm]; c = B.ctx; ab = lpB.age_bins
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(B.sp, [:plot_id, :eco_id]))))
  sim = Dict{NTuple{5,Int},Float64}()
  for k in 1:BREPS
    r = P.fit_params(c.dual_b.ref_soa, p, c.msy, length(species_list), eco_species_ids, c.spdf_plts, c.ssy, false, c.spc, lpB;
      debug=false, search_tier=3, dual_b=c.dual_b, seeds=[UInt64(1000 + k)])
    for (pid, sy, esp, age, bio) in r[1][2]
      haskey(plot2eco, Int(pid)) || continue
      cy = get(c.dual_b.cycle_map, (Int(pid), Int(sy)), 0); cy == 0 && continue
      b = PU.find_age_bin(Int(ceil(Float64(age))), ab); b >= SIMB_MIN_BIN || continue
      kk = (Int(pid), cy, plot2eco[Int(pid)], Int(esp), b); sim[kk] = get(sim, kk, 0.0) + Float64(bio) / BREPS
    end
  end
  obs = Dict{NTuple{5,Int},Float64}()
  for r in eachrow(DF.subset(B.sp, :sim_year => DF.ByRow(>(0))))
    cy = get(c.dual_b.cycle_map, (Int(r.plot_id), Int(r.sim_year)), 0); cy == 0 && continue
    b = PU.find_age_bin(Int(ceil(Float64(r.age_calc))), ab); b >= SIMB_MIN_BIN || continue
    kk = (Int(r.plot_id), cy, Int(r.eco_id), Int(r.eco_species_id), b); obs[kk] = get(obs, kk, 0.0) + Float64(r.agb_sum)
  end
  agg = Dict{NTuple{4,Int},Tuple{Float64,Float64}}()
  for kk in union(keys(sim), keys(obs))
    k4 = kk[1:4]; o, s = get(agg, k4, (0.0, 0.0)); agg[k4] = (o + get(obs, kk, 0.0), s + get(sim, kk, 0.0))
  end
  sp_of = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id) for r in eachrow(unique(DF.select(B.sp, [:eco_id, :eco_species_id, :species_id]))))
  pr = DF.DataFrame(plot=Int[], cyc=Int[], eco=Int[], esp=Int[], sp=Int[], obs_agb=Float64[], sim_agb=Float64[])
  for ((pl, cy, e, esp), (o, s)) in agg
    haskey(sp_of, (e, esp)) || continue
    push!(pr, (pl, cy, e, esp, sp_of[(e, esp)], o, s))
  end
  pr.pgrp = [grp(s, e) for (s, e) in zip(pr.sp, pr.eco)]; pr.plot_id = offset .+ pr.plot
  pr
end
function bstats(pr)
  out = Dict{String,Any}()
  for (s, gp) in sort(unique([(r.sp, r.pgrp) for r in eachrow(pr)]))
    d = DF.subset(pr, [:sp, :pgrp] => DF.ByRow((a, b) -> a == s && b == gp)); length(unique(d.plot_id)) < 3 && continue
    br = boot_ratio(d, MARGINS); out[label(s, gp)] = (n=br.n, mean=br.mean, tests=br.tests)
  end
  cs = DF.combine(DF.groupby(pr, :cyc), :sim_agb => sum => :s, :obs_agb => sum => :o)
  cyc = Dict(Int(r.cyc) => (r.s / r.o - 1) * 100 for r in eachrow(cs) if r.o > 0)
  out, cyc
end
summ(gs) = (ng=length(gs), p05=count(x -> x.tests[1].equiv, values(gs)), p10=count(x -> x.tests[2].equiv, values(gs)),
            p20=count(x -> x.tests[3].equiv, values(gs)), mab=Statistics.median([abs(expm1(x.mean)) * 100 for x in values(gs)]))
rows = NamedTuple[]
for i in pick
  r = arch[i, :]; p = JLD2.load_object(joinpath(rundir, "mo_archive", r.file))
  row = Dict{Symbol,Any}(:idx => r.idx, :role => string(coalesce(r.role, "")), :local_train => r.sumA, :global_train => r.sumW)
  for (k, nm) in enumerate(SPLITS)
    a = summ(group_stats(_paired_one(p, CTXS[k], 0), MARGINS))
    bs, cyc = bstats(bpairs(p, nm, 0)); b = summ(bs)
    for (f, v) in pairs(a); row[Symbol("A_$(nm)_$(f)")] = v; end
    for (f, v) in pairs(b); row[Symbol("B_$(nm)_$(f)")] = v; end
    for c in sort(collect(keys(cyc))); row[Symbol("B_$(nm)_cyc$(c)_bias")] = round(cyc[c]; digits=2); end
  end
  push!(rows, (; row...))
  s1 = SPLITS[1]; gk(k) = row[Symbol("$(k[1])_$(s1)_$(k[2])")]
  @printf("TEVAL %3d %-8s local=%.3f global=%.3f | A %s ±20=%d/%d mab=%.1f%% | B %s ±20=%d/%d mab=%.1f%%\n", r.idx, row[:role], r.sumA, r.sumW,
          s1, gk(("A", "p20")), gk(("A", "ng")), gk(("A", "mab")), s1, gk(("B", "p20")), gk(("B", "ng")), gk(("B", "mab")))
end
allk = union((collect(keys(x)) for x in rows)...)
CSV.write(outcsv, DF.DataFrame([Dict(k => get(x, k, missing) for k in allk) for x in rows]))
