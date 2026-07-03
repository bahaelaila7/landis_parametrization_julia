# Sweep a CMA-MAE per-species-breadth archive: for every elite, evaluate its VALIDATION TOST
# (# species equivalent within ±margin on the held-out split) and its breadth (3rd-axis level).
# Reports TWO solutions:
#   • representative  = elite with the HIGHEST validation TOST (best generalizer)
#   • broadest        = elite at the HIGHEST breadth level (most non-dominated species), min-aggregate tie-break
# Writes archive_sweep.csv (per elite) + best_by_valtost.json / best_by_breadth.json.
#   Run:  ./julia_gdal.sh --project=. test/sweep_archive_tost.jl <config.yml> [equiv_margin=0.2] [alpha=0.05]
using Pan
import JLD2, YAML, Statistics, DataFrames, HypothesisTests, Printf, JSON3
const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames; const HT = HypothesisTests

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
EQM = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.2
ALPHA = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.05
Δ = log1p(EQM)
outdir = cfg["output_dir"]
P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true))
P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false))
P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false))
D.FIA_CYCLE_MERGE[] = Int(g("fia_cycle_merge", 1))
P.INIT_PERTURB_FRAC[] = 0.0
no_estab = Bool(g("no_establishment", false))
rng = P.RNGType(UInt64(Int(g("seed", 1))))
val_frac = Float64(g("val_frac", 0.0)); split_rng = val_frac > 0 ? P.RNGType(UInt64(Int(g("split_seed", 42)))) : nothing
splots, eco_list, species_list, eco_species_ids, splots_val = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=val_frac, split_rng=split_rng,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  filter_extent=(haskey(cfg, "filter_extent") ? String(cfg["filter_extent"]) : nothing),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  filter_plots=NTuple{4,Int}[NTuple{4,Int}(Int.(p)) for p in get(cfg, "filter_plots", [])], RNG=rng)
n_species = length(species_list)
bins = Int.(g("bins_idx", [20, 60, 120]))
loss_params = PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=true), smoothing_weights=P.FloatType[1.0], lambda=P.FloatType(g("loss_lambda", 1.0)))
st = JLD2.load_object(joinpath(outdir, "search_state_latest.jld2"))

# --- # species equivalent (val TOST) for a given parameter set ---
function val_equiv_count(params)
  sp = splots_val
  (isnothing(sp) || DF.nrow(sp) == 0) && return (0, 0)
  max_age = Int(maximum(sp.age_calc)); spdf = PU.smoothen_ref_years(sp, loss_params, max_age; debug=false)
  spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(sp)
  inj = (no_estab || P.OVERRIDE_INJECTION[]) ? D.get_injection_cohorts(sp; all_cohorts=P.OVERRIDE_INJECTION[]) : nothing
  rs = P.make_sites(sp, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)
  idict = isnothing(inj) ? nothing : P._build_injection_dict(inj, rs); iyears = isnothing(inj) ? Set{Int}() : Set(Int.(inj.sim_year))
  msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
  res = P.fit_params(rs, params, msy, n_species, eco_species_ids, spdf_plts, ssy, false, spc, loss_params;
    debug=false, search_tier=3, injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)])
  cached = res[1][2]
  simdf = DF.DataFrame(plot_id=Int[], sim_year=Int[], esp=Int[], agb=Float64[])
  for (pid, sy, esp, _a, bio) in cached; push!(simdf, (Int(pid), Int(sy), Int(esp), Float64(bio))); end
  sim_agg = DF.combine(DF.groupby(simdf, [:plot_id, :sim_year, :esp]), :agb => sum => :sim_agb)
  obs = DF.combine(DF.groupby(DF.subset(sp, :sim_year => DF.ByRow(>(0))), [:plot_id, :sim_year, :eco_id, :eco_species_id]), :agb_sum => sum => :obs_agb)
  DF.rename!(obs, :eco_species_id => :esp)
  paired = DF.innerjoin(obs, sim_agg, on=[:plot_id, :sim_year, :esp])
  if P.OVERRIDE_INJECTION_DISTURBANCE[] in (:exclude_overwrite, :exclude_noscale) && !isnothing(inj)
    excl = Set((Int(r.plot_id), Int(r.sim_year), Int(r.eco_species_id)) for r in eachrow(inj) if r.disturbance_drop_pct > 0)
    paired = DF.filter(row -> !((row.plot_id, row.sim_year, row.esp) in excl), paired)
  end
  plot2eco = Dict(Int(r.plot_id) => Int(r.eco_id) for r in eachrow(unique(DF.select(sp, [:plot_id, :eco_id]))))
  especo2sp = Dict((Int(r.eco_id), Int(r.eco_species_id)) => Int(r.species_id) for r in eachrow(unique(DF.select(sp, [:eco_id, :eco_species_id, :species_id]))))
  paired.eco = [plot2eco[p] for p in paired.plot_id]; paired.sp = [especo2sp[(e, esp)] for (e, esp) in zip(paired.eco, paired.esp)]
  equiv = 0; total = 0
  for e in unique(paired.eco), s in unique(DF.subset(paired, :eco => DF.ByRow(==(e))).sp)
    d = DF.subset(paired, :eco => DF.ByRow(==(e)), :sp => DF.ByRow(==(s)))
    DF.nrow(d) < 3 && continue
    dd = log1p.(Float64.(d.sim_agb)) .- log1p.(Float64.(d.obs_agb))
    md = Statistics.mean(dd); total += 1
    if Statistics.std(dd) < 1e-9
      abs(md) < Δ && (equiv += 1)
    else
      p = max(HT.pvalue(HT.OneSampleTTest(dd, -Δ); tail=:right), HT.pvalue(HT.OneSampleTTest(dd, Δ); tail=:left))
      p < ALPHA && (equiv += 1)
    end
  end
  (equiv, total)
end

# dominance-species mask (must match the run's, so the reconstructed per-species objectives align)
domset = let mode = lowercase(String(g("dominance_species", "exact")))
  mode == "all" ? Set{Int}() : Set(s for s in 1:n_species if P._dom_include_name(species_list[s], mode))
end
P.DOMINANCE_GSP[] = domset

# reconstruct per-(global species) (ΣW_s, ΣAGB_s) from a candidate's fx.objectives by replaying the
# _mo_objectives ordering (block = eco-block cycling eco_species_ids; pairs filtered to domset).
function per_sp_WA(objs)
  ne = length(eco_species_ids); npairs = length(objs) ÷ 2
  W = zeros(Float64, n_species); A = zeros(Float64, n_species)
  pos = 1; consumed = 0; blk = 0
  while consumed < npairs
    blk += 1; eco = (blk - 1) % ne + 1
    for gsp in eco_species_ids[eco]
      (isempty(domset) || gsp in domset) || continue
      consumed >= npairs && break
      W[gsp] += Float64(objs[pos]); A[gsp] += Float64(objs[pos + 1]); pos += 2; consumed += 1
    end
  end
  (W, A)
end

# --- dedup unique candidates ---
uniq = Dict{UInt,Any}()
for cand in st.cell_cand
  cand === nothing && continue
  h = hash(cand.x); haskey(uniq, h) || (uniq[h] = cand)
end
cands = collect(values(uniq))
N = length(cands)
println("Archive: $(count(!isnothing, st.cell_cand)) occupied cells, $N unique candidates; dominance species: $(isempty(domset) ? n_species : length(domset))")

# per-candidate per-species (W,A), ΣW, ΣAGB, aggregate, val-TOST
spW = Vector{Vector{Float64}}(undef, N); spA = Vector{Vector{Float64}}(undef, N)
SW = zeros(N); SA = zeros(N); AGG = zeros(N); VE = zeros(Int, N); VT = zeros(Int, N)
sp_list = isempty(domset) ? collect(1:n_species) : sort(collect(domset))
for i in 1:N
  w, a = per_sp_WA(cands[i].fx.objectives)
  spW[i] = w; spA[i] = a; SW[i] = sum(w); SA[i] = sum(a); AGG[i] = convert(Float64, cands[i].fx.aggregate)
  VE[i], VT[i] = val_equiv_count(cands[i].x)
end

# POST-HOC breadth: # species on which candidate i is non-dominated vs the FINAL set (fixes first-mover bias)
BR = zeros(Int, N)
for i in 1:N, s in sp_list
  dominated = false
  for j in 1:N
    j == i && continue
    if spW[j][s] <= spW[i][s] && spA[j][s] <= spA[i][s] && (spW[j][s] < spW[i][s] || spA[j][s] < spA[i][s])
      dominated = true; break
    end
  end
  dominated || (BR[i] += 1)
end

# 3-objective non-dominated sort over (ΣW↓, ΣAGB↓, breadth↑) → "total dominance" front rank
function nd3(SW, SA, BR)
  n = length(SW); dom(i, j) = (SW[i] <= SW[j] && SA[i] <= SA[j] && BR[i] >= BR[j]) && (SW[i] < SW[j] || SA[i] < SA[j] || BR[i] > BR[j])
  front = fill(0, n); rem = Set(1:n); fr = 0
  while !isempty(rem)
    fr += 1; nd = [i for i in rem if !any(j -> dom(j, i), rem)]; for i in nd; front[i] = fr; delete!(rem, i); end
  end
  front
end
front = nd3(SW, SA, BR)
nfronts = maximum(front)
order = sortperm([(front[i], -VE[i], AGG[i]) for i in 1:N])   # front, then val-TOST desc, then aggregate

open(joinpath(outdir, "archive_sweep.csv"), "w") do io
  println(io, "rank,front,breadth,sumW,sumAGB,aggregate,val_equiv,val_total")
  for (r, i) in enumerate(order)
    println(io, join([r, front[i], BR[i], round(SW[i], digits=4), round(SA[i], digits=3), round(AGG[i], digits=3), VE[i], VT[i]], ","))
  end
end

# 3 CORNERS of front-1 to illustrate the tradeoff (NOT picked by val-TOST — that would be p-hacking;
# val-TOST is reported only as narrative). Corners: lowest ΣW, lowest ΣAGB, highest #sp (breadth).
f1 = [i for i in 1:N if front[i] == 1]
corners = unique([f1[argmin([SW[i] for i in f1])], f1[argmin([SA[i] for i in f1])], f1[argmax([BR[i] for i in f1])]])
labels = ["lowW", "lowAGB", "mostSp"]
println("\n=== total-dominance (ΣW,ΣAGB,#sp) fronts: $nfronts ; front-1 size: $(length(f1)) ===")
println("=== 3 FRONT-1 CORNERS (story only; not a selection) ===")
manifest = String[]
for (k, i) in enumerate(corners)
  tag = "AGB$(round(SA[i],digits=2))_W$(round(SW[i],digits=3))_sp$(BR[i])"   # subfolder name encodes the coords
  fn = "corner_$(labels[min(k,end)])_$(tag)"
  JLD2.save_object(joinpath(outdir, "$(fn).jld2"), cands[i].x)
  open(joinpath(outdir, "$(fn).json"), "w") do io; JSON3.write(io, cands[i].x); end
  push!(manifest, "$(fn).jld2\t$(fn)")
  println("  $(labels[min(k,end)]): ΣW=$(round(SW[i],digits=4))  ΣAGB=$(round(SA[i],digits=3))  #sp=$(BR[i])  agg=$(round(AGG[i],digits=3))  val_equiv=$(VE[i])/$(VT[i]) (narrative)  → $(fn).{jld2,json}")
end
# manifest: <params_jld2>\t<subfolder> for the plot step
open(joinpath(outdir, "corners_manifest.tsv"), "w") do io; for l in manifest; println(io, l); end; end
println("\nwrote archive_sweep.csv ($N elites) + corners_manifest.tsv")
