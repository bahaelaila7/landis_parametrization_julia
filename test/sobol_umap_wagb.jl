# UMAP of the Sobol sample coloured by ΣW alone AND ΣAGB alone (brighter/yellow = lower = better).
# The DB only stored the scalar aggregate, so each candidate is RE-EVALUATED (1 rep, same dual config)
# to recover measure()=(ΣW,ΣAGB). Results cached to sobol_wagb.csv so re-plots are free.
# Same param-feature embedding as test/sobol_umap.jl → identical layout, only the colour changes.
#   Run:  ./julia_gdal.sh --project=. test/sobol_umap_wagb.jl <config.yml>
using Pan
import DuckDB, DataFrames, Serialization, UMAP, Random, Statistics, CairoMakie, YAML
const MK = CairoMakie; const P = Pan; const PU = P.PU; const D = P.Data; const DF = DataFrames

cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
outdir = cfg["output_dir"]
dbpath = String(g("sobol_candidates_db", joinpath(outdir, "losses.duckdb")))
P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true))
P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false))
P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false)); D.FIA_CYCLE_MERGE[] = Int(g("fia_cycle_merge", 1))
P.DUAL_MODE[] = let v = g("dual_mode", "off"); v === true ? :joint : v === false ? :off : Symbol(lowercase(String(v))) end
P.TIER_B[] = Int(g("tier_b", 4)); P.INIT_PERTURB_FRAC[] = 0.0
no_estab = Bool(g("no_establishment", false)); rng = P.RNGType(UInt64(Int(g("seed", 1))))
splots, eco_list, species_list, eco_species_ids, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]), output_dir=String(cfg["tablename"]),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false,
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  filter_extent=(haskey(cfg, "filter_extent") ? String(cfg["filter_extent"]) : nothing),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])), RNG=rng)
n_species = length(species_list)
bins = Int.(g("bins_idx", [20, 60, 120]))
sw = Int(g("smoothing_window_size", 0))
smooth = sw > 0 ? PU.get_smoothing_window(; smoothing_window=sw, smoothing_variance=P.FloatType(g("smoothing_variance", 1.0))) : P.FloatType[1.0]
# unbinned_w (Sim A): mirror parametrize() — per-year 1:400 grid, no age-smoothing → W₁ is the EXACT per-year EMD.
# `bins` (config bins_idx) is still the COARSE grid used for the count-balance reweight below.
unbinned_w = Bool(g("unbinned_w", false))
loss_params = unbinned_w ?
  PU.LossParams(age_bins=PU.AgeBins(bins_idx=collect(1:400), last_bin_open=true), smoothing_weights=P.FloatType[one(P.FloatType)], lambda=P.FloatType(g("loss_lambda", 1.0))) :
  PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=true), smoothing_weights=smooth, lambda=P.FloatType(g("loss_lambda", 1.0)))

spdf = PU.smoothen_ref_years(splots, loss_params, Int(maximum(splots.age_calc)); debug=false)
spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(splots)
inj = (no_estab || P.OVERRIDE_INJECTION[]) ? D.get_injection_cohorts(splots; all_cohorts=P.OVERRIDE_INJECTION[]) : nothing
ref = P.make_sites(splots, eco_species_ids; rng=rng, spinup=false, no_establishment=no_estab)
idict = isnothing(inj) ? nothing : P._build_injection_dict(inj, ref); iyears = isnothing(inj) ? Set{Int}() : Set(Int.(inj.sim_year))
msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
cycle_years = Real(g("cycle_years", 8))
# n_bins must match sp_age_cdf length, which derives from loss_params.age_bins (1:400 when unbinned_w,
# else the coarse `bins`) — mirror parametrize() line 1731, not the config `bins`, or the t4_ref build mismatches.
n_bins = length(loss_params.age_bins.bins_idx) + Int(loss_params.age_bins.last_bin_open)
cmap, ncyc = D.build_cycle_map(splots; cycle_years=cycle_years)
t4_ref = [[zeros(P.FloatType, length(eco_species_ids[e]), n_bins) for _ in 1:ncyc] for e in eachindex(eco_list)]
for ((pid, eid), yd) in spdf_plts, (sy, gt) in yd
  c = get(cmap, (Int(pid), Int(sy)), 0); c == 0 && continue
  for (sp_eco, rec) in gt.records; t4_ref[eid][c][sp_eco, :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum; end
end
# Replicate the RUN's loss shaping so the re-evaluation (ΣW, ΣAGB, aggregate, NDS fronts) is on the SAME
# NORMALIZED loss the Sobol run optimized — previously these flags were unset, so the re-eval ran on the
# RAW/unnormalized loss and mismatched the stored mean_loss. Values + defaults mirror parametrize().
PU.LOSS_ALPHA[]          = P.FloatType(g("loss_alpha", 1.0))
PU.AGB_HINGE[]           = Bool(g("agb_hinge", false))
PU.AGB_HINGE_THRESHOLD[] = P.FloatType(g("agb_hinge_threshold", 10.0))
PU.AGB_HINGE_PCT[]       = P.FloatType(g("agb_hinge_pct", 0.0))
PU.AGB_HINGE_PCT_MIN[]   = P.FloatType(g("agb_hinge_pct_min", 0.0))
PU.AGB_HINGE_PCT_MAX[]   = P.FloatType(g("agb_hinge_pct_max", Inf))
PU.AGB_HINGE_L2[]        = Bool(g("agb_hinge_l2", false))
PU.AGB_HINGE_P[]         = P.FloatType(g("agb_hinge_p", 2.0))
PU.AGB_HINGE_BETA[]      = P.FloatType(g("agb_hinge_beta", 1.0))
PU.AGB_NORMALIZE[]       = Bool(g("agb_normalize", false))
PU.W_NORMALIZE[]         = Bool(g("w_normalize", false))
PU.CELL_NORM[]           = Bool(g("cell_normalize", false))
PU.W_SCALE_FACTOR[]      = P.FloatType(g("w_scale_factor", 1.0))
PU.CELL_NORM[] && (PU.RANKW[] = zeros(P.FloatType, 0, 0); PU.CELL_NORM_FREEZE[] = false)
PU.W_SOFTPLUS[]          = Bool(g("w_smooth", false))
PU.W_SMOOTH_BAND[]       = P.FloatType(g("w_smooth_band", 0.05)); PU.W_SMOOTH_CONC[] = P.FloatType(g("w_smooth_conc", 0.5))
PU.W_SMOOTH_BETA[]       = P.FloatType(g("w_smooth_beta", 1.0)); PU.W_SMOOTH_AUTO_KNEE[] = Bool(g("w_smooth_auto_knee", false))
PU.W_P[]                 = P.FloatType(g("w_p", 1.0))
PU.LOSS_PIECEWISE[]      = Bool(g("loss_piecewise", false)); PU.W_PIVOT[] = P.FloatType(g("w_pivot", 1.0)); PU.AGB_PIVOT[] = P.FloatType(g("agb_pivot", 1.0))
P._set_loss_scales!(Int(g("tier", 3)), spdf_plts, t4_ref, loss_params, eco_species_ids, n_species)  # populate W/AGB scales + RANKW from the train reference
println("norm: W=$(PU.W_NORMALIZE[]) AGB=$(PU.AGB_NORMALIZE[]) CELL=$(PU.CELL_NORM[]) | RANKW $(size(PU.RANKW[])) W_SCALE_A $(size(PU.W_SCALE_A[]))")
# Count-balance reweight (survivorship correction) — mirror parametrize() lines 945-950 so the re-eval
# aggregate matches the stored mean_loss. Frozen from the TRAIN reference on the coarse `bins` grid;
# Sim A looks it up per-year via CBAL_COARSE. Defaults off → runs without w_count_balance are unaffected.
PU.CBAL_ON[]   = Bool(g("w_count_balance", false))
PU.CBAL_MODE[] = Symbol(lowercase(String(g("w_count_balance_mode", "both"))))
PU.CBAL_BETA[] = Float64(g("w_count_beta", 0.99))
if PU.CBAL_ON[]
  PU._set_cbal_weights!(splots, PU.AgeBins(bins_idx=bins, last_bin_open=true), loss_params.age_bins, eco_species_ids, n_species; beta=PU.CBAL_BETA[])
  println("CBAL ON mode=$(PU.CBAL_MODE[]) β=$(PU.CBAL_BETA[]); coarse=$(length(bins)) bins+open, Sim-A W bins=$(length(loss_params.age_bins.bin_widths))")
end

dual_b = P.DUAL_MODE[] == :off ? nothing :
  P._build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cmap, ncyc, inj, spc, rng, no_estab; b_only=false)

# measure() = (ΣW, ΣAGB) over the dominance species for one candidate
function wagb(params)
  res = P.fit_params(ref, params, msy, n_species, eco_species_ids, spdf_plts, ssy, false, spc, loss_params;
    debug=false, search_tier=Int(g("tier", 3)), t4_ref=t4_ref, cycle_map=cmap, n_cycles=ncyc,
    injection_dict=idict, injection_years=iyears, seeds=[rand(rng, UInt64)], dual_b=dual_b)
  fx = P.MOLBSA.MOFitness(P._mo_objectives(res[1][3], eco_species_ids), convert(Float64, PU.get_total_loss(res[1][1])))
  m = P.CMAMAE.measure(fx); (Float64(m[1]), Float64(m[2]), Float64(fx.aggregate), fx.objectives)
end

function feat(p)
  v = Float64[]
  for gsp in eachindex(p.SPECIES_LIST); push!(v, Float64(p.D[gsp]), Float64(p.LONGEVITY[gsp]), Float64(length(p.MATURITY) >= gsp ? p.MATURITY[gsp] : 0), Float64(p.SHADE_TOL[gsp])); end
  for eco_id in eachindex(p.ECO_LIST), sp_local in eachindex(p.ECO_SPECIES_IDS[eco_id])
    gsp = Int(p.ECO_SPECIES_IDS[eco_id][sp_local])
    push!(v, Float64(p.S[gsp]), Float64(p.ANPP_MAX_SPP[eco_id][sp_local]), Float64(p.B_MAX_SPP[eco_id][sp_local]), Float64(p.PROB_MORT_SPP[eco_id][sp_local]), Float64(length(p.PROB_ESTAB_SPP) >= eco_id ? p.PROB_ESTAB_SPP[eco_id][sp_local] : 0))
  end
  for eco_id in eachindex(p.ECO_LIST); push!(v, Float64(p.MIN_REL_BIOMASS[eco_id][1])); end
  v
end

con = DuckDB.connect(DuckDB.DB(dbpath))
# losses.duckdb may hold several appended Sobol runs (same params, different configs). Restrict to the
# MOST RECENT run_id (ISO-timestamp string → max()) so we use exactly that run's candidates.
# EVERYTHING comes from the run's OWN recorded columns — W/AGB/aggregate from sumW/sumAGB/mean_loss and the
# NDS objective vector from objs_blob (= the run's _mo_objectives = the 2-D [ΣW, ΣAGB]; for a single sim
# that IS the full MO vector — _mo_objectives is deliberately 2-D, not per-species). So NO re-simulation is
# needed; the old per-candidate fit_params re-eval was redundant with these stored columns.
res = DuckDB.execute(con, "SELECT params_blob, mean_loss, sumW, sumAGB, objs_blob FROM sobol_results WHERE run_id = (SELECT max(run_id) FROM sobol_results) ORDER BY mean_loss ASC") |> DF.DataFrame
N = DF.nrow(res); println("Reading $N Sobol candidates from the DB (NO re-sim); OBJ = objs_blob = the run's [ΣW, ΣAGB] objectives…")
feats = Vector{Float64}[]; W = Float64[]; A = Float64[]; AG = Float64[]; OBJ = Vector{Vector{Float32}}()
for (i, r) in enumerate(eachrow(res))
  p = Serialization.deserialize(IOBuffer(r.params_blob))
  ob = Float32.(Serialization.deserialize(IOBuffer(r.objs_blob)))   # stored [ΣW,ΣAGB]; == [sumW,sumAGB]. No fit_params needed.
  push!(feats, feat(p)); push!(W, Float64(r.sumW)); push!(A, Float64(r.sumAGB)); push!(AG, Float64(r.mean_loss)); push!(OBJ, ob)
  i % 500 == 0 && println("  $i/$N")
end
# EXACT non-dominated sorting (fast non-dominated sort) on the FULL objective vector — the same
# objectives MO-CMA-ES ranks — giving each candidate its Pareto-front index (1 = non-dominated).
FR = let Nn = length(OBJ)
  domc = zeros(Int, Nn); domd = [Int[] for _ in 1:Nn]
  for p in 1:Nn, q in 1:Nn
    p == q && continue
    if P.MOLBSA.dominates(OBJ[p], OBJ[q]); push!(domd[p], q)
    elseif P.MOLBSA.dominates(OBJ[q], OBJ[p]); domc[p] += 1 end
  end
  rank = zeros(Int, Nn); fr = [p for p in 1:Nn if domc[p] == 0]; r = 1
  while !isempty(fr)
    for p in fr; rank[p] = r end
    nx = Int[]; for p in fr, q in domd[p]; domc[q] -= 1; domc[q] == 0 && push!(nx, q) end
    fr = nx; r += 1
  end
  rank
end
println("exact NDS on (ΣW, ΣAGB): $(maximum(FR)) fronts; front-1 (Pareto) size = $(count(==(1), FR))/$N")
cache = joinpath(outdir, "sobol_wagb.csv")
open(cache, "w") do io
  println(io, "idx,W,AGB,aggregate,nds_front")
  for i in 1:N; println(io, "$i,$(W[i]),$(A[i]),$(AG[i]),$(FR[i])"); end
end
println("cached → $cache")

X = reduce(hcat, feats); mu = Statistics.mean(X; dims=2); sd = Statistics.std(X; dims=2); sd[sd.==0] .= 1
Z = (X .- mu) ./ sd
Random.seed!(7)
emb = UMAP.fit(Z, 2; n_neighbors=max(2, min(15, size(Z, 2) - 1)), min_dist=0.4).embedding
function plot_umap(cval, label, fname)
  fig = MK.Figure(size=(860, 680))
  MK.Label(fig[0, 1:2], "Sobol sample ($N) — UMAP — colour = $label (yellow = lower = better)"; fontsize=13, font=:bold)
  ax = MK.Axis(fig[1, 1]; xlabel="UMAP-1", ylabel="UMAP-2")
  MK.scatter!(ax, emb[1, :], emb[2, :]; color=cval, colormap=MK.cgrad(:viridis; rev=true), markersize=11, strokecolor=:black, strokewidth=0.4)
  MK.Colorbar(fig[1, 2]; colormap=MK.cgrad(:viridis; rev=true), colorrange=(minimum(cval), maximum(cval)), label="$label (yellow = lower = better)")
  MK.save(joinpath(outdir, fname), fig); println("wrote $(joinpath(outdir, fname))")
end
plot_umap(W, "ΣW (age-distribution loss)", "sobol_umap_W.png")
plot_umap(A, "ΣAGB (biomass loss)", "sobol_umap_AGB.png")
plot_umap(AG, "aggregate loss (get_total_loss)", "sobol_umap_aggregate.png")
plot_umap(Float64.(FR), "exact non-dominated front rank (ΣW, ΣAGB)", "sobol_umap_nds.png")
println("done. W∈[$(round(minimum(W),sigdigits=3)),$(round(maximum(W),sigdigits=3))] AGB∈[$(round(minimum(A),sigdigits=3)),$(round(maximum(A),sigdigits=3))]")
