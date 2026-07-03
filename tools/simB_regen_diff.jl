# Sim B (free natural REGEN) tier-4 comparison for an archive: per (species × age-bin × cycle),
# MAPE = |mean(simB over REPS) − ref| / ref, with error bars = std(simB)/ref. NATURAL/regen strata only.
# Two phases: (1) SIMULATE — run REPS Sim-B evals/candidate, CACHE the raw t4_sim arrays; (2) PLOT from
# the cache (fast). Pass --resim (or change REPS) to re-simulate; otherwise the cache is reused so metric
# tweaks re-plot in seconds. Drives the real Sim B via fit_params + Pan.CAPTURE_T4SIM.
#   Run: ./julia_gdal.sh --project=. tools/simB_regen_diff.jl runs/...stageB_joint.yml [REPS=50] [--resim]
using Pan, YAML, DataFrames, JLD2, CairoMakie, Statistics
const P = Pan; const D = P.Data; const PU = P.PU; const BSP = P.BiomassSuccessionPlugin
const MK = CairoMakie; const DF = DataFrames
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
outdir = cfg["output_dir"]
REPS = let a = filter(x -> tryparse(Int, x) !== nothing, ARGS); isempty(a) ? 50 : parse(Int, a[1]); end
RESIM = "--resim" in ARGS
CACHE = joinpath(outdir, "simB_t4sim_cache.jld2")

# colours: HUE = cycle, SHADE = ref(light)/sim(full)
const CYCHUE = [MK.RGBf(0.15, 0.38, 0.70), MK.RGBf(0.85, 0.45, 0.10), MK.RGBf(0.20, 0.55, 0.25), MK.RGBf(0.55, 0.20, 0.60)]
lighten(c, f) = MK.RGBf(c.r + (1 - c.r) * f, c.g + (1 - c.g) * f, c.b + (1 - c.b) * f)
simcol(cyc) = CYCHUE[mod1(cyc, length(CYCHUE))]
sanitize(s) = replace(String(s), r"[^A-Za-z0-9]" => "_")

# ============================ PHASE 1: SIMULATE (cached) ============================
if isfile(CACHE) && !RESIM
  cc = JLD2.load(CACHE)
  allsims = cc["allsims"]; t4_ref = cc["t4_ref"]; eco_list = cc["eco_list"]
  species_list = cc["species_list"]; eco_species_ids = cc["eco_species_ids"]; n_cycles = cc["n_cycles"]; bins = cc["bins"]
  println("loaded cache $CACHE — $(length(allsims)) candidates × $(length(allsims[1])) reps (pass --resim to redo)")
else
  P.OVERRIDE_INJECTION[] = Bool(g("override_injection", true)); P.OVERRIDE_INJECTION_SYNC[] = Bool(g("override_injection_sync", true))
  P.OVERRIDE_INJECTION_REPLACE[] = Bool(g("override_injection_replace", false))
  P.OVERRIDE_INJECTION_DISTURBANCE[] = Symbol(g("override_injection_disturbance", "off"))
  D.USE_FIA_CYCLE[] = Bool(g("fia_cycle", false))
  let dm = g("dual_mode", false); P.DUAL_MODE[] = dm == "joint" ? :joint : dm == "b" ? :b : :off; end
  P.TIER_B[] = Int(g("tier_b", 4)); PU.CELL_NORM[] = Bool(g("cell_normalize", false))
  P.SIMB_SPINUP[] = Bool(g("simB_spinup", true))   # honor non-spinup Sim B (start from observed cohorts)
  no_estab = Bool(g("no_establishment", false)); rng = P.RNGType(UInt64(Int(g("seed", 1))))
  splots, eco_list, species_list, eco_species_ids, _ = D.prepare_parametrization_data(;
    cohorts_db_path=cfg["cohorts_db_path"], eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]),
    output_dir=String(cfg["tablename"]), filter_eco_field=String(g("filter_eco_field", "epa_l4")),
    filter_ecos=String.(get(cfg, "filter_ecos", String[])), filter_plots=NTuple{4,Int}[],
    min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
    skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=0.0,
    single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)), RNG=rng)
  n_species = length(species_list)
  bins = Int.(g("bins_idx", [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]))
  loss_params = PU.LossParams(age_bins=PU.AgeBins(bins_idx=bins, last_bin_open=true), smoothing_weights=P.FloatType[1.0], lambda=P.FloatType(g("loss_lambda", 1.0)))
  n_bins = length(bins) + 1
  spinup = Bool(g("spinup", false)); max_age = Int(maximum(splots.age_calc))
  spdf = PU.smoothen_ref_years(splots, loss_params, max_age; debug=false)
  spdf_plts = D.make_spdf_dict(spdf, eco_species_ids); ssy = D.get_site_sim_years(spdf); spc = D.get_spinup_cohorts(splots)
  inj = (no_estab || P.OVERRIDE_INJECTION[]) ? D.get_injection_cohorts(splots; all_cohorts=P.OVERRIDE_INJECTION[]) : nothing
  cycle_map, n_cycles = D.build_cycle_map(splots; cycle_years=Float64(g("cycle_years", 8)))
  t4_ref = [[zeros(P.FloatType, length(eco_species_ids[e]), n_bins) for _ in 1:n_cycles] for e in eachindex(eco_list)]
  for ((plot_id, eco_id), yd) in spdf_plts, (sy, gt) in yd
    cyc = get(cycle_map, (Int(plot_id), Int(sy)), 0); cyc == 0 && continue
    for (sp_eco, rec) in gt.records
      t4_ref[eco_id][cyc][Int(sp_eco), :] .+= diff([0f0; rec.sp_age_cdf]) .* rec.sp_agb_sum
    end
  end
  ref_soa = P.make_sites(splots, eco_species_ids; rng=rng, spinup=spinup, no_establishment=no_estab)
  dual_b = P._build_dual_b(splots, eco_species_ids, eco_list, spdf_plts, loss_params, t4_ref, cycle_map, n_cycles, inj, spc, rng, no_estab; b_only=false)
  msy = maximum(maximum.(filter(!isempty, ssy.sim_years)))
  arch = collect(JLD2.load_object(joinpath(outdir, "search_state_latest.jld2")).archive)
  println("archive: $(length(arch)) candidates; simulating $REPS Sim-B reps each …")
  allsims = Vector{Any}(undef, length(arch))
  for (ci, c) in enumerate(arch)
    reps = Vector{Any}(undef, REPS)
    for rep in 1:REPS
      P.CAPTURE_T4SIM[] = :want
      P.fit_params(dual_b.ref_soa, c.x, msy, n_species, eco_species_ids, dual_b.spdf_plts, ssy, dual_b.spinup, dual_b.spinup_cohorts, dual_b.loss_params;
        debug=false, search_tier=4, t4_ref=dual_b.t4_ref, cycle_map=dual_b.cycle_map, n_cycles=dual_b.n_cycles,
        seeds=[rand(rng, UInt64)], injection_dict=dual_b.injection_dict, injection_years=dual_b.injection_years,
        disturbance_dict=dual_b.disturbance_dict, disturbance_years=dual_b.disturbance_years)
      cap = P.CAPTURE_T4SIM[]; P.CAPTURE_T4SIM[] = nothing
      cap === :want && error("candidate $ci rep $rep: no t4_sim captured")
      reps[rep] = [[Float32.(m) for m in eco] for eco in cap.sim]
    end
    allsims[ci] = reps; println("  candidate $ci: $REPS reps done")
  end
  JLD2.save(CACHE, "allsims", allsims, "t4_ref", t4_ref, "eco_list", eco_list,
    "species_list", species_list, "eco_species_ids", eco_species_ids, "n_cycles", n_cycles, "bins", bins)
  println("cached → $CACHE")
end

# ============================ PHASE 2: PLOT (fast, from cache) ============================
n_bins = size(t4_ref[1][1], 2)
binlabels = vcat(["<$(bins[1])"], ["$(bins[i-1])–$(bins[i])" for i in 2:length(bins)], ["≥$(bins[end])"])
natural = [e for e in eachindex(eco_list) if occursin("natural", lowercase(eco_list[e]))]
regenlbl(e) = replace(replace(String(eco_list[e]), "|lu=natural" => " — natural regeneration"), "_" => " ")

for (ci, reps) in enumerate(allsims)
  REPN = length(reps)
  sumS = [[zeros(Float64, size(m)) for m in eco] for eco in reps[1]]
  sumSq = [[zeros(Float64, size(m)) for m in eco] for eco in reps[1]]
  for s in reps, e in eachindex(s), cyc in 1:n_cycles
    ss = Float64.(s[e][cyc]); sumS[e][cyc] .+= ss; sumSq[e][cyc] .+= ss .^ 2
  end
  cd = joinpath(outdir, "candidates", "candidate_$(ci)"); mkpath(cd); nwrote = 0
  for e in natural
    # SMAPE% = |mean(simB) − ref| / ((|mean(simB)|+|ref|)/2); bounded [0,200], handles ref=0 (over-production →
    # 200%) and sim=0 (→200%). Tolerance: denom < 0.1 g/m² → negligible AGB → NaN (no error → skip).
    # Error bar = sd(simB)/denom.
    mape = Vector{Matrix{Float64}}(undef, n_cycles); mapeSD = Vector{Matrix{Float64}}(undef, n_cycles)
    for cyc in 1:n_cycles
      r = Float64.(t4_ref[e][cyc]); ms = sumS[e][cyc] ./ REPN
      sds = sqrt.(max.(0.0, sumSq[e][cyc] ./ REPN .- ms .^ 2))
      den = (abs.(ms) .+ abs.(r)) ./ 2
      mape[cyc] = @. ifelse(den >= 0.1, abs(ms - r) / den * 100, NaN)
      mapeSD[cyc] = @. ifelse(den >= 0.1, sds / den * 100, NaN)
    end
    sp_present = [sp for sp in 1:length(eco_species_ids[e]) if any(any(.!isnan.(mape[cyc][sp, :])) for cyc in 1:n_cycles)]
    isempty(sp_present) && continue
    ncol = min(3, length(sp_present)); nrow = cld(length(sp_present), ncol)
    fig = MK.Figure(size=(380 * ncol + 80, 250 * nrow + 90))
    MK.Label(fig[0, 1:ncol], "$(regenlbl(e)) — Sim B natural regeneration: SMAPE per age bin = |mean(simB)−ref|/((|simB|+|ref|)/2) ($REPN reps), candidate $ci"; fontsize=12, font=:bold)
    ngrp = n_cycles; w = 0.7 / ngrp
    for (pi, sp) in enumerate(sp_present)
      gsp = eco_species_ids[e][sp]; r, c = fldmod1(pi, ncol)
      ax = MK.Axis(fig[r, c]; title=species_list[gsp], xlabel="age bin", ylabel="SMAPE (%)",
        xticks=(1:n_bins, binlabels), xticklabelrotation=π/4, xticklabelsize=8, limits=(nothing, nothing, 0, 200))
      for cyc in 1:n_cycles
        v = mape[cyc][sp, :]; ok = findall(.!isnan.(v))
        isempty(ok) && continue
        MK.barplot!(ax, (ok .+ (cyc - (ngrp + 1) / 2) * w), v[ok]; width=w * 0.85, color=simcol(cyc))
      end
    end
    MK.Legend(fig[nrow+1, 1:ncol], [MK.PolyElement(color=simcol(cyc)) for cyc in 1:n_cycles],
      ["cycle $cyc" for cyc in 1:n_cycles]; orientation=:horizontal, framevisible=false)
    MK.save(joinpath(cd, "simB_smape_$(sanitize(eco_list[e])).png"), fig); nwrote += 1
  end
  println("  candidate $ci: wrote $nwrote SMAPE charts")
end
println("done")
