# Pan 100-yr projection WITH establishment (AUTOES analogue): candidate GROWTH params (D,LONGEVITY,S,
# ANPP_MAX,B_MAX) + ESTABLISHMENT params from file — SHADE_TOL, MATURITY, PROB_ESTAB from
# runs/prob_estab_all_species.csv (groups n_seedling-weighted), MIN_REL_BIOMASS=[0.1,0.2,0.3,0.4,0.5]/eco.
# Start from first-measurement cohorts; make_sites(no_establishment=false) so establishment fires.
#   Run: ./julia_gdal.sh --project=. tools/pan_project100_autoes.jl <config.yml> <candidate params.jld2>
using Pan
import JLD2, YAML, DuckDB, DataFrames, CSV, Statistics, Setfield
const P = Pan; const D = P.Data; const BSP = P.BiomassSuccessionPlugin; const DF = DataFrames
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d); candpath = ARGS[2]
const HORIZONS = [25, 50, 75, 100]; const BINS = [10,20,30,40,50,60,80,100,120,150]
binidx(a) = (for (i, b) in enumerate(BINS); a < b && return i; end; length(BINS)+1)
binlabel(i) = i == 1 ? "≤$(BINS[1])" : i <= length(BINS) ? "$(BINS[i-1])–$(BINS[i])" : "$(BINS[end])+"

splots, eco_list, species_list, eco_species_ids, _ = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]),
  output_dir=String(cfg["tablename"]), filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])),
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false,
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  RNG=P.RNGType(UInt64(Int(g("seed", 1)))))
n_species = length(species_list); n_eco = length(eco_list)
l3_of(e) = String(split(eco_list[e], "|")[1])

# --- establishment params from file (n_seedling-weighted for groups) ---
f = CSV.read(joinpath(@__DIR__, "..", "runs", "prob_estab_all_species.csv"), DF.DataFrame)
f.category = String.(f.category); f.l3 = String.(f.l3)
wmean(v, w) = (s = sum(w); s > 0 ? sum(w .* v) / s : Statistics.mean(v))
# per (category,l3): shade_tol, maturity, prob_estab
byl3 = DF.combine(DF.groupby(f, [:category, :l3]),
  [:shade_tol, :n_seedling] => ((s, w) -> wmean(Float64.(s), Float64.(w))) => :st,
  [:maturity,  :n_seedling] => ((m, w) -> wmean(Float64.(m), Float64.(w))) => :mat,
  [:prob_estab,:n_seedling] => ((p, w) -> wmean(Float64.(p), Float64.(w))) => :pe)
pe_l3 = Dict((r.category, r.l3) => Float64(r.pe) for r in eachrow(byl3))
# per category (species-level trait): shade_tol, maturity over all rows
bycat = DF.combine(DF.groupby(f, :category),
  [:shade_tol, :n_seedling] => ((s, w) -> wmean(Float64.(s), Float64.(w))) => :st,
  [:maturity,  :n_seedling] => ((m, w) -> wmean(Float64.(m), Float64.(w))) => :mat)
st_cat  = Dict(r.category => Float64(r.st)  for r in eachrow(bycat))
mat_cat = Dict(r.category => Float64(r.mat) for r in eachrow(bycat))

SHADE_TOL = P.UIntType[clamp(round(P.UIntType, get(st_cat, species_list[gsp], 3.0)), 1, 5) for gsp in 1:n_species]
MATURITY  = P.FloatType[get(mat_cat, species_list[gsp], 20.0) for gsp in 1:n_species]
PROB_ESTAB_SPP = [P.FloatType[get(pe_l3, (species_list[Int(gsp)], l3_of(e)), 0.3) for gsp in eco_species_ids[e]] for e in 1:n_eco]
MIN_REL_BIOMASS = [P.FloatType[0.1, 0.2, 0.3, 0.4, 0.5] for _ in 1:n_eco]
println("establishment from file → MATURITY[1:5]=", round.(MATURITY[1:min(5,end)];digits=1),
        " SHADE_TOL[1:5]=", Int.(SHADE_TOL[1:min(5,end)]), " PROB_ESTAB eco1[1:5]=", round.(PROB_ESTAB_SPP[1][1:min(5,end)];digits=2))

# --- candidate growth params + establishment params ---
# Default: establishment {SHADE_TOL, MATURITY, PROB_ESTAB, MIN_REL} pulled from file (data seed), only
# growth+PROB_MORT taken from the candidate (the original Sim-A comparison behaviour). Set
# PAN_USE_CAND_ESTAB=1 to instead KEEP the candidate's OWN calibrated establishment — required for a
# Sim-B candidate, whose whole point is the fitted PROB_ESTAB (else the calibration is thrown away and
# only the calibrated PROB_MORT survives, an inconsistent mix).
base = JLD2.load_object(candpath)
p = base
if haskey(ENV, "PAN_USE_CAND_ESTAB")
  println("PAN_USE_CAND_ESTAB set → keeping candidate's OWN calibrated establishment (file ignored)")
else
  p = Setfield.@set p.SHADE_TOL = SHADE_TOL
  p = Setfield.@set p.MATURITY = MATURITY
  p = Setfield.@set p.PROB_ESTAB_SPP = PROB_ESTAB_SPP
  p = Setfield.@set p.MIN_REL_BIOMASS = MIN_REL_BIOMASS
end

# establishment is STOCHASTIC (PROB_ESTAB draws) → run NREP reseeded replicates and average.
const NREP = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
soa_ref = P.make_sites(splots, eco_species_ids; rng=P.RNGType(1), spinup=false, no_establishment=false)  # establishment ENABLED
eco_params = BSP.generate_eco_params(p); ctx = (eco_params=eco_params,)
site_pid = [Int(P.getsite(soa_ref, i).mapcode) for i in 1:soa_ref.n]
pid2key = Dict(Int(r.plot_id) => join((Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)), "_")
               for r in eachrow(unique(DF.select(splots, [:plot_id, :statecd, :unitcd, :countycd, :plot]))))
# Σ AGB per (plotkey, offset, eff, agebin) across reps → later ÷NREP = mean
acc = Dict{Tuple{String,Int,String,Int},Float64}()
reptot = Dict(off => Float64[] for off in HORIZONS)   # per-rep total AGB per horizon (rep spread)
for rep in 1:NREP
  soa = P.copy_and_reseed_soa(soa_ref, UInt64(rep))
  for y in 1:maximum(HORIZONS)
    P.PanCore.process_plugin!(soa, BSP.BiomassSuccession, y; ctx=ctx)
    y in HORIZONS || continue
    t = 0.0
    for i in 1:soa.n
      s = P.getsite(soa, i); pk = get(pid2key, site_pid[i], "?")
      for j in 1:Int(s.live)
        eff = species_list[Int(s.c_species[j])]; ab = binidx(round(Int, Float64(s.c_age[j]))); b = Float64(s.c_bio[j])
        acc[(pk, y, eff, ab)] = get(acc, (pk, y, eff, ab), 0.0) + b; t += b
      end
    end
    push!(reptot[y], t / soa.n)   # mean plot AGB this rep
  end
  print("rep $rep "); flush(stdout)
end
println()
rows = [(plotkey=pk, offset=off, eff=eff, agebin=ab, agb=v / NREP) for ((pk, off, eff, ab), v) in acc]
coh = DF.DataFrame(rows); coh.agebin_label = binlabel.(coh.agebin)
out = joinpath(dirname(candpath), "pan_cohorts_100_autoes.csv"); CSV.write(out, coh)
println("wrote $out  ($(DF.nrow(coh)) rows; mean over $NREP reps)")
rs = DF.DataFrame(offset=HORIZONS,
  mean_agb=[Statistics.mean(reptot[o]) for o in HORIZONS],
  std_agb=[length(reptot[o]) > 1 ? Statistics.std(reptot[o]) : 0.0 for o in HORIZONS], nreps=NREP)
CSV.write(joinpath(dirname(candpath), "pan_autoes_repstats.csv"), rs)
for r in eachrow(rs)
  println("  yr $(r.offset): mean plot AGB = $(round(r.mean_agb)) ± $(round(r.std_agb)) g/m²  (over $NREP reps)")
end
