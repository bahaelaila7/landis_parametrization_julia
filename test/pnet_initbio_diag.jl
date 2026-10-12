# How many observed (injected) cohorts would the PnET InitBiomass death rule kill?  (spec 6.5(2): TotalBiomass < InitBiomass at December of year>0)
#   PAN_DATA=... PAN_OUT=... ./julia_gdal.sh --project=. --threads=auto test/pnet_initbio_diag.jl <config.yml> trials=12 igel_mu=4 output_dir=...
# Runs a (short) search, then, with the newest best_params, (1) solves TotalBiomass for every observed cohort (sim-year-0 cohorts of the
# reference SoA and all injected cohorts) and counts TB < InitBiomass; (2) steps the reference SoA one year with the rule ON and OFF and
# reports the surviving-cohort fractions.
using Pan, JLD2, DataFrames, Statistics, Printf
const P = Pan
const PN = Pan.Plugins.PnETPlugin

function diag(d)
  outd = d.output_dir
  ck = sort(filter(f -> occursin(r"^best_params@\d+\.jld2$", f), readdir(outd)); by=f -> parse(Int, match(r"@(\d+)", f)[1]))
  best = JLD2.load_object(joinpath(outd, last(ck)))
  eps_ = PN.generate_pnet_eco_params(best)
  cst = PN.FIXED[]
  println("DIAG PFT map: ", join(["$(s)=$(Int(p))" for (s, p) in zip(best.SPECIES_LIST, best.PFT)], " "))
  # (1) injected + initial cohorts, solved with the best params
  rows = d.splots   # observed cohorts incl. every measurement year; unique (plot, species, age) at first sight = what gets injected
  n = 0; nlow = 0; agb_all = 0.0; agb_low = 0.0
  seen = Set{Tuple{Int,Int,Int}}()
  for r in eachrow(rows)
    key = (Int(r.plot_id), Int(r.eco_species_id), Int(r.age_calc) - Int(r.sim_year))     # birth year identifies a cohort across visits
    key in seen && continue
    push!(seen, key)
    ep = eps_[Int(r.eco_id)]; s = Int(r.eco_species_id)
    tb = PN.solve_total_biomass(Float64(r.agb_sum), ep, s)
    n += 1; agb_all += r.agb_sum
    if tb < ep.initbiomass[s]
      nlow += 1; agb_low += r.agb_sum
    end
  end
  @printf("DIAG InitBiomass: %d distinct observed cohorts, %d (%.1f%%) have solved TotalBiomass < InitBiomass (they would be killed at their first December); they carry %.2f%% of the observed cohort AGB\n",
    n, nlow, 100nlow / n, 100agb_low / agb_all)
  # (2) one-year survival with the rule on/off on the reference SoA
  for flag in (true, false)
    soa = P.PanCore.copy_and_reseed_soa(d.ref_soa, 1)
    cst2 = PN.PnETConst(; (f => (f === :init_biomass_death ? flag : getfield(cst, f)) for f in fieldnames(PN.PnETConst))...)
    ctx = (eco=eps_, cst=cst2, clim=PN.CLIMATE[], scratch=PN.scratch_pool())
    l0 = sum(soa.scalar.live)
    P.PanCore.process_plugin!(soa, PN.PnET, 1; ctx=ctx)
    l1 = sum(soa.scalar.live)
    @printf("DIAG one-year survival (init_biomass_death=%s): %d -> %d cohorts (%.1f%% removed)\n", flag, l0, l1, 100 * (l0 - l1) / l0)
  end
end

P.POST_SEARCH_HOOK[] = diag
ov = Dict{String,Any}()
for a in ARGS[2:end]
  k, v = split(a, '='; limit=2)
  ov[k] = something(tryparse(Int, v), tryparse(Float64, v), v)
end
P.run_from_yaml(ARGS[1]; overrides=ov)
