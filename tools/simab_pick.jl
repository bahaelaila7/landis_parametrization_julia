# Helpers of runs_local/simab_campaign.sh. Nothing here uses the TEST split.
#   simab_pick.jl knee  <A_dir> <out_dir>                          → <out_dir>/knee.jld2 (the gdigel archive's knee) + refs.env (CALIB_AREF/WREF[/DREF/KAPPA])
#   simab_pick.jl seeds <A_dir> <refined.jld2> <out.duckdb> <mu> <cfgB.yml>   → Sim-B seed DB (sobol_results): the refined knee + (mu-1) archive points spread over ΣA,
#                                  each REMAPPED BY NAME into Sim B's own species/eco lists (as tools/seed_simB_start.jl: Sim A and Sim B tables tier species differently)
#   simab_pick.jl pickB <B_dir> <out.jld2>                         → knee of the Sim-B archive on (ΣB_W, ΣB_AGB) (latest search_state@N.jld2)
using Pan, JLD2, CSV, DataFrames, Statistics, DuckDB, Serialization
cmd = ARGS[1]
if cmd == "knee"
  adir, outd = ARGS[2], ARGS[3]; mkpath(outd)
  A = CSV.read(joinpath(adir, "mo_archive.csv"), DataFrame)
  kn = findfirst(r -> occursin("knee", String(coalesce(r.role, ""))), eachrow(A)); kn === nothing && error("no knee in $adir/mo_archive.csv")
  cp(joinpath(adir, "mo_archive", A.file[kn]), joinpath(outd, "knee.jld2"); force=true)
  u = JLD2.load_object(joinpath(adir, "mo_archive_u.jld2"))
  open(joinpath(outd, "refs.env"), "w") do io
    println(io, "export CALIB_AREF=$(u.aref)"); println(io, "export CALIB_WREF=$(u.wref)")
    hasproperty(u, :dref) && u.dref != 1.0 && println(io, "export CALIB_DREF=$(u.dref)")
    isfile(joinpath(adir, "kappa.txt")) && println(io, "export CALIB_KAPPA=$(strip(read(joinpath(adir, "kappa.txt"), String)))")
  end
  println("knee = $(A.file[kn]) → $outd/knee.jld2")
elseif cmd == "seeds"
  adir, refined, outdb, mu, cfgBp = ARGS[2], ARGS[3], ARGS[4], parse(Int, ARGS[5]), ARGS[6]
  A = CSV.read(joinpath(adir, "mo_archive.csv"), DataFrame)
  kn = findfirst(r -> occursin("knee", String(coalesce(r.role, ""))), eachrow(A))
  others = [i for i in 1:nrow(A) if i != kn]
  pick = isempty(others) ? Int[] : others[unique(round.(Int, range(1, length(others); length=min(mu - 1, length(others)))))]
  raw = vcat([(JLD2.load_object(refined), A.sumA[kn], A.sumW[kn])], [(JLD2.load_object(joinpath(adir, "mo_archive", A.file[i])), A.sumA[i], A.sumW[i]) for i in pick])
  # ---- Sim B context (own species/eco lists) and the remap of tools/seed_simB_start.jl
  P = Pan; BSP = P.BiomassSuccessionPlugin; cfg = P.load_config(cfgBp)
  _, eco_list, species_list, eco_species_ids, _, _ = P.prepare_from_cfg(cfg)
  BSP.SHADE_TOL_TABLE[] = P._load_shade_tol_table(cfg["shade_tol_csv"])
  BSP.MATURITY_TABLE[] = P._load_maturity_table(cfg["maturity_csv"])
  pet = isfile(cfg["maturity_csv"]) ? CSV.read(cfg["maturity_csv"], DataFrame) : nothing
  function remap(simA)
    p = BSP.generate_biomass_params(species_list, eco_list, eco_species_ids; rng=P.RNGType(UInt64(1)), no_establishment=false)
    ai = Dict((simA.ECO_LIST[e], simA.SPECIES_LIST[Int(gid)]) => (e, j) for (e, ids) in enumerate(simA.ECO_SPECIES_IDS) for (j, gid) in enumerate(ids))
    asp = Dict(s => k for (k, s) in enumerate(simA.SPECIES_LIST)); nhit = 0; nmiss = 0
    for (e, ids) in enumerate(eco_species_ids), (j, gid) in enumerate(ids)
      key = (eco_list[e], species_list[gid])
      if haskey(ai, key)
        ea, ja = ai[key]
        p.S[e][j] = simA.S[ea][ja]; p.B_MAX_SPP[e][j] = simA.B_MAX_SPP[ea][ja]; p.ANPP_MAX_SPP[e][j] = simA.ANPP_MAX_SPP[ea][ja]
        p.PROB_MORT_SPP[e][j] = simA.PROB_MORT_SPP[ea][ja]; nhit += 1
      else
        p.PROB_MORT_SPP[e][j] = 0f0; nmiss += 1
      end
    end
    for (k, s) in enumerate(species_list); haskey(asp, s) || continue; a = asp[s]; p.D[k] = simA.D[a]; p.LONGEVITY[k] = simA.LONGEVITY[a]; end
    st = BSP.SHADE_TOL_TABLE[]
    for (k, s) in enumerate(species_list)
      v = st === nothing ? nothing : get(st, uppercase(s), nothing)
      p.SHADE_TOL[k] = v !== nothing ? v : (endswith(s, "_HST") ? 4 : endswith(s, "_LST") ? 2 : 3)
    end
    P._seed_maturity!(p, species_list)
    for (e, ids) in enumerate(eco_species_ids), (j, gid) in enumerate(ids)
      if pet === nothing; p.PROB_ESTAB_SPP[e][j] = 0.1f0; continue; end
      nm = uppercase(species_list[gid]); l3 = String(first(split(eco_list[e], "|lu=")))
      sel = pet[(uppercase.(String.(pet.category)) .== nm) .& (String.(string.(pet.l3)) .== l3), :prob_estab]
      isempty(sel) && (sel = pet[(uppercase.(String.(pet.species)) .== nm) .& (String.(string.(pet.l3)) .== l3), :prob_estab])
      isempty(sel) && (sel = pet[uppercase.(String.(pet.species)) .== nm, :prob_estab])
      v = isempty(sel) ? Statistics.median(skipmissing(pet.prob_estab)) : Statistics.median(skipmissing(sel))
      p.PROB_ESTAB_SPP[e][j] = Float32(clamp(v, 0.0, 1.0))
    end
    P._seed_min_rel!(p, eco_list)
    println("remap: growth/PROB_MORT for $nhit (eco,species) cells of Sim B, $nmiss new cells (PROB_MORT 0)")
    p
  end
  seeds = [(remap(p), a, w) for (p, a, w) in raw]
  isfile(outdb) && rm(outdb)
  db = DuckDB.DB(outdb); con = DuckDB.connect(db)
  DuckDB.execute(con, "CREATE TABLE sobol_results (run_id VARCHAR, sobol_idx INTEGER, mean_loss DOUBLE, std_loss DOUBLE, median_loss DOUBLE, sumW DOUBLE, sumAGB DOUBLE, params_blob BLOB)")
  for (i, (p, a, w)) in enumerate(seeds)
    buf = IOBuffer(); Serialization.serialize(buf, p)
    DuckDB.execute(con, "INSERT INTO sobol_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ["simab_seeds", i, Float64(i), 0.0, Float64(i), Float64(w), Float64(a), take!(buf)])   # mean_loss = rank (refined knee first)
  end
  close(db); println("SEEDS=$(length(seeds)) → $outdb")
elseif cmd == "pickB"
  bdir, outp = ARGS[2], ARGS[3]
  ck = sort(filter(f -> occursin(r"^search_state@\d+\.jld2$", f), readdir(bdir)); by=f -> parse(Int, match(r"@(\d+)", f)[1]))
  isempty(ck) && error("no search_state in $bdir")
  arch = collect(JLD2.load_object(joinpath(bdir, last(ck))).archive)
  pts = [(sum(Float64.(c.fx.objectives[1:2:end])), sum(Float64.(c.fx.objectives[2:2:end]))) for c in arch]
  nd = [i for i in eachindex(pts) if !any(j -> pts[j][1] <= pts[i][1] && pts[j][2] <= pts[i][2] && pts[j] != pts[i], eachindex(pts))]
  xs = [pts[i][1] for i in nd]; ys = [pts[i][2] for i in nd]
  # knee = the front point farthest below the chord joining the two extremes in normalised objective space (same rule as the Sim A archive,
  # src/gdigel_driver.jl _write_mo_archive); objectives divided by their maximum over the front. Chosen on the TRAIN objectives only.
  o = sortperm(xs); nxs = xs[o] ./ maximum(xs); nys = ys[o] ./ maximum(ys); nn = length(o)
  if nn >= 3
    x1, y1, x2, y2 = nxs[1], nys[1], nxs[nn], nys[nn]; L = hypot(x2 - x1, y2 - y1) + 1e-300
    dist = [((y2 - y1) * (nxs[i] - x1) - (x2 - x1) * (nys[i] - y1)) / L for i in 1:nn]
    best = nd[o[argmin(dist)]]
  else
    best = nd[o[argmin(max.(nxs, nys))]]
  end
  JLD2.save_object(outp, arch[best].x); println("Sim B pick: cand $best of $(length(nd)) non-dominated (ΣW=$(round(pts[best][1], digits=4)), ΣAGB=$(round(pts[best][2], digits=4))) → $outp")
else
  error("unknown command $cmd")
end
