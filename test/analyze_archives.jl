# Post-run analysis of CMA-MAE (or any MO) archives. For EACH run output dir it writes, IN that dir:
#   - archive_candidates.csv : one row per candidate × land-use class × species, with the candidate's
#                              train_loss + val_loss and its parameters.
#   - archive_umap.png       : UMAP of that run's recovered parameter sets, coloured by validation loss.
# val_loss comes from archive_eval.csv (written by the driver's held-out re-evaluation); falls back to
# train loss if absent.
#   Run:  ./julia_gdal.sh --project=. test/analyze_archives.jl <out_dir1> [<out_dir2> ...]
using Pan
import JLD2, UMAP, Random, Statistics, DuckDB
import CairoMakie
const MK = CairoMakie

dirs = isempty(ARGS) ?
  ["runs/cmame_lu_v1_base_outputs", "runs/cmame_lu_v2_sobol_outputs", "runs/cmame_lu_v3_alpha05_outputs", "runs/cmame_lu_v4_explore05_outputs"] :
  ARGS
label(d) = replace(basename(d), "cmame_lonbh_" => "hinge:", "cmame_lonb_" => "nobin:", "cmame_lu_" => "", "_outputs" => "")
const HEADER = "candidate,train_loss,val_loss,eco_landuse,species,D,LONGEVITY,MATURITY,SHADE_TOL,S,ANPP_MAX,B_MAX,PROB_MORT,PROB_ESTAB,MIN_REL_BIOMASS"

# candidate -> val_loss from the driver's archive_eval.csv (empty Dict if missing)
function load_val(d)
  f = joinpath(d, "archive_eval.csv"); v = Dict{Int,Float64}()
  isfile(f) || return v
  for (li, ln) in enumerate(eachline(f)); li == 1 && continue
    p = split(ln, ","); v[parse(Int, p[1])] = parse(Float64, p[3])
  end
  return v
end

# Flatten one candidate's BiomassSuccessionParams into a fixed-order numeric feature vector (for UMAP).
function feat(p)
  v = Float64[]
  for gsp in eachindex(p.SPECIES_LIST)
    push!(v, Float64(p.D[gsp]), Float64(p.LONGEVITY[gsp]),
            Float64(length(p.MATURITY) >= gsp ? p.MATURITY[gsp] : 0), Float64(p.SHADE_TOL[gsp]))
  end
  for eco_id in eachindex(p.ECO_LIST), sp_local in eachindex(p.ECO_SPECIES_IDS[eco_id])
    gsp = Int(p.ECO_SPECIES_IDS[eco_id][sp_local])     # S is global per-species now
    push!(v, Float64(p.S[gsp]), Float64(p.ANPP_MAX_SPP[eco_id][sp_local]),
            Float64(p.B_MAX_SPP[eco_id][sp_local]), Float64(p.PROB_MORT_SPP[eco_id][sp_local]),
            Float64(length(p.PROB_ESTAB_SPP) >= eco_id ? p.PROB_ESTAB_SPP[eco_id][sp_local] : 0))
  end
  for eco_id in eachindex(p.ECO_LIST); push!(v, Float64(p.MIN_REL_BIOMASS[eco_id][1])); end
  return v
end

# one long CSV row per (candidate, eco, species)
function candidate_rows!(io, ci, tr, vl, p)
  for eco_id in eachindex(p.ECO_LIST), (sp_local, gsp) in enumerate(p.ECO_SPECIES_IDS[eco_id])
    gsp = Int(gsp)
    mat = length(p.MATURITY) >= gsp ? round(Float64(p.MATURITY[gsp]), digits=1) : ""
    pes = length(p.PROB_ESTAB_SPP) >= eco_id ? round(Float64(p.PROB_ESTAB_SPP[eco_id][sp_local]), digits=4) : ""
    println(io, join([ci, round(tr, digits=3), vl === missing ? "" : round(vl, digits=3),
      p.ECO_LIST[eco_id], p.SPECIES_LIST[gsp],
      round(Float64(p.D[gsp]), digits=3), round(Float64(p.LONGEVITY[gsp]), digits=1), mat, Int(p.SHADE_TOL[gsp]),
      round(Float64(p.S[gsp]), digits=4), round(Float64(p.ANPP_MAX_SPP[eco_id][sp_local]), digits=1),
      round(Float64(p.B_MAX_SPP[eco_id][sp_local]), digits=1), round(Float64(p.PROB_MORT_SPP[eco_id][sp_local]), digits=5),
      pes, round(Float64(p.MIN_REL_BIOMASS[eco_id][1]), digits=4)], ","))
  end
end

for d in dirs
  f = joinpath(d, "search_state_latest.jld2")
  isfile(f) || (println("skip (no state): $d"); continue)
  st = JLD2.load_object(f); v = label(d); valmap = load_val(d)
  feats = Vector{Float64}[]; trains = Float64[]; vals = Float64[]
  open(joinpath(d, "archive_candidates.csv"), "w") do io
    println(io, HEADER)
    for (ci, m) in enumerate(st.archive)
      p = m.x; tr = Float64(m.fx.aggregate); vl = get(valmap, ci, missing)
      candidate_rows!(io, ci, tr, vl, p)
      push!(feats, feat(p)); push!(trains, tr); push!(vals, vl === missing ? tr : vl)
    end
  end
  println("$v: archive=$(length(st.archive)) → archive_candidates.csv" * (isempty(valmap) ? " (no val)" : " (+val)"))

  # ---- per-run UMAP of this archive's parameter sets, coloured by validation loss ----
  if length(feats) >= 3
    X = reduce(hcat, feats); mu = Statistics.mean(X; dims=2); sd = Statistics.std(X; dims=2); sd[sd .== 0] .= 1
    Z = (X .- mu) ./ sd
    Random.seed!(7)
    emb = UMAP.fit(Z, 2; n_neighbors=max(2, min(15, size(Z, 2) - 1)), min_dist=0.4).embedding
    cval = isempty(valmap) ? trains : vals
    clab = isempty(valmap) ? "train loss" : "validation loss"
    fig = MK.Figure(size=(820, 660))
    MK.Label(fig[0, 1:2], "$v — UMAP of archive parameter sets ($(length(feats)) elites) — colour = $clab"; fontsize=13, font=:bold)
    ax = MK.Axis(fig[1, 1]; xlabel="UMAP-1", ylabel="UMAP-2")
    MK.scatter!(ax, emb[1, :], emb[2, :]; color=cval, colormap=MK.cgrad(:viridis; rev=true), markersize=13, strokecolor=:black, strokewidth=0.5)
    MK.Colorbar(fig[1, 2]; colormap=MK.cgrad(:viridis; rev=true), colorrange=(minimum(cval), maximum(cval)), label="$clab (yellow = lower = better)")
    MK.save(joinpath(d, "archive_umap.png"), fig)
    println("   wrote $(joinpath(d, "archive_umap.png"))")
  else
    println("   too few elites for a UMAP ($(length(feats)))")
  end

  # ---- per-run metrics trajectory: best train+val loss (one graph); pop & archive size (another) ----
  mf = joinpath(d, "metrics.csv")
  if isfile(mf)
    it = Int[]; tr = Float64[]; vl = Union{Float64,Missing}[]; ps = Float64[]; asz = Float64[]
    for (li, ln) in enumerate(eachline(mf)); li == 1 && continue
      c = split(ln, ",")
      push!(it, parse(Int, c[1])); push!(tr, parse(Float64, c[2]))
      push!(vl, isempty(c[3]) ? missing : parse(Float64, c[3]))
      push!(ps, parse(Float64, c[4])); push!(asz, parse(Float64, c[5]))
    end
    fig = MK.Figure(size=(920, 780))
    ax1 = MK.Axis(fig[1, 1]; xlabel="iteration (generation)", ylabel="loss", title="$v — best train & validation loss")
    MK.lines!(ax1, it, tr; color=:steelblue, linewidth=2, label="best train loss")
    vmask = .!ismissing.(vl)
    any(vmask) && MK.lines!(ax1, it[vmask], Float64.(vl[vmask]); color=:firebrick, linewidth=2, label="best val loss")
    MK.axislegend(ax1; position=:rt)
    ax2 = MK.Axis(fig[2, 1]; xlabel="iteration (generation)", ylabel="count", title="$v — population & archive size")
    MK.lines!(ax2, it, ps; color=:seagreen, linewidth=2, label="population size (λ)")
    MK.lines!(ax2, it, asz; color=:darkorange, linewidth=2, label="archive size")
    MK.axislegend(ax2; position=:rb)
    MK.save(joinpath(d, "archive_metrics.png"), fig)
    println("   wrote $(joinpath(d, "archive_metrics.png"))  ($(length(it)) iterations)")
  elseif isfile(joinpath(d, "losses.duckdb"))
    # MOLBSA (and other writers without metrics.csv): convergence from losses.duckdb's per-new-best rows.
    db = DuckDB.connect(DuckDB.DB(joinpath(d, "losses.duckdb")))
    cv = DuckDB.execute(db, "SELECT iteration, total_loss, archive_size FROM total_loss ORDER BY iteration")
    it = Int[]; tr = Float64[]; asz = Float64[]
    for r in cv; push!(it, Int(r.iteration)); push!(tr, Float64(r.total_loss)); push!(asz, Float64(r.archive_size)); end
    if !isempty(it)
      fig = MK.Figure(size=(920, 780))
      ax1 = MK.Axis(fig[1, 1]; xlabel="iteration", ylabel="best train loss", yscale=log10, title="$v — convergence (best-so-far train loss)")
      MK.lines!(ax1, it, tr; color=:steelblue, linewidth=2)
      MK.scatter!(ax1, it, tr; color=:steelblue, markersize=5)
      ax2 = MK.Axis(fig[2, 1]; xlabel="iteration", ylabel="archive size", title="$v — archive size")
      MK.lines!(ax2, it, asz; color=:darkorange, linewidth=2)
      MK.save(joinpath(d, "archive_metrics.png"), fig)
      println("   wrote $(joinpath(d, "archive_metrics.png"))  ($(length(it)) new-best steps, from losses.duckdb)")
    end
  end
end
println("=== ARCHIVE ANALYSIS DONE ===")
