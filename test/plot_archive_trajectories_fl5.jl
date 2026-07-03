# Long-horizon projections for a SAMPLE of plots per (eco × land-use) split. For each of the 4 splits we
# greedily sample N=10 species-diverse plots, initialise each from its STARTING condition (observed
# sim_year-0 cohorts) and project HORIZON yr forward under every archive candidate (free run: model
# growth + mortality; no_establishment per Sim A; no future data to sync/disturb). HORIZON defaults to the
# max species longevity so every cohort reaches senescence/death. One figure per split, one panel per
# sampled plot: a line per (cohort × candidate) COLOURED BY SPECIES (legend), observed pts overlaid.
# Local competition is preserved (each plot's cohorts compete on their own site). Same-species/same-age
# cohorts are summed per year so a cohort is one continuous line (no key-collision zig-zags).
#   Run:  ./julia_gdal.sh --project=. test/plot_archive_trajectories_fl5.jl <config.yml> [horizon=maxLongevity] [nplots=10]
using Pan
import JLD2, YAML, CairoMakie, DataFrames, Random, DuckDB
const MK = CairoMakie; const P = Pan; const D = P.Data; const BSP = P.BiomassSuccessionPlugin
const DF = DataFrames

cfgpath = ARGS[1]
HORIZON_ARG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : nothing   # default: max species longevity (see below)
NPLOTS  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
cfg = YAML.load_file(cfgpath); g(k, d) = get(cfg, k, d)
outdir = cfg["output_dir"]
simB = get(ENV, "PAN_SIMB", "0") == "1"            # Sim B view: force free establishment (regeneration on)
no_estab = simB ? false : Bool(g("no_establishment", false))
rng = P.RNGType(UInt64(Int(g("seed", 1))))

# --- run data; pool TRAIN+VAL ---
val_frac = Float64(g("val_frac", 0.0))
split_rng = val_frac > 0 ? P.RNGType(UInt64(Int(g("split_seed", 42)))) : nothing
splots, eco_list, species_list, eco_species_ids, splots_val = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], eco_field=String(cfg["eco_field"]), tablename=String(cfg["tablename"]),
  output_dir=String(cfg["tablename"]), filter_eco_field=String(g("filter_eco_field", "epa_l4")),
  filter_ecos=String.(get(cfg, "filter_ecos", String[])), filter_plots=NTuple{4,Int}[],
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=val_frac, split_rng=split_rng,
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  RNG=P.RNGType(UInt64(Int(g("seed", 1)))))
n_eco = length(eco_list)
allplots = val_frac > 0 && splots_val !== nothing ? vcat(splots, splots_val) : splots

# --- archive ---
st = JLD2.load_object(joinpath(outdir, "search_state_latest.jld2"))
archive = collect(st.archive)
losses = [Float64(m.fx.aggregate) for m in archive]
lo, hi = minimum(losses), maximum(losses)
const CMAP = MK.cgrad(:viridis; rev=true); order = sortperm(losses; rev=true)
# default horizon = max species longevity over the archive → every cohort reaches senescence/death
maxlong = maximum(maximum(Float64.(m.x.LONGEVITY)) for m in archive)
HORIZON = HORIZON_ARG === nothing ? ceil(Int, maxlong) : HORIZON_ARG
println("archive: $(length(archive)) candidates; loss $(round(lo;digits=4))–$(round(hi;digits=4)); HORIZON=$(HORIZON) (maxLONGEVITY=$(round(maxlong))), N=$(NPLOTS)/split")
sanitize(s) = replace(String(s), r"[^A-Za-z0-9]" => "_")

# --- canonical species colour scheme (ported from test/fig_fl5_species_panels.jl): softwood blue→green,
#     hardwood yellow→red, shaded by within-class order over species_list ---
let fcon = DuckDB.connect(DuckDB.DB(cfg["cohorts_db_path"]))
  ref = DataFrames.DataFrame(DuckDB.execute(fcon, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh FROM REF_SPECIES"))
  global sym_sh = Dict(String(r.sym) => String(r.sh) for r in eachrow(ref) if !ismissing(r.sh))
  global grpclass = Dict{Int,String}()
  try
    gg = DataFrames.DataFrame(DuckDB.execute(fcon, "SELECT SPGRPCD spgrpcd, UPPER(TRIM(CLASS)) class FROM REF_SPECIES_GROUP"))
    global grpclass = Dict(Int(r.spgrpcd) => String(r.class) for r in eachrow(gg))
  catch; end
end
function is_soft(s)
  s = uppercase(strip(s))
  s == "_S" && return true; s == "_H" && return false
  if startswith(s, "_GRP_"); n = tryparse(Int, s[6:end]); return startswith(get(grpclass, something(n, -1), ""), "S"); end
  return get(sym_sh, s, "H") == "S"            # unknown → hardwood
end
soft = [s for s in species_list if is_soft(s)]
hard = [s for s in species_list if !is_soft(s)]
soft_grad = MK.cgrad([:navy, :dodgerblue, :darkturquoise, :seagreen, :limegreen])
hard_grad = MK.cgrad([:gold, :orange, :orangered, :red, :darkred])
shade(i, n) = n <= 1 ? 0.5 : (i - 1) / (n - 1)
color_of = Dict{String,MK.RGBAf}()
for (i, s) in enumerate(soft); color_of[s] = MK.RGBAf(soft_grad[shade(i, length(soft))]); end
for (i, s) in enumerate(hard); color_of[s] = MK.RGBAf(hard_grad[shade(i, length(hard))]); end
spcol(gsp) = color_of[species_list[gsp]]

# species present per plot (at sim_year 0) — for the greedy species-diverse sample
function sample_diverse(rows, n)
  init = rows[rows.sim_year .== 0, :]
  sp_of = Dict{Int,Set{Int}}()
  for r in eachrow(init); push!(get!(sp_of, Int(r.plot_id), Set{Int}()), Int(r.eco_species_id)); end
  pids = sort(collect(keys(sp_of)))            # deterministic
  covered = Set{Int}(); chosen = Int[]
  while length(chosen) < n && length(chosen) < length(pids)
    rest = setdiff(pids, chosen)
    # maximize newly-covered species, tie-break by total species then plot_id
    best = rest[argmax([(length(setdiff(sp_of[p], covered)), length(sp_of[p]), -p) for p in rest])]
    push!(chosen, best); union!(covered, sp_of[best])
  end
  chosen
end

for e in 1:n_eco
  rows_e = allplots[allplots.eco_id .== e, :]
  isempty(rows_e) && continue
  chosen = sample_diverse(rows_e, NPLOTS)
  sub = rows_e[in.(rows_e.plot_id, Ref(Set(chosen))), :]
  # Sim B starts from a SPINUP DEFICIT (bare sites) then spinup_cohorts! regenerates the stand from each
  # cohort's inferred establishment year (free establishment) — that's where the candidates' establishment
  # params diverge; Sim A starts from the observed cohorts and grows them.
  soa_ref = P.make_sites(sub, eco_species_ids; rng=P.RNGType(1), spinup=simB, no_establishment=no_estab)
  spinup_sub = simB ? D.get_spinup_cohorts(sub) : nothing
  site_pid = [Int(P.getsite(soa_ref, i).mapcode) for i in 1:soa_ref.n]
  keyrow = Dict(Int(r.plot_id) => (Int(r.statecd), Int(r.unitcd), Int(r.countycd), Int(r.plot)) for r in eachrow(sub))
  # observed (age,agb) per (site, local species) → overlaid on each plot's panel, coloured by species
  obs = Dict{Tuple{Int,Int},Vector{Tuple{Float64,Float64}}}()
  for r in eachrow(sub)
    si = findfirst(==(Int(r.plot_id)), site_pid); si === nothing && continue
    push!(get!(obs, (si, Int(r.eco_species_id)), Tuple{Float64,Float64}[]), (Float64(r.age_calc), Float64(r.agb_sum)))
  end

  # project HORIZON yr for each candidate; track each cohort by (site, sp_local, birthkey=year-age)
  function trajectory(params)
    soa = P.copy_and_reseed_soa(soa_ref, UInt64(1))
    eco_params = BSP.generate_eco_params(params)
    ctx = (BiomassSuccession=(eco_params=eco_params,),)
    simB && (soa = BSP.spinup_cohorts!(soa, spinup_sub, eco_params))   # regenerate stand from the deficit → year-0 state
    rec = Dict{Tuple{Int,Int,Int},Vector{Tuple{Float64,Float64}}}()
    snap!(y) = begin
      ag = Dict{Tuple{Int,Int,Int},Float64}()           # (site,sp,roundage) → Σbiomass (merge dup cohorts → no collision)
      for i in 1:soa.n
        s = P.getsite(soa, i)
        for j in 1:Int(s.live)
          k = (i, Int(s.c_species[j]), Int(round(Float64(s.c_age[j]))))
          ag[k] = get(ag, k, 0.0) + Float64(s.c_bio[j])
        end
      end
      for ((i, sp, ra), b) in ag
        push!(get!(rec, (i, sp, y - ra), Tuple{Float64,Float64}[]), (Float64(ra), b))
      end
    end
    snap!(0)
    for y in 1:HORIZON
      P.PanCore.process_plugin!(soa, BSP.BiomassSuccession, y; ctx=ctx.BiomassSuccession)
      snap!(y)
    end
    rec
  end
  trajs = [trajectory(m.x) for m in archive]

  # corruption scan (candidate 1): within a cohort, age must rise ~1/yr — duplicates/decreases mean a
  # key collision or merge; gaps mean a dropped year; biomass must be finite & non-negative.
  let collide = 0, gaps = 0, nonfin = 0, ncoh = 0
    for (_, seg) in trajs[1]
      ncoh += 1; ages = first.(seg)
      for t in 2:length(ages)
        d = ages[t] - ages[t-1]
        d <= 0 && (collide += 1); d > 1.5 && (gaps += 1)
      end
      any(x -> !isfinite(x) || x < 0, last.(seg)) && (nonfin += 1)
    end
    println("  [scan $(eco_list[e])] cohorts=$(ncoh) | collisions(Δage≤0)=$(collide) | age-gaps(Δ>1)=$(gaps) | bad-biomass=$(nonfin)")
  end

  np = soa_ref.n; ncol = min(5, np); nrow = cld(np, ncol)
  fig = MK.Figure(size=(330 * ncol + 180, 250 * nrow + 50))
  MK.Label(fig[0, 1:(ncol+1)], "$(eco_list[e]) — $(np) sampled plots projected $(HORIZON) yr from start, $(simB ? "Sim B (free establishment)" : "Sim A (no establishment)") (cohorts coloured by species, ● = observed); all $(length(archive)) candidates"; fontsize=12, font=:bold)
  for si in 1:np
    r, c = fldmod1(si, ncol)
    ax = MK.Axis(fig[r, c]; xlabel="cohort age (yr)", ylabel="AGB g/m²", title=join(keyrow[site_pid[si]], "-"))
    for k in order, (key, seg) in trajs[k]
      key[1] == si || continue                          # this plot's cohorts; one line per (cohort × candidate)
      length(seg) < 2 && continue
      MK.lines!(ax, first.(seg), last.(seg); color=spcol(eco_species_ids[e][key[2]]), linewidth=0.7, alpha=0.18)
    end
    for ((s, sp), pts) in obs
      s == si || continue
      MK.scatter!(ax, first.(pts), last.(pts); color=spcol(eco_species_ids[e][sp]), markersize=8, marker=:circle, strokecolor=(:black, 0.4), strokewidth=0.4, alpha=0.5)
    end
  end
  present = sort(unique(s[2] for s in keys(obs)))        # species present in this split → legend
  handles = [MK.LineElement(color=spcol(eco_species_ids[e][li]), linewidth=4) for li in present]
  labels = [species_list[eco_species_ids[e][li]] for li in present]
  MK.Legend(fig[1:nrow, ncol+1], handles, labels, "Species"; framevisible=true, labelsize=10)
  out = joinpath(outdir, "trajectories_$(sanitize(eco_list[e])).png")
  MK.save(out, fig); println("wrote $out  ($(np) plots, $(length(present)) species)")
end
