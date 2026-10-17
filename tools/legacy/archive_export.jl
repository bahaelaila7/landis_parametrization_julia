# Export a finished MO-CMA-ES archive (front-1 candidates): a flat parameter table archive_candidates.csv
# (one row/candidate: objectives + aggregate + every tunable param), and one candidates/candidate_<i>/
# params.jld2 per candidate (for the per-candidate scatter/TOST/trajectory plots).
#   Run: ./julia_gdal.sh --project=. tools/archive_export.jl [output_dir]
#   (default = A-only outputs; pass the stage-B output dir for the Sim B archive.)
using Pan, JLD2, DataFrames, CSV
const OUTDIR = length(ARGS) >= 1 ? ARGS[1] : "runs/fl5_l4cover_mocmaes_Aonly_Sglobal_ipop_v2_outputs"
st = JLD2.load_object(joinpath(OUTDIR, "search_state_latest.jld2"))
arch = collect(st.archive)
p0 = arch[1].x; sl = String.(p0.SPECIES_LIST); el = String.(p0.ECO_LIST); esi = p0.ECO_SPECIES_IDS
nobj = length(arch[1].fx.objectives)
# name the objective columns: 4 = joint dual (A_W,A_AGB,B_W,B_AGB); 2 = single sim (W,AGB); else objN
objnames = nobj == 4 ? ["A_W", "A_AGB", "B_W", "B_AGB"] : nobj == 2 ? ["A_W", "A_AGB"] : ["obj$(j)" for j in 1:nobj]
println("archive front-1: ", length(arch), " candidates; objectives/candidate = ", nobj, " ", objnames)

# S is now per-(eco,species); collapse to a per-species mean over the ecos that contain the species for this flat export.
_meanS(p, g) = (v = Float64[]; for e in eachindex(p.ECO_SPECIES_IDS); k = findfirst(==(g), p.ECO_SPECIES_IDS[e]); k === nothing || push!(v, Float64(p.S[e][k])); end; isempty(v) ? 0.0 : sum(v) / length(v))
function flat!(d, p)
  for g in eachindex(sl)
    d["S_$(sl[g])"] = _meanS(p, g); d["D_$(sl[g])"] = Float64(p.D[g]); d["LONGEVITY_$(sl[g])"] = Float64(p.LONGEVITY[g])
    length(p.MATURITY) >= g && (d["MATURITY_$(sl[g])"] = Float64(p.MATURITY[g]))
    d["SHADE_$(sl[g])"] = Float64(p.SHADE_TOL[g])
  end
  for e in eachindex(el), (li, g) in enumerate(esi[e])
    d["ANPP_$(el[e])_$(sl[g])"] = Float64(p.ANPP_MAX_SPP[e][li]); d["BMAX_$(el[e])_$(sl[g])"] = Float64(p.B_MAX_SPP[e][li])
    length(p.PROB_ESTAB_SPP) >= e && (d["PESTAB_$(el[e])_$(sl[g])"] = Float64(p.PROB_ESTAB_SPP[e][li]))
  end
  for e in eachindex(el); d["MINREL_$(el[e])"] = Float64(p.MIN_REL_BIOMASS[e][1]); end
end

rows = Dict{String,Any}[]
for (i, c) in enumerate(arch)
  d = Dict{String,Any}("candidate" => i, "aggregate" => c.fx.aggregate)
  for (j, o) in enumerate(c.fx.objectives); d[objnames[j]] = Float64(o); end
  flat!(d, c.x); push!(rows, d)
  cd = joinpath(OUTDIR, "candidates", "candidate_$(i)"); mkpath(cd)
  JLD2.save_object(joinpath(cd, "params.jld2"), c.x)
  # per-candidate params as a long-format CSV (parameter,value): meta first, then params sorted
  meta = ["candidate"; objnames; "aggregate"]
  ks = vcat(intersect(meta, collect(keys(d))), sort(setdiff(collect(keys(d)), meta)))
  CSV.write(joinpath(cd, "params.csv"), DataFrame(parameter=ks, value=[d[k] for k in ks]))
end
df = DataFrame(rows)
# column order: candidate, objectives, aggregate, then params
front = ["candidate"; objnames; "aggregate"]
select!(df, intersect(front, names(df))..., setdiff(names(df), front)...)
CSV.write(joinpath(OUTDIR, "archive_candidates.csv"), df)
println("wrote ", joinpath(OUTDIR, "archive_candidates.csv"), "  (", nrow(df), " rows × ", ncol(df), " cols)")
println("wrote per-candidate params.jld2 under ", joinpath(OUTDIR, "candidates"))
