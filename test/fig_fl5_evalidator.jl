# Fig 3b: FL5 total live AGB by species from FIA EvaliDator (TOTAL all-live-tree stocking, short tons),
# grouped to the SAME 14 tiered species + soft→hard colors + common-name labels as Fig 3a.
#   Run: ./julia_gdal.sh --project=. test/fig_fl5_evalidator.jl
using DataFrames, CairoMakie, DuckDB, CSV
const MK = CairoMakie
const EV = "/workspace/EVALIDATOR_FL5_DRYBIO_AG_LIVE_FL5_2015+.csv"
const FIADB = "/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb"

ev = CSV.read(EV, DataFrame)
sc1 = names(ev)[2]   # ALL_LIVE_STOCKING grouping column ("Total" = species total over stocking classes)
tot = ev[(string.(ev[!, sc1]) .== "Total") .& (string.(ev.SPECIES) .!= "Total"), :]
tot.spcd = [(m = match(r"SPCD\s+0*(\d+)", String(s)); m === nothing ? missing : parse(Int, m[1])) for s in tot.SPECIES]
tot = tot[.!ismissing.(tot.spcd), :]
println("Evalidator species (Total stocking): ", nrow(tot), "  total estimate(tons)=", round(Int, sum(tot.ESTIMATE)))

# SPCD → symbol (REF_SPECIES) ; symbol → spgrpcd (curated, same source as the analysis tiering) → tier
con = DBInterface.connect(DuckDB.DB(FIADB))
ref = DataFrame(DBInterface.execute(con, "SELECT SPCD spcd, UPPER(TRIM(SPECIES_SYMBOL)) sym, UPPER(TRIM(SFTWD_HRDWD)) sh, COMMON_NAME cn FROM REF_SPECIES"))
spcd_sym = Dict(Int(r.spcd) => String(r.sym) for r in eachrow(ref) if !ismissing(r.spcd))
spcd_sh  = Dict(Int(r.spcd) => (ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(ref) if !ismissing(r.spcd))
common = Dict(String(r.sym) => String(r.cn) for r in eachrow(ref) if !ismissing(r.cn))
sym_sh = Dict(String(r.sym) => (ismissing(r.sh) ? "H" : String(r.sh)) for r in eachrow(ref) if !ismissing(r.sym))
grp_of = Dict(String(r.sym) => Int(r.grp) for r in eachrow(DataFrame(DBInterface.execute(con, "SELECT DISTINCT UPPER(TRIM(species_symbol)) sym, spgrpcd grp FROM curated_cohorts_landis"))))
EXACT = Set(["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS"]); GRPS = Set([41,43])
function tier_spcd(spcd)
  haskey(spcd_sym, spcd) || return "_H"
  sym = spcd_sym[spcd]
  sym in EXACT && return sym
  get(grp_of, sym, -1) in GRPS && return "_GRP_$(grp_of[sym])"
  "_" * (get(spcd_sh, spcd, "H") in ("S", "H") ? spcd_sh[spcd] : "H")
end
tot.eff = [tier_spcd(s) for s in tot.spcd]
agg = combine(groupby(tot, :eff), :ESTIMATE => sum => :tons, :VARIANCE => sum => :var)
totd = Dict(r.eff => r.tons for r in eachrow(agg))
sed = Dict(r.eff => sqrt(max(r.var, 0.0)) for r in eachrow(agg))   # SE of the tier sum ≈ √Σvar (species independent)

# shared color/label scheme
species_list = ["ACRU","LIST2","NYBI","PIEL","PIPA2","PITA","QULA3","QUNI","QUVI","TAAS","_GRP_41","_GRP_43","_H","_S"]
isoft(s) = (su = uppercase(strip(s)); su == "_S" ? true : su == "_H" ? false : startswith(su, "_GRP_") ? false : get(sym_sh, su, "H") == "S")
soft = [s for s in species_list if isoft(s)]; hard = [s for s in species_list if !isoft(s)]
sgrad = MK.cgrad([:navy,:dodgerblue,:darkturquoise,:seagreen,:limegreen]); hgrad = MK.cgrad([:gold,:orange,:orangered,:red,:darkred])
shade(i, n) = n <= 1 ? 0.5 : (i - 1) / (n - 1)
color_of = Dict{String,MK.RGBAf}()
for (i, s) in enumerate(soft); color_of[s] = MK.RGBAf(sgrad[shade(i, length(soft))]); end
for (i, s) in enumerate(hard); color_of[s] = MK.RGBAf(hgrad[shade(i, length(hard))]); end
stack_order = vcat(soft, hard)
function label_for(s)
  su = uppercase(strip(s)); startswith(su, "_GRP_") && return "Group $(s[6:end]) ($s)"
  su == "_H" && return "Other Hardwoods (_H)"; su == "_S" && return "Other Softwoods (_S)"
  cn = get(common, su, nothing); isnothing(cn) ? s : "$cn ($s)"
end

ord = [s for s in stack_order if get(totd, s, 0.0) > 0]
fig = MK.Figure(size = (860, 560))
ax = MK.Axis(fig[1, 1]; xticks = (1:length(ord), [label_for(s) for s in ord]), xticklabelrotation = π/3,
  ylabel = "live AGB (short tons)", title = "FL5 by EvaliDator — total live AGB by species (all-live stocking)")
MK.barplot!(ax, 1:length(ord), [totd[s] for s in ord]; color = [color_of[s] for s in ord])
MK.errorbars!(ax, 1:length(ord), [totd[s] for s in ord], [sed[s] for s in ord]; color = :black, whiskerwidth = 8, linewidth = 1.2)
MK.save("runs/fig3b_fl5_evalidator_aggregate.png", fig; px_per_unit = 3)
for s in ord; println("  ", rpad(label_for(s), 30), round(Int, totd[s]), " tons"); end
println("wrote runs/fig3b_fl5_evalidator_aggregate.png")
