# Export a calibrated parameter set (JLD2) as a self-contained YAML roster (src/config/params_yaml.jl), with the mapping metadata
# a scenario needs to place NEW plots into the parameter space.
#   ./julia_gdal.sh --project=. tools/export_params_yaml.jl <training_config.yml> <params.jld2> <out.yaml> "<name>" ["<description>"] ['<provenance json>']
# species_map: every FIA tree species symbol in REF_SPECIES → its parameter species, by the SAME rule the training data
#   prep used (Data.assign_tiered_species!): an exact parameter species keeps its own name; otherwise the
#   "_GRP_<spgrpcd>_<shade tier>" group when that group is a parameter species; otherwise the softwood/hardwood catch-all
#   "_S_<tier>" / "_H_<tier>" (tier = HST/LST from the shade-tolerance table, NA when unknown), falling back to "_S"/"_H"
#   variants that exist. Species with no possible target are omitted (and reported).
# eco_mapping: the eco-cell rule of the config (EPA L3 field; 4-cell site class from the dominant forested condition's
#   SITECLCD in the listed L3s, one cell "X" elsewhere).
using Pan, JLD2, DuckDB, DataFrames, JSON3
const P = Pan
cfg = P.load_config(ARGS[1]); p = P._load_params_from_path(ARGS[2]); out = ARGS[3]; name = ARGS[4]   # (upgrades older HBAD param files)
desc = length(ARGS) >= 5 ? ARGS[5] : ""
prov = length(ARGS) >= 6 ? Dict{String,Any}(String(k) => v for (k, v) in JSON3.read(ARGS[6])) : Dict{String,Any}()
db = DuckDB.DB(cfg["cohorts_db_path"]; readonly=true); con = DBInterface.connect(db)
# (symbol → species group, softwood/hardwood) exactly as the training cohort table carries them; species absent from the
# data fall back to REF_SPECIES (eastern group code E_SPGRPCD).
tbl = cfg["tablename"]
ref = DataFrame(DBInterface.execute(con, """
  WITH d AS (SELECT UPPER(TRIM(species_symbol)) sym, any_value(spgrpcd) spgrpcd, any_value(sftwd_hrdwd) sh FROM $(tbl) GROUP BY 1),
       r AS (SELECT UPPER(TRIM(SPECIES_SYMBOL)) sym, E_SPGRPCD spgrpcd, SFTWD_HRDWD sh FROM REF_SPECIES WHERE SPECIES_SYMBOL IS NOT NULL)
  SELECT * FROM d UNION ALL SELECT * FROM r WHERE sym NOT IN (SELECT sym FROM d)"""))
tier = Dict{String,String}()
if cfg["shade_tier_csv"] !== nothing
  st = DataFrame(DBInterface.execute(con, "SELECT UPPER(TRIM(sym)) sym, shade_class FROM read_csv_auto('$(cfg["shade_tier_csv"])')"))
  for r in eachrow(st); tier[String(r.sym)] = Int(r.shade_class) <= 3 ? "LST" : "HST"; end
end
spset = Set(String.(p.SPECIES_LIST))
tiered = cfg["shade_tier_csv"] !== nothing
smap = Dict{String,String}(); missing_ = String[]
for r in eachrow(ref)
  s = uppercase(strip(String(r.sym)))
  if s in spset; smap[s] = s; continue; end
  t = tiered ? get(tier, s, "NA") : ""; suf = isempty(t) ? "" : "_" * t
  sh = coalesce(r.sh, "H") |> string
  cands = String[]
  r.spgrpcd === missing || push!(cands, "_GRP_$(r.spgrpcd)$(suf)")
  push!(cands, "_$(sh)$(suf)", "_$(sh)", "_$(sh)_NA", "_H$(suf)", "_H_NA", "_H")
  k = findfirst(in(spset), cands)
  k === nothing ? push!(missing_, s) : (smap[s] = cands[k])
end
eco_mapping = Dict{String,Any}(
  "field" => cfg["eco_field"],
  "site_class_from" => "COND.SITECLCD of the dominant forested condition (max CONDPROP_UNADJ, COND_STATUS_CD = 1)",
  "site_class_cells" => Dict("A" => [1, 2, 3], "B" => [4], "C" => [5], "D" => [6, 7]),
  "cell_l3s" => cfg["siteclass_l3s"],
  "other_cell" => "X", "missing_cell" => "NA",
  "fallback" => "exact cell → 'NA' cell → the L3's most common cell (scenario runner)")
region = Dict{String,Any}("states" => cfg["filter_states"], "l3" => cfg["filter_ecos"],
  "eco_cells" => String.(p.ECO_LIST))
prov["source_jld2"] = get(prov, "source_jld2", abspath(ARGS[2])); prov["training_config"] = get(prov, "training_config", abspath(ARGS[1]))
y = P.params_to_yaml(p; name=name, description=desc, region=region, provenance=prov, species_map=smap, eco_mapping=eco_mapping)
q, meta = P.params_from_yaml(y)
isempty(P.params_diff(p, q)) || error("round trip failed: $(P.params_diff(p, q))")
write(out, y)
println("wrote $out: plugin $(meta.plugin), $(length(p.SPECIES_LIST)) species, $(length(p.ECO_LIST)) eco cells, species_map $(length(smap)) FIA symbols ($(length(missing_)) unmapped); warnings: $(meta.warnings)")
