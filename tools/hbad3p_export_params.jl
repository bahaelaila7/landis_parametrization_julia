# Export a fitted HBAD3P parameter set (prototype pipeline directory results/hbad3/fit3p_<tag>/: species.csv, globals.csv, recruitment.csv, meta.txt, log.txt) as the plugin's parameter directory
# (params_species.csv, params_global.csv, params_recruitment.csv, meta.txt), optionally the plot-normal water table and the yaml roster of the scenario engine.
#   ./julia_gdal.sh --project=. tools/hbad3p_export_params.jl <fit3p dir> <out dir> [--region GAFL]
#        [--intervals <intervals.arrow> --water <interval_water.csv>]        writes <out dir>/plot_normal_water.csv (st,u,c,p,wn): wn = aetpet_n of the plot's intervals (normal per plot)
#        [--yaml <out.yaml> --name "<name>" [--description "<text>"] [--base <roster.yaml>] [--wn-path <path of plot_normal_water.csv relative to the repo root>]]
# --base (default results/params_yaml/r4_hbad2_v2j.yaml) supplies region / eco cells / eco_mapping (the parameters do not depend on the cell); species_map = every FIA symbol of REF_SPECIES (database PAN_FIA_DB, read only) ->
# its own key if the set has a row for it, else `_S` / `_H` by SFTWD_HRDWD (the rule of Data.default_key_resolver).
using Pan, CSV, DataFrames, Arrow, Statistics, DuckDB, YAML, SHA
const P = Pan
function parse_args(argv)
  pos = String[]; opt = Dict{String,String}(); i = 1
  while i <= length(argv)
    if startswith(argv[i], "--"); opt[argv[i][3:end]] = argv[i+1]; i += 2; else push!(pos, argv[i]); i += 1; end
  end
  pos, opt
end
pos, opt = parse_args(ARGS)
length(pos) == 2 || error("usage: hbad3p_export_params.jl <fit3p dir> <out dir> [--region R] [--intervals f --water f] [--yaml f --name n]")
fit, out = pos; mkpath(out)
repo = dirname(@__DIR__)
for f in ("species.csv", "globals.csv", "recruitment.csv", "meta.txt"); isfile(joinpath(fit, f)) || error("$(joinpath(fit, f)) missing"); end

# --- parameter tables: the fit's own numbers, unchanged (lnε = log(eps) is formed by the plugin loader exactly as the prototype's load_fit3p does)
S = CSV.read(joinpath(fit, "species.csv"), DataFrame)
G = CSV.read(joinpath(fit, "globals.csv"), DataFrame)
R = CSV.read(joinpath(fit, "recruitment.csv"), DataFrame)
need = collect(String.(P.HBAD3PPlugin.GLOB_NAMES))
miss = setdiff(need, String.(G.name)); isempty(miss) || error("globals.csv lacks $(miss) (an older fit: the plugin needs all 32 globals)")
keys_ = String.(S.key)
for k in keys_; "lnF:" * k in R.name || error("recruitment.csv lacks lnF:$k"); end
CSV.write(joinpath(out, "params_species.csv"), select(S, :key, :eps, :Hmax, :ST, :longevity))
CSV.write(joinpath(out, "params_global.csv"), DataFrame(name=need, value=[Float64(G.value[findfirst(==(n), String.(G.name))]) for n in need]))
CSV.write(joinpath(out, "params_recruitment.csv"), R)
# --- meta: the fit's meta.txt + provenance
region = get(opt, "region", "")
if isempty(region) && isfile(joinpath(fit, "log.txt"))
  m = match(r"^region (\S+)", first(eachline(joinpath(fit, "log.txt")))); m === nothing || (region = String(m[1]))
end
open(joinpath(out, "meta.txt"), "w") do io
  for l in eachline(joinpath(fit, "meta.txt")); isempty(strip(l)) || println(io, l); end
  println(io, "source=", abspath(fit))
  println(io, "region=", region)
  if isfile(joinpath(fit, "log.txt"))
    for l in eachline(joinpath(fit, "log.txt")); startswith(l, "FIT ") && println(io, "fit_log=", strip(l)); end
  end
end
println("wrote parameter directory $out ($(length(keys_)) keys: $(join(keys_, ",")); region $(region))")

# --- plot-normal water index
if haskey(opt, "intervals") && haskey(opt, "water")
  iv = DataFrame(Arrow.Table(opt["intervals"])); w = CSV.read(opt["water"], DataFrame)
  d = innerjoin(select(iv, :iid, :st, :u, :c, :p), select(w, :iid, :aetpet_n), on=:iid)
  g = combine(groupby(d, [:st, :u, :c, :p]), :aetpet_n => minimum => :lo, :aetpet_n => maximum => :hi, :aetpet_n => mean => :wn, nrow => :n)
  spread = maximum(g.hi .- g.lo)
  spread < 1e-9 || @warn "the normal differs between the intervals of a plot (max spread $(spread)); the mean is used"
  sort!(g, [:st, :u, :c, :p])
  CSV.write(joinpath(out, "plot_normal_water.csv"), select(g, :st, :u, :c, :p, :wn))
  println("wrote $(joinpath(out, "plot_normal_water.csv")): $(nrow(g)) plots (from $(nrow(d)) intervals), wn median $(round(median(g.wn); digits=4)), range [$(round(minimum(g.wn); digits=3)), $(round(maximum(g.wn); digits=3))], max spread within a plot $(spread)")
end

# --- yaml roster
if haskey(opt, "yaml")
  base = YAML.load_file(get(opt, "base", joinpath(repo, "results", "params_yaml", "r4_hbad2_v2j.yaml")))
  name = get(opt, "name", "HBAD3P"); desc = get(opt, "description", "")
  t = P.HBAD3PPlugin.load_3p_tables(out)
  scen = P.ScenarioConfig()
  db = DuckDB.DB(scen.fia_db; readonly=true); con = DBInterface.connect(db)
  ref = DataFrame(DBInterface.execute(con, "SELECT UPPER(TRIM(SPECIES_SYMBOL)) sym, any_value(SFTWD_HRDWD) sh FROM REF_SPECIES WHERE SPECIES_SYMBOL IS NOT NULL GROUP BY 1 ORDER BY 1"))
  resolve = P.Data.default_key_resolver(String.(t.keys))
  smap = Dict{String,String}(String(r.sym) => resolve(String(r.sym), String(coalesce(r.sh, "H"))) for r in eachrow(ref))
  eco = String.(base["pan_params"]["region"]["eco_cells"])
  ids = [collect(1:length(t.keys)) for _ in eco]
  p = P.HBAD3PPlugin.HBAD3PParams(eco, String.(t.keys), [P.UIntType.(i) for i in ids], String.(t.keys), t.sp, t.rec, t.glob, t.farea)
  prov = Dict{String,Any}("params_dir" => relpath(out, repo), "source_fit" => abspath(fit), "species_map_source" => "REF_SPECIES.SFTWD_HRDWD in $(scen.fia_db)", "eco_cells_from" => get(opt, "base", "results/params_yaml/r4_hbad2_v2j.yaml"),
                           "fit_region" => region, "fit_meta" => join(filter(!isempty, strip.(readlines(joinpath(fit, "meta.txt")))), "; "))
  for l in eachline(joinpath(out, "meta.txt")); startswith(l, "fit_log=") && (prov["fit_log"] = l[9:end]); end
  data = Dict{String,Any}()
  wnp = get(opt, "wn-path", isfile(joinpath(out, "plot_normal_water.csv")) ? relpath(joinpath(out, "plot_normal_water.csv"), repo) : "")
  isempty(wnp) || (data["plot_normal_water"] = wnp)
  regd = Dict{String,Any}(String(k) => v for (k, v) in base["pan_params"]["region"])
  y = P.params_to_yaml(p; name=name, description=desc, region=regd, provenance=prov, species_map=smap,
                       eco_mapping=Dict{String,Any}(String(k) => v for (k, v) in base["eco_mapping"]), data=data)
  q, meta = P.params_from_yaml(y)
  isempty(P.params_diff(p, q)) || error("round trip failed: $(P.params_diff(p, q))")
  write(opt["yaml"], y)
  println("wrote $(opt["yaml"]): plugin $(meta.plugin), $(length(t.keys)) keys, $(length(eco)) eco cells, species_map $(length(smap)) FIA symbols, data $(data)")
end
