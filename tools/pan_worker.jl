# Pan scenario worker — JSON lines over stdin/stdout (docs/scenario_campaign_plan.md "Worker protocol").
#   cd <pan repo> && source wt_env.sh && ./julia_gdal.sh --project=. --threads=auto tools/pan_worker.jl
# Request:  {"id": <any>, "op": "...", ...}            Response: {"id": …, "ok": true, "result": …} | {"id": …, "ok": false, "error": "…"}
# Progress: {"id": …, "progress": {"done": n, "total": m}} lines may precede the response of a long op.
# ops: ping | load_params {yaml} | default_params {plugin} | list_params | simulate {params_id, plots:[{cn, acres}], plan, climate,
#      establishment?, seed?} | climate_preview {plots, climate, start_year, years} | shutdown
# Everything Pan prints is redirected to stderr, so stdout carries protocol lines only.
using Pan, JSON3, Dates
const P = Pan
const OUT = stdout
redirect_stdout(stderr)

const PARAMS_DIR = get(ENV, "PAN_PARAMS_DIR", joinpath(dirname(@__DIR__), "results", "params_yaml"))
# defaults: EPA Region 4 sets (cover all R4 ecoregions incl. GA+FL); GA+FL-specific, cross-validated sets ship as samples
const DEFAULTS = Dict("biomass" => "r4_biomass_v1.yaml", "pnet" => "r4_pnet_v1.yaml", "hbad" => "r4_hbad_v1.yaml", "hbad2" => "r4_hbad2_v2g.yaml")
const SCEN = P.ScenarioConfig()          # data sources of the scenario engine, read from the PAN_* env once at startup (owns the lazy tables)
const CACHE = Dict{String,Any}()        # params_id => ParamSet

send(x) = (println(OUT, JSON3.write(x)); flush(OUT))
_d(x) = x === nothing ? nothing : x isa JSON3.Object ? Dict{String,Any}(String(k) => _d(v) for (k, v) in pairs(x)) :
        x isa JSON3.Array ? Any[_d(v) for v in x] : x

function _describe(ps)
  p = ps.params; m = ps.meta
  (params_id=ps.id, plugin=m.plugin, name=m.name, description=m.description, n_species=length(p.SPECIES_LIST), n_ecos=length(p.ECO_LIST),
   region=m.region, provenance=m.provenance, species=String.(p.SPECIES_LIST), ecos=String.(p.ECO_LIST), warnings=m.warnings)
end
function _load(text::AbstractString)
  ps = P.load_paramset(text, SCEN); CACHE[ps.id] = ps; ps
end
function _default(plugin::AbstractString)
  f = joinpath(PARAMS_DIR, get(DEFAULTS, lowercase(plugin), "")); isfile(f) || error("no default parameter set for plugin '$plugin' ($f)")
  _load(read(f, String))
end

function op_simulate(id, req)
  pid = get(req, "params_id", nothing)
  ps = pid === nothing ? _default(String(get(req, "plugin", "biomass"))) : get(CACHE, String(pid), nothing)
  ps === nothing && error("unknown params_id $(pid) — upload the parameter file again (worker restarted?)")
  want = get(req, "plugin", nothing)
  (want === nothing || String(want) == ps.meta.plugin) ||
    error("the parameter set '$(ps.meta.name)' is for plugin '$(ps.meta.plugin)', but the selected engine is '$(want)'")
  plots = _plots(req)
  plan = req["plan"]; climate = get(req, "climate", nothing)
  t0 = time()
  res, warns = P.run_scenario(ps, plots, plan; climate=climate, seed=Int(get(req, "seed", 1)),
    establishment=Bool(get(req, "establishment", true)), progress=(c, n) -> send((id=id, progress=(done=c, total=n))))
  out = Dict{String,Any}()
  for (cn, r) in res
    out[cn] = (status=r.status, message=r.message,
      metrics=[(year=m.year, ba=m.ba, tpa=m.tpa, qmd=m.qmd, tcuft=m.tcuft, bdft=m.bdft, rem_tcuft=m.rem_tcuft, rem_bdft=m.rem_bdft,
                carbon_total=m.carbon_total, carbon_live_ag=m.carbon_live_ag, carbon_removed=m.carbon_removed, agb=m.agb) for m in r.metrics],
      dist=Dict(string(y) => [[Int(k[1]), k[2], v[1], v[2]] for (k, v) in d] for (y, d) in r.dist),
      spsym=Dict(string(k) => v for (k, v) in r.spsym))
  end
  approx = "Pan volumes: FIA Region-4 species equations at each cohort's QMD and height (total-stem ft³; board feet International ¼-inch); carbon_total = live above + belowground (roots $(P.SC_ROOT_FRAC)×AG); no dead-wood pools"
  (plots=out, warnings=vcat(warns, [approx]), engine_info=(params_name=ps.meta.name, plugin=ps.meta.plugin, params_id=ps.id),
   seconds=round(time() - t0; digits=1))
end

# plots: [{cn, acres, l4?, lon?, lat?}] — l4/lon/lat = AOI location of the plot's pixels (site from location, vegetation from the plot)
_plots(req) = [(cn=String(p["cn"]), acres=Float64(get(p, "acres", 1.0)), l4=String(something(get(p, "l4", ""), "")),
                lon=Float64(something(get(p, "lon", NaN), NaN)), lat=Float64(something(get(p, "lat", NaN), NaN))) for p in req["plots"]]

function handle(id, req)
  op = String(get(req, "op", ""))
  op == "ping" && return (version="pan-worker/1", plugins=["biomass", "pnet", "hbad", "hbad2"], threads=Threads.nthreads(), params_dir=PARAMS_DIR)
  op == "load_params" && return _describe(_load(String(req["yaml"])))
  op == "default_params" && return _describe(_default(String(req["plugin"])))
  op == "list_params" && return [(file=f, path=joinpath(PARAMS_DIR, f)) for f in sort(readdir(PARAMS_DIR)) if endswith(f, ".yaml")]
  op == "simulate" && return op_simulate(id, req)
  if op == "climate_preview"
    plots = _plots(req)
    return P.climate_preview(SCEN, plots, get(req, "climate", nothing); start_year=Int(get(req, "start_year", 2025)), years=Int(get(req, "years", 100)))
  end
  error("unknown op '$op'")
end

@info "pan_worker ready (threads=$(Threads.nthreads()), params dir $(PARAMS_DIR))"
send((id=nothing, ok=true, result=(ready=true,)))
while !eof(stdin)
  line = readline(stdin); isempty(strip(line)) && continue
  id = nothing
  try
    req = _d(JSON3.read(line)); id = get(req, "id", nothing)
    get(req, "op", "") == "shutdown" && (send((id=id, ok=true, result="bye")); break)
    send((id=id, ok=true, result=handle(id, req)))
  catch e
    bt = sprint(showerror, e, catch_backtrace())
    @error "request failed" exception = bt
    send((id=id, ok=false, error=sprint(showerror, e)))
  end
end
