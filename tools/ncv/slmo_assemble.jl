# Species-level MO experiment (tools/ncv/slmo_bs.sh): assemble ONE Biomass Succession parameter set from per-group choices made on the inner VALIDATION split.
#   ./julia_gdal.sh --project=. tools/ncv/slmo_assemble.jl <config.yml> <cands dir> <out.jld2>
# <cands dir> holds the candidate sets (list.txt = their order, as given to tools/tost_select.jl), tost_select_val.csv and tost_select_val_groups.csv.
# Per group (tools/tost_common.jl label: species, or species·cell): the candidate with most val passes at ±20 %, then ±10 %, then ±5 %, then the lower
# mean |log1p(sim) - log1p(obs)|. The base is the best single candidate on val (row 1 of tost_select_val.csv); every cell (eco, species) of a group takes
# S, B_MAX_SPP, ANPP_MAX_SPP from the group's candidate; the species-level D from the candidate of the species' group with most val pairs. Groups without
# val pairs keep the base. Writes <out.jld2> and <out>_choices.csv.
using Pan, JLD2, CSV, DataFrames
cfg = Pan.load_config(ARGS[1]); cd_ = ARGS[2]; out = ARGS[3]
SPLIT = "val"
include(joinpath(@__DIR__, "..", "tost_common.jl"))           # species_list, eco_list, eco_species_ids, grp, label (the val data are prepared; not simulated here)
files = filter(!isempty, strip.(readlines(joinpath(cd_, "list.txt"))))
sets = [JLD2.load_object(f) for f in files]
C = CSV.read(joinpath(cd_, "tost_select_val.csv"), DataFrame); G = CSV.read(joinpath(cd_, "tost_select_val_groups.csv"), DataFrame)
base_i = C.cand[1]; base = deepcopy(sets[base_i])
G.key = [(-Int(r.pass_20), -Int(r.pass_10), -Int(r.pass_5), Float64(r.mean_abs_lr)) for r in eachrow(G)]
chosen = Dict{String,Int}(); npairs = Dict{String,Int}()
for g in groupby(G, :group)
  i = argmin(g.key); chosen[String(g.group[1])] = Int(g.cand[i]); npairs[String(g.group[1])] = Int(g.n[i])
end
ncopy = 0; dsrc = Dict{Int,Tuple{Int,Int}}()            # species -> (n pairs, candidate) for D
for e in eachindex(eco_species_ids), (j, gsp) in enumerate(eco_species_ids[e])
  lbl = label(Int(gsp), grp(Int(gsp), e)); haskey(chosen, lbl) || continue
  src = sets[chosen[lbl]]
  base.S[e][j] = src.S[e][j]; base.B_MAX_SPP[e][j] = src.B_MAX_SPP[e][j]; base.ANPP_MAX_SPP[e][j] = src.ANPP_MAX_SPP[e][j]; global ncopy += 1
  (n = npairs[lbl]; (!haskey(dsrc, Int(gsp)) || n > dsrc[Int(gsp)][1]) && (dsrc[Int(gsp)] = (n, chosen[lbl])))
end
for (s, (_, c)) in dsrc; base.D[s] = sets[c].D[s]; end
JLD2.save_object(out, base)
ch = DataFrame(group=collect(keys(chosen)), cand=collect(values(chosen)))
ch.file = [basename(files[c]) for c in ch.cand]; ch.n_val_pairs = [npairs[g] for g in ch.group]; sort!(ch, :group)
CSV.write(replace(out, ".jld2" => "_choices.csv"), ch)
println("assembled from $(length(unique(ch.cand))) of $(length(files)) candidates over $(nrow(ch)) groups ($(ncopy) cells, D for $(length(dsrc)) species); base = cand $base_i $(basename(files[base_i]))")
show(stdout, MIME("text/plain"), combine(groupby(ch, :file), nrow => :groups); allrows=true); println()
