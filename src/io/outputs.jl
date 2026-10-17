# Persist the exact train/val/test plot assignment (+ eco cell, + dominant species when train_cap ran) so floor
# builders / analysis scripts reuse the run's split instead of re-deriving it (scripts drift from training kwargs).
function _write_split_plots(output_dir::AbstractString, eco_list, splots, sv, st, outs::Data.DataOutputs)
  mkpath(output_dir)
  ids = [:statecd, :unitcd, :countycd, :plot]
  parts = DataFrame[]
  for (set, d) in (("train", splots), ("val", sv), ("test", st))
    (d === nothing || nrow(d) == 0) && continue
    u = unique(select(d, vcat(ids, [:eco_id])))
    u.eco = [eco_list[i] for i in u.eco_id]; u.set .= set
    push!(parts, select(u, vcat(ids, [:eco, :set])))
  end
  isempty(parts) && return
  out = vcat(parts...)
  dd = outs.split_dom
  out = dd === nothing ? out : leftjoin(out, select(dd, vcat(ids, [:dom])), on=ids)
  sort!(out, vcat([:set], ids))
  CSV.write(joinpath(output_dir, "split_plots.csv"), out)
  CSV.write(joinpath(output_dir, "species_map.csv"), outs.species_map)   # symbol → category (for floor builders)
  @info "split_plots.csv: $(nrow(out)) plots → $(output_dir)"
end
