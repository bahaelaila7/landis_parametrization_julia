"""
    record_best_losses!(losses_db, iter, total, run_result, eco_losses, params; eco_list, species_list, n_species, loss_params, print_breakdown=false)

The `losses.duckdb` rows of a new best of a single-objective driver (LBSA, CMA-ES, SA): `total_loss` (with the serialized parameters), one
`ecoregion_loss` row per ecoregion and one `species_loss` row per (ecoregion, species) with a nonzero age-distribution loss. `print_breakdown`
also prints the raw per-ecoregion / per-species breakdown, sorted by contribution, to see where the remaining error concentrates.
"""
function record_best_losses!(losses_db, iter, total, run_result, eco_losses, params; eco_list, species_list, n_species, loss_params, print_breakdown::Bool=false)
  let buf = IOBuffer()
    Serialization.serialize(buf, params)
    DuckDB.execute(losses_db, "INSERT INTO total_loss VALUES (?, ?, ?, ?, ?)", [iter, run_result.num_sites, run_result.num_obs, total, take!(buf)])
  end
  isnothing(eco_losses) && return nothing
  for (eco_id, eco_loss) in enumerate(eco_losses)
    eco_name = _eco_name(eco_list, eco_id)
    eco_total = convert(Float64, PU.get_total_loss(eco_loss, loss_params))
    DuckDB.execute(losses_db, "INSERT INTO ecoregion_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, eco_loss.num_sites, eco_loss.num_obs, eco_total])
    n = max(1, eco_loss.num_sites)
    for gsp in 1:n_species
      eco_loss.sp_w_loss[gsp] == 0f0 && continue
      DuckDB.execute(losses_db, "INSERT INTO species_loss VALUES (?, ?, ?, ?, ?)", [iter, eco_name, species_list[gsp], eco_loss.sp_w_loss[gsp] / n, eco_loss.sp_agb_loss[gsp] / n])
    end
  end
  if print_breakdown
    println("Raw SiteLoss breakdown @ $iter (total=$(round(total, sigdigits=6))):")
    for eco_id in sort(collect(eachindex(eco_losses)); by=e -> -convert(Float64, PU.get_total_loss(eco_losses[e], loss_params)))
      el = eco_losses[eco_id]
      println("  [$(_eco_name(eco_list, eco_id))] eco_total=$(round(convert(Float64, PU.get_total_loss(el, loss_params)), sigdigits=5))  sites=$(el.num_sites) obs=$(el.num_obs)")
      for gsp in sort([g for g in 1:n_species if el.sp_w_loss[g] != 0f0]; by=g -> -el.sp_w_loss[g])
        println("      $(rpad(species_list[gsp], 10)) w=$(round(el.sp_w_loss[gsp], sigdigits=4))  agb=$(round(el.sp_agb_loss[gsp], sigdigits=4))")
      end
    end
  end
  return nothing
end
