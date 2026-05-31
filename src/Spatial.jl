module Spatial

using ..PanCore
import Term.Progress as TProgress

export run_spatial!

function run_spatial!(soa, plugin_type::Type, ctx, output_dir::String, emit_year!;
  timehorizon::Int, output_every::Int)
  mkpath(output_dir)
  emit_year!(soa, 0)
  PanCore.gc_and_trim!()

  TProgress.@track for year in 1:timehorizon
    PanCore.process_plugin!(soa, plugin_type, year; ctx=ctx)
    @info "Year $year: SoA $(round(Base.summarysize(soa)/1e9, digits=2)) GB, RSS: $(round(PanCore.current_rss_gb(), digits=2)) GB, peak RSS: $(round(Sys.maxrss() / 1e9, digits=2)) GB"

    if year % output_every == 0
      emit_year!(soa, year)
    end
  end
end


end # module Spatial
