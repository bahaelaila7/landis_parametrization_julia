# Integration check for B2 (needs the GA+FL data; PAN_DATA=/workspace/landis_region4/data_r4):
#   PAN_PARALLEL=candidate ./julia_gdal.sh --project=. --threads=2 test/integration/parallel_sites_restore.jl <out_dir>
# A 1-trial igelmo run in candidate mode returns early from the driver. Site-parallelism must be restored both for the
# POST_SEARCH_HOOK and afterwards (it used to stay false for the rest of the process: on c184e00 it prints false/false).
import YAML; using Pan, Test
include(joinpath(@__DIR__, "..", "golden", "scenarios.jl"))
base, ov, _ = SCENARIOS["bs_igelmo"]
over = merge(common(), ov, Dict{String,Any}("output_dir" => ARGS[1], "trials" => 1, "igel_mu" => 2))
hook_ps = Ref{Any}(nothing)
Pan.POST_SEARCH_HOOK[] = d -> (hook_ps[] = Pan.PARALLEL_SITES[])
Pan.run_from_yaml(base; overrides=over)
@test hook_ps[] === true
@test Pan.PARALLEL_SITES[] === true
