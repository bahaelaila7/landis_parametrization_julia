# Fast parity gate of the HBAD2 plugin: the vendored PROTOTYPE kernel (test/ref_hbad2_proto.jl, verbatim copy of branch hbad-v2 src/plugins/HBAD2/HBAD2.jl) vs the Pan plugin on synthetic
# subplot sites, state compared cohort by cohort after every year. Free runs (recruitment, vigour, planted stands, empty sites) and Sim A style runs (no recruitment).
# The full gate on the real VAL sites is tools/hbad2_parity.jl.
module UnitHBAD2Parity
using Test, Random, Pan
module Proto
include("ref_hbad2_proto.jl")
end
include("hbad2_parity_lib.jl")
const PR = Proto.HBAD2

@testset "hbad2 parity vs prototype" begin
  P, G, R, keys_ = proto_params(PR, HB2.DEFAULT_PARAMS_DIR)
  pcfg = Pan.PluginConfig(HB2.HBAD2Context(keys_, [collect(1:length(keys_))]; verbose=false))
  init = synth_init(Random.Xoshiro(11), 60, length(keys_))
  recP, dropped = proto_free(PR, init, P, G, R, 25)
  @test dropped == 0
  c = compare_rec(recP, pan_run(pcfg, pan_soa(pcfg, init; no_establishment=false), 25))
  @info "free-run parity" c
  @test c.mismatches == 0 && c.maxrel <= 1e-12 && c.cohorts > 10_000
  nyr = [3 + s % 9 for s in 1:60]
  recA = proto_simA(PR, init, P, G, nyr)
  cA = compare_rec(recA, pan_run(pcfg, pan_soa(pcfg, init; no_establishment=true), maximum(nyr); nyr=nyr); nyr=nyr)
  @info "Sim A parity" cA
  @test cA.mismatches == 0 && cA.maxrel <= 1e-12 && cA.cohorts > 500
end
end
