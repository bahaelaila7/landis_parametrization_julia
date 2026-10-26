# profile one simulate! of a cached set (fold product HBAD2_DATA_DIR): tools/hbad4/prof4.jl <hbad3p fit dir> [set]
include(joinpath(@__DIR__, "data4.jl")); include(joinpath(@__DIR__, "params4.jl")); using Profile
set = length(ARGS) > 1 ? ARGS[2] : "train"
t0 = time(); CS = cached_sets(["train", "val", "test"]); println("sets ", round(time() - t0; digits=1), " s")
D = CS.sets[set]; e3, f3, g3 = hbad3p_start(ARGS[1], CS.keys)
SP = space4(CS.keys, CS.TR; start_eps=e3, start_lnF=f3, start_glob=g3); P, G, Ob = SP.decode(SP.z0)
HBAD4P.simulate!(D.S, P, G); t = time(); HBAD4P.simulate!(D.S, P, G); println("simulate ", round(time() - t; digits=2), " s threads ", Threads.nthreads())
println("alloc ", @allocated HBAD4P.simulate!(D.S, P, G))
Profile.clear(); @profile HBAD4P.simulate!(D.S, P, G)
Profile.print(IOContext(stdout, :displaysize => (200, 250)); format=:flat, sortedby=:count, mincount=30, noisefloor=2)
