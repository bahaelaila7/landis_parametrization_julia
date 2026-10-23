# Per-fold references of the HBAD v2 joint scalarisation (docs/nested_cv_gafl.md §2): the j15 recipe normalises fA (Sim A NLL) and fB (Sim B margins)
# by an ideal and a nadir point. On a nested-CV fold the train set differs, so:
#   ideal fA = fA(start) on train (the start is this fold's Sim A optimum, as j15's ideal fA was the v2f train optimum)
#   nadir fB = fB(start) on train (as j15: "fB: the start")
#   nadir fA = ideal fA + (j15 nadir fA - j15 ideal fA) * nA / nA_j15      ideal fB = j15 ideal fB * nB / nB_j15
# with nA = Sim A train cohorts, nB = Sim B train sites (j15: 81,680 / 2,128). Prints one line: "ideal: [fA, fB] nadir: [fA, fB] nA nB".
#   ./julia_gdal.sh --project=. --threads=N tools/ncv/hbad2_refs.jl <joint config.yml>
using Pan, Printf
const HC = Pan.HBAD2Calib
# nA in Pan's unit (observed cohorts): 81,680 on the j15 config itself (2026-10-23; the prototype log's 245,040 counts the 3 vigour sub-cohorts = 3 x 81,680);
# the same check reproduced j15's start exactly (train fA -273,387.6, fB 23,153.1) and its 2,128 Sim B sites.
const J15 = (ideal=(-272975.3, 14619.3), nadir=(-236466.5, 24102.3), nA=81680, nB=2128)
cfg = Pan.load_config(ARGS[1])
prob = Pan.hbad2_calib_problem(cfg; output_dir=mktempdir(), sets=[:train])
θ = prob.θ0
fA = HC.simA_scores(prob, θ, :train).total
fB = HC.fb_value(HC.simB_scores(prob, θ, :train), prob.fb)
nA = size(prob.A[:train].coh, 1); nB = length(prob.B[:train].live0)
idA = fA; ndB = fB
ndA = idA + (J15.nadir[1] - J15.ideal[1]) * nA / J15.nA
idB = J15.ideal[2] * nB / J15.nB
@printf("ideal: [%.1f, %.1f] nadir: [%.1f, %.1f] nA %d nB %d\n", idA, idB, ndA, ndB, nA, nB)
