# Per-fold references of the HBAD v2 joint scalarisation (docs/nested_cv_gafl.md §2): the j15 recipe normalises fA (Sim A NLL) and fB (Sim B margins)
# by an ideal and a nadir point, f' = (f - ideal) / (nadir - ideal). Both rules below return j15's ideal / nadir exactly on j15's own data (checked):
#   fA: anchored at the start (this fold's Sim A optimum, as j15 started at the v2fv optimum): the start keeps j15's normalised position
#       xs = (-273387.6 - ideal)/spanA = -0.0113 and the span scales with the Sim A train cohorts (spanA = j15 spanA * nA / nA_j15).
#   fB: a sum of per-site terms, so ideal and nadir scale with the Sim B train sites (x nB / nB_j15) — NOT anchored at the start, so a start worse than
#       j15's (a cold Sim A fit that lands lower) does not move the fB target; if the start is above the scaled nadir, the nadir is raised to it (the start
#       stays inside the box) and a warning is printed.
# Prints one line: "ideal: [fA, fB] nadir: [fA, fB] nA nB".
#   ./julia_gdal.sh --project=. --threads=N tools/ncv/hbad2_refs.jl <joint config.yml>
using Pan, Printf
const HC = Pan.HBAD2Calib
# nA in Pan's unit (observed cohorts): 81,680 on the j15 config itself (2026-10-23; the prototype log's 245,040 counts the 3 vigour sub-cohorts = 3 x 81,680);
# the same check reproduced j15's start exactly (train fA -273,387.6, fB 23,153.1) and its 2,128 Sim B sites.
const J15 = (ideal=(-272975.3, 14619.3), nadir=(-236466.5, 24102.3), start=(-273387.6, 23153.1), nA=81680, nB=2128)
const XS = (J15.start[1] - J15.ideal[1]) / (J15.nadir[1] - J15.ideal[1])
cfg = Pan.load_config(ARGS[1])
prob = Pan.hbad2_calib_problem(cfg; output_dir=mktempdir(), sets=[:train])
θ = prob.θ0
fA = HC.simA_scores(prob, θ, :train).total
fB = HC.fb_value(HC.simB_scores(prob, θ, :train), prob.fb)
nA = size(prob.A[:train].coh, 1); nB = length(prob.B[:train].live0)
spA = (J15.nadir[1] - J15.ideal[1]) * nA / J15.nA
idA = fA - XS * spA; ndA = idA + spA
idB = J15.ideal[2] * nB / J15.nB; ndB = J15.nadir[2] * nB / J15.nB
fB > ndB && (@warn "start fB $(round(fB; digits=1)) is above the scaled nadir fB $(round(ndB; digits=1)): nadir raised to the start"; ndB = fB)
@printf("ideal: [%.1f, %.1f] nadir: [%.1f, %.1f] nA %d nB %d\n", idA, idB, ndA, ndB, nA, nB)
