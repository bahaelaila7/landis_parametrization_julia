#!/usr/bin/env bash
# One-time depot warm-up — RE-RUN AFTER ANY `git pull` THAT TOUCHES src/ (Pan's cache key changes, else the first
# job recompiles on the clock). Precompiles Pan + deps into the depot ON A MATCHING-ARCH node, so the real training
# job (JULIA_PKG_PRECOMPILE_AUTO=0) never compiles. pkgimages are CPU-arch tagged → warm on the SAME CPU family you
# run on. Mode-agnostic: this warms the package for BOTH PAN_PARALLEL=soa and =candidate.
#
#   sbatch tools/warmup.sbatch                    # fire-and-forget on a Genoa node
#   # or interactively:  srun --constraint=Genoa --cpus-per-task=8 --mem=16G --time=30 --pty bash tools/warmup.sbatch

#SBATCH --job-name=pan-warmup
#SBATCH --partition=batch
#SBATCH --constraint=Genoa           # SAME arch as the training job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # precompile fans out across a few cores
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
# PAN_ROOT/JULIA_DEPOT_PATH honor a pre-set value so you can warm an ISOLATED env (a worktree checkout of a
# struct-changing branch) into its OWN depot: e.g. PAN_ROOT=$SCRATCH/landis_split
# JULIA_DEPOT_PATH=$SCRATCH/.julia_split:$SCRATCH/.julia sbatch tools/warmup.sbatch  (layered: writes new to
# .julia_split, reuses shared deps/artifacts from .julia read-only → never touches the running jobs' cache).
export SCRATCH="${SCRATCH:-/scratch/$USER}"
export PAN_ROOT="${PAN_ROOT:-$SCRATCH/landis_parametrization_julia}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$SCRATCH/.julia}" # MUST match the training job's depot
export JULIA_PKG_PRECOMPILE_AUTO=1        # force precompilation now (opposite of the training job)

cd "$PAN_ROOT"
echo "warmup: depot=$JULIA_DEPOT_PATH  host=$(hostname)  $(date)"
./julia_gdal.sh --project=. --threads=8 -e '
  using Pkg; Pkg.instantiate()          # ensure deps are present (no-op if already)
  using Pan                             # triggers precompile of Pan + deps into the depot
  println("warmup OK: Pan loaded; pkgimages in ", joinpath(DEPOT_PATH[1], "compiled"))'
echo "warmup done $(date) — training jobs can now run with JULIA_PKG_PRECOMPILE_AUTO=0"
