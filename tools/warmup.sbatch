#!/usr/bin/env bash
# One-time depot warm-up (re-run after any `git pull` that touches src/). Precompiles Pan + deps into the
# depot ON A MATCHING-ARCH node, so the real training job (JULIA_PKG_PRECOMPILE_AUTO=0) never compiles on the
# clock. pkgimages are CPU-arch tagged → warm on the SAME CPU family you'll run on.
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
export SCRATCH="/scratch/$USER"
export PAN_ROOT="$SCRATCH/landis_parametrization_julia"
export JULIA_DEPOT_PATH="$SCRATCH/.julia" # MUST match the training job's depot
export JULIA_PKG_PRECOMPILE_AUTO=1        # force precompilation now (opposite of the training job)

cd "$PAN_ROOT"
echo "warmup: depot=$JULIA_DEPOT_PATH  host=$(hostname)  $(date)"
./julia_gdal.sh --project=. --threads=8 -e '
  using Pkg; Pkg.instantiate()          # ensure deps are present (no-op if already)
  using Pan                             # triggers precompile of Pan + deps into the depot
  println("warmup OK: Pan loaded; pkgimages in ", joinpath(DEPOT_PATH[1], "compiled"))'
echo "warmup done $(date) — training jobs can now run with JULIA_PKG_PRECOMPILE_AUTO=0"
