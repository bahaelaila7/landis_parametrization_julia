#!/bin/bash

# Persistent, project-local Julia depot: survives sandbox restarts (unlike ~/.julia, which lives in $HOME and
# gets wiped). Only used when the caller hasn't ALREADY set JULIA_DEPOT_PATH — on the cluster, submit.sbatch
# exports the pre-warmed $SCRATCH/.julia depot, which must win. Trailing ':' appends the default depots so the
# bundled stdlibs/artifacts stay reachable. Populate it once with:  make prepare  (Pkg.instantiate lands here).
if [[ -z "${JULIA_DEPOT_PATH:-}" ]]; then
    _PAN_DEPOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.julia_depot"
    [[ -d "$_PAN_DEPOT" ]] && export JULIA_DEPOT_PATH="${_PAN_DEPOT}:"
fi

# Extract --project from args
PROJECT_PATH=""
for arg in "$@"; do
    if [[ "$arg" == --project=* ]]; then
        PROJECT_PATH="${arg#--project=}"
        break
    fi
done

# Fallback to current directory if no --project given
PROJECT_PATH="${PROJECT_PATH:-.}"
CACHE_FILE="${PROJECT_PATH}/.gdal_driver_path_cache"

# Rebuild cache if missing or Manifest.toml is newer
if [[ ! -f "$CACHE_FILE" ]] || \
   [[ "${PROJECT_PATH}/Manifest.toml" -nt "$CACHE_FILE" ]]; then
    julia --project="${PROJECT_PATH}" -e \
        'using GDAL_jll; print(joinpath(dirname(GDAL_jll.libgdal_path), "gdalplugins"))' \
        > "$CACHE_FILE"
fi

export GDAL_DRIVER_PATH=$(cat "$CACHE_FILE")

# Pin OpenBLAS to 1 thread: the only linalg is tiny per-offspring CMA-ES covariance eigendecomps (a few params
# each). Multi-threaded BLAS spawns all cores per call → catastrophic oversubscription with many Julia threads.
# Single-thread BLAS is far faster for these small matrices. Override by exporting OPENBLAS_NUM_THREADS yourself.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

SYSIMAGE_ARG=""
if [[ -f "${PROJECT_PATH}/Pan.so" ]]; then
    SYSIMAGE_ARG="--sysimage=${PROJECT_PATH}/Pan.so"          # full image (Pan + deps)
elif [[ -f "${PROJECT_PATH}/Pan_deps.so" ]]; then
    SYSIMAGE_ARG="--sysimage=${PROJECT_PATH}/Pan_deps.so"     # deps-only image; Pan recompiles incrementally on top
fi

exec julia $SYSIMAGE_ARG "$@"
