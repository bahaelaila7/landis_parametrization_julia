#!/bin/bash

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
exec julia "$@"
