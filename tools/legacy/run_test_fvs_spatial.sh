#!/usr/bin/env bash
# Run the FVS raster pipeline test and log output for Claude to inspect.
# Usage: bash tools/run_test_fvs_spatial.sh TREEMAP.tif [DB] [HORIZON] [EVERY] [VERSION]
set -u
cd "$(dirname "$0")/.."
mkdir -p tmp
LOG=tmp/test_fvs_spatial.log
echo "args: $*" | tee "$LOG"
./julia_gdal.sh --project=. tools/test_fvs_spatial.jl "$@" >>"$LOG" 2>&1
echo "exit=$?" >>"$LOG"
echo "DONE -> $LOG"
