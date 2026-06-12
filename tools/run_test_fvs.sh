#!/usr/bin/env bash
# Run the FVS module test and log output for Claude to inspect.
# Usage: bash tools/run_test_fvs.sh [DB] [ECO] [MAXPLOTS]
set -u
cd "$(dirname "$0")/.."
mkdir -p tmp
LOG=tmp/test_fvs.log
echo "args: $*" | tee "$LOG"
# Use the project's Julia wrapper so deps resolve; inherits LD_LIBRARY_PATH
# from the current (mamba) env so FVS finds libgfortran.
./julia_gdal.sh --project=. tools/test_fvs.jl "$@" >>"$LOG" 2>&1
echo "exit=$?" >>"$LOG"
echo "DONE -> $LOG"
