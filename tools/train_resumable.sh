#!/usr/bin/env bash
# Generic resumable training launcher. With the deterministic stratified split (fixed 2026-07),
# a resume recomputes the IDENTICAL train/val partition → identical loss normalization → correct
# continuation (no archive collapse / scale shift). Resumes from the highest search_state@N.jld2
# (or fresh), writes .train_done on clean completion. Relaunch to resume after a reap.
# ONE Pan job at a time.  tools/train_resumable.sh <config.yml> <output_dir> [threads]
set -u
cd /workspace/landis_parametrization_julia
CFG="$1"; O="$2"; TH="${3:-8}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a RESUME_STATUS.txt; }
latest(){ ls "$1"/search_state@*.jld2 2>/dev/null | sed -E 's/.*@([0-9]+)\.jld2/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-; }
[ -f "$O/.train_done" ] && { log "already done: $(basename "$O")"; exit 0; }
CK=$(latest "$O"); RES=""
if [ -n "$CK" ]; then
  # PRESERVE losses.duckdb across resumes (do NOT delete). The deterministic split ⇒ identical loss scale, and
  # the loss tables use CREATE TABLE IF NOT EXISTS, so the resumed run APPENDS gen N+ to the existing curve →
  # unbroken per-gen history. A stale .wal from a hard kill is replayed/recovered by DuckDB on open. The full
  # per-iteration ARCHIVES were always safe (search_state@N.jld2); this only keeps the loss CURVE continuous too.
  RES=", overrides=Dict(\"resume_from\"=>\"$CK\")"; log "resume $(basename "$O") from $(basename "$CK") — losses.duckdb preserved (appending)"
else log "fresh $(basename "$O")"; fi
if ./julia_gdal.sh --project=. --threads=$TH -e "using Pan; Pan.run_from_yaml(\"$CFG\"$RES)" >> "$O.log" 2>&1; then
  touch "$O/.train_done"; log "TRAIN complete $(basename "$O") ($(ls "$O"/search_state@*.jld2 2>/dev/null|wc -l) ckpts)"
else log "TRAIN interrupted $(basename "$O") — relaunch to resume"; exit 1; fi
