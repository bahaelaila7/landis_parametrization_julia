#!/usr/bin/env bash
# Generic resumable training launcher. With the deterministic stratified split (fixed 2026-07),
# a resume recomputes the IDENTICAL train/val partition → identical loss normalization → correct
# continuation (no archive collapse / scale shift). Resumes from the highest search_state@N.jld2
# (or fresh), writes .train_done on clean completion. Relaunch to resume after a reap.
# ONE Pan job at a time.  tools/train_resumable.sh <config.yml> <output_dir> [threads]
set -u
ROOT="${PAN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}}"
cd "$ROOT"
CFG="$1"; O="$2"; TH="${3:-8}"
# Outputs live under $PAN_OUT (e.g. scratch) so the project stays clean; unset ⇒ the path as passed (local default).
# $O is authoritative: it drives checkpoint scan/log/.train_done AND is passed as the run's output_dir override.
[ -n "${PAN_OUT:-}" ] && O="$PAN_OUT/$(basename "$O")"
mkdir -p "$O"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$O/RESUME_STATUS.txt"; }
latest(){ ls "$1"/search_state@*.jld2 2>/dev/null | sed -E 's/.*@([0-9]+)\.jld2/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-; }
[ -f "$O/.train_done" ] && { log "already done: $(basename "$O")"; exit 0; }
CK=$(latest "$O"); OVR="\"output_dir\"=>\"$O\""
if [ -n "$CK" ]; then
  # PRESERVE losses.duckdb across resumes (deterministic split ⇒ identical loss scale; CREATE TABLE IF NOT EXISTS ⇒
  # the resumed run APPENDS gen N+ to the existing curve). Archives (search_state@N.jld2) were always safe.
  OVR="$OVR,\"resume_from\"=>\"$CK\""; log "resume $(basename "$O") from $(basename "$CK") — losses.duckdb preserved (appending)"
else log "fresh $(basename "$O")"; fi
if ./julia_gdal.sh --project=. --threads=$TH -e "using Pan; Pan.run_from_yaml(\"$CFG\"; overrides=Dict($OVR))" >> "$O/run.log" 2>&1; then
  touch "$O/.train_done"; log "TRAIN complete $(basename "$O") ($(ls "$O"/search_state@*.jld2 2>/dev/null|wc -l) ckpts)"
else log "TRAIN interrupted $(basename "$O") — relaunch to resume"; exit 1; fi
