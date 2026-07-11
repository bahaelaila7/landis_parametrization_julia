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

# Ramdisk output (OPT-IN, default OFF): the run WRITES to fast tmpfs so the async checkpoint writer never stalls
# on a slow shared FS, and an event/backstop sync + exit trap flush it to the DURABLE $O (ramdisk is volatile —
# the sync is what makes resume survive a crash/preemption). Enable per submission:
#   PAN_RAMDISK=auto          → use $SLURM_TMPDIR if set (local SSD/tmpfs), else /dev/shm
#   PAN_RAMDISK=/dev/shm      → that specific base
#   PAN_RAMDISK unset|none|off|0 → DISABLED, write straight to $O (default)
# Note: tmpfs (/dev/shm) usage is charged to the job's --mem cgroup — watch total (peak RSS + output size).
W="$O"   # W = where the run actually writes (ramdisk if enabled+usable, else the durable dir)
case "${PAN_RAMDISK:-}" in
  ""|none|off|0|false) RAM_BASE="" ;;                       # disabled
  auto)                RAM_BASE="${SLURM_TMPDIR:-/dev/shm}" ;;
  *)                   RAM_BASE="$PAN_RAMDISK" ;;
esac
if [ -n "$RAM_BASE" ] && [ -d "$RAM_BASE" ] && [ -w "$RAM_BASE" ]; then
  W="$RAM_BASE/pan_$(basename "$O")_${SLURM_JOB_ID:-$$}"
  mkdir -p "$W"
  [ -n "$(ls -A "$O" 2>/dev/null)" ] && cp -a "$O/." "$W/" 2>/dev/null   # seed from durable so resume finds prior ckpts/losses.duckdb
  SYNC_SEC="${PAN_SYNC_SEC:-600}"; SYNC_PIDS=""
  if command -v inotifywait >/dev/null 2>&1; then
    # EVENT-DRIVEN: the instant a checkpoint/param file finishes writing (close_write), push it out — near-zero
    # data-at-risk, and it only fires when there's actually something new (no polling, no idle tree scans).
    ( inotifywait -m -q -e close_write "$W" | while read -r _; do rsync -a "$W/" "$O/" 2>/dev/null; done ) & SYNC_PIDS="$!"
    log "ramdisk output $W → event-sync (inotify close_write) to $O + ${SYNC_SEC}s backstop"
  else
    log "ramdisk output $W → ${SYNC_SEC}s periodic sync to $O  (install inotify-tools for event-driven)"
  fi
  # coarse periodic backstop (also covers losses.duckdb, which stays open so never emits close_write mid-run)
  ( while :; do sleep "$SYNC_SEC"; rsync -a "$W/" "$O/" 2>/dev/null; done ) & SYNC_PIDS="$SYNC_PIDS $!"
  # flush on ANY exit so completed checkpoints land in $O. TERM (SLURM preemption) / INT forward to the single
  # EXIT handler (exit re-triggers it) so it runs exactly once — no double rsync/rm race.
  _flush(){ kill $SYNC_PIDS 2>/dev/null; rsync -a "$W/" "$O/" 2>/dev/null; rm -rf "$W"; log "flushed ramdisk → $O"; }
  trap _flush EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
fi

CK=$(latest "$W"); OVR="\"output_dir\"=>\"$W\""
if [ -n "$CK" ]; then
  # PRESERVE losses.duckdb across resumes (deterministic split ⇒ identical loss scale; CREATE TABLE IF NOT EXISTS ⇒
  # the resumed run APPENDS gen N+ to the existing curve). Archives (search_state@N.jld2) were always safe.
  OVR="$OVR,\"resume_from\"=>\"$CK\""; log "resume $(basename "$O") from $(basename "$CK") — losses.duckdb preserved (appending)"
else log "fresh $(basename "$O")"; fi
if ./julia_gdal.sh --project=. --threads=$TH -e "using Pan; Pan.run_from_yaml(\"$CFG\"; overrides=Dict($OVR))" >> "$O/run.log" 2>&1; then
  touch "$O/.train_done"; log "TRAIN complete $(basename "$O") ($(ls "$W"/search_state@*.jld2 2>/dev/null|wc -l) ckpts) — flushing ramdisk on exit"
else log "TRAIN interrupted $(basename "$O") — relaunch to resume"; exit 1; fi
