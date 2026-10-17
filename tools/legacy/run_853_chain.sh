#!/usr/bin/env bash
# Priority: 8.5.3 igelmo+cbalpct training (fresh), then resume the paused NSGA-II p101 diagnostics (idempotent).
# One Pan job at a time. Relaunch this script to resume if reaped (8.5.3 training resumes from latest ckpt).
set -u
ROOT="${PAN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}}"
cd "$ROOT"
O=runs/fl853_igelmo_cbalpct_simA_l1_8020_stdorg_outputs
CFG=runs/fl853_igelmo_cbalpct_simA_l1_8020_stdorg.yml
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a RESUME_STATUS.txt; }
latest(){ ls "$1"/search_state@*.jld2 2>/dev/null | sed -E 's/.*@([0-9]+)\.jld2/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-; }

log "===== 8.5.3 igelmo+cbalpct training ====="
if [ ! -f "$O/.train_done" ]; then
  CK=$(latest "$O"); RES=""
  if [ -n "$CK" ]; then rm -f "$O"/losses.duckdb "$O"/losses.duckdb.wal; RES=", overrides=Dict(\"resume_from\"=>\"$CK\")"; log "resume from $(basename "$CK")"; else log "fresh"; fi
  if ./julia_gdal.sh --project=. --threads=8 -e "using Pan; Pan.run_from_yaml(\"$CFG\"$RES)" >> "$O.log" 2>&1; then
    touch "$O/.train_done"; log "8.5.3 training complete ($(ls "$O"/search_state@*.jld2 2>/dev/null | wc -l) ckpts)"
  else log "8.5.3 training interrupted (resumes on relaunch)"; exit 1; fi
else log "8.5.3 already trained — skip"; fi

log "===== resume NSGA-II p101 diagnostics ====="
bash tools/p101_pipeline.sh runs/fl5_l4cover_nsga2_simA_l1_8020_pruned_stdorg.yml runs/fl5_l4cover_nsga2_simA_l1_8020_pruned_stdorg_outputs 8
log "===== ALL DONE ====="
