#!/usr/bin/env bash
# RESUME-ON-RELAUNCH MASTER — idempotent single pass; re-run after any reap and it continues from disk.
# Container/session teardown kills all processes, but /workspace persists, so every stage checkpoints there:
#   - training (cbal): resume from the highest-numbered search_state@N.jld2 (symlink may be absent after a
#     reap), or start fresh; writes .train_done only on clean completion.
#   - pipelines: p101_pipeline.sh is idempotent (skips finished sweeps/candidates/plots).
# ONE Pan job at a time (concurrent data-loads exhaust RAM and get reaped). Relaunch each session:
#   bash tools/resume_master.sh          (or via harness background)
set -u
cd /workspace/landis_parametrization_julia
STATUS=RESUME_STATUS.txt
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }

# highest-numbered search_state@N.jld2 (robust to a missing search_state_latest symlink)
latest_ckpt(){ ls "$1"/search_state@*.jld2 2>/dev/null | sed -E 's/.*@([0-9]+)\.jld2/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-; }

train_resumable(){  # $1=cfg $2=outdir
  local CFG="$1" OUT="$2" NM CK RES; NM=$(basename "$OUT")
  if [ -f "$OUT/.train_done" ]; then log "TRAIN skip (done): $NM"; return 0; fi
  CK=$(latest_ckpt "$OUT"); RES=""
  if [ -n "$CK" ]; then
    rm -f "$OUT"/losses.duckdb "$OUT"/losses.duckdb.wal          # clear stale DuckDB lock (cache only; state is in JLD2)
    RES=", overrides=Dict(\"resume_from\"=>\"$CK\")"
    log "TRAIN resume $NM from $(basename "$CK")"
  else
    log "TRAIN fresh $NM"
  fi
  if ./julia_gdal.sh --project=. --threads=8 -e "using Pan; Pan.run_from_yaml(\"$CFG\"$RES)" >> "$OUT.log" 2>&1; then
    touch "$OUT/.train_done"; log "TRAIN complete $NM ($(ls "$OUT"/search_state@*.jld2 2>/dev/null | wc -l) ckpts)"
  else
    log "TRAIN interrupted $NM (resumes next pass)"; return 1
  fi
}

CBAL_CFG=runs/fl5_l4cover_ccigel_modeA_cbalpct_simA_l1_8020_pruned_stdorg.yml
CBAL_OUT=runs/fl5_l4cover_ccigel_modeA_cbalpct_simA_l1_8020_pruned_stdorg_outputs
MA_CFG=runs/fl5_l4cover_ccigel_modeA_simA_l1_8020_pruned_stdorg.yml
MA_OUT=runs/fl5_l4cover_ccigel_modeA_simA_l1_8020_pruned_stdorg_outputs
N_CFG=runs/fl5_l4cover_nsga2_simA_l1_8020_pruned_stdorg.yml
N_OUT=runs/fl5_l4cover_nsga2_simA_l1_8020_pruned_stdorg_outputs

log "===== MASTER PASS START ====="
train_resumable "$CBAL_CFG" "$CBAL_OUT" || true                  # 1) cbal training (priority; resumable)
[ -f "$CBAL_OUT/.train_done" ] && { log "PIPE cbal";   bash tools/p101_pipeline.sh "$CBAL_CFG" "$CBAL_OUT" 8; }   # 2) cbal p101 (only once trained)
log "PIPE modeA-noreweight"; bash tools/p101_pipeline.sh "$MA_CFG" "$MA_OUT" 8   # 3) already-trained; idempotent
log "PIPE nsga2";            bash tools/p101_pipeline.sh "$N_CFG"  "$N_OUT"  8   # 4) already-trained; idempotent
log "===== MASTER PASS COMPLETE ====="
