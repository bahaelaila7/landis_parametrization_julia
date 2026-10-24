#!/usr/bin/env bash
# Species-level selection over a MIXED pool (user request 2026-10-24): the 21 species-level MO sweep sets of tools/ncv/slmo_bs.sh PLUS the role's 8
# production candidates (bias family raw + refined, MO picks raw + refined), so every TOST group chooses on the inner VAL split between bias-targeted
# and accuracy-targeted parameters (tools/ncv/slmo_assemble.jl: ±20 -> ±10 -> ±5 % -> mean |log ratio|). Val then decides between the assembled set and
# the best single set of the pool (same rule); the outer TEST fold reports the val-chosen set once (+ the other one and the production selection, as
# comparisons). No new search. Needs slmo_bs.sh <role> done. Resumable.
#   tools/ncv/slmo_pool.sh [role=k1]
set -uo pipefail
NCV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; WS="$(dirname "$NCV")"; cd "$NCV"; source wt_env.sh
ROLE="${1:-k1}"; R="$NCV/runs_local/ncv"; D="$R/bs/$ROLE"; S="$NCV/runs_local/ncv_slmo/bs_$ROLE"; X="$NCV/runs_local/ncv_slmo/bs_${ROLE}_pool"; mkdir -p "$X"
export PAN_DATA="$WS/landis_region4/data_r4" PAN_OUT="$X" TOST_REAL_ONLY=1
PY="$WS/standmap/.venv/bin/python"; JL="./julia_gdal.sh --project=. --threads=${THREADS:-4}"; C="$D/config.yml"
[ -f "$S/steps/cands.done" ] || { echo "run tools/ncv/slmo_bs.sh $ROLE first"; exit 1; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$X/pool.log"; }
step() { local id=$1; shift; local mk="$X/steps/$id.done" lg="$X/logs/$id.log"; [ -f "$mk" ] && return 0; mkdir -p "$(dirname "$mk")" "$(dirname "$lg")"
         log "run  $id"; local t0=$SECONDS; if ( "$@" ) < /dev/null >> "$lg" 2>&1; then date > "$mk"; log "done $id ($((SECONDS - t0)) s)"; else log "FAIL $id ($lg)"; tail -15 "$lg"; exit 1; fi; }
step cands bash -c "set -e; mkdir -p $X/cands; : > $X/cands/list.txt
  for f in \$(cat $S/cands/list.txt); do cp \$f $X/cands/; echo $X/cands/\$(basename \$f) >> $X/cands/list.txt; done
  for f in $D/cands/*.jld2; do cp \$f $X/cands/prod_\$(basename \$f); echo $X/cands/prod_\$(basename \$f) >> $X/cands/list.txt; done; wc -l $X/cands/list.txt"
step val bash -c "TOST_RANK=0.2,0.1,0.05,acc $JL tools/tost_select.jl $C \$(paste -sd, $X/cands/list.txt) 64 val 0.05,0.1,0.2"
step assemble $JL tools/ncv/slmo_assemble.jl "$C" "$X/cands" "$X/assembled.jld2"
step final_files bash -c "set -e; mkdir -p $X/final; cp $X/assembled.jld2 $X/final/a_assembled.jld2; cp $X/cands/selected_val/params.jld2 $X/final/b_best_single.jld2; cp $D/cands/selected_val/params.jld2 $X/final/c_production.jld2"
step final_val bash -c "mkdir -p $X/final/val && cp $X/final/a_assembled.jld2 $X/final/b_best_single.jld2 $X/final/val/ && TOST_RANK=0.2,0.1,0.05,acc $JL tools/tost_select.jl $C $X/final/val/a_assembled.jld2,$X/final/val/b_best_single.jld2 2 val 0.05,0.1,0.2"
step final_test bash -c "mkdir -p $X/final/test/pairs && cp $X/final/a_assembled.jld2 $X/final/b_best_single.jld2 $X/final/c_production.jld2 $X/final/test/ && \
     TOST_DUMP=$X/final/test/pairs $JL tools/tost_select.jl $C $X/final/test/a_assembled.jld2,$X/final/test/b_best_single.jld2,$X/final/test/c_production.jld2 3 test 0.05,0.1,0.2"
step summary "$PY" -I -c "
import pandas as p
X='$X'; ch=p.read_csv(X+'/assembled_choices.csv', keep_default_na=False)
ch['family']=['bias' if 'bias' in f else ('mo_pick' if f.startswith('prod_') else 'species_sweep') for f in ch.file]
md=['# Mixed-pool species-level selection — bs_$ROLE', '', f'pool {sum(1 for _ in open(X+\"/cands/list.txt\"))} sets; groups by chosen family: '+', '.join(f'{k} {v}' for k,v in ch.family.value_counts().items()), '']
nm={1:'a_assembled',2:'b_best_single',3:'c_production'}
for split,f in (('val (decides: assembled vs best single)',X+'/final/val/tost_select_val.csv'),('test (report; c = production selection, comparison)',X+'/final/test/tost_select_test.csv')):
    t=p.read_csv(f).sort_values('cand'); md+=['## '+split,'','| set | groups | ±5 % | ±10 % | ±20 % | median abs bias % | mean abs log ratio |','|---|---|---|---|---|---|---|']
    md+=[f'| {nm[int(r.cand)]} | {r.groups} | {r.pass_5} | {r.pass_10} | {r.pass_20} | {r.median_abs_bias} | {r.mean_abs_lr} |' for r in t.itertuples()]+['']
md+=['best single = '+open(X+'/cands/selected_val/source.txt').read().strip().split('/')[-1],'','## per-group choices','','| group | chosen | family |','|---|---|---|']+[f'| {r.group} | {r.file} | {r.family} |' for r in ch.itertuples()]
open(X+'/summary.md','w').write('\n'.join(md)); print('\n'.join(md[:24]))"
log "POOL DONE $ROLE"
