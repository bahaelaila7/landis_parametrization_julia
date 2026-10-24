#!/usr/bin/env bash
# Corrected selection for a species-level experiment dir (slmo_bs.sh / slmo_pool.sh output; review 2026-10-24):
#  * NOT circular: the role's inner VAL plots are split into two halves by 0.25 deg spatial block (seeded); per-group choices AND the best single set
#    are made on half A, the decision assembled-vs-best-single on half B only; the outer TEST fold reports once (+ the production selection).
#  * D-consistent assembly: a tiered species (several TOST groups, one species-level D) takes ONE candidate for all its cells (summed group results;
#    tools/ncv/slmo_assemble.jl --species-tied).
# Writes <dir>/v2/. Resumable.   tools/ncv/slmo_select.sh <experiment dir> [role=k1]
set -uo pipefail
NCV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; WS="$(dirname "$NCV")"; cd "$NCV"; source wt_env.sh
X="$1"; ROLE="${2:-k1}"; R="$NCV/runs_local/ncv"; D="$R/bs/$ROLE"; V="$X/v2"; mkdir -p "$V"
export PAN_DATA="$WS/landis_region4/data_r4" PAN_OUT="$V" TOST_REAL_ONLY=1
PY="$WS/standmap/.venv/bin/python"; JL="./julia_gdal.sh --project=. --threads=${THREADS:-4}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$V/select.log"; }
step() { local id=$1; shift; local mk="$V/steps/$id.done" lg="$V/logs/$id.log"; [ -f "$mk" ] && return 0; mkdir -p "$(dirname "$mk")" "$(dirname "$lg")"
         log "run  $id"; local t0=$SECONDS; if ( "$@" ) < /dev/null >> "$lg" 2>&1; then date > "$mk"; log "done $id ($((SECONDS - t0)) s)"; else log "FAIL $id ($lg)"; tail -15 "$lg"; exit 1; fi; }
# 1. half files: the role's capped split (prep/<role>/split/split_plots.csv); val plots -> half A / half B by spatial block; test stays test
step halves "$PY" -I -c "
import pandas as p, numpy as np
k=['statecd','unitcd','countycd','plot']
s=p.read_csv('$R/prep/$ROLE/split/split_plots.csv'); f=p.read_csv('$R/folds/folds.csv')[k+['block']]
s=s.merge(f,on=k,how='left'); v=s[s.set=='val']; bl=sorted(v.block.unique()); rng=np.random.default_rng(20261024); rng.shuffle(bl)
half={b:('A' if i%2==0 else 'B') for i,b in enumerate(bl)}
for h in 'AB':
    o=s[k].copy(); o['set']=np.where(s.set=='test','test',np.where((s.set=='val')&(s.block.map(half)==h),'val','train')); o.to_csv('$V/split_val'+h+'.csv',index=False)
    print(h, int((o.set=='val').sum()), 'val plots in', sum(1 for b in bl if half[b]==h), 'blocks')"
for h in A B; do
  step cfg_$h "$PY" -I tools/ncv/mkcfg.py "$D/config.yml" "$V/val$h.yml" split_csv="$V/split_val$h.csv" train_cap=null output_dir="$V/run_$h"
done
# 2. all candidates on half A; species-tied assembly from half A
step cands bash -c "set -e; mkdir -p $V/candsA; : > $V/candsA/list.txt; for f in \$(cat $X/cands/list.txt); do cp \$f $V/candsA/; echo $V/candsA/\$(basename \$f) >> $V/candsA/list.txt; done"
step valA bash -c "TOST_RANK=0.2,0.1,0.05,acc $JL tools/tost_select.jl $V/valA.yml \$(paste -sd, $V/candsA/list.txt) 64 val 0.05,0.1,0.2"
step assemble $JL tools/ncv/slmo_assemble.jl "$V/valA.yml" "$V/candsA" "$V/assembled.jld2" --species-tied
# 3. decision on half B only: assembled vs the best single of half A
step final_files bash -c "set -e; mkdir -p $V/final; cp $V/assembled.jld2 $V/final/a_assembled.jld2; cp $V/candsA/selected_val/params.jld2 $V/final/b_best_single.jld2; cp $D/cands/selected_val/params.jld2 $V/final/c_production.jld2"
step valB bash -c "mkdir -p $V/final/valB && cp $V/final/a_assembled.jld2 $V/final/b_best_single.jld2 $V/final/valB/ && TOST_RANK=0.2,0.1,0.05,acc $JL tools/tost_select.jl $V/valB.yml $V/final/valB/a_assembled.jld2,$V/final/valB/b_best_single.jld2 2 val 0.05,0.1,0.2"
# 4. test report (role config: the outer test fold)
step test bash -c "mkdir -p $V/final/test/pairs && cp $V/final/a_assembled.jld2 $V/final/b_best_single.jld2 $V/final/c_production.jld2 $V/final/test/ && \
     TOST_DUMP=$V/final/test/pairs $JL tools/tost_select.jl $D/config.yml $V/final/test/a_assembled.jld2,$V/final/test/b_best_single.jld2,$V/final/test/c_production.jld2 3 test 0.05,0.1,0.2"
step summary "$PY" -I -c "
import pandas as p
V='$V'; nm={1:'a_assembled',2:'b_best_single',3:'c_production'}
win='a_assembled' if p.read_csv(V+'/final/valB/tost_select_val.csv').cand.iloc[0]==1 else 'b_best_single'
md=['# Species-level selection v2 (split val, species-tied D) — '+V.split('/')[-2], '', 'choices on val half A; decision on val half B: **'+win+'**; best single (half A) = '+open(V+'/candsA/selected_val/source.txt').read().strip().split('/')[-1], '']
for split,f in (('val half B (decides)',V+'/final/valB/tost_select_val.csv'),('test (report; c = production selection, comparison)',V+'/final/test/tost_select_test.csv')):
    t=p.read_csv(f).sort_values('cand'); md+=['## '+split,'','| set | groups | ±5 % | ±10 % | ±20 % | median abs bias % | mean abs log ratio |','|---|---|---|---|---|---|---|']
    md+=[f'| {nm[int(r.cand)]} | {r.groups} | {r.pass_5} | {r.pass_10} | {r.pass_20} | {r.median_abs_bias} | {r.mean_abs_lr} |' for r in t.itertuples()]+['']
open(V+'/summary.md','w').write('\n'.join(md)); print('\n'.join(md))"
log "SELECT V2 DONE"
