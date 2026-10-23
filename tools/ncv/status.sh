#!/usr/bin/env bash
# Progress of a nested-CV campaign (tools/ncv/campaign.sh): done steps per model/role, the running step (lock held?) and the last log lines.
#   tools/ncv/status.sh [root]      (default runs_local/ncv)
NCV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; R="${1:-$NCV/runs_local/ncv}"
[ -d "$R" ] || { echo "no campaign at $R"; exit 1; }
if ( exec 9> "$R/.lock"; flock -n 9 ); then echo "campaign: NOT running (lock free)"; else echo "campaign: RUNNING"; fi
echo "done steps:"; ( cd "$R/steps" 2>/dev/null && find . -name '*.done' | sed 's#^\./##; s#\.done$##' | sort | awk -F/ '{k=$1"/"$2; a[k]=a[k]" "$NF} END{for (k in a) print "  " k ":" a[k]}' | sort )
echo "last campaign log lines:"; tail -n "${N:-8}" "$R/campaign.log" 2>/dev/null | sed 's/^/  /'
cur=$(grep -E '\] run  ' "$R/campaign.log" 2>/dev/null | tail -1 | sed 's/.*run  //')
if [ -n "$cur" ] && [ ! -f "$R/steps/$cur.done" ]; then echo "current step $cur — last lines of its log:"; tail -n 3 "$R/logs/$cur.log" | cut -c1-200 | sed 's/^/  /'; fi
