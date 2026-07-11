#!/usr/bin/env bash
# Extract representative archive candidates from a MO-CMA-ES checkpoint at a given generation.
# Saves params.jld2 for best-W (min ΣW), best-AGB (min ΣAGB) and best-agg (min ΣW+ΣAGB) under
# <output_dir>/cand_<role>_<gen>/params.jld2, and prints each candidate's (A_W, A_AGB).
# Use the saved params.jld2 with a scatter/eval script via PAN_PARAMS.
#   tools/extract_candidates.sh <output_dir> <gen>
set -euo pipefail
ROOT="${PAN_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)}}"
cd "$ROOT"
O="$1"; G="$2"
[ -n "${PAN_OUT:-}" ] && O="$PAN_OUT/$(basename "$O")"   # find checkpoints under $PAN_OUT (scratch) if set
CK="$O/search_state@$G.jld2"
[ -f "$CK" ] || { echo "no checkpoint: $CK"; echo "available: $(ls "$O"/search_state@*.jld2 2>/dev/null | sed -E 's/.*@([0-9]+)\.jld2/\1/' | sort -n | tr '\n' ' ')"; exit 1; }
./julia_gdal.sh --project=. --threads=1 -e "
using Pan; import JLD2                                    # using Pan so params load as the real struct (not ReconstructedMutable)
st=JLD2.load_object(\"$CK\"); arch=collect(st.archive)
W(c)=sum(Float64.(c.fx.objectives[1:2:end])); A(c)=sum(Float64.(c.fx.objectives[2:2:end]))
roles=[(\"bestW\",argmin([W(c) for c in arch])),(\"bestAGB\",argmin([A(c) for c in arch])),(\"bestagg\",argmin([W(c)+A(c) for c in arch]))]
for (r,i) in roles
  d=\"$O/cand_\$(r)_$G\"; mkpath(d); JLD2.save_object(joinpath(d,\"params.jld2\"), arch[i].x)
  println(\"ROLE \$r  idx=\$i  A_W=\", round(W(arch[i]),digits=5), \"  A_AGB=\", round(A(arch[i]),digits=4), \"  -> \$d/params.jld2\")
end
println(\"n_arch=\", length(arch))"
