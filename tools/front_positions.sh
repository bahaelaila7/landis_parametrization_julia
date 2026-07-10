#!/usr/bin/env bash
# Cache the 5 designated p101-front positions (role -> candidate idx) from p101_candidates/manifest.csv,
# computed on the p101 union front itself (self-normalized to its own [Wmin,Wmax]×[Amin,Amax] box):
#   extreme_W   = min A_W   (first, = candidate 1)
#   extreme_AGB = min A_AGB  (last)
#   best_agg    = min (A_W + A_AGB)
#   knee        = max perpendicular distance from the chord joining the two extremes (normalized axes)
#   median      = closest point to the segment from the extremes' midpoint (0.5,0.5) to the lower-left
#                 corner (0,0) in normalized space (mirrors sweep_fold_percentiles.jl designate!)
# Writes <outdir>/p101_candidates/front_positions.csv.
#   tools/front_positions.sh <run_output_dir>
set -u
OUT="$1"; C="$OUT/p101_candidates"; M="$C/manifest.csv"
[ -f "$M" ] || { echo "no manifest: $M"; exit 1; }
awk -F, '
NR>1 { n++; idx[n]=$1; w[n]=$2+0; a[n]=$3+0; g[n]=$4
       if(w[n]<wmin||n==1)wmin=w[n]; if(w[n]>wmax||n==1)wmax=w[n]
       if(a[n]<amin||n==1)amin=a[n]; if(a[n]>amax||n==1)amax=a[n]
       s=w[n]+a[n]; if(s<bagg||n==1){bagg=s;bi=n} }
END{
  # extremes (manifest is sorted by A_W asc): first = min-W, last = min-AGB
  eW=1; eA=n
  dw=wmax-wmin; da=amax-amin; if(dw==0)dw=1; if(da==0)da=1
  # chord between extremes in normalized space
  x1=(w[eW]-wmin)/dw; y1=(a[eW]-amin)/da; x2=(w[eA]-wmin)/dw; y2=(a[eA]-amin)/da
  d12=sqrt((x2-x1)^2+(y2-y1)^2); kbest=-1; ki=eW
  # median: segment from midpoint(0.5,0.5) to lower-left corner (0,0)
  ax=0.5; ay=0.5; bx=0; by=0; ABx=bx-ax; ABy=by-ay; AB2=ABx*ABx+ABy*ABy; mbest=1e9; mi=eW
  for(i=1;i<=n;i++){
    xi=(w[i]-wmin)/dw; yi=(a[i]-amin)/da
    perp=(d12>0)?((x2-x1)*(y1-yi)-(x1-xi)*(y2-y1))/d12:0; if(perp<0)perp=-perp
    if(perp>kbest){kbest=perp;ki=i}
    t=(AB2>0)?(((xi-ax)*ABx+(yi-ay)*ABy)/AB2):0; if(t<0)t=0; if(t>1)t=1
    dx=xi-(ax+t*ABx); dy=yi-(ay+t*ABy); dd=sqrt(dx*dx+dy*dy)
    if(dd<mbest){mbest=dd;mi=i}
  }
  print "role,candidate,A_W,A_AGB,source_gen"
  printf "extreme_W,%s,%s,%s,%s\n",   idx[eW], w[eW], a[eW], g[eW]
  printf "extreme_AGB,%s,%s,%s,%s\n", idx[eA], w[eA], a[eA], g[eA]
  printf "knee,%s,%s,%s,%s\n",        idx[ki], w[ki], a[ki], g[ki]
  printf "median,%s,%s,%s,%s\n",      idx[mi], w[mi], a[mi], g[mi]
  printf "best_agg,%s,%s,%s,%s\n",    idx[bi], w[bi], a[bi], g[bi]
}' "$M" > "$C/front_positions.csv"
echo "wrote $C/front_positions.csv"; cat "$C/front_positions.csv"
