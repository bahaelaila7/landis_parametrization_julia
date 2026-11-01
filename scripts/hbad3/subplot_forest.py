"""Forested fraction of every FIA subplot and microplot visit (accessible forest land, COND_STATUS_CD = 1) for the states of the R4 interval product:
  f_subp = sum of SUBPCOND_PROP over the forest conditions of the subplot, f_micr = the same with MICRCOND_PROP (the sapling tally area).
FIA tallies trees only on forest conditions, so a tree list of a subplot covers f_subp of its area and a sapling list f_micr of the microplot; a subplot
with no trees is either an unstocked forest subplot (f_subp > 0) or not forest at all (f_subp = 0). Used by the HBAD3P recruitment fit and free runs.
Source: the FIA truth DB (read-only; D59) raw SUBP_COND x COND x PLOT; visit date = PLOT MEASYEAR / MEASMON / MEASDAY (the MEASDATE of curated_trees).
  python -I scripts/hbad3/subplot_forest.py [truth DB path, default <WORKSPACE>/FIA/truth/fia_truth_r4.duckdb] -> results/hbad3/subplot_forest_r4.csv"""
import os, sys, duckdb
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WS = os.environ.get('WORKSPACE', os.path.dirname(REPO))
db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WS, 'FIA', 'truth', 'fia_truth_r4.duckdb')
out = os.environ.get('HBAD3_FOREST_CSV', os.path.join(REPO, 'results', 'hbad3', 'subplot_forest_r4.csv'))
out = out if os.path.isabs(out) else os.path.join(REPO, out)
c = duckdb.connect(db, read_only=True)
d = c.sql('''
  SELECT p.STATECD st, p.UNITCD u, p.COUNTYCD c, p.PLOT p, make_date(p.MEASYEAR::INT, p.MEASMON::INT, p.MEASDAY::INT) md, s.SUBP sb,
         sum(CASE WHEN k.COND_STATUS_CD = 1 THEN s.SUBPCOND_PROP ELSE 0 END) f_subp, sum(CASE WHEN k.COND_STATUS_CD = 1 THEN s.MICRCOND_PROP ELSE 0 END) f_micr,
         sum(s.SUBPCOND_PROP) tot_subp, sum(s.MICRCOND_PROP) tot_micr
  FROM raw.SUBP_COND s JOIN raw.PLOT p ON p.STATECD = s.STATECD AND p.UNITCD = s.UNITCD AND p.COUNTYCD = s.COUNTYCD AND p.PLOT = s.PLOT AND p.INVYR = s.INVYR
       JOIN raw.COND k ON k.STATECD = s.STATECD AND k.UNITCD = s.UNITCD AND k.COUNTYCD = s.COUNTYCD AND k.PLOT = s.PLOT AND k.INVYR = s.INVYR AND k.CONDID = s.CONDID
  WHERE p.STATECD IN (1, 12, 13, 21, 28, 37, 45, 47) AND p.MEASYEAR IS NOT NULL AND p.MEASMON IS NOT NULL AND p.MEASDAY IS NOT NULL
  GROUP BY ALL ORDER BY 1, 2, 3, 4, 5, 6''').df()
dup = d.duplicated(['st', 'u', 'c', 'p', 'md', 'sb']).sum()
assert dup == 0, f'{dup} duplicated (plot, date, subplot) rows'
bad = ~((d.tot_subp - 1).abs() <= 1e-6) | ~((d.tot_micr - 1).abs() <= 1e-6)              # proportions missing / not summing to 1: status unknown (NaN)
d.loc[bad, ['f_subp', 'f_micr']] = float('nan'); d['f_subp'] = d.f_subp.clip(0, 1); d['f_micr'] = d.f_micr.clip(0, 1)
d.to_csv(out, index=False)
print(f'{len(d)} subplot visits -> {out} | unknown {bad.mean():.3f} | f_subp=0 {(d.f_subp == 0).mean():.3f}, partial {((d.f_subp > 0) & (d.f_subp < 1)).mean():.3f} | '
      f'proportions summing to 1: subp {(abs(d.tot_subp - 1) < 1e-6).mean():.4f} micr {(abs(d.tot_micr - 1) < 1e-6).mean():.4f}')
