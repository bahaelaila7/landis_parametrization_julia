"""Build donor table for TreeMap 2022 plots lacking curated cohort data.
Run: /workspace/standmap/.venv/bin/python -I tools/build_treemap_donors.py
"""
import duckdb, numpy as np, pandas as pd
from dbfread import DBF
from sklearn.neighbors import NearestNeighbors

TM = '/workspace/FVSclaude/treemap/Data/'
DB = '/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb'
OUT = '/workspace/landis_scen/data/treemap_donors.csv'
GEO_W = 0.5
MIN_FT = 20

cols = ['Value', 'TM_ID', 'PLT_CN', 'FORTYPCD', 'FLDTYPCD', 'STDSZCD', 'BALIVE', 'QMD',
        'STANDHT', 'TPA_LIVE', 'CANOPYPCT', 'ALSTK']
dbf = DBF(TM + 'TreeMap2022_CONUS.tif.vat.dbf', load=False)
df = pd.DataFrame(iter(dbf))[cols]
df = df.dropna(subset=['TM_ID', 'PLT_CN']).drop_duplicates('TM_ID').reset_index(drop=True)
df['TM_ID'] = df.TM_ID.astype('int64')
df['cn'] = df.PLT_CN.astype('int64').astype(str)
print('treemap plots', len(df))

# tree table: live BA, softwood fraction
tt = pd.read_csv(TM + 'TreeMap2022_CONUS_Tree_Table.csv',
                 usecols=['TM_ID', 'STATUSCD', 'TPA_UNADJ', 'SPCD', 'DIA'])
tt = tt[tt.STATUSCD == 1].copy()
tt['ba'] = 0.005454154 * tt.DIA.fillna(0) ** 2 * tt.TPA_UNADJ.fillna(0)
g = tt.groupby('TM_ID').agg(ba_tot=('ba', 'sum'), n=('ba', 'size'))
g['ba_sw'] = tt[tt.SPCD < 300].groupby('TM_ID').ba.sum()
g['ba_sw'] = g.ba_sw.fillna(0)
df = df.merge(g, left_on='TM_ID', right_index=True, how='left')
df['n'] = df.n.fillna(0)
df['swfrac'] = np.where(df.ba_tot > 0, df.ba_sw / df.ba_tot.where(df.ba_tot > 0, 1), 0.0)

con = duckdb.connect(DB, read_only=True)
plot = con.execute("""select CN::VARCHAR as cn, STATECD, UNITCD, COUNTYCD, PLOT, LAT, LON, MEASYEAR
                      from PLOT""").df()
plot['key'] = (plot.STATECD.astype(str) + '_' + plot.UNITCD.astype(str) + '_' +
               plot.COUNTYCD.astype(str) + '_' + plot.PLOT.astype(str))
keys = set(r[0] for r in con.execute("""select distinct plt_cn from curated_cohorts_landis_stdorg
    where age_calc > 0 and coalesce(intro_type,'') <> 'phantom_backfill'""").fetchall())
df = df.merge(plot.rename(columns={'STATECD': 'statecd', 'LAT': 'lat', 'LON': 'lon',
                                   'MEASYEAR': 'measyear'})[['cn', 'key', 'statecd', 'lat', 'lon', 'measyear']],
              on='cn', how='left')
print('no PLOT match:', df.key.isna().sum())
df['has'] = df.key.isin(keys)
print('with data', int(df.has.sum()), 'without', int((~df.has).sum()))

df['nonstocked'] = ((df.BALIVE == 0) & (df.TPA_LIVE == 0)) | (df.n == 0)
feat = np.column_stack([np.log1p(df.BALIVE), np.log1p(df.QMD), df.STANDHT, np.log1p(df.TPA_LIVE),
                        df.CANOPYPCT, df.swfrac, df.lat, df.lon]).astype(float)
for j in range(feat.shape[1]):
    c = feat[:, j]
    m = np.nanmean(c); feat[np.isnan(c), j] = m
mu, sd = feat.mean(0), feat.std(0); sd[sd == 0] = 1
Z = (feat - mu) / sd
Z[:, 6:] *= GEO_W
df['fg'] = (df.FORTYPCD.fillna(-1).astype(int) // 10 * 10)
df['ft'] = df.FORTYPCD.fillna(-1).astype(int)

cand = df[df.has & ~df.nonstocked]  # donors must themselves be stocked
cand_idx = cand.index.values
rows = []
tgt = df[~df.has]
def nn(sel_idx, t_idx):
    nbr = NearestNeighbors(n_neighbors=1).fit(Z[sel_idx])
    d, i = nbr.kneighbors(Z[t_idx])
    return sel_idx[i[:, 0]], d[:, 0]
res = {}
for ti in tgt.index[tgt.nonstocked]:
    res[ti] = (None, np.nan, 'nonstocked')
ts = tgt[~tgt.nonstocked]
ft_cnt = cand.ft.value_counts(); fg_cnt = cand.fg.value_counts()
for ft, sub in ts.groupby('ft'):
    if ft_cnt.get(ft, 0) >= MIN_FT:
        pool, reason = cand_idx[(cand.ft == ft).values], 'fortypcd'
        di, d = nn(pool, sub.index.values)
        for t, a, b in zip(sub.index, di, d): res[t] = (a, b, reason)
    else:
        for fg, s2 in sub.groupby('fg'):
            if fg_cnt.get(fg, 0) >= 1 and fg >= 0:
                pool, reason = cand_idx[(cand.fg == fg).values], 'fortypgrp'
            else:
                pool, reason = cand_idx, 'any'
            di, d = nn(pool, s2.index.values)
            for t, a, b in zip(s2.index, di, d): res[t] = (a, b, reason)
out = []
for t in tgt.index:
    a, d, r = res[t]
    row = dict(cn=df.cn[t], donor_cn=(df.cn[a] if a is not None else ''), reason=r,
               dist=(round(float(d), 4) if a is not None else ''),
               cn_measyear=df.measyear[t], donor_measyear=(df.measyear[a] if a is not None else np.nan),
               cn_statecd=df.statecd[t], donor_statecd=(df.statecd[a] if a is not None else np.nan))
    out.append(row)
o = pd.DataFrame(out)
for c in ['cn_measyear', 'donor_measyear', 'cn_statecd', 'donor_statecd']:
    o[c] = o[c].astype('Int64')
o.to_csv(OUT, index=False)
print('wrote', OUT, len(o)); print(o.reason.value_counts())

# validation
byid = df.set_index('cn')
v = o[o.donor_cn != ''].copy()
def stats(v, label):
    print('==', label, len(v))
    print('same state', round(float((v.cn_statecd == v.donor_statecd).mean()), 4))
    for c in ['BALIVE', 'QMD', 'STANDHT', 'TPA_LIVE']:
        a = byid.loc[v.cn, c].values.astype(float); b = byid.loc[v.donor_cn, c].values.astype(float)
        ad = np.abs(a - b); pc = ad / np.where(a != 0, np.abs(a), np.nan)
        print(c, 'median|diff|', round(float(np.nanmedian(ad)), 3), 'median|%diff|', round(float(np.nanmedian(pc) * 100), 2))
    print('median dist', round(float(v.dist.astype(float).median()), 3))
stats(v, 'all'); stats(v[v.cn_statecd.isin([1, 12, 13, 21, 28, 37, 45, 47])], 'region4')
print('median measyear target/donor', v.cn_measyear.median(), v.donor_measyear.median())
print(v.sample(5, random_state=1).to_string())
