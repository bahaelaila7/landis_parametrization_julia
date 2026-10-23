"""Check that every model of a nested-CV campaign used the role's plots (docs/nested_cv_gafl.md §5.3). Exit 1 on any FAIL.
  python -I tools/ncv/verify_splits.py <root> <models,comma> <roles,comma>
Per role: the train-capped split (prep/<role>/split/split_plots.csv) keeps the role file's test plots exactly, train ⊆ role train, and
val = role val ∪ train surplus; the BS / HBAD v1 / PnET runs wrote that same split (their split_plots.csv); HBAD v2 read it; the HBAD3P interval
product carries it (every GA+FL interval's set = its plot's set)."""
import sys, os, pandas as pd, pyarrow.ipc as ipc
R, models, roles = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(',')
ids = ['statecd', 'unitcd', 'countycd', 'plot']; fails = []
def check(name, ok, msg=''):
    print(f"{'PASS' if ok else 'FAIL'} {name} {msg}"); ok or fails.append(name)
def sets(f): d = pd.read_csv(f); return d[ids + ['set']].sort_values(ids).reset_index(drop=True)
train_by_model = {}
for r in roles:
    role = sets(f'{R}/folds/split_{r}.csv'); cap = sets(f'{R}/prep/{r}/split/split_plots.csv')
    m = cap.merge(role, on=ids, how='left', suffixes=('', '_role'))
    check(f'{r} capped split plots are role plots', m.set_role.notna().all(), f'{len(cap)} of {len(role)} role plots carry cohorts')
    check(f'{r} test = role test', ((m.set == 'test') == (m.set_role == 'test')).all(), f'test {int((m.set == "test").sum())}')
    check(f'{r} train ⊆ role train', (m.set_role[m.set == 'train'] == 'train').all(), f'train {int((m.set == "train").sum())} of role {int((m.set_role == "train").sum())}')
    check(f'{r} val = role val ∪ train surplus', m.set_role[m.set == 'val'].isin(['val', 'train']).all() and (m.set[m.set_role == 'val'] == 'val').all())
    for mod in models:
        if mod in ('bs', 'hbad', 'pnet'):
            for sub in ('gd_bias', 'gd_mo'):
                f = f'{R}/{mod}/{r}/{sub}/split_plots.csv'
                if os.path.exists(f):
                    s = sets(f); check(f'{r} {mod}/{sub} used the capped split', s.equals(cap))
                    sub == 'gd_bias' and train_by_model.setdefault(r, {}).__setitem__(mod, set(map(tuple, s[s.set == 'train'][ids].values)))
        elif mod == 'hbad2':
            for sub in ('simA', 'joint'):
                f = f'{R}/hbad2/{r}/{sub}/split_plots.csv'
                if os.path.exists(f): check(f'{r} hbad2/{sub} used the capped split', sets(f).equals(cap))
        elif mod == 'hbad3p':
            iv = ipc.open_file(f'{R}/prep/{r}/product/intervals.arrow').read_all().to_pandas()[['st', 'u', 'c', 'p', 'set']]
            iv.columns = ids + ['set']; j = iv.merge(cap, on=ids, how='left', suffixes=('', '_cap'))
            ga = j[j.statecd.isin([12, 13])]
            check(f'{r} hbad3p product sets = capped split', (ga.set.fillna('') == ga.set_cap.fillna('')).all() and j[~j.statecd.isin([12, 13])].set.isna().all(),
                  ', '.join(f'{s} {int((ga.set == s).sum())}' for s in ('train', 'val', 'test')) + ' intervals')
    tm = train_by_model.get(r, {})
    if len(tm) > 1: check(f'{r} identical train plots across {sorted(tm)}', len({frozenset(v) for v in tm.values()}) == 1)
print('ALL PASS' if not fails else f'FAILED: {fails}'); sys.exit(1 if fails else 0)
