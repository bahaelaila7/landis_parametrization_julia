#!/usr/bin/env python3
"""Compare two golden output roots scenario by scenario.

  .venv/bin/python -I compare.py <golden_root> <candidate_root> [--rtol 1e-5] [--atol 1e-6] [scenario ...]

Compared per scenario:
  - status.txt (exact),
  - the file listing of run/ (names, except .jld2/.png/.log),
  - every CSV (numeric columns with tolerance, others exact; columns that look like timings are ignored),
  - every losses.duckdb table (params_blob ignored),
  - every JSON (numbers with tolerance).
Exit code 0 = all scenarios match.
"""
import sys, os, json, math, re, argparse
import duckdb, pandas as pd

TIME_COL = re.compile(r"(time|wall|elapsed|sec\b|_s$|date|stamp|ns$|^run_id$)", re.I)
SKIP_EXT = (".jld2", ".png", ".pdf", ".log", ".svg", ".txt")


def close(a, b, rtol, atol):
    if isinstance(a, float) or isinstance(b, float):
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= atol + rtol * max(abs(a), abs(b))
    return a == b


def cmp_frames(x, y, what, rtol, atol, out):
    x = x[[c for c in x.columns if not TIME_COL.search(c)]]
    y = y[[c for c in y.columns if not TIME_COL.search(c)]]
    if list(x.columns) != list(y.columns):
        out.append(f"{what}: columns differ {list(x.columns)} vs {list(y.columns)}")
        return
    if len(x) != len(y):
        out.append(f"{what}: {len(x)} vs {len(y)} rows")
        return
    nbad = 0; first = None
    for c in x.columns:
        num = pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c])
        for i, (a, b) in enumerate(zip(x[c].tolist(), y[c].tolist())):
            ok = close(float(a), float(b), rtol, atol) if num else (str(a) == str(b))
            if not ok:
                nbad += 1
                if first is None:
                    first = f"row {i} col {c}: {a!r} vs {b!r}"
    if nbad:
        out.append(f"{what}: {nbad} cells differ (first: {first})")


def cmp_json(a, b, path, rtol, atol, out):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys differ {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b)):
            cmp_json(a[k], b[k], f"{path}.{k}", rtol, atol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}"); return
        for i, (u, v) in enumerate(zip(a, b)):
            cmp_json(u, v, f"{path}[{i}]", rtol, atol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not close(float(a), float(b), rtol, atol):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def files(root):
    res = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            res.append(os.path.relpath(os.path.join(dp, f), root))
    return sorted(res)


def cmp_scenario(ga, gb, rtol, atol):
    out = []
    for f in ("status.txt",):
        a = open(os.path.join(ga, f)).read().strip() if os.path.exists(os.path.join(ga, f)) else "<missing>"
        b = open(os.path.join(gb, f)).read().strip() if os.path.exists(os.path.join(gb, f)) else "<missing>"
        if a != b:
            out.append(f"status: {a!r} vs {b!r}")
    fa, fb = files(ga), files(gb)
    la = [f for f in fa if not f.endswith(SKIP_EXT) and not f.startswith("error")]
    lb = [f for f in fb if not f.endswith(SKIP_EXT) and not f.startswith("error")]
    if la != lb:
        out.append(f"file listing differs: only-golden={sorted(set(la) - set(lb))} only-candidate={sorted(set(lb) - set(la))}")
    for f in sorted(set(la) & set(lb)):
        pa, pb = os.path.join(ga, f), os.path.join(gb, f)
        if f.endswith(".csv"):
            try:
                cmp_frames(pd.read_csv(pa), pd.read_csv(pb), f, rtol, atol, out)
            except pd.errors.EmptyDataError:
                if os.path.getsize(pa) != os.path.getsize(pb):
                    out.append(f"{f}: empty vs non-empty")
        elif f.endswith(".json"):
            cmp_json(json.load(open(pa)), json.load(open(pb)), f, rtol, atol, out)
        elif f.endswith(".duckdb"):
            ca, cb = duckdb.connect(pa, read_only=True), duckdb.connect(pb, read_only=True)
            ta = sorted(t for (t,) in ca.sql("show tables").fetchall())
            tb = sorted(t for (t,) in cb.sql("show tables").fetchall())
            if ta != tb:
                out.append(f"{f}: tables {ta} vs {tb}")
            for t in sorted(set(ta) & set(tb)):
                xa = ca.sql(f"select * from {t}").df(); xb = cb.sql(f"select * from {t}").df()
                def _blob(df):  # serialized blobs (params_blob, ...) are not comparable across code versions
                    return [c for c in df.columns if c == "params_blob" or (len(df) and isinstance(df[c].iloc[0], (bytes, bytearray, memoryview)))]
                xa = xa.drop(columns=_blob(xa)); xb = xb.drop(columns=_blob(xb))
                keys = list(xa.columns)
                if keys == list(xb.columns) and len(keys):
                    xa = xa.sort_values(keys, kind="mergesort").reset_index(drop=True)
                    xb = xb.sort_values(keys, kind="mergesort").reset_index(drop=True)
                cmp_frames(xa, xb, f"{f}:{t}", rtol, atol, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden"); ap.add_argument("cand")
    ap.add_argument("--rtol", type=float, default=1e-5); ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    names = a.names or sorted(d for d in os.listdir(a.golden) if os.path.isdir(os.path.join(a.golden, d)))
    bad = 0
    for n in names:
        ga, gb = os.path.join(a.golden, n), os.path.join(a.cand, n)
        if not os.path.isdir(gb):
            print(f"MISSING {n}"); bad += 1; continue
        diffs = cmp_scenario(ga, gb, a.rtol, a.atol)
        if diffs:
            bad += 1; print(f"DIFF    {n}")
            for d in diffs[:12]:
                print(f"        {d}")
            if len(diffs) > 12:
                print(f"        ... {len(diffs) - 12} more")
        else:
            print(f"MATCH   {n}")
    print(f"{len(names) - bad}/{len(names)} scenarios match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
