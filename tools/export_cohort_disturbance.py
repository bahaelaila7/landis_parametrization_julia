"""
export_cohort_disturbance.py
============================
Export PARTIAL *and* FULL per-cohort disturbance rows to a Parquet file (read by Pan's `disturbance_full_removals`).

Source of the logic: /workspace/FIA/FIA_DATA_PREP/cohort_disturbance.py (NOT modified). Its `load()` and the module
constants (SUBPLOT_ID, COHORT_ID, SEVERITY_FRAC, GF_FLOOR/GF_CEIL, DISTURB_FATES, NATURAL_FATES) are imported as-is.
`compute()` (source lines 62-173) cannot be reused because it drops full-cohort removals (n_survivors == 0) internally
(source lines 160-165); `compute_all()` below is a COPY of that function with exactly one change: the full removals are
kept, plus two extra columns (`n_alive_cur`, `full_removal`). The partial rows (n_survivors > 0) are therefore identical
to the source's output (validated against the `cohort_disturbance` table in slim_r4.duckdb, see --validate).

Definitions
  full_removal    : no tree of the cohort is alive at T (n_alive_cur == 0): the cohort is GONE from the inventory at T
                    (its trees were cut / disturbance-killed / disappeared). drop = 1.0.
  rows with n_survivors == 0 but n_alive_cur > 0 (only severely damaged survivors remain, so the cohort still has a live
                    cohort row at T) are exported as partial (full_removal False); the legacy table dropped them.
  measdate        : visit T at which the loss is observed; age_calc = sim_year(T) - birth_sim_year (age had it survived).

The FIA DuckDB is opened READ-ONLY.

Usage: export_cohort_disturbance.py [--db FIADB.duckdb] [--out parquet] [--validate slim.duckdb] STATECD ...
"""
import sys, argparse
sys.path.insert(0, "/workspace/FIA/FIA_DATA_PREP")
import duckdb
import polars as pl
import cohort_disturbance as cd
from cohort_disturbance import (SUBPLOT_ID, COHORT_ID, SEVERITY_FRAC, GF_FLOOR, GF_CEIL, DISTURB_FATES, NATURAL_FATES)


def compute_all(df):
    """COPY of cohort_disturbance.compute (lines 62-173) keeping full-cohort removals."""
    df = df.with_columns([
        (pl.col("drybio_g") * pl.col("tpa")).alias("w"),
        (pl.col("sim_year") - pl.col("age_calc")).alias("birth_sim_year"),
        (
            (pl.col("statuscd") == 1)
            | (pl.col("statuscd").is_null() & pl.col("intro_type").is_in(["phantom_backfill", "gap_interpolated"]))
        ).alias("alive"),
        ((pl.col("damsev1") > 40) | (pl.col("damsev2") > 40)).fill_null(False).alias("severe_dmg"),
    ])
    visits = (
        df.select(SUBPLOT_ID + ["measdate", "sim_year"]).unique()
        .sort(SUBPLOT_ID + ["measdate"])
        .with_columns(pl.col("measdate").rank("dense").over(SUBPLOT_ID).cast(pl.Int32).alias("vrank"))
    )
    df = df.join(visits.select(SUBPLOT_ID + ["measdate", "vrank"]), on=SUBPLOT_ID + ["measdate"], how="left")
    alive = df.filter(pl.col("alive")).select(COHORT_ID + ["tree", "vrank", "w", "severe_dmg", "terminal_fate"])
    prev = (
        alive.select(COHORT_ID + ["tree", "vrank", "w", "terminal_fate"])
        .rename({"vrank": "vrank_prev", "w": "w_prev"})
        .with_columns((pl.col("vrank_prev") + 1).alias("vrank"))
    )
    cur = (
        alive.select(COHORT_ID + ["tree", "vrank", "w", "severe_dmg"])
        .rename({"w": "w_cur", "severe_dmg": "dmg_cur"})
    )
    pair = prev.join(cur, on=COHORT_ID + ["tree", "vrank"], how="left")
    disturbed_removal = pl.col("w_cur").is_null() & pl.col("terminal_fate").is_in(DISTURB_FATES)
    natural_removal = pl.col("w_cur").is_null() & pl.col("terminal_fate").is_in(NATURAL_FATES)
    other_gone = pl.col("w_cur").is_null() & ~(disturbed_removal | natural_removal)
    survivor = pl.col("w_cur").is_not_null() & ~pl.col("dmg_cur").fill_null(False)
    damaged_alive = pl.col("w_cur").is_not_null() & pl.col("dmg_cur").fill_null(False)
    pair = pair.with_columns([
        disturbed_removal.alias("is_disturbed_removal"),
        natural_removal.alias("is_natural_removal"),
        other_gone.alias("is_other_gone"),
        survivor.alias("is_survivor"),
        damaged_alive.alias("is_damaged_alive"),
    ])
    surv = (
        pair.filter(pl.col("is_survivor"))
        .group_by(COHORT_ID + ["vrank"])
        .agg([pl.col("w_prev").sum().alias("surv_w_prev"), pl.col("w_cur").sum().alias("surv_w_cur")])
        .with_columns(
            pl.when(pl.col("surv_w_prev") > 0)
            .then((pl.col("surv_w_cur") / pl.col("surv_w_prev")).clip(GF_FLOOR, GF_CEIL))
            .otherwise(1.0)
            .alias("gf")
        )
    )
    pair = pair.join(surv.select(COHORT_ID + ["vrank", "gf", "surv_w_cur"]), on=COHORT_ID + ["vrank"], how="left")
    pair = pair.with_columns(pl.col("gf").fill_null(1.0).alias("gf"))
    pair = pair.with_columns((pl.col("w_prev") * pl.col("gf")).alias("w_exp"))
    pair = pair.with_columns([
        (
            pl.col("is_disturbed_removal")
            | (pl.col("is_damaged_alive")
               & ((pl.col("w_exp") - pl.col("w_cur")).clip(0.0, None) > SEVERITY_FRAC * pl.col("w_exp")))
        ).alias("counted"),
    ])
    pair = pair.with_columns([
        pl.when(pl.col("counted")).then(pl.col("w_exp")).otherwise(0.0).alias("counted_would_be"),
        pl.when(pl.col("counted"))
          .then(pl.col("w_exp") - pl.col("w_cur").fill_null(0.0)).otherwise(0.0)
          .clip(0.0, None).alias("counted_dropped"),
        pl.when(pl.col("w_cur").is_not_null() & ~pl.col("counted")).then(pl.col("w_cur")).otherwise(0.0)
          .alias("noncounted_live_w"),
    ])
    coh = (
        pair.group_by(COHORT_ID + ["vrank"])
        .agg([
            pl.col("counted_would_be").sum().alias("would_be_disturbed"),
            pl.col("counted_dropped").sum().alias("dropped"),
            pl.col("noncounted_live_w").sum().alias("live_w"),
            pl.col("counted").sum().alias("n_disturbed"),
            pl.col("is_survivor").sum().alias("n_survivors"),
            pl.col("w_cur").is_not_null().sum().alias("n_alive_cur"),      # NEW (not in source)
        ])
        .with_columns((pl.col("would_be_disturbed") + pl.col("live_w")).alias("would_be"))
        .filter(pl.col("dropped") > 0)
        .with_columns((pl.col("dropped") / pl.col("would_be")).alias("disturbance_drop_pct"))
        .with_columns((pl.col("n_alive_cur") == 0).alias("full_removal"))   # NEW; source dropped n_survivors == 0 here
    )
    vr = visits.select(SUBPLOT_ID + ["measdate", "sim_year", "vrank"])
    coh = (
        coh.join(vr, on=SUBPLOT_ID + ["vrank"], how="left")
        .with_columns((pl.col("sim_year") - pl.col("birth_sim_year")).cast(pl.Int32).alias("age_calc"))
        .select(SUBPLOT_ID + ["species_symbol", "measdate", "age_calc", "disturbance_drop_pct", "dropped", "would_be",
                              "n_disturbed", "n_survivors", "n_alive_cur", "full_removal"])
    )
    return coh


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/workspace/FIA/FIASQLITE2PGSQL/FIADB.duckdb")
    ap.add_argument("--out", default="/workspace/landis_mogd/data/cohort_disturbance_full.parquet")
    ap.add_argument("--validate", default="/workspace/landis_region4/data_r4/slim_r4.duckdb")
    ap.add_argument("states", nargs="*", type=int)
    a = ap.parse_args()
    states = a.states or [12, 13]
    con = duckdb.connect(a.db, read_only=True)
    coh = compute_all(cd.load(con, states))
    con.close()
    n_undated = coh.filter((pl.col("measdate").is_null() | pl.col("age_calc").is_null())).height
    coh = coh.filter(pl.col("measdate").is_not_null() & pl.col("age_calc").is_not_null())   # removal after a subplot's LAST recorded visit: no visit T to attach it to
    print(f"[export] dropped {n_undated:,} rows without a visit T (cohort lost after the last recorded visit)")
    coh = coh.with_columns(pl.col("measdate").cast(pl.Datetime("ms"))).sort(SUBPLOT_ID + ["species_symbol", "measdate", "age_calc"])
    coh.write_parquet(a.out)
    print(f"[export] {coh.height:,} rows -> {a.out}")
    for s in states:
        c = coh.filter(pl.col("statecd") == s)
        print(f"  state {s}: total {c.height:,}  full_removal {c.filter(pl.col('full_removal')).height:,}  "
              f"partial {c.filter(~pl.col('full_removal')).height:,}  (legacy partial n_survivors>0: {c.filter(pl.col('n_survivors') > 0).height:,})")
    if a.validate:
        v = duckdb.connect(a.validate, read_only=True)
        ref = v.execute("select * from cohort_disturbance").pl().with_columns(pl.col("measdate").cast(pl.Datetime("ms")))
        v.close()
        ref = ref.filter(pl.col("statecd").is_in(states))
        key = SUBPLOT_ID + ["species_symbol", "measdate", "age_calc"]
        mine = coh.filter(pl.col("n_survivors") > 0)
        j = ref.join(mine, on=key, how="left", suffix="_new")
        found = j.filter(pl.col("disturbance_drop_pct_new").is_not_null())
        close = found.filter((pl.col("disturbance_drop_pct") - pl.col("disturbance_drop_pct_new")).abs() < 1e-6)
        print(f"[validate] reference rows {ref.height:,}; key matched {found.height:,} ({found.height/max(ref.height,1):.4%}); "
              f"drop equal (1e-6) {close.height:,} ({close.height/max(ref.height,1):.4%})")
        extra = mine.join(ref.select(key), on=key, how="anti")
        print(f"[validate] legacy-partial rows not in reference table: {extra.height:,}")
