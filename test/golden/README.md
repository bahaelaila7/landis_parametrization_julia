# Golden harness (refactor safety net)

`scenarios.jl` defines 46 small scenarios. Together they cover every search driver, tier, injection mode and loss knob, all three
plugins, Sim B and plot_only. Each runs on the 80 fixed GA+FL plots in `plots.csv` with seed 1337, n_reps 1, a 2-fold split (fold 1)
and a handful of evaluations.

    test/golden/run.sh <repo_root> <out_root> [jobs] [scenario ...]     # one Julia process per scenario, 1 thread
    python -I test/golden/compare.py <golden_root> <candidate_root>     # needs duckdb + pandas

- Each scenario writes:
  - `status.txt` (OK or the first error line);
  - the run outputs: losses.duckdb, CSVs, best_params JSON;
  - a probe: one fixed parameter set evaluated at tier 3 on the training context. It records the per-(eco × species) losses, the u
    round trip and the pre-injection cohort state.
- `probe_v0.jl` is the adapter for the pre-refactor API (`Pan.POST_SEARCH_HOOK`, a global). `probe_v1.jl` produces the same files through
  the explicit `run_from_yaml(...; post_search_hook=)` of Step 3e; `run_scenario.jl` picks the adapter by `isdefined(Pan, :POST_SEARCH_HOOK)`.
- Recorded goldens live outside the repo: /workspace/pan_refactor/golden_out/pre_refactor (tag `pre-refactor`).
- The data comes from /workspace/landis_region4/data_r4 (PAN_DATA).
