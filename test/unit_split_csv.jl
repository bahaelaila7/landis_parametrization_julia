# split_csv (nested CV roles, tools/ncv): every plot's set comes from the file; plots the file does not list are dropped; the species-balanced train cap
# moves train surplus to VAL only (the test plots stay exactly the file's); a file without test plots gives no test split; config validation.
module UnitSplitCSV
using Test, Pan, DataFrames, CSV, Dates, Random
const D = Pan.Data
@testset "split_csv" begin
  # 40 plots, one cohort row each; eco 1 for plots 1-30 (species A dominant), eco 2 for 31-40 (species B)
  n = 40
  df = DataFrame(statecd=fill(12, n), unitcd=fill(1, n), countycd=fill(1, n), plot=1:n, eco_id=[p <= 30 ? 1 : 2 for p in 1:n],
                 effective_species=[p <= 30 ? "A" : "B" for p in 1:n], measdate=fill(Date(2010, 6, 1), n), agb=fill(100.0, n))
  sets = [p % 5 == 0 ? "test" : p % 5 == 1 ? "val" : "train" for p in 1:n]
  dir = mktempdir(); f = joinpath(dir, "split.csv")
  keep = 1:38                                    # plots 39, 40 are not in the file -> dropped
  CSV.write(f, DataFrame(statecd=12, unitcd=1, countycd=1, plot=collect(keep), set=sets[keep]))
  idc = [:statecd, :unitcd, :countycd, :plot]
  tr, va, te, dom = D._split_from_csv(df, idc, f, Random.Xoshiro(1))
  @test sort(tr.plot) == [p for p in keep if sets[p] == "train"]
  @test sort(va.plot) == [p for p in keep if sets[p] == "val"]
  @test sort(te.plot) == [p for p in keep if sets[p] == "test"]
  @test dom === nothing
  # train cap 5 per (eco, dominant species): eco 1 has 18 train plots -> 13 surplus to val; eco 2 has 4 (31..38 minus val/test) -> none
  tr2, va2, te2, dom2 = D._split_from_csv(df, idc, f, Random.Xoshiro(1); train_cap=5)
  @test sort(te2.plot) == sort(te.plot)                                   # test untouched
  @test count(<=(30), tr2.plot) == 5 && count(>(30), tr2.plot) == count(p -> p > 30 && sets[p] == "train", keep)
  @test isempty(intersect(Set(tr2.plot), Set(va2.plot))) && isempty(intersect(Set(tr2.plot), Set(te2.plot)))
  @test sort(vcat(tr2.plot, va2.plot)) == sort(vcat(tr.plot, va.plot))   # surplus went to val, nothing lost
  @test dom2 isa DataFrame
  # the same rng gives the same capped train set (reproducible)
  tr3, _, _, _ = D._split_from_csv(df, idc, f, Random.Xoshiro(1); train_cap=5); @test sort(tr3.plot) == sort(tr2.plot)
  # no test plot in the file (final fit) -> no test split
  CSV.write(f, DataFrame(statecd=12, unitcd=1, countycd=1, plot=collect(keep), set=[s == "test" ? "train" : s for s in sets[keep]]))
  _, _, te4, _ = D._split_from_csv(df, idc, f, Random.Xoshiro(1)); @test te4 === nothing
  # duplicate plot rows are an error
  CSV.write(f, DataFrame(statecd=12, unitcd=1, countycd=1, plot=[1, 1, 2], set=["train", "val", "test"]))
  @test_throws ErrorException D._split_from_csv(df, idc, f, Random.Xoshiro(1))
  # config validation: split_csv excludes drawn splits
  db = joinpath(dir, "x.duckdb"); touch(db); CSV.write(f, DataFrame(statecd=12, unitcd=1, countycd=1, plot=[1], set=["train"]))
  @test D.split_active(D.DataConfig(; cohorts_db_path=db, split_csv=f))
  @test_throws ErrorException D.validate_data_config(D.DataConfig(; cohorts_db_path=db, split_csv=f, val_frac=0.2))
  @test_throws ErrorException D.validate_data_config(D.DataConfig(; cohorts_db_path=db, split_csv=joinpath(dir, "missing.csv")))
end
end
