# Species-balanced train cap (Data._stratified_split_df train_cap): determinism, partition, cap, unset ⇒ unchanged.
#   ./julia_gdal.sh --project=. test/split_cap_test.jl
using Pan, Test, DataFrames, Random, Dates
const D = Pan.Data
function synth(; nsite=600, seed=7)
  rng = Random.Xoshiro(seed); rows = NamedTuple[]
  for s in 1:nsite
    eco = 1 + (s % 3)
    domsp = rand(rng) < 0.7 ? "PIEL" : rand(rng, ["QUNI", "LIST2", "ACRU", "NYBI"])   # 70 % pine-dominated
    for (v, d) in enumerate((Date(2005, 1, 1), Date(2012, 1, 1)))
      push!(rows, (statecd=12, unitcd=1, countycd=s ÷ 50, plot=s, eco_id=eco, measdate=d, effective_species=domsp, agb=100.0 * v))
      push!(rows, (statecd=12, unitcd=1, countycd=s ÷ 50, plot=s, eco_id=eco, measdate=d, effective_species="TAAS", agb=10.0))
    end
  end
  DataFrame(rows)
end
ids = [:statecd, :unitcd, :countycd, :plot]
plots(d) = Set(Tuple.(eachrow(unique(d[:, ids]))))
df = synth()
split(cap) = D._stratified_split_df(df, ids, 0.25, Random.Xoshiro(42); test_frac=0.25, train_cap=cap)
@testset "train_cap" begin
  a0 = split(nothing); b0 = D._stratified_split_df(df, ids, 0.25, Random.Xoshiro(42); test_frac=0.25)
  @test all(isequal(a0[i], b0[i]) for i in 1:3)                    # unset kwarg ⇒ identical to legacy call
  for cap in (30, "auto")
    a = split(cap); b = split(cap)
    @test all(isequal(a[i], b[i]) for i in 1:3)                    # deterministic
    tr, va, te = plots.(a)
    @test isempty(intersect(tr, va)) && isempty(intersect(tr, te)) && isempty(intersect(va, te))
    @test union(tr, va, te) == plots(df)                           # nothing lost
    @test length(tr) < length(plots(a0[1]))                        # cap actually removed train plots
    dd = D.LAST_SPLIT_DOM[]; trd = filter(r -> (r.statecd, r.unitcd, r.countycd, r.plot) in tr, dd)
    n = combine(groupby(trd, [:eco_id, :dom]), nrow => :n)
    cap isa Int && @test maximum(n.n) <= cap
    @test length(va) + length(te) > length(plots(a0[2])) + length(plots(a0[3]))   # surplus went to val/test
    println("cap=$cap  train=$(length(tr)) val=$(length(va)) test=$(length(te))  max/stratum=$(maximum(n.n))")
  end
end
