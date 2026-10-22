# prototype free run of 30 GA+FL val plots for 10 years, W = Wn (projection convention), no fraction; plot BA at t = 0 and 10 (m2/ha)
R = ENV["HBAD3_REPO"]
include(joinpath(R, "tools", "hbad3", "common3.jl")); include(joinpath(R, "src", "plugins", "HBAD3", "HBAD3P.jl")); import .HBAD3P
include(joinpath(R, "tools", "hbad3", "lib3p.jl"))
FD = joinpath(R3DIR, "fit3p_P3_GAFL"); FAREA = meta_farea(FD)
tr = load_trees(); keys_ = sort(unique(tr.key)); nsp = length(keys_); kix = Dict(k => i for (i, k) in enumerate(keys_))
M = load_fit3p(FD, keys_); P, Rg = load_recruit3p(FD, M.P); G = M.G
T = interval_tables(keys_); wd = water_pairs(); IF = interval_forest(T.iv, forest_table()); HBAD3P.set_quadrature!(VQ_Z, VQ_W)
ivd = Dict(r.iid => r for r in eachrow(T.iv)); regiids = Set(tr.iid)
allp = combine(groupby(T.iv, [:st, :u, :c, :p]), :iid => (x -> all(i -> i in regiids && !ivd[i].planted, x)) => :ok, nrow => :n, :set => first => :set)
plots = allp[allp.ok .& (allp.n .>= 2) .& (coalesce.(allp.set, "") .== "val"), :][1:30, :]
ivbp = Dict((k.st, k.u, k.c, k.p) => sort(DataFrame(g), :k) for (k, g) in pairs(groupby(T.iv, [:st, :u, :c, :p])))
B = build_free(plots, ivbp, T, IF, wd, keys_, kix, FAREA)
for (pi_, r) in enumerate(B.res)                        # W = Wn = the plot normal (= the first interval's aetpet_n)
  for s in findall(==(pi_), B.pl_site); B.Ws[s] = B.Wns[s]; end
end
F0 = free_store(B, nsp); v0 = plot_values(F0, B)
F = free_store(B, nsp); HBAD3P.run_free!(F, P, G, Rg, fill(10, length(B.live)), (s, y) -> nothing)
v1 = plot_values(F, B)
out = DataFrame(st=Int[], u=Int[], c=Int[], p=Int[], d1=Date[], ba0=Float64[], ba10=Float64[], n12_10=Float64[])
for (pi_, r) in enumerate(B.res)
  push!(out, (r.st, r.u, r.c, r.p, ivbp[(r.st, r.u, r.c, r.p)].d1[1], v0.sim[(v0.var .== "BA")][pi_], v1.sim[(v1.var .== "BA")][pi_], v1.sim[(v1.var .== "N12")][pi_]))
end
CSV.write(ARGS[1], out); println(first(out, 5))
