# Reproduce the run's exact train/val split (val_frac + split_seed, eco×lu×species stratified) and emit a
# LaTeX table of plot counts per (eco × land use) × {train, val}.
#   Run: ./julia_gdal.sh --project=. tools/split_counts_latex.jl runs/fl5_l4cover_mocmaes_Aonly_Sglobal_ipop_v2.yml
using Pan, YAML, DataFrames
const P = Pan; const D = Pan.Data
cfg = YAML.load_file(ARGS[1]); g(k, d) = get(cfg, k, d)
val_frac = Float64(g("val_frac", 0.2)); split_rng = P.RNGType(UInt64(Int(g("split_seed", 42))))
splots, eco_list, species_list, eco_species_ids, splots_val = D.prepare_parametrization_data(;
  cohorts_db_path=cfg["cohorts_db_path"], eco_field=cfg["eco_field"], tablename=cfg["tablename"],
  output_dir=cfg["tablename"], filter_eco_field=cfg["filter_eco_field"],
  filter_ecos=String.(get(cfg, "filter_ecos", String[])), filter_plots=NTuple{4,Int}[],
  min_trees=Int(g("min_trees", 100)), min_agb_frac=Float64(g("min_agb_frac", 0.05)),
  skip_disturbances=Bool(g("skip_disturbances", true)), spinup=false, val_frac=val_frac, split_rng=split_rng,
  single_ecoregion=Bool(g("single_ecoregion", false)), stratify_landuse=Bool(g("stratify_landuse", false)),
  RNG=P.RNGType(UInt64(Int(g("seed", 1)))))
nplots(df) = df === nothing ? Dict{Int,Int}() :
  Dict(r.eco_id => r.n for r in eachrow(combine(groupby(df, :eco_id),
    [:statecd, :unitcd, :countycd, :plot] => ((a, b, c, d) -> length(unique(collect(zip(a, b, c, d))))) => :n)))
trd = nplots(splots); vad = nplots(splots_val)
println("\n% train/val plot counts per eco×land-use (val_frac=$val_frac, split_seed=$(Int(g("split_seed",42))), stratified)")
println("\\begin{table}[ht]")
println("\\centering")
println("\\begin{tabular}{lrrr}")
println("\\toprule")
println("Ecoregion \$\\times\$ land use & Train & Val & Total \\\\")
println("\\midrule")
tt = sum(values(trd)); tv = sum(values(vad))
for eid in 1:length(eco_list)
  nt = get(trd, eid, 0); nv = get(vad, eid, 0)
  lab = replace(String(eco_list[eid]), "_" => "\\_")
  println("$(lab) & $(nt) & $(nv) & $(nt + nv) \\\\")
end
println("\\midrule")
println("\\textbf{Total} & \\textbf{$(tt)} & \\textbf{$(tv)} & \\textbf{$(tt + tv)} \\\\")
println("\\bottomrule")
println("\\end{tabular}")
println("\\caption{Train/validation plot counts stratified by ecoregion (EPA Level~III) and land use, for the FL5 study region. The split assigns $(round(Int, val_frac*100))\\% of plots within each (ecoregion~\$\\times\$~land use~\$\\times\$~species) stratum to validation (\\texttt{split\\_seed}=$(Int(g("split_seed",42)))).}")
println("\\label{tab:trainval-split}")
println("\\end{table}")
