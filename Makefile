THREADS ?= auto
RUN ?= run.yml
JULIA_CMD ?= ./julia_gdal.sh --project=.
JULIA_RUN ?= 'using Pan;Pan.main()'
JULIA_RUN_YAML ?= 'using Pan;Pan.run_from_yaml("$(RUN)")'

all: run

prepare:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.resolve(); Pkg.update(); Pkg.instantiate()'

run1: prepare
	$(JULIA_CMD) --threads=1 -e $(JULIA_RUN)

run: prepare
	$(JULIA_CMD) --threads=$(THREADS) -e $(JULIA_RUN)

debug1: prepare
	JULIA_DEBUG=Pan $(JULIA_CMD) --threads=1 -e $(JULIA_RUN)

debug: prepare
	JULIA_DEBUG=Pan $(JULIA_CMD) --threads=$(THREADS) -e $(JULIA_RUN)
                                     
exec: prepare
	$(JULIA_CMD) --threads=$(THREADS) -e 'using PackageCompiler;create_app(".", "build";filter_stdlibs=true,precompile_execution_file="src/entry.jl")'
                                     
# FULL sysimage (Pan + deps). One compile pass: instantiate WITHOUT auto-precompile (so the depot isn't
# compiled first and then AGAIN into the image), then create_sysimage. No `prepare` dep → no Pkg.update churn.
# Assumes deps are downloaded (run `make prepare` once after cloning). Rebuild after any src/ change.
sysimage:
	JULIA_PKG_PRECOMPILE_AUTO=0 $(JULIA_CMD) -e 'using Pkg; Pkg.instantiate()'
	JULIA_NUM_THREADS=$(THREADS) $(JULIA_CMD) --threads=$(THREADS) -e 'using PackageCompiler; create_sysimage(["Pan"]; sysimage_path="Pan.so",precompile_execution_file="precompile_exec.jl")'

# INCREMENTAL sysimage: bake only the (stable, heavy) DEPENDENCIES — not Pan. Pan then recompiles on top via
# Julia's per-package cache, so editing src/ recompiles only Pan (seconds), not the whole image. Rebuild this
# one only when the Manifest changes. The wrapper uses Pan.so if present, else Pan_deps.so.
sysimage-deps:
	JULIA_PKG_PRECOMPILE_AUTO=0 $(JULIA_CMD) -e 'using Pkg; Pkg.instantiate()'
	JULIA_NUM_THREADS=$(THREADS) $(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg, PackageCompiler; create_sysimage(collect(keys(Pkg.project().dependencies)); sysimage_path="Pan_deps.so",precompile_execution_file="precompile_exec.jl")'

# LEAN sysimage for TRAINING (low build memory). Bakes only the fitting-critical deps — NOT the plotting stacks
# (CairoMakie/Plots/PythonPlot), UMAP, ImageFiltering or GDAL — which is why the full/deps builds OOM on a 32 GB
# node. Pan + Makie load from the depot cache on top (training never plots, so no Makie JIT). Produces Pan_deps.so
# (wrapper auto-detects it). Override the list with FIT_DEPS="..." if you add fitting deps.
FIT_DEPS ?= DuckDB DataFrames Distributions JLD2 CSV Sobol StatsBase Statistics YAML Setfield JSON3 Term DataStructures
sysimage-fit:
	JULIA_PKG_PRECOMPILE_AUTO=0 $(JULIA_CMD) -e 'using Pkg; Pkg.instantiate()'
	JULIA_NUM_THREADS=$(THREADS) $(JULIA_CMD) --threads=$(THREADS) -e 'using PackageCompiler; create_sysimage(split("$(FIT_DEPS)"); sysimage_path="Pan_deps.so",precompile_execution_file="precompile_exec.jl")'

spatial: prepare
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pan; Pan.spatial_main()'

debug-spatial: prepare
	JULIA_DEBUG=Pan $(JULIA_CMD) --threads=$(THREADS) -e 'using Pan; Pan.spatial_main()'

landis:
	$(JULIA_CMD) --threads=1 -e 'using Pan; Pan.landis_main()'

debug-landis: prepare
	JULIA_DEBUG=Pan $(JULIA_CMD) --threads=$(THREADS) -e 'using Pan; Pan.landis_main()'

yaml:
	$(JULIA_CMD) --threads=$(THREADS) -e $(JULIA_RUN_YAML)

yaml1:
	$(JULIA_CMD) --check-bounds=yes --threads=1 -e $(JULIA_RUN_YAML)

debug-yaml:
	JULIA_DEBUG=Pan $(JULIA_CMD) --threads=$(THREADS) -e $(JULIA_RUN_YAML)

export-landis: prepare
	$(JULIA_CMD) --threads=1 -e 'using Pan; Pan.export_landis_main()'
