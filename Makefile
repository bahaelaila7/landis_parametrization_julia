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
                                     
sysimage: prepare
	$(JULIA_CMD) --threads=$(THREADS) -e 'using PackageCompiler; create_sysimage(["Pan"]; sysimage_path="Pan.so",precompile_execution_file="precompile_exec.jl")'

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
