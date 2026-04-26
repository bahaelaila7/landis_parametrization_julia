THREADS ?= auto
JULIA_CMD ?= ./julia_gdal.sh --project=.
JULIA_RUN ?= 'using Pan;Pan.main()'

all: run

prepare:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate()'

update:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.update()'

run1:
	$(JULIA_CMD) --threads=1 -e $(JULIA_RUN)

run:
	$(JULIA_CMD) --threads=$(THREADS) -e $(JULIA_RUN)

debug1:
	JULIA_DEBUG=Pan $(JULIA_CMD) --threads=1 -e $(JULIA_RUN)

debug:
	JULIA_DEBUG=Pan $(JULIA_CMD) --threads=$(THREADS) -e $(JULIA_RUN)
                                     
exec:                                
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate();using PackageCompiler;create_app(".", "build";filter_stdlibs=true,precompile_execution_file="src/entry.jl")'
                                     
sysimage:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate();using PackageCompiler; create_sysimage(["Pan"]; sysimage_path="Pan.so",precompile_execution_file="precompile_exec.jl")'

spatial:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pan; Pan.spatial_main()'

debug-spatial:
	JULIA_DEBUG=Pan $(JULIA_CMD) --threads=$(THREADS) -e 'using Pan; Pan.spatial_main()'
