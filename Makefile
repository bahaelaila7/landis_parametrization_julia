THREADS ?= auto
JULIA_CMD ?= ./julia_gdal.sh --project=.

all: run

prepare:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate()'

update:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.update()'

run1:
	$(JULIA_CMD) --threads=1 -e 'using BiomassSuccession;BiomassSuccession.main("")'

run:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using BiomassSuccession;BiomassSuccession.main("")'
                                     
exec:                                
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate();using PackageCompiler;create_app(".", "build";filter_stdlibs=true,precompile_execution_file="src/entry.jl")'
                                     
sysimage:                            
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate();using PackageCompiler; create_sysimage(["BiomassSuccession"]; sysimage_path="BiomassSuccession.so",precompile_execution_file="precompile_exec.jl")'
