THREADS ?= auto
JULIA_CMD ?= ./julia_gdal.sh --project=.

all: run

prepare:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate()'

update:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.update()'

run1:
	$(JULIA_CMD) --threads=1 -e 'using Pan;Pan.main("")'

run:
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pan;Pan.main("")'
                                     
exec:                                
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate();using PackageCompiler;create_app(".", "build";filter_stdlibs=true,precompile_execution_file="src/entry.jl")'
                                     
sysimage:                            
	$(JULIA_CMD) --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate();using PackageCompiler; create_sysimage(["Pan"]; sysimage_path="Pan.so",precompile_execution_file="precompile_exec.jl")'
