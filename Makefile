THREADS ?= auto

all: run

prepare:
	julia --project=. --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate()'

run1:
	julia --project=. --threads=1 -e 'using BiomassSuccession;BiomassSuccession.main("")'

run:
	julia --project=. --threads=$(THREADS) -e 'using BiomassSuccession;BiomassSuccession.main("")'
                                     
exec:                                
	julia --project=. --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate();using PackageCompiler;create_app(".", "build";filter_stdlibs=true,precompile_execution_file="src/entry.jl")'
                                     
sysimage:                            
	julia --project=. --threads=$(THREADS) -e 'using Pkg; Pkg.instantiate();using PackageCompiler; create_sysimage(["BiomassSuccession"]; sysimage_path="BiomassSuccession.so",precompile_execution_file="precompile_exec.jl")'
