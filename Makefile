all: run

run:
	julia --project=. -e 'using BiomassSuccession;BiomassSuccession.main("")'

exec:
	julia --project=. -e 'using Pkg; Pkg.instantiate();using PackageCompiler;create_app(".", "build";filter_stdlibs=true,precompile_execution_file="src/entry.jl")'

sysimage:
	julia --project=. -e 'using Pkg; Pkg.instantiate();using PackageCompiler; create_sysimage(["BiomassSuccession"]; sysimage_path="BiomassSuccession.so",precompile_execution_file="precompile_exec.jl")'
