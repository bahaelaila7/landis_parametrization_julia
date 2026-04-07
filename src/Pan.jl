module Pan
include("PanCore.jl")
include("Plugins.jl")
include("Parametrization.jl")
#for file in filter(f -> endswith(f,"Plugin.jl"), readdir("src/plugins";join=false))
#    println("including plugin $(file)")
#    include(joinpath("plugins", file))
#end
#include("plugins/BiomassSuccessionPlugin.jl")
using .PanCore
using .Plugins: BaseSitePlugin, BiomassSuccessionPlugin
import .Parametrization as PU
import .Parametrization.BiomassSuccessionParametrization as BSP


import Random
import Distributions as Dists



const ActivePlugins = (BaseSitePlugin.BaseSite,BiomassSuccessionPlugin.BiomassSuccession)
const ActiveSoA = SiteSoA{Tuple{(ActivePlugins)...}}



function main(ARGS)
    #
    println("hello")
    rng = Random.Xoshiro(2123)
    #Random.seed!(42)
    #println(typeof(ActiveSoA))
    #println(
    n = 1000
    n_species = 14
    species_list = ["s_$(i)" for i in 1:n_species]
    eco_list = ["e_1"]
    eco_species_id::Array{Array{Int64}} = [vec(1:n_species) .|> Int64]
    bio_params = BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_id; rng = rng)
    param_dists =  BSP.make_biomass_param_dists(length(species_list), length(eco_list), eco_species_id)
    #println(param_dists)
    #println(bio_params)

    cohort_counts = rand(rng, Int32(1):Int32(8), n)
    species_count = fill(Int32(n_species), n)

    #nteractiveUtils.@code_typed ActiveSoA((cohort=cohort_counts,species = species_count))
    #soa = ActiveSoA((cohort=cohort_counts,species = species_count))
    counts_tuple = (cohort=cohort_counts,species = species_count)
    #@code_typed site = getsite(soa,1)
    #site = getsite(soa,n)
    #z =getproperty(site, :c_bio)
    #z .= 1.0f0
    #@code_llvm z =getproperty(site, :c_bio)
    
    @time for trial in 1:300
        soa = ActiveSoA(counts_tuple)
        println(trial)
        for i in 1:n
            site = getsite(soa, i)
            #println(site)
            cap = length(site.c_species)
            site.rng = Random.Xoshiro(rand(rng, Int64))
            site.eco_id = 1
            #site.eco_params = eco_params[1]
            site.live = rand(site.rng, UIntType(1):UIntType(cap))
            site.c_age .= 1.0f0
            site.c_species .= rand(site.rng, UIntType(1):UIntType(n_species), cap)
            site.c_bio .= 2.0f0
        end
        bio_params = PU.mutate_params(bio_params, param_dists; rng = rng)
        eco_params = BiomassSuccessionPlugin.generate_eco_params(bio_params)
        for t in 1:130
            #println("\ttimestep $(t)")
            simulate_timestep!(soa, t; ctx = (BiomassSuccession = (eco_params = eco_params,),))
        end
        #println(soa.refs.cohort[end]-1)
    end
    #println(soa)

end

end
