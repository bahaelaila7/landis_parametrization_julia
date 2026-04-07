module Pan
include("PanCore.jl")
include("Plugins.jl")
#for file in filter(f -> endswith(f,"Plugin.jl"), readdir("src/plugins";join=false))
#    println("including plugin $(file)")
#    include(joinpath("plugins", file))
#end
#include("plugins/BiomassSuccessionPlugin.jl")
using .PanCore
using .Plugins
using .Plugins.BaseSite
import .Plugins.BiomassSuccessionPlugin


import Random

const ActivePlugins = (BaseSite,BiomassSuccessionPlugin.BiomassSuccessionPlugin)
const ActiveSoA = SiteSoA{Tuple{(ActivePlugins)...}}



function main(ARGS)
    println("hello")
    rng = Random.Xoshiro(123)
    #println(typeof(ActiveSoA))
    #println(
    n = 1000
    n_species = 14
    species_list = ["s_$(i)" for i in 1:n_species]
    eco_list = ["e_1"]
    eco_species_id::Array{Array{Int64}} = [vec(1:n_species) .|> Int64]
    bio_params = BiomassSuccessionPlugin.generate_biomass_params(species_list, eco_list, eco_species_id; rng = rng)

    cohort_counts = rand(rng, Int32(1):Int32(8), n)
    species_count = fill(Int32(n_species), n)

    #nteractiveUtils.@code_typed ActiveSoA((cohort=cohort_counts,species = species_count))
    #soa = ActiveSoA((cohort=cohort_counts,species = species_count))
    counts_tuple = (cohort=cohort_counts,species = species_count)
    soa = ActiveSoA(counts_tuple)
    #@code_typed site = getsite(soa,1)
    #site = getsite(soa,n)
    #z =getproperty(site, :c_bio)
    #z .= 1.0f0
    #@code_llvm z =getproperty(site, :c_bio)
    eco_params = BiomassSuccessionPlugin.generate_eco_params(bio_params)
    
    for i in 1:n
        site = getsite(soa, i)
        #println(site)
        site.cap = length(site.c_species)
        site.rng = Random.Xoshiro(rand(Int64))
        site.eco_params = eco_params[1]
        site.live = rand(site.rng, UIntType(1):UIntType(site.cap))
        site.c_age .= 1.0f0
        site.c_species .= rand(rng, UIntType(1):UIntType(n_species), length(site.c_species))
        site.c_bio .= 2.0f0
    end
    @time for t in 1:200
        simulate_timestep!(soa, t)

    end
    #println(soa)

end

end
