module ES
import Random
import StatsBase


export main;

# hyperparams: populatiopn, selection strategy, survival strategy, recombination type, 
# TODO: 1-generate population of params 
# 2-fit each population (ie run simulation and )
# 

#mutate
Base.@kwdef struct Candidate
    var1::Vector{Float64}
    var2::Vector{Float64}
    var3::Vector{Vector{Float64}}
    var4::Vector{Vector{Float64}}
    var5::Vector{Vector{Float64}}
end
Base.@kwdef struct Sigma
    var1::Vector{Float64}
    var2::Vector{Float64}
    var3::Vector{Vector{Float64}}
    var4::Vector{Vector{Float64}}
    var5::Vector{Vector{Float64}}
end
struct Instance
    vars::Candidate
    sigma::Sigma
end
function new_instance(len::Int;rng = Random.AbstractRNG)::Instance
    Instance(
    Candidate(
        var1 = randn(rng,len),
        var2 = randn(rng,len),
        var3 = [randn(rng, len) for _ in 1:1],
        var4 = [randn(rng, len) for _ in 1:1],
        var5 = [randn(rng, len) for _ in 1:1],
    ),
    Sigma(
        var1 = abs.(randn(rng,len)),
        var2 = abs.(randn(rng,len)),
        var3 = [abs.(randn(rng, len)) for _ in 1:1],
        var4 = [abs.(randn(rng, len)) for _ in 1:1],
        var5 = [abs.(randn(rng, len)) for _ in 1:1],
    ))
end
struct Variable
    name::Symbol
    dims::Int
    min::Float64
    max::Float64

end


function fit_func(instance::Instance; show = false)
    xs = Vector{Float64}()
    for varname in fieldnames(typeof(instance.vars))
        var = getproperty(instance.vars, varname)
        #println("$(varname)")
        #println("$(var)")
        #println("$(eachindex(var))")
        for dim1 in eachindex(var)
            #println("$(dim1)")
            #println("$(typeof(dim1))")
            #println("$(var[dim1])")
            #println("$(size(var[dim1]))")
            #println("$(length(size(var[dim1])))")
            len = length(size(var[dim1])) 

            if len > 1
                for dim2 in eachindex(var[dim1][dim2])
                    #println("$(var[dim1][dim2])")
                    push!(xs, var[dim1][dim2])
                end
            elseif len == 1
                append!(xs,var[dim1])
            else
                push!(xs, var[dim1])
            end
        end
    end
    if show
        println(xs)
        println(size(xs))
    end
    return ackley(xs)
end
               
           
       
       
       
       
function mutate!(instances::Array{Instance};rng = Random.AbstractRNG, TAU_G, TAU, Pm)
    log_purturb = randn(rng,Float64) 
    vars, sigmas = instance
    sigmas .*= rand(rng, length(sigmas))
    vars .+= sigmas
end
@inline function mutate(val, sigma, var_prop, perturb_g, TAU, SIGMA_EPS, rng)
            new_sigma = max(sigma*exp(perturb_g + TAU * randn(rng,Float64)), SIGMA_EPS)
            new_val = clamp(val + new_sigma* randn(rng,Float64),var_prop.min, var_prop.max)
            return new_val, new_sigma
end
@inline function recombine(p1, p2, p1_sigma, p2_sigma, rng)
                            a = rand(rng, Float64)
                            val = a * (p1-p2) + p2
                            sigma = a * (p1_sigma-p2_sigma) + p2_sigma
                            return val, sigma
end
function generate_offspring(pop; lambda=10, SIGMA_EPS = 1e-35, Pc=1.0,Pm=1.0, TAU_G, TAU, rng=Random.AbstractRNG)
    cbp = [Variable(:var1,1,-30.0,+30.0),Variable(:var2,1,-30.0,+30.0),Variable(:var3,2,-30.0,+30.0),Variable(:var4,2,-30.0,+30.0),Variable(:var5,2,-30.0,+30.0)]
    cbp_dict = Dict(v.name => v for v in cbp)
    #for each dimension, choose two parents
    offspring = Array{Instance}(undef,lambda)
    should_recombine = Pc == 1.0 || (Pc > 0.0 && rand(rng, Float64) <= Pc)
    should_mutate = Pm == 1.0 || (Pm > 0.0 && rand(rng,Float64) <= Pm)
    for child_id = 1:lambda
        purturb_g = TAU_G*randn(rng, Float64)
        child  = deepcopy(rand(rng, pop)) #Array{Variable}(undef,params_per_instance)
        for varname in fieldnames(typeof(child.vars))
            var = getproperty(child.vars,varname)
            var_sigma = getproperty(child.sigma,varname)
            var_prop = get(cbp_dict,varname,nothing)
            for dim1 in eachindex(var)
                if length(size(var[dim1])) >  0
                    for dim2 in eachindex(var[dim1])
                        if should_recombine
                            parents = StatsBase.sample(pop, 2; replace=false)
                            parent1 = parents[1]
                            parent2 = parents[2]
                            p1 = getproperty(parent1.vars, varname)[dim1][dim2]
                            p2 = getproperty(parent2.vars, varname)[dim1][dim2]
                            p1_sigma = getproperty(parent1.sigma, varname)[dim1][dim2]
                            p2_sigma = getproperty(parent2.sigma, varname)[dim1][dim2]
                            var[dim1][dim2], var_sigma[dim1][dim2] = recombine(p1,p2,p1_sigma,p2_sigma,rng)
                        end
                        if should_mutate
                            var[dim1][dim2], var_sigma[dim1][dim2] = mutate(var[dim1][dim2], var_sigma[dim1][dim2], var_prop, purturb_g, TAU, SIGMA_EPS, rng)
                        end
                    end
                else
                        if  should_recombine
                            parents = StatsBase.sample(pop, 2; replace=false)
                            parent1 = parents[1]
                            parent2 = parents[2]
                            p1 = getproperty(parent1.vars, varname)[dim1]
                            p2 = getproperty(parent2.vars, varname)[dim1]
                            p1_sigma = getproperty(parent1.sigma, varname)[dim1]
                            p2_sigma = getproperty(parent2.sigma, varname)[dim1]
                            var[dim1], var_sigma[dim1] = recombine(p1,p2,p1_sigma,p2_sigma,rng)
                        end
                        if should_mutate
                            var[dim1], var_sigma[dim1] = mutate(var[dim1], var_sigma[dim1], var_prop, purturb_g, TAU, SIGMA_EPS, rng)
                        end
                end
            end
        end
        offspring[child_id] = child
    end
    return offspring
end
function select_parents(pop; parents::Int) 

end
function select_next_pop(parents::Array{Instance}, offspring::Array{Instance})

end
@inline function ackley(xs)
    Float64(-20.0 * exp(-0.2 * sqrt(sum(xs.*xs)/length(xs))) - exp(sum(cos.(2*pi*xs))/length(xs)) + ℯ + 20.0)
end

function track_best(best, fit_best, pop,fit)::Tuple{Instance, Float64, Bool}
    changed = false 
    max_idx = argmin(fit)
    cur, fit_cur = pop[max_idx], fit[max_idx]
    if fit_cur < fit_best
        changed = true
        best, fit_best = cur, fit_cur
    end
    return best, fit_best, changed
end

function make_evolutionary_strategy(;POP_SIZE::Int, lambda::Int, SIGMA_EPS=1e-35, instance_size::Int=10, rng= Random.AbstractRNG)
    instance_dim = (1 + 1 + 1 + 1 +1)*instance_size
    println(instance_dim)
    TAU_G = 1/sqrt(0.2*instance_dim)
    TAU = 1/sqrt(0.2*sqrt(instance_dim))
    println("making population")
    pop = [new_instance(instance_size; rng=rng) for _ in 1:POP_SIZE]
    println(" population")
    fit = fit_func.(pop)
    #return
    best, fit_best,changed = track_best(nothing, Inf, pop,fit)
    fit_func(best;show = true)
    println("Best: $(fit_best)")
    i = 0
    while true
        i+=1
        offspring = generate_offspring(pop; rng = rng, lambda = lambda, SIGMA_EPS = SIGMA_EPS, TAU_G=TAU_G, TAU=TAU)
        fit_offspring = fit_func.(offspring)
        best, fit_best, changed = track_best(best, fit_best, offspring, fit_offspring)
        if changed
            println("Best@$(i): $(fit_best)")
        end
        # mu plus lambda
        new_pop = vcat(pop, offspring)
        new_fit = vcat(fit,fit_offspring)
        new_pop_ids = sortperm(new_fit)[1:POP_SIZE]
        pop = new_pop[new_pop_ids]
        fit = new_fit[new_pop_ids]

    end
end

#recombination
#linear recombination

function main(ARGS)
    #can =  Instance(Candidate(zeros(Float64, 10),zeros(Float32, 10),[zeros(Float32, 10) for _ in 1:10],[zeros(Float32, 10) for _ in 1:10] ),Sigma(ones(Float32, 10),ones(Float32, 10),[ones(Float32, 10) for _ in 1:10],[ones(Float32, 10) for _ in 1:10] ))
    #println("$(can)")
    #println(fit_func(can))
    println("Ackly")
    println(ackley(randn(15)))
    println("done")
    make_evolutionary_strategy(;POP_SIZE=12, lambda=12*7, instance_size = 16, rng = Random.Xoshiro(42))

end

end

ES.main("")