module BaseSite
    export BaseSite, scalar_arrays
    using ..PanCore
    struct BaseSite <: AbstractPlugin end
    import Random

    PanCore.scalar_arrays(::Type{BaseSite}, n::Int) = 
        (
            active = Vector{Bool}(undef, n),
            rng = Vector{Random.AbstractRNG}(undef, n),
            mapcode = Vector{Int}(undef, n),
            ecocode = Vector{Int}(undef, n),
            eco_id = Vector{Int}(undef, n),
            ref_cn = Vector{Int}(undef, n),
            cap = Vector{Int}(undef, n),
            old = Vector{Int}(undef, n),
            live = Vector{Int}(undef, n),
        )
end
