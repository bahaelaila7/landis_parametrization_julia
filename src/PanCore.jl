module PanCore

export AbstractPlugin, SiteSoA, SiteView, getsite, scalar_arrays, csr_fields, csr_arrays, process_site!, simulate_timestep!, FloatType, UIntType, Plugins
abstract type AbstractPlugin end

include("types.jl")

function scalar_arrays end
function csr_arrays end

scalar_arrays(::Type{<:AbstractPlugin}, ::Int) = NamedTuple()

csr_fields(::Type{<:AbstractPlugin})  = NamedTuple()
csr_arrays(::Type{<:AbstractPlugin}, ::NamedTuple)  = NamedTuple()

        


struct SiteSoA{Plugins <: Tuple, Refs <: NamedTuple, Scalars <: NamedTuple, Csr <: NamedTuple}
    n ::Int
    refs :: Refs
    scalar :: Scalars
    csr :: Csr
end

struct SiteView{S}
    soa :: S
    i :: Int
end

function process_plugin!(::SiteView, ::Type{<:AbstractPlugin}, ::Int)  end

@inline function build_refs(counts::Vector{Int32})::Vector{Int32}
    n       = length(counts)
    refs    = Vector{Int32}(undef, n + 1)
    refs[1] = Int32(1)
    for i in 1:n
        refs[i+1] = refs[i] + counts[i]
    end
    return refs
end

function _build_refs_expr(plugin_types)
    seen_keys = Set{Symbol}() # keys are the counts like cohort counts per site
    ref_pairs_keys = Symbol[]
    ref_pairs_vals = Expr[]
    for P in plugin_types
        for (field, key) in pairs(csr_fields(P))
            if key ∉ seen_keys
                push!(seen_keys, key)
                push!(ref_pairs_keys, key)
                push!(ref_pairs_vals, :(build_refs(counts[$(QuoteNode(key))])))
            end
        end
    end
    isempty(ref_pairs_keys) && return :(NamedTuple())
    :(NamedTuple{$(Tuple(ref_pairs_keys))}(tuple($(ref_pairs_vals...))))
    #refs :: NamedTuple{(:cohort, :species), Tuple{Vector{Int32}, Vector{Int32}}}
end

function _build_refs_expr2(plugin_types)
    ref_pairs_keys = Symbol[]
    ref_pairs_vals = Expr[]
    for P in plugin_types
        for (field, key) in pairs(csr_fields(P))
            push!(ref_pairs_keys, field)
            push!(ref_pairs_vals, :(build_refs(counts[$(QuoteNode(key))])))
        end
    end
    isempty(ref_pairs_keys) && return :(NamedTuple())
    :(NamedTuple{$(Tuple(ref_pairs_keys))}(tuple($(ref_pairs_vals...))))
end

function _build_scalar_expr(plugin_types)
    foldl(
        (a,b) -> :(merge($a,$b)),
        [:(scalar_arrays($P, n)) for P in plugin_types];
        init = :(NamedTuple())
    )
end

function _field_to_key_map(plugin_types)
    map = Dict{Symbol, Symbol}()
    for P in plugin_types
        for (field, key) in pairs(csr_fields(P))
            map[field] = key
        end
    end
    return map
end

function _build_getproperty_expr(scalar_names, csr_names, field_to_key)
    expr = :(error("field ", f, " not found in SiteView"))

    for fn in reverse(csr_names)
        key = field_to_key[fn]   # compile-time lookup
        expr = quote
            if f === $(QuoteNode(fn))
                # key = :cohort  --> soa.refs.cohort[i] which is a vector saying how many cohorts for site[i]
                refs      = getfield(getfield(soa, :refs), $(QuoteNode(key)))
                csr_field = getfield(getfield(soa, :csr),  $(QuoteNode(fn)))
                lo        = Int(refs[i])
                hi        = Int(refs[i + 1]) - 1
                return @view csr_field[lo:hi]
            else
                $expr
            end
        end
    end

    for fn in reverse(scalar_names)
        expr = quote
            if f === $(QuoteNode(fn))
                return getfield(getfield(soa, :scalar), $(QuoteNode(fn)))[i]
            else
                $expr
            end
        end
    end

    expr
end

@generated function Base.getproperty(sv::SiteView{S}, f::Symbol) where {S}
    scalar_type    = fieldtype(S, :scalar)
    csr_type      = fieldtype(S, :csr)
    scalar_names   = fieldnames(scalar_type)
    csr_names     = fieldnames(csr_type)

    # S = {Plugins, Refs, Scalars, Csr}
    Plugins       = S.parameters[1] # Tuple{BaseSite, BiomassSuccessionPlugin, ...}
    plugin_types  = Plugins.parameters # (BaseSite, BiomassSuccessionPlugin, ...)
    field_to_key  = _field_to_key_map(plugin_types) # Dict(c_bio => :cohort, sp_mature => :species, ...)

    expr = _build_getproperty_expr(scalar_names, csr_names, field_to_key)
    quote
        f === :soa && return getfield(sv, :soa)
        f === :i   && return getfield(sv, :i)
        soa = getfield(sv, :soa)
        i   = getfield(sv, :i)
        $expr
    end
end
#function _build_csr_expr(plugin_types)
#    foldl(
#        (a,b) -> :(merge($a,$b)),
#        [:(csr_arrays($P, nnz)) for P in plugin_types];
#        init = :(NamedTuple())
#    )
#end
#@generated function SiteSoA{Plugins}(count::C) where {Plugins, C<: NamedTuple}
#    # some meta-programming to reduce runtime pointer jumping and 
#    # utilize more cache hits
#    # this function constructs the effective site struct after designating which plugins to use
#    # basically merging the fields into one struct before compilation
#    plugin_types = Plugins.parameters #fieldtypes(Plugins)
#    refs_expr = _build_refs_expr(plugin_types)
#    scalar_expr = _build_scalar_expr(plugin_types)
#    csr_expr = _build_csr_expr(plugin_types)
#    quote
#        n = length(first(counts))
#        nnz = map(v -> Int(sum(v)), counts)
#        refs = $refs_expr
#        scalar = $scalar_expr
#        csr = $csr_expr
#        SiteSoA{Plugins}(n, refs, scalar, csr)
#
#    end
#end
function SiteSoA{Plugins}(counts::C) where {Plugins, C <: NamedTuple}
    plugin_types = Plugins.parameters
    n   = length(first(counts))
    nnz = map(v -> Int(sum(v)), counts)

    #refs  = mapreduce(merge, plugin_types; init=NamedTuple()) do P
    #    nt = csr_fields(P)
    #    NamedTuple{Tuple(keys(nt))}(map(k -> build_refs(counts[k]), values(nt)))
    #end
    # csr_fields(::Type{BiomassSuccessionPlugin}) = (c_age = :cohort, sp_mature = :species)
    # ie the counts of c_age per site is going to be passed by refs.cohort while 
    # for sp_mature is refs.species
    # refs::NamedTuple{Tuple{:cohort, :species}} = (cohort = build_refs(counts.cohort), species = build_refs(counts.species))
    # The map will be built later when getproperty gets invoked
    refs =  begin 
        seen_keys = Symbol[]
        for P in plugin_types
            for (field, key) in pairs(csr_fields(P))
                key ∉ seen_keys && push!(seen_keys, key)
            end
        end
        NamedTuple{Tuple(seen_keys)}(map(key -> build_refs(counts[key]), Tuple(seen_keys)))
    end

    scalar = mapreduce(P -> scalar_arrays(P, n),   merge, plugin_types; init=NamedTuple())
    csr   = mapreduce(P -> csr_arrays(P, nnz),   merge, plugin_types; init=NamedTuple())

    SiteSoA{Plugins, typeof(refs), typeof(scalar), typeof(csr)}(n, refs, scalar, csr)
end

#@generated function SiteSoAA{Plugins}(count::C) where {Plugins, C<: NamedTuple}
#    # some meta-programming to reduce runtime pointer jumping and 
#    # utilize more cache hits
#    # this function constructs the effective site struct after designating which plugins to use
#    # basically merging the fields into one struct before compilation
#    plugin_types = Plugins.parameters #fieldtypes(Plugins)
#
#    scalar_expr = foldl(
#        (a,b) -> :(merge($a, $b)),
#        [:(scalar_arrays($P, n)) for P in plugin_types]
#    )
#
#    # this is for attributes that have different counts per site
#    # think sp_mature has its counts based on the number of species at the site
#    # while c_bio is basically by number of cohorts
#    # so each csr field has a count_key (:species, :cohort,...) with it
#    ref_pairs = Expr[]
#    #@show Plugins
#    #@show typeof(Plugins)
#    #@show Plugins.parameters[1].parameters
#    for P in plugin_types
#        #@show P
#        #@show typeof(P)
#        #@show csr_fields(P)
#        for (field, count_key) in pairs(csr_fields(P))
#            push!(ref_pairs, :($(QuoteNode(field)) => build_refs(counts[$(QuoteNode(count_key))])))
#        end
#    end
#
#    # making all into one named tuple
#    refs_expr = :(NamedTuple{$(Tuple(first.(ref_pairs)))}(tuple(last.(ref_pairs)...)))
#    
#    csr_expr = foldl(
#        (a,b) -> :(merge($a, $b)),
#        [:(csr_arrays($P, nnz)) for P in plugin_types],
#    )
#
#    # ok now assembling
#    quote
#
#        n = length(first(counts))
#        nnz = map(v -> Int(sum(v)), counts)
#        refs = $refs_expr
#        scalar = $scalar_expr
#        csr = $csr_expr
#        SiteSoA{Plugins}(n, refs, scalar, csr)
#    end
#
#end

function _build_process_calls(plugin_types)
    [:(process_plugin!(sv, $P, t)) for P in plugin_types]
end

function _build_getproperty_expr(scalar_names, csr_names)
    expr = :(error("field ", f, " not found in SiteView"))
    for fn in reverse(csr_names)
        expr = quote
            if f === $(QuoteNode(fn))
                refs      = getfield(getfield(soa, :refs), $(QuoteNode(fn)))
                csr_field = getfield(getfield(soa, :csr),  $(QuoteNode(fn)))
                lo        = Int(refs[i])
                hi        = Int(refs[i + 1]) - 1
                return @view csr_field[lo:hi]
            else
                $expr
            end
        end
    end
    for fn in reverse(scalar_names)
        expr = quote
            if f === $(QuoteNode(fn))
                return getfield(getfield(soa, :scalar), $(QuoteNode(fn)))[i]
            else
                $expr
            end
        end
    end
    expr
end
function _build_setproperty_expr(scalar_names, csr_names)
    expr = :(error("field ", f, " not found in SiteView"))

    for fn in reverse(csr_names)
        expr = quote
            if f === $(QuoteNode(fn))
                error("use .= on CSR field $($(QuoteNode(fn)))")
            else
                $expr
            end
        end
    end

    for fn in reverse(scalar_names)
        expr = quote
            if f === $(QuoteNode(fn))
                getfield(getfield(soa, :scalar), $(QuoteNode(fn)))[i] = v
                return v
            else
                $expr
            end
        end
    end

    expr
end

#@generated function Base.getproperty(sv::SiteView{S}, f::Symbol) where {S}
#    scalar_types = fieldtype(S, :scalar)
#    csr_types = fieldtype(S, :csr)
#    scalar_names = fieldnames(scalar_types)
#    csr_names   = fieldnames(csr_types)
#    expr        = _build_getproperty_expr(scalar_names, csr_names)
#    quote
#        f === :soa && return getfield(sv, :soa)
#        f === :i   && return getfield(sv, :i)
#        soa = getfield(sv, :soa)
#        i   = getfield(sv, :i)
#        $expr
#    end
#end
@generated function Base.setproperty!(sv::SiteView{S}, f::Symbol, v) where {S}
    scalar_types = fieldtype(S, :scalar)
    csr_types = fieldtype(S, :csr)
    scalar_names = fieldnames(scalar_types)
    csr_names   = fieldnames(csr_types)
    expr        = _build_setproperty_expr(scalar_names, csr_names)
    quote
        f === :soa && error("cannot set soa")
        f === :i   && error("cannot set i")
        soa = getfield(sv, :soa)
        i   = getfield(sv, :i)
        $expr
    end
end

@generated function process_site!(sv::SiteView{S}, t::Int) where {S}
    plugin_types = S.parameters[1].parameters
    calls        = _build_process_calls(plugin_types)
    quote $(calls...) end
end

@inline getsite(soa::SiteSoA, i::Int) = SiteView(soa, i)

#@generated function Base.getproperty(sv::SiteView{S}, f::Symbol) where {S}
#        # redefining getproperty based on plugins before compilation
#        scalar_names = fieldnames(fieldtype(S, :scalar))
#        csr_names = fieldnames(fieldtype(S, :csr))
#        
#        # building a nested if/else chain to find the field within the struct
#        expr = :(error("field ", f, " not found in SiteView"))
#        
#        for fn in reverse(csr_names)
#            expr = quote
#                    if f === $(QuoteNode(fn))
#                        soa = getfield(sv, :soa)
#                        i = getfield(sv, :i)
#                        refs = getfield(getfield(soa, :refs), $(QuoteNode(fn)))
#                        csr = getfield(getfield(soa, :csr), $(QuoteNode(fn)))
#                        lo = Int(refs[i])
#                        hi = Int(refs[i+1]) - 1
#                        return @view csr[lo:hi]
#                    else
#                        $expr
#                    end
#            end
#        end
#
#        for fn in reverse(scalar_names)
#            expr = quote
#                    if f === $(QuoteNode(fn))
#                        soa = getfield(sv, :soa)
#                        i = getfield(sv, :i)
#                        return getfield(getfield(soa, :scalar), $(QuoteNode(fn)))[i]
#                    else
#                        $expr
#                    end
#            end
#        end
#
#        quote
#            f === :soa && return getfield(sv, :soa)
#            f === :i && return getfield(sv, :i)
#            $expr
#        end
#end
#
#
#@generated function process_site!(sv::SiteView{S}, t::Int) where {S}
#    Plugins = fieldtype(S, :Plugins)
#    calls   = [:(process_plugin!(sv, $P, t)) for P in fieldtypes(Plugins)]
#    quote $(calls...) end
#end

function simulate_timestep!(soa::SiteSoA{P}, t::Int) where {P}
    Threads.@threads :static for i in 1:soa.n
        process_site!(getsite(soa, i), t)
    end
end

end
