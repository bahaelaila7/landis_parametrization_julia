module PanCore

export AbstractPlugin, SiteSoA, SiteView, getsite, scalar_arrays, csr_fields, csr_arrays, process_site!, simulate_timestep!, FloatType, UIntType, Plugins
abstract type AbstractPlugin end

include("types.jl")

function scalar_arrays end
function csr_arrays end

scalar_arrays(::Type{<:AbstractPlugin}, ::Int) = NamedTuple()

csr_fields(::Type{<:AbstractPlugin})  = NamedTuple()
csr_arrays(::Type{<:AbstractPlugin}, ::NamedTuple)  = NamedTuple()

        


mutable struct SiteSoA{Plugins <: Tuple, Refs <: NamedTuple, Scalars <: NamedTuple, Csr <: NamedTuple}
    n ::Int
    refs :: Refs
    scalar :: Scalars
    csr :: Csr
end

const AnySoA{P} = SiteSoA{P, <: NamedTuple, <: NamedTuple, <: NamedTuple}

struct SiteView{S}
    soa :: S
    i :: Int
end

function process_plugin!(::AnySoA{P}, ::Type{<:AbstractPlugin}, ::Int; ctx::C) where {P, C<:NamedTuple}  end

function build_refs!(refs::Vector{Int32}, counts::Vector{Int32})
    refs[1] = Int32(1)
    for i in 1:length(counts)
        refs[i+1] = refs[i] + counts[i]
    end
    return refs
end

@inline function build_refs(counts::Vector{Int32})::Vector{Int32}
    build_refs!(Vector{Int32}(undef, length(counts) + 1), counts)
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

@generated function _csr_field_to_key(::Val{P}) where {P}
    plugin_types = P.parameters
    field_names  = Symbol[]
    key_values   = Symbol[]
    for PT in plugin_types
        for (field, key) in pairs(csr_fields(PT))
            push!(field_names, field)
            push!(key_values,  key)
        end
    end
    nt_type = NamedTuple{Tuple(field_names), NTuple{length(field_names), Symbol}}
    :($nt_type($(Tuple(key_values))))
end

@kwdef struct RegrowOptions
    grow_only     :: Bool = false   # default is compacting
    multiple_of_2 :: Bool = true   # default is exact
end

function effective_counts(new_counts::Vector{Int32},
                          old_refs::Vector{Int32},
                          opts::RegrowOptions)
    n = length(new_counts)
    ec = Vector{Int32}(undef, n)
    for i in 1:n
        old_count = old_refs[i+1] - old_refs[i]
        c = opts.grow_only ? max(new_counts[i], old_count) : new_counts[i]
        c = max(c, 1)
        if opts.multiple_of_2
            c = Int32(2^ceil(Int, log2(max(c, 1))))
        end
        @assert c > 0
        ec[i] = c
    end
    return ec
end

function readjust_soa!(soa::SiteSoA{P,Refs,Scalars,Csr},
                       new_counts::NamedTuple,
                       options::NamedTuple = NamedTuple()) where {P,Refs,Scalars,Csr}

    field_to_key = _csr_field_to_key(Val(P)) # compile-time, need to lift types
    old_refs     = getfield(soa, :refs)
    old_csr      = getfield(soa, :csr)
    n            = soa.n

    for key in keys(new_counts)
        opts     = hasfield(typeof(options), key) ? getfield(options, key) : RegrowOptions()
        ec       = effective_counts(new_counts[key], getfield(old_refs, key), opts)
        cur_refs = getfield(old_refs, key)
        new_refs = build_refs(ec)
        new_nnz  = Int(new_refs[end]) - 1
        #println("new counts: $(ec)")
        #println("cur_refs: $(cur_refs)")
        #println("new_refs: $(new_refs)")

        all_growing = true
        all_shrinking = true
        for i in 1:n
            all_growing &= ec[i] >= (cur_refs[i+1] - cur_refs[i])
            all_shrinking &= ec[i] <= (cur_refs[i+1] - cur_refs[i])
        end

        #check if no mix of growing and shrinking
        #if mix, we can only allocat and copy, not resize and shift
        #all_growing = all(new_refs[i] >= cur_refs[i] for i in 1:n+1)
        #all_shrinking = !all_growing && all(new_refs[i] <= cur_refs[i] for i in 1:n+1)

        updated_csr = old_csr

        for csr_array_name in keys(old_csr)
            getfield(field_to_key, csr_array_name) === key || continue
            #println(csr_array_name)
            arr = getfield(old_csr, csr_array_name)

            if all_growing
                #println("all_growing")
                #println(length(arr))
                resize!(arr, new_nnz)
                #println(length(arr))
                for i in n:-1:1
                    old_lo = Int(cur_refs[i])
                    old_hi = Int(cur_refs[i+1]) - 1
                    new_hi = Int(new_refs[i+1]) - 1
                    new_lo = Int(new_refs[i])
                    new_hi = Int(new_refs[i+1]) - 1
                    new_lo == old_lo && continue
                    n_copy = old_hi - old_lo + 1
                    #if(new_lo == old_lo) #tinue # no shift in location, no need to copy
                    #    println("\t Nopying s$(i) $(n_copy): arr[$(old_lo):$(old_hi)] to arr[$(new_lo):$(new_hi)]")
                    #else
                    #    println("\t copying s$(i) $(n_copy): arr[$(old_lo):$(old_hi)] to arr[$(new_lo):$(new_hi)]")
                        copyto!(arr, new_lo, arr, old_lo, n_copy)
                    #end
                end
            elseif all_shrinking
                #println("all_shrinking")
                #println(length(arr))
                for i in 1:n
                    old_lo = Int(cur_refs[i])
                    new_lo = Int(new_refs[i])
                    old_hi = Int(cur_refs[i+1]) - 1
                    new_hi = Int(new_refs[i+1]) - 1
                    new_lo == old_lo && continue
                    n_copy = new_hi - new_lo + 1
                    #if new_lo == old_lo
                    #    println("\t Nopying s$(i) $(n_copy): arr[$(old_lo):$(old_hi)] to arr[$(new_lo):$(new_hi)]")
                    #else
                    #    println("\t copying s$(i) $(n_copy): arr[$(old_lo):$(old_hi)] to arr[$(new_lo):$(new_hi)]")
                        copyto!(arr, new_lo, arr, old_lo, n_copy)
                    #end
                end
                resize!(arr, new_nnz)
                #println(length(arr))

            else
                #println("mix")
                # some grow some shrin
                tmp = similar(arr, new_nnz)
                #println(length(arr))
                #println(length(tmp))
                for i in 1:n
                    old_lo = Int(cur_refs[i])
                    old_hi = Int(cur_refs[i+1]) - 1
                    new_lo = Int(new_refs[i])
                    new_hi = Int(new_refs[i+1]) - 1
                    n_copy = min(old_hi - old_lo + 1, new_hi - new_lo + 1)
                    #println("\t copying s$(i) $(n_copy): arr[$(old_lo):$(old_hi)] to tmp[$(new_lo):$(new_hi)]")
                    copyto!(tmp, new_lo, arr, old_lo, n_copy)
                end
                updated_csr = merge(updated_csr, NamedTuple{(csr_array_name,)}((tmp,)))
            end
        end

        soa.refs = merge(old_refs, NamedTuple{(key,)}((new_refs,)))
        soa.csr  = updated_csr
    end

    return soa
end

@generated function csr_count_key(::AnySoA{P}, ::Val{F}) where {P, F}
    plugin_types = P.parameters
    for PT in Plugins
        for (field, key) in pairs(csr_fields(PT))
            field === F && return QuoteNode(key)
        end
    end
    :(error("field $F not found"))
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


@inline getsite(soa::SiteSoA, i::Int) = SiteView(soa, i)


@generated function get_ctx(ctx::C, ::Type{P}) where {C<:NamedTuple, P}
    key = nameof(P)   # e.g. :CarbonPlugin
    hasfield(C, key) ? :(getfield(ctx, $(QuoteNode(key)))) : :(NamedTuple())
end

@generated function process_soa!(soa::AnySoA{P}, t::Int; ctx::C=NamedTuple()) where {P, C<:NamedTuple}
    plugin_types = P.parameters
    calls        = [:(process_plugin!(soa, $PT, t; ctx=get_ctx(ctx, $PT))) for PT in plugin_types]
    quote $(calls...) end
end
function simulate_timestep!(soa::AnySoA{P}, t::Int; ctx::C=NamedTuple()) where {P, C<:NamedTuple}
    process_soa!(soa, t; ctx = ctx)
end




end
