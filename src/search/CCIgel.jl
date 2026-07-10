module CCIgel
# Cooperative-Coevolutionary Igel MO-(1+1)-CMA-ES ("ccigel"). A CC layer wrapped around the existing
# IgelMOCMAES engine (search/IgelMOCMAES.jl). The population is n_species × K whole param-set individuals,
# organised into per-species GROUPS of K. Two alternating phases:
#
#   Specialization (cc_spec_gens gens): each species group g runs a (1+1)-CMA step RESTRICTED to species
#   g's u-dimensions (perturb only g's slots, all other coords fixed → stable competition context; a group
#   cannot cheat by weakening rivals). Each individual is FULLY simulated (real competition) but scored on
#   species g's 2-objective loss (W_g, A_g) only. NSGA-II selects K within the group. This is implemented by
#   giving each group its own IgelMOCMAES.IgelState with mu=K and blocks=[species_slots[g]] — Igel's
#   block-diagonal ask/tell already perturbs and updates ONLY that block, leaving the rest of u untouched.
#
#   Recombination (between phases): build a fresh n_species × K whole population; for each new individual and
#   each species s, copy species s's u-block from a RANDOMLY chosen member of species-s's group.
#
#   Integration (cc_integ_gens gens): standard Igel over the whole vector, scored on the aggregate 2-obj —
#   this is where the Pareto archive / metrics / checkpoints fire. Driven from Pan by the verbatim Igel loop.
#
# This module only owns the CC-specific plumbing (block map, per-species IgelStates, recombination). The
# actual (1+1) mechanics are IgelMOCMAES's; the integration IgelState lives in the Pan driver.
import Random
import ..IgelMOCMAES
import ..MOLBSA: MOFitness, MOCandidate

export species_slot_map, CCGroups, recombine_us

# Build species_slots[g] = the u-dim indices (into build_slots order) that target global species g.
# For a SpeciesSampler slot the target `idx` IS the (global) species id. For an EcoSpeciesSampler slot
# `idx == (eco, local)` maps to the global species eco_species_ids[eco][local]. Global / Eco slots belong
# to no single species and are left OUT of specialization (they stay fixed across a spec phase).
function species_slot_map(param_dists, slots, eco_species_ids, n_species)
  species_slots = [Int[] for _ in 1:n_species]
  for (dim, (pi, idx)) in enumerate(slots)
    samp = param_dists.params[pi].sampler
    gsp = if samp isa PU_SpeciesSampler_type(param_dists)
      idx isa Integer ? Int(idx) : 0
    elseif samp isa PU_EcoSpeciesSampler_type(param_dists)
      (idx isa Tuple) ? Int(eco_species_ids[Int(idx[1])][Int(idx[2])]) : 0
    else
      0
    end
    gsp >= 1 && gsp <= n_species && push!(species_slots[gsp], dim)
  end
  return species_slots
end

# The sampler concrete types live in Pan's PU (parametrize/utils.jl); resolve them via a sample param so we
# don't hard-import the symbols (keeps this module dependency-light, mirroring how the driver passes things).
_SpeciesSampler_ref = Ref{Any}(nothing)
_EcoSpeciesSampler_ref = Ref{Any}(nothing)
PU_SpeciesSampler_type(_) = _SpeciesSampler_ref[]
PU_EcoSpeciesSampler_type(_) = _EcoSpeciesSampler_ref[]
# Register the concrete sampler types once (called from the Pan driver with PU.SpeciesSampler etc.).
function register_sampler_types!(species_sampler_type, eco_species_sampler_type)
  _SpeciesSampler_ref[] = species_sampler_type
  _EcoSpeciesSampler_ref[] = eco_species_sampler_type
  return nothing
end

# Container for the specialization sub-states: one IgelMOCMAES.IgelState per species group (only groups with
# ≥1 slot AND ≥1 competing dim; groups with no slots are skipped). `groups` maps species→state or nothing.
struct CCGroups
  states::Vector                            # per global species: its group's IgelState (nothing ⇒ no slots)
  species_slots::Vector{Vector{Int}}
end

# Recombine the current groups into a fresh n_species×K whole-individual u population. For each new whole and
# each species s (with a group), copy s's u-slots from a random member of s's group; the base vector is a
# random group member's full u (so global/eco coords come along coherently from one specialist).
function recombine_us(cc::CCGroups, n_species::Int, K::Int, base_us::Vector{Vector{Float64}}, rng::Random.AbstractRNG)
  new_us = Vector{Vector{Float64}}(undef, n_species * K)
  members(s) = cc.states[s] === nothing ? nothing : cc.states[s].pop
  for j in 1:(n_species * K)
    u = copy(base_us[rand(rng, 1:length(base_us))])
    for s in 1:n_species
      isempty(cc.species_slots[s]) && continue
      m = members(s); m === nothing && continue
      donor = m[rand(rng, 1:length(m))].x
      @inbounds for d in cc.species_slots[s]; u[d] = donor[d]; end
    end
    new_us[j] = u
  end
  return new_us
end

end
