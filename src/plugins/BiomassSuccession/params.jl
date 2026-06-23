function generate_eco_params(params::BiomassSuccessionParams)::Vector{BiomassSuccessionEcoParams}

  return [
    BiomassSuccessionEcoParams(
      SPINUP_MORTALITY_FRACTION=params.SPINUP_MORTALITY_FRACTION,
      SUFFICIENT_LIGHT=params.SUFFICIENT_LIGHT, D=(@dimslice params.D species),
      S=(@dimslice params.S species),
      LONGEVITY=(@dimslice params.LONGEVITY species),
      MATURITY=(@dimslice params.MATURITY species),
      SHADE_TOL=(@dimslice params.SHADE_TOL species),
      PROB_RESPROUT=(@dimslice params.PROB_RESPROUT species),
      ANPP_MAX_SPP=params.ANPP_MAX_SPP[eco_id],
      B_MAX_SPP=params.B_MAX_SPP[eco_id],
      B_MAX_ECO=maximum(params.B_MAX_SPP[eco_id]),
      PROB_MORT_SPP=(params.PROB_MORT_SPP[eco_id]),
      PROB_ESTAB_SPP=(params.PROB_ESTAB_SPP[eco_id]),
      MIN_REL_BIOMASS=params.MIN_REL_BIOMASS[eco_id],
    )




    for (eco_id, species) in enumerate(params.ECO_SPECIES_IDS)]


end

function generate_biomass_params(species_list::Vector{String}, eco_list::Vector{String}, eco_species_ids::Vector{Vector{Int}}; rng::Random.AbstractRNG, no_establishment::Bool=false)
  n_species = length(species_list) |> UIntType
  n_ecoregions = length(eco_list) |> UIntType
  SPINUP_MORTALITY_FRACTION = 0.15f0 #rand(Dists.Uniform(0f0,0.20f0))
  #println(typeof(SPINUP_MORTALITY_FRACTION))

  S = rand(rng, Dists.truncated(Dists.Normal(0.5, 1.0), 0.01, 1.0), n_species) .|> FloatType #Random.rand(rng, FloatType, n_species),#
  #println(typeof(S))
  D = rand(rng, Dists.truncated(Dists.Normal(15, 10), 5, 25), n_species) .|> FloatType
  #println(typeof(D))
  #LONGEVITY = rand(rng, Dists.truncated(Dists.Normal(200, 100), 100, 300), n_species) .|> FloatType
  LONGEVITY = fill(FloatType(400), n_species)
  #println(typeof(LONGEVITY))
  SHADE_TOL = rand(rng, Dists.DiscreteUniform(1, 5), n_species) .|> UIntType # ::Vector{FloatType}
  #println(typeof(SHADE_TOL))
  MATURITY = if no_establishment
    zeros(FloatType, n_species)
  else
    rand(rng, Dists.DiscreteUniform(3, 40), n_species) .|> FloatType #::Vector{FloatType}
  end
  PROB_RESPROUT = zeros(FloatType, n_species)
  #println(typeof(MATURITY))

  PROB_MORT_SPP = [rand(rng, Dists.Uniform(), length(eco_species)) .|> FloatType  #::Matrix{FloatType}
                   for eco_species in eco_species_ids]
  #println(typeof(PROB_MORT_SPP))
  PROB_ESTAB_SPP = if no_establishment
    [zeros(FloatType, length(eco_species)) for eco_species in eco_species_ids]
  else
    [rand(rng, Dists.Uniform(), length(eco_species)) .|> FloatType for eco_species in eco_species_ids]
  end
  #println(typeof(PROB_ESTAB_SPP))
  ANPP_MAX_SPP = [rand(rng, Dists.truncated(Dists.Normal(700, 100), 100, 1500), length(eco_species)) .|> FloatType
                  for eco_species in eco_species_ids]
  #println(typeof(ANPP_MAX_SPP))
  # B_MAX_SPP lives on a 100-grid (see make_biomass_param_dists quantum=100); seed it on-grid too.
  B_MAX_SPP = [rand(rng, Dists.truncated(Dists.Normal(25000, 1000), 20000, 35000), length(eco_species)) .|> (x -> FloatType(round(x / 100) * 100))
               for eco_species in eco_species_ids]


  #println(typeof(B_MAX_SPP))
  #println(B_MAX_SPP)
  #B_MAX_ECO = [maximum(b_max_spp_eco) for b_max_spp_eco in B_MAX_SPP] #FloatType[0.0,0.0],#::Vector{FloatType}
  #println(typeof(B_MAX_ECO))
  #println(B_MAX_ECO)
  # (species.shade_tol 1-5 X (site.shade_class+1) 1-6) -> prob of sufficient light
  # shade_tol: 1 (least tolerant to shade) to 5 (most tolerant to shade)
  # shade_class: 1 (no shade on site) to 6 (full shade)
  # Biomass Documentation shadeclass 0 (no shade) to 5 (full shade) but julia arrays are 1-indexed
  # julia is column major, much faster to pickout the site's shade class as one chunk
  # then reference the species' shade_tol within
  SUFFICIENT_LIGHT_MATRIX = FloatType[
    1.00 0.50 0.25 0.00 0.00 0.00;
    1.00 1.00 0.50 0.25 0.00 0.00;
    1.00 1.00 1.00 0.50 0.25 0.00;
    1.00 1.00 1.00 1.00 0.50 0.25;
    1.00 1.00 1.00 1.00 1.00 0.50]
  # transforming to (shade_class -> shade_tol) to be able to save to json Vector{Vector{FloatType}}, Matrix{FloatType} is not easily serde'd
  SUFFICIENT_LIGHT = [vec(SUFFICIENT_LIGHT_MATRIX[:, shade_class]) for shade_class in axes(SUFFICIENT_LIGHT_MATRIX, 2)]
  #[1.0, 1.0, 1.0, 1.0, 1.0]  for shade_class = 1 (no shade), plants of all shade tolerance can reproduce
  #[0.5, 1.0, 1.0, 1.0, 1.0]
  #[0.25, 0.5, 1.0, 1.0, 1.0]
  #[0.0, 0.25, 0.5, 1.0, 1.0]
  #[0.0, 0.0, 0.25, 0.5, 1.0]
  #[0.0, 0.0, 0.0, 0.25, 0.5] for shade_class = 6 (full shade), only plants with highest shade tol (4,5) have a chance

  #println(typeof(SUFFICIENT_LIGHT))

  # by ecoregion
  firstMINRel = rand(rng, Dists.Uniform(0.0, 0.5), n_ecoregions) .|> FloatType
  #MIN_REL_BIOMASS = FloatType[[0.25, 0.45, 0.56, 0.70, 0.90] for _ in 1:2]
  # (shade_class x eco) -> bio percent
  # again, column major, pickout the relevant column for ecoregion
  #MIN_REL_BIOMASS = [fmin + k * 0.10f0
  #                   for k in 0:4,
  #                   fmin in firstMINRel]
  # transforming to eco -> shade_class
  MIN_REL_BIOMASS = [[fmin + k * 0.10f0 for k in 0:4]
                     for fmin in firstMINRel]
  #println(typeof(MIN_REL_BIOMASS))
  #println(MIN_REL_BIOMASS)

  p = BiomassSuccessionParams(
    SPINUP_MORTALITY_FRACTION=SPINUP_MORTALITY_FRACTION,
    SUFFICIENT_LIGHT=SUFFICIENT_LIGHT, D=D,
    S=S,
    LONGEVITY=LONGEVITY,
    SHADE_TOL=SHADE_TOL,
    MATURITY=MATURITY, ANPP_MAX_SPP=ANPP_MAX_SPP,
    B_MAX_SPP=B_MAX_SPP,
    PROB_RESPROUT=PROB_RESPROUT,
    PROB_MORT_SPP=PROB_MORT_SPP,
    PROB_ESTAB_SPP=PROB_ESTAB_SPP,
    MIN_REL_BIOMASS=MIN_REL_BIOMASS, SPECIES_LIST=species_list,
    ECO_LIST=eco_list,
    ECO_SPECIES_IDS=eco_species_ids,
  )
  return p


end
