module Parametrization
    using ..PanCore
    include("parametrize/utils.jl")
    include("parametrize/BiomassSuccession.jl")
    using .BiomassSuccessionParametrization
end

