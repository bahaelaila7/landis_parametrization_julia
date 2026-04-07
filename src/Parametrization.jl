module Parametrization
    using ..PanCore
    include("parametrize/types.jl")
    include("parametrize/BiomassSuccession.jl")
    using .BiomassSuccessionParametrization
end

