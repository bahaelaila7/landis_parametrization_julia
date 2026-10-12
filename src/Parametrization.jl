module Parametrization
    using ..PanCore
    include("parametrize/utils.jl")
    include("parametrize/BiomassSuccession.jl")
    include("parametrize/PnET.jl")
    using .BiomassSuccessionParametrization
    using .PnETParametrization
end

