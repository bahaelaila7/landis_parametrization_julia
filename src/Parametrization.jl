module Parametrization
    using ..PanCore
    include("parametrize/utils.jl")
    include("parametrize/BiomassSuccession.jl")
    include("parametrize/PnET.jl")
    include("parametrize/HBAD.jl")
    using .BiomassSuccessionParametrization
    using .PnETParametrization
    using .HBADParametrization
end

