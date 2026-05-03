module Plugins
    using ..PanCore
    include("plugins/BaseSitePlugin.jl")
    include("plugins/BiomassSuccession/BiomassSuccessionPlugin.jl")
    using .BaseSitePlugin
    using .BiomassSuccessionPlugin
end

