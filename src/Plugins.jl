module Plugins
    using ..PanCore
    include("plugins/BaseSitePlugin.jl")
    include("plugins/BiomassSuccessionPlugin.jl")
    using .BaseSitePlugin
    using .BiomassSuccessionPlugin
end

