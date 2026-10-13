module Plugins
    using ..PanCore
    include("plugins/BaseSitePlugin.jl")
    include("plugins/BiomassSuccession/BiomassSuccessionPlugin.jl")
    include("plugins/PnET/PnETPlugin.jl")
    include("plugins/HBAD/HBADPlugin.jl")
    using .BaseSitePlugin
    using .BiomassSuccessionPlugin
    using .PnETPlugin
    using .HBADPlugin
end

