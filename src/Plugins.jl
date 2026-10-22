module Plugins
    using ..PanCore
    include("plugins/BaseSitePlugin.jl")
    include("plugins/BiomassSuccession/BiomassSuccessionPlugin.jl")
    include("plugins/PnET/PnETPlugin.jl")
    include("plugins/HBAD/HBADPlugin.jl")
    include("plugins/HBAD2/HBAD2Plugin.jl")
    include("plugins/HBAD3P/HBAD3PPlugin.jl")
    using .BaseSitePlugin
    using .BiomassSuccessionPlugin
    using .PnETPlugin
    using .HBADPlugin
    using .HBAD2Plugin
    using .HBAD3PPlugin
end

