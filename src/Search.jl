module Search
    include("search/common.jl")
    include("search/SA.jl")
    include("search/LBSA.jl")
    include("search/MOLBSA.jl")
    include("search/CMAES.jl")
    include("search/MOCMAES.jl")
    include("search/IgelMOCMAES.jl")
    include("search/CMAMAE.jl")
    include("search/NSGA2.jl")
    include("search/CCIgel.jl")
    include("search/MOSA.jl")
    include("search/GDIgel.jl")
    include("search/PlainCMA.jl")
    using .SA
    using .LBSA
    using .MOLBSA
    using .CMAES
    using .MOCMAES
    using .IgelMOCMAES
    using .CMAMAE
    using .NSGA2
    using .CCIgel
    using .MOSA
    using .GDIgel
    using .PlainCMA
end
