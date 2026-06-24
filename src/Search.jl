module Search
    include("search/SA.jl")
    include("search/LBSA.jl")
    include("search/MOLBSA.jl")
    include("search/CMAES.jl")
    include("search/MOCMAES.jl")
    include("search/IgelMOCMAES.jl")
    include("search/CMAMAE.jl")
    include("search/MOSA.jl")
    using .SA
    using .LBSA
    using .MOLBSA
    using .CMAES
    using .MOCMAES
    using .IgelMOCMAES
    using .CMAMAE
    using .MOSA
end
