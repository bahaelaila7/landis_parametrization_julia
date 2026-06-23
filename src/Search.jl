module Search
    include("search/SA.jl")
    include("search/LBSA.jl")
    include("search/MOLBSA.jl")
    include("search/CMAES.jl")
    include("search/MOCMAES.jl")
    using .SA
    using .LBSA
    using .MOLBSA
    using .CMAES
    using .MOCMAES
end
