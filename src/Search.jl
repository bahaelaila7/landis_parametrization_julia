module Search
    include("search/SA.jl")
    include("search/LBSA.jl")
    include("search/MOLBSA.jl")
    using .SA
    using .LBSA
    using .MOLBSA
end
