module Search
    include("search/SA.jl")
    include("search/LBSA.jl")
    using .SA
    using .LBSA
end
