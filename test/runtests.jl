using Regressions, Test, LinearAlgebra, Random
DO_SKLEARN = false; include("testutils.jl")

m("UTILS"); include("utils.jl")

m("LOSS-PENALTY"); begin
    include("loss-penalty/generic.jl")
    include("loss-penalty/utils.jl")
end

m("GLR"); begin
    include("glr/constructors.jl")
    include("glr/tools.jl")
    include("glr/gradhess.jl")
end

m("FIT"); begin
    include("fit/analytical.jl")
end
