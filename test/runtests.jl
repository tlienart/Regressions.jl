using Regressions, Test, LinearAlgebra, Random
DO_SKLEARN = true; include("testutils.jl")

m("UTILS"); include("utils.jl")

m("LOSS-PENALTY"); include("loss-penalty.jl")

m("GLR"); begin
    include("glr/constructors.jl")
    include("glr/tools.jl")
    include("glr/gradhess.jl")
end

m("FIT"); begin
    include("fit/analytical.jl")
end
