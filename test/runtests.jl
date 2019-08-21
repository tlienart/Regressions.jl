using Regressions, Test, LinearAlgebra, Random
DO_SKLEARN = true; include("testutils.jl")

m("UTILS"); include("utils.jl")

m("LOSS-PENALTY"); begin
    include("loss-penalty/generic.jl")
    include("loss-penalty/utils.jl")
    include("loss-penalty/robust.jl")
end

m("GLR"); begin
    include("glr/constructors.jl")
    include("glr/tools-utils.jl")
    include("glr/grad-hess-prox.jl")
end

m("FIT"); begin
    include("fit/analytical.jl")
    include("fit/newton.jl")
    include("fit/proxgrad.jl")
end
