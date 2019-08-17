using Regressions, Test, LinearAlgebra, Random
const R = Regressions
m(s) = println("\n== $s ==\n")

m("UTILS"); include("utils.jl")

@show ENV

m("LOSS-PENALTY"); include("loss-penalty.jl")

m("GLR"); begin
    include("glr/constructors.jl")
    include("glr/tools.jl")
    include("glr/gradhess.jl")
end

m("FIT"); begin
    include("fit/analytical.jl")
end
