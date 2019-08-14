using Regressions, Test, LinearAlgebra, Random
const R = Regressions
m(s) = println("\n== $s ==\n")

m("UTILS"); include("utils.jl")

m("LOSS-PENALTY"); include("loss-penalty.jl")

m("GLR"); include("glr.jl")

m("FIT"); begin
    include("fit/analytical.jl")
end
