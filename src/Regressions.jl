module Regressions

using Parameters, DocStringExtensions
using LinearAlgebra, IterativeSolvers
import LinearMaps: LinearMap
import IterativeSolvers: cg
import Optim

import Base.+, Base.-, Base.*, Base./, Base.convert

const AVR = AbstractVector{<:Real}

include("utils.jl")

include("loss-penalty/generic.jl")
include("loss-penalty/standard.jl")

include("glr/constructors.jl")
include("glr/tools.jl")
include("glr/gradhess.jl")

include("fit/solvers.jl")
include("fit/default.jl")
include("fit/analytical.jl")
include("fit/newton.jl")

end # module
