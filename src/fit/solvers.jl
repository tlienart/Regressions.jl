export Analytical, CG,
        Newton, NewtonCG
        #        ProxGD, ISTA,
#        QuasiNewton, BFGS, LBFGS

abstract type Solver end

@with_kw struct Analytical <: Solver
    iterative::Bool = false
    max_inner::Int  = 200
end

CG() = Analytical(; iterative=true)

struct Newton <: Solver end
