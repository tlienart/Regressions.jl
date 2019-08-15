export Analytical, CG,
        Newton, NewtonCG,
        LBFGS
        #        ProxGD, ISTA,
#        QuasiNewton, BFGS, LBFGS

abstract type Solver end

@with_kw struct Analytical <: Solver
    iterative::Bool = false
    max_inner::Int  = 200
end

CG() = Analytical(; iterative=true)

struct Newton <: Solver end

struct NewtonCG <: Solver end

abstract type QuasiNewton <: Solver end

struct LBFGS <: QuasiNewton end

struct BFGS <: QuasiNewton end
