export Analytical, CG,
        Newton, NewtonCG,
        LBFGS,
        ProxGrad, FISTA

# =====
# TODO
# * all - pick linesearch
# * NewtonCG number of inner iter
# * FISTA field to enforce descent
# ====

abstract type Solver end

# ===================== analytical.jl

@with_kw struct Analytical <: Solver
    iterative::Bool = false
    max_inner::Int  = 200
end

CG() = Analytical(; iterative=true)

# ===================== newton.jl

struct Newton <: Solver end

struct NewtonCG <: Solver end

struct LBFGS <: Solver end

# struct BFGS <: Solver end

# ===================== pgrad.jl

struct ProxGrad <: Solver end

struct AccelProxGrad <: Solver end
const FISTA = AccelProxGrad
