export LPLoss, LPPenalty,
        L1Loss, L1Penalty,
        L2Loss, L2Penalty,
        LogisticLoss

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# No Loss / No Penalty
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

(l::NoLoss)(a::AVR, b::AVR) = 0.0
(p::NoPenalty)(θ::AVR)      = 0.0

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# LP Losses and Penalties
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

"""
$TYPEDEF

Scaled L-p loss of the residual.

``L(x,y) = ||x-y||_{p}^{p}/p``

The scaling simplifies expressions in the common L2 case.
"""
struct LPLoss{p} <: AtomicLoss where p <: Real end


"""
$TYPEDEF

Scaled L-p norm of the parameter vector.

``P(θ) = ||θ||_{p}^{p}/p``

The scaling simplifies expressions in the common L2 case.
"""
struct LPPenalty{p} <: AtomicPenalty where p <: Real end


# Useful Shortcuts
const L1Loss    = LPLoss{1}
const L1Penalty = LPPenalty{1}
const L2Loss    = LPLoss{2}
const L2Penalty = LPPenalty{2}
const LPCost{p} = Union{LPLoss{p},LPPenalty{p}}

const L1R = ScaledPenalty{L1Penalty}
const L2R = Union{NoPenalty,ScaledPenalty{L2Penalty}}


"""
$SIGNATURES

Return the `p` in an `LPCost{p}`.
"""
getp(lpc::LPCost{p}) where p = p


"""
$SIGNATURES

Compute the lp norm to the `p`-th power of a vector given `p` scaled by `p`.
"""
function lp(v::AbstractVector{<:Real}, p)
    p == Inf && return maximum(v)
    p == 1   && return sum(abs.(v))
    p == 2   && return sum(abs2.(v)) / 2
    p  > 0   && return sum(abs.(v).^p) / p
    throw(DomainError("[lp] `p` has to be greater than 0"))
end

(l::LPLoss)(a::AVR, b::AVR) = lp(a .- b, getp(l))
(p::LPPenalty)(θ::AVR)      = lp(θ, getp(p))

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Logistic loss
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

"""
$TYPEDEF

``L(x, y) = -∑logσ(xᵢyᵢ)``

where `logσ` is the log of the sigmoid function.
See [`logsigmoi`](@ref).
"""
struct LogisticLoss <: AtomicLoss end

(::LogisticLoss)(x::AVR, y::AVR) = -sum(logsigmoid.(x .* y))
