export Cost,
        NoLoss, NoPenalty,
        AtomicLoss, AtomicPenalty,
        ScaledLoss, ScaledPenalty,
        CompositeLoss, CompositePenalty

abstract type Cost end

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Loss: (x, y) -> L(x, y)
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

abstract type Loss <: Cost end

struct NoLoss <: Loss end

abstract type AtomicLoss <: Loss end

mutable struct ScaledLoss{AL} <: Loss where AL <: AtomicLoss
    loss::AL
    scale::Real
end

mutable struct CompositeLoss <: Loss
    losses::Vector{ScaledLoss}
end

(sl::ScaledLoss)(x::AVR, y::AVR)    = sl.scale * sl.loss(x, y)
(cl::CompositeLoss)(x::AVR, y::AVR) = sum(loss(x, y) for loss ∈ cl.losses)

getscale(n::NoLoss)     = 0.0
getscale(l::AtomicLoss) = 1.0
getscale(l::ScaledLoss) = l.scale

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Penalty: θ -> P(θ)
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

abstract type Penalty <: Cost end

struct NoPenalty <: Penalty end

abstract type AtomicPenalty <: Penalty end

mutable struct ScaledPenalty{AP} <: Penalty where AP <: AtomicPenalty
    penalty::AP
    scale::Real
end

mutable struct CompositePenalty <: Penalty
    penalties::Vector{ScaledPenalty}
end

(sp::ScaledPenalty)(θ::AVR)    = sp.scale * sp.penalty(θ)
(cl::CompositePenalty)(θ::AVR) = sum(penalty(θ) for penalty ∈ cl.penalties)

getscale(n::NoPenalty)     = 0.0
getscale(p::AtomicPenalty) = 1.0
getscale(p::ScaledPenalty) = p.scale

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Objective function: (x, y, θ) -> L(x, y) + P(θ)
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

mutable struct ObjectiveFunction{L<:Loss,P<:Penalty} <: Cost
    loss::L
    penalty::P
end

(J::ObjectiveFunction)(y, ŷ, θ) = J.loss(y, ŷ) + J.penalty(θ)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Composition of Loss & Penalty functions
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const AL = AtomicLoss
const AP = AtomicPenalty
const SL = ScaledLoss
const CL = CompositeLoss
const SP = ScaledPenalty
const CP = CompositePenalty
const NL = NoLoss
const NP = NoPenalty
const OF = ObjectiveFunction

scale1(a::AL) = ScaledLoss(a, 1)
scale1(a::AP) = ScaledPenalty(a, 1)

# Combinations with NoLoss (NL)
*(::NL, ::Real)  = NoLoss()
+(::NL, l::Loss) = l
+(l::Loss, ::NL) = l

# Combinations with NoPenalty (NP)
*(::NP, ::Real)     = NoPenalty()
+(::NP, p::Penalty) = p
+(p::Penalty, ::NP) = p

# Combinations with AtomicLoss (AL)
+(a::AL, b::AL)           = scale1(a) + scale1(b)
+(a::AL, b::Union{SL,CL}) = scale1(a) + b
+(b::Union{SL,CL}, a::AL) = a + b
*(a::AL, c::Real)         = ScaledLoss(a, c)

# Combinations with AtomicPenalty (AP)
+(a::AP, b::AP)           = scale1(a) + scale1(b)
+(a::AP, b::Union{SP,CP}) = scale1(a) + b
+(b::Union{SP,CP}, a::AP) = a + b
*(a::AP, c::Real)         = ScaledPenalty(a, c)

# Combinations with Scaled Losses and Combined Losses
+(a::SL, b::SL)   = CL([a, b])
+(a::CL, b::CL)   = CL(vcat(a.losses, b.losses))
+(a::SL, c::CL)   = CL(vcat(c.losses, a))
+(c::CL, a::SL)   = a + c
*(a::SL, c::Real) = SL(a.loss, c * a.scale)
*(a::CL, c::Real) = CL(a.losses * c)

# Combinations with Scaled Penalties and Combined Penalties
+(a::SP, b::SP)   = CP([a, b])
+(a::CP, b::CP)   = CP(vcat(a.penalties, b.penalties))
+(a::SP, c::CP)   = CP(vcat(c.penalties, a))
+(c::CP, a::SP)   = a + c
*(a::SP, c::Real) = SP(a.penalty, c * a.scale)
*(a::CP, c::Real) = CP(a.penalties .* c)

# higher combinations ==> OF
+(l::Loss, p::Penalty) = OF(l, p)
+(p::Penalty, l::Loss) = l + p
+(o::OF, l::Loss)      = OF(o.loss + l, o.penalty)
+(l::Loss, o::OF)      = o + l
+(o::OF, p::Penalty)   = OF(o.loss, o.penalty + p)
+(p::Penalty, o::OF)   = o + p
+(a::OF, b::OF)        = OF(a.loss+b.loss, a.penalty+b.penalty)
*(o::OF, a::Real)      = OF(a * o.loss, a * o.penalty)

# Symetric relations
*(a::Real, c::Cost) = c * a

# - and / operations with Objective Functions (just use + and *)
-(a::Cost, b::Cost) = a + (-1 * b)
/(a::Cost, c::Real) = a * (1 / c)
