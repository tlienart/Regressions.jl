export HuberLoss

abstract type RobustLoss <: AtomicLoss end

"""
$TYPEDEF

Huber Loss corresponding to

``ρ(rᵢ) = rᵢ²/2  if |rᵢ| ≤ M`` and ``ρ(rᵢ) = M(|rᵢ|-M/2)`` otherwise.
"""
struct HuberLoss{M} <: RobustLoss where M <: Real
    HuberLoss(M::Real=1.35) = new{M}()
end

ρ(::HuberLoss{M}, r, w) where {M} = sum(r.^2/2 .* w .+ M .* (abs.(r) .- M/2) .* (1.0 .- w))
ψ(::HuberLoss{M}, r, w) where {M} = r .* w .+ M .* sign.(r) .* (1.0 .- w)
ϕ(::HuberLoss{M}, r, w) where {M} = w

ρ(hl::HuberLoss{M}, r) where {M} = ρ(hl, r, abs.(r) .<= M)

(l::HuberLoss)(x::AVR, y::AVR) = ρ(l, x .- y)
(l::HuberLoss)(r::AVR)         = ρ(l, r)
