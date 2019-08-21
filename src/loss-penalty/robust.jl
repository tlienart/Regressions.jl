export HuberLoss

"""
$TYPEDEF

Huber Loss corresponding to

``ρ(rᵢ) = rᵢ²/2  if |rᵢ|≤δ`` and ``ρ(rᵢ)=δ(|rᵢ|-δ/2)`` otherwise.
"""
struct HuberLoss{δ} <: AtomicLoss where δ <: Real
    HuberLoss(δ::Real) = new{δ}()
end

function ρ(::HuberLoss{δ}, r) where {δ}
    ar = abs.(r)
    m  = ar .<= δ
    return sum(r.^2/2 .* m .+ δ .* (ar .- δ/2) .* .!m)
end

(l::HuberLoss)(x::AVR, y::AVR) = ρ(l, x .- y)
(l::HuberLoss)(r::AVR)         = ρ(l, r)
