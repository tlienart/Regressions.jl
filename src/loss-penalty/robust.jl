export RobustLoss, HuberRho

abstract type RobustRho end

struct RobustLoss{ρ} <: AtomicLoss where ρ <: RobustRho
   rho::ρ
end

(rl::RobustLoss)(x::AVR, y::AVR) = rl(x .- y)
(rl::RobustLoss)(r::AVR) = rl.rho(r)

"""
$TYPEDEF

Huber weighing of the residualss corresponding to

``ρ(rᵢ) = rᵢ²/2  if |rᵢ|≤δ`` and ``ρ(rᵢ)=δ(|rᵢ|-δ/2)`` otherwise.
"""
struct HuberRho{δ} <: RobustRho where δ <: Real
   HuberRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end

(::HuberRho{δ})(r::AVR) where δ = begin
   ar = abs.(r)
   m  = ar .<= δ
   return sum(r.^2/2 .* m .+ δ .* (ar .- δ/2) .* .!m)
end
