# ------------------------ #
#  -- Robust Regression -- #
# ------------------------ #
# ->     r  = Xθ - y
# ->   ρ(r) is the robust penalty assoc with residual
# ->   ψ(r) = ρ'(r) (first deriv)
# ->   ϕ(r) = ψ'(r) (second deriv; may be discont)
# ->   Λ(r) = diag(ϕ(r))
# ->  f(θ)  = ∑ρ.(r) + λ|θ|₂²
# -> ∇f(θ)  = X'ψ.(r) + λθ
# -> ∇²f(θ) = X'Λ(r)X + λI
# ---------------------------------------------------------

function fgh!(glr::GLR{HuberLoss{δ},<:L2R}, X, y) where {δ}
    p  = size(X, 2)
    λ  = getscale(glr.penalty)
    if glr.fit_intercept
        (f, g, H, θ) -> begin
            r = apply_X(X, θ) .- y
            w = convert.(Float64, abs.(r) .<= δ) # note that ϕr = w
            g === nothing || begin
                ψr  = r .* w .+ δ .* sign.(r) .* (1.0 .- w)
                mul!(view(g, 1:p), X', ψr)
                g[end] = sum(ψr)
                g .+= λ .* θ
            end
            H === nothing || begin
                ΛX = w .* X
                mul!(view(H, 1:p, 1:p), X', ΛX)
                ΛXt1 = sum(ΛX, dims=1)
                @inbounds for i in 1:p
                    H[i, end] = H[end, i] = ΛXt1[i]
                end
                H[end, end] = sum(w)
                add_λI!(H, λ)
            end
            f === nothing || return glr.loss(r) + glr.penalty(θ)
        end
    else
        (f, g, H, θ) -> begin
            r = apply_X(X, θ) .- y
            w = convert.(Float64, abs.(r) .<= δ) # note that ϕr = w
            g === nothing || begin
                ψr  = r .* w .+ δ .* sign.(r) .* (1.0 .- w)
                mul!(g, X', ψr)
                g .+= λ .* θ
            end
            H === nothing || (mul!(H, X', w .* X); add_λI!(H, λ))
            f === nothing || return glr.loss(r) + glr.penalty(θ)
        end
    end
end


function Hv!(glr::GLR{HuberLoss{δ},<:L2R}, X, y) where {δ}
    p = size(X, 2)
    λ = getscale(glr.penalty)
    # see d_logistic.jl for more comments on this (≈ procedure)
    if glr.fit_intercept
        (Hv, θ, v) -> begin
            r    = apply_X(X, θ) .- y
            w    = convert.(Float64, abs.(r) .<= δ)
            a    = 1:p
            Hvₐ  = view(Hv, a)
            vₐ   = view(v, a)
            XtΛ1 = X' * w
            vₑ   = v[end]
            # update for first p rows
            mul!(Hvₐ, X', w .* (X * vₐ))
            Hvₐ .+= λ .* vₐ .+ XtΛ1 .* vₑ
            # update for the last row
            Hv[end] = dot(XtΛ1, vₐ) + (sum(w)+λ) * vₑ
        end
    else
        (Hv, θ, v) -> begin
            r = apply_X(X, θ) .- y
            w = convert.(Float64, abs.(r) .<= δ)
            mul!(Hv, X', w .* (X * v))
            Hv .+= λ .* v
        end
    end
end
