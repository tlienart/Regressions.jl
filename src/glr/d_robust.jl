# ------------------------ #
#  -- Robust Regression -- #
# ------------------------ #
# ->        r̄ = (Xθ - y) / η²
# ->    ρ(r̄ᵢ) is the robust penalty assoc with residual
# ->    ψ(r̄ᵢ) = ρ'(r̄ᵢ) (first deriv)
# ->    ϕ(r̄ᵢ) = ψ'(r̄ᵢ) (second deriv; may be discont)
# ->     Λ(r̄) = diag(ϕ.(r̄)) / η²
# ->     J(θ) = nη² + ∑ρ.(r̄)η² + λ|θ|₂²
# ->    ∇J(θ) = X'ψ.(r̄) + λθ
# ->   ∇²J(θ) = X'Λ(r̄)X + λI
# ->     ∂_ηJ = 2η(n + ∑(ρ.(r̄) - r̄ .* ψ.(r̄)))
# ->   ∂²_ηηJ = (∂J/∂η)/η - 2η ∑ (r̄ .* ϕ.(r̄))
# -> ∂_η∇J(θ) = X'(ϕ.(r̄) .* (-2r̄/η))
# ---------------------------------------------------------


# function fgh!(glr::GLR{HuberLoss{M},<:L2R}, X, y) where {M}
#     n, p = size(X)
#     n_   = ifelse(M≤1, n * (M^2+1), n) # 1+M^2 avoids degeneracy when M≤1; see Owen 2006 p9.
#     p_   = p + Int(glr.fit_intercept)
#     λ    = getscale(glr.penalty)
#     (f, g, H, θ) -> begin
#         vθ = view(θ, 1:p_)
#         σ  = θ[p_+1]
#         r  = apply_X(X, vθ) .- y
#         r̄  = r ./ σ
#         w  = convert.(Float64, abs.(r̄) .<= M)
#         ρr̄ = ρ(glr.loss, r̄, w)
#         ψr̄ = ψ(glr.loss, r̄, w)
#         ∂J = n_ + ρr̄ - sum(r̄ .* ψr̄)
#         g === nothing || begin
#             vg = view(g, 1:p)
#             mul!(vg, X', ψr̄)
#             glr.fit_intercept && (g[p+1] = sum(ψr̄))
#             g .+= λ .* θ
#             g[p_ + 1] = ∂J
#         end
#         H === nothing || begin
#             ϕr̄  = ϕ(glr.loss, r̄, w)
#             ϕr̄σ = r̄ .* ϕr̄ ./ σ
#             ΛX  = ϕr̄σ .* X
#             ∂∇J = ΛX' * (-r̄)
#             mul!(view(H, 1:p, 1:p), X', ΛX)
#             @inbounds for i in 1:p
#                 H[i, end] = H[end, i] = ∂∇J[i]
#             end
#             if glr.fit_intercept
#                 H[p+1, end] = H[end, p+1] = sum(v)
#                 ΛXt1 = sum(ΛX, dims=1)
#                 @inbounds for i in 1:p
#                     H[i, p+1] = H[p+1, i] = ΛXt1[i]
#                 end
#                 H[p+1, p+1] = sum(ϕr̄σ)
#             end
#             add_λI!(H, λ)
#             # note: we don't regularize the scaling factor
#             H[end, end] = sum(r̄.^2 .* ϕr̄σ)
#         end
#         f === nothing || return ρr̄ + glr.penalty(vθ)
#     end
# end


function fgh!(glr::GLR{HuberLoss{M},<:L2R}, X, y) where {M}
    n, p = size(X)
    n_   = ifelse(M≤1, n * (M^2+1), n) # 1+M^2 avoids degeneracy when M≤1; see Owen 2006 p9.
    p_   = p + Int(glr.fit_intercept)
    λ    = getscale(glr.penalty)
    (f, g, H, θ) -> begin
        vθ = view(θ, 1:p_)
        η  = θ[p_+1]
        r  = apply_X(X, vθ) .- y
        r̄  = r ./ η^2
        w  = convert.(Float64, abs.(r̄) .<= M)
        ρr̄ = ρ(glr.loss, r̄, w)
        ψr̄ = ψ(glr.loss, r̄, w)
        ∂J = 2η*(n_ + ρr̄ - sum(r̄ .* ψr̄))
        g === nothing || begin
            vg = view(g, 1:p)
            mul!(vg, X', ψr̄)
            glr.fit_intercept && (g[p+1] = sum(ψr̄))
            g .+= λ .* θ
            g[end] = ∂J
        end
        H === nothing || begin
            ϕr̄  = ϕ(glr.loss, r̄, w)
            r̄ϕr̄ = r̄ .* ϕr̄
            ΛX  = ((ϕr̄ ./ η^2) .* X)
            v   = -2r̄ϕr̄ ./ η
            ∂∇J = X' * v
            mul!(view(H, 1:p, 1:p), X', ΛX)
            @inbounds for i in 1:p
                H[i, end] = H[end, i] = ∂∇J[i]
            end
            if glr.fit_intercept
                H[p+1, end] =  H[end, p+1] = sum(v)
                ΛXt1 = sum(ΛX, dims=1)
                @inbounds for i in 1:p
                    H[i, p+1] = H[p+1, i] = ΛXt1[i]
                end
                H[p+1, p+1] = sum(ϕr̄) / η^2
            end
            add_λI!(H, λ)
            # note: we don't regularize the scaling factor
            H[end, end] = ∂J/η .- 2η * sum(r̄ϕr̄) # ∂²J
        end
        f === nothing || return ρr̄ + glr.penalty(view(θ, 1:p_))
    end
end


function Hv!(glr::GLR{HuberLoss{M},<:L2R}, X, y) where {M}
    p = size(X, 2)
    λ = getscale(glr.penalty)
    # see d_logistic.jl for more comments on this (≈ procedure)
    if glr.fit_intercept
        (Hv, θ, v) -> begin
            r    = apply_X(X, θ) .- y
            w    = convert.(Float64, abs.(r) .<= M)
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
            w = convert.(Float64, abs.(r) .<= M)
            mul!(Hv, X', w .* (X * v))
            Hv .+= λ .* v
        end
    end
end
