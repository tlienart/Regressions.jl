# fg! -- objectivₑ function and gradient (avoiding recomputations)
# Hv! -- application of the Hessian

# ----------------------- #
#  -- Ridge Regression -- #
# ----------------------- #
# ->  f(θ)  = |Xθ - y|₂²/2 + λ|θ|₂²
# -> ∇f(θ)  = X'(Xθ - y) + λθ
# -> ∇²f(θ) = X'X + λI
# NOTE:
# * Hv! used in iterativₑ solution
# ---------------------------------

function Hv!(glr::GLR{L2Loss,<:L2R}, X, y)
    n, p = size(X)
    λ    = getscale(glr.penalty)
    if glr.fit_intercept
        # H = [X 1]'[X 1] + λ I
        # rows a 1:p = [X'X + λI | X'1]
        # row  e end = [1'X'     | n+λ]
        (Hv, v) -> begin
            # view on the first p rows
            a   = 1:p
            Hvₐ = view(Hv, a)
            vₐ  = view(v,  a)
            Xt1 = vec(sum(X, dims=1))
            vₑ  = v[end]
            # update for the first p rows -- (X'X + λI)v[1:p] + (X'1)v[end]
            mul!(Hvₐ, X', X * vₐ)
            Hvₐ .+= λ .* vₐ .+ Xt1 .* vₑ
            # update for the last row -- (X'1)'v + n v[end]
            Hv[end] = dot(Xt1, vₐ) + (n+λ) * vₑ
        end
    else
        (Hv, v) -> (mul!(Hv, X', X * v); Hv .+= λ .* v)
    end
end

# ------------------------------- #
#  -- Logistic Regression (L2) -- #
# ------------------------------- #
# ->  f(θ)  = -∑logσ(yXθ) + λ|θ|₂²
# -> ∇f(θ)  = -X'(yσ(-yXθ)) + λθ
# -> ∇²f(θ) = X'(σ(yXθ))X + λI
# NOTE:
# * y ∈ {±1} so that y² = 1
# * -σ(-x) ==(σ(x)-1)
# -------------------------------

function fgh!(glr::GLR{LogisticLoss,<:L2R}, X, y)
    J    = obj(glr) # GLR objectivₑ (loss+penalty)
    n, p = size(X)
    λ    = getscale(glr.penalty)
    if glr.fit_intercept
        (f, g, H, θ) -> begin
            v = X * θ[1:p] .+ θ[end]
            # precompute σ(yXθ) use -σ(-x) = (σ(x)-1)
            w = σ.(v .* y)
            g === nothing || begin
                tmp = y .* (w .- 1.0)
                mul!(view(g, 1:p), X', tmp)
                g[end] = sum(tmp)
                g .+= λ .* θ
            end
            H === nothing || begin
                ΛX = Diagonal(w) * X
                mul!(view(H, 1:p, 1:p), X', ΛX)
                ΛXt1 = sum(ΛX, dims=1)
                @inbounds for i = 1:p
                    H[i, end] = H[end, i] = ΛXt1[i]
                end
                H[end, end] = sum(w)
                add_λI!(H, λ)
            end
            f === nothing || return J(y, v, θ)
        end
    else
        (f, g, H, θ) -> begin
            v = X * θ
            # precompute σ(yXθ) use -σ(-x) = σ(x)(σ(x)-1)
            w = σ.(y .* v)
            g === nothing || (mul!(g, X', y .* (w .- 1.0)); g .+= λ .* θ)
            H === nothing || (mul!(H, X', Diagonal(w) * X); add_λI!(H, λ))
            f === nothing || return J(y, v, θ)
        end
    end
end


function Hv!(glr::GLR{LogisticLoss,<:L2R}, X, y)
    n, p = size(X)
    λ    = getscale(glr.penalty)
    if glr.fit_intercept
        # H = [X 1]'Λ[X 1] + λ I
        # rows a 1:p = [X'ΛX + λI | X'Λ1]
        # row  e end = [1'ΛX      | sum(a)+λ]
        (Hv, θ, v) -> begin
            # precompute σ(yXθ) use -σ(-x) = (σ(x)-1)
            w = σ.((X * θ[1:p] .+ θ[end]) .* y)
            # view on the first p rows
            a    = 1:p
            Hvₐ  = view(Hv, a)
            vₐ   = view(v,  a)
            XtΛ1 = X' * (w .* ones(n))     # X'Λ1; O(np)
            vₑ   = v[end]
            # update for the first p rows -- (X'X + λI)v[1:p] + (X'1)v[end]
            mul!(Hvₐ, X', w .* (X * vₐ)) # (X'ΛX)vₐ
            Hvₐ .+= λ .* vₐ .+ XtΛ1 .* vₑ
            # update for the last row -- (X'1)'v + n v[end]
            Hv[end] = dot(XtΛ1, vₐ) + (sum(w)+λ) * vₑ
        end
    else
        (Hv, θ, v) -> begin
            w = σ.((X * θ) .* y)
            mul!(Hv, X', w .* (X * v))
            Hv .+= λ .* v
        end
    end
end
