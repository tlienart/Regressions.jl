# fg! -- objective function and gradient (avoiding recomputations)
# Hv! -- application of the Hessian

# ----------------------- #
#  -- Ridge Regression -- #
# ----------------------- #
# ->  f(θ)  = |Xθ - y|₂²/2 + λ|θ|₂²
# -> ∇f(θ)  = X'(Xθ - y) + λθ
# -> ∇²f(θ) = X'X + λI
# NOTE:
# * Hv! used in iterative solution
# ---------------------------------------------------------

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
# * yᵢ ∈ {±1} so that y² = 1
# * -σ(-x) ==(σ(x)-1)
# ---------------------------------------------------------

function fgh!(glr::GLR{LogisticLoss,<:L2R}, X, y)
    J    = obj(glr) # GLR objective (loss+penalty)
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
                ΛX = w .* X
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
            H === nothing || (mul!(H, X', w .* X); add_λI!(H, λ))
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


# ---------------------------------- #
#  -- Multinomial Regression (L2) -- #
# ---------------------------------- #
# ->  c is the number of classes, θ has dims p * c
# ->  P = X * θ
# -> Zᵢ = ∑ exp(Pᵢ)
# -> Λ  = Diagonal(-Z)
# ->  f(θ)   = ∑(log Zᵢ - P[i, y[i]]) +  λ|θ|₂²
# -> ∇f(θ)   = reshape(X'ΛM, c * p)
# -> ∇²f(θ)v = via R operator
# NOTE:
# * yᵢ ∈ {1, 2, ..., c}
# ---------------------------------------------------------

function fg!(glr::GLR{MultinomialLoss,<:L2R}, X, y)
    n, p = size(X)
    c    = maximum(y)
    λ    = getscale(glr.penalty)
    if glr.fit_intercept
        (f, g, θ) -> begin
            P = apply_X(X, θ[1:end-c], c) .+ θ[end-c+1:end]     # O(npc) dims n * c
            M = exp.(P)                                         # O(npc) dims n * c
            g === nothing || begin
                ΛM = M ./ sum(M, dims=2)                        # O(nc)  dims n * c
                Q  = BitArray(y[i] == j for i = 1:n, j=1:c)
                g[1:end-c]     .= reshape(X'ΛM .+ X'Q, p * c)   # O(npc)
                g[end-c+1:end] .= sum(ΛM, dims=1) .+ sum(Q, dims=1)
                g .+= λ .* θ
            end
            f === nothing || begin
                # we re-use pre-computations here, see also MultinomialLoss
                ms = maximum(P, dims=2)
                ss = sum(M ./ exp(ms), dims=2)
                @inbounds ps = [P[i, y[i]] for i in eachindex(y)]
                return sum(log.(ss) .+ ms .- ps) + λ * norm(θ)^2/2
            end
        end
    else
        (f, g, θ) -> begin
            P = apply_X(X, θ, c)
            M = exp.(P)
            g === nothing || begin
                ΛM  = M ./ sum(M, dims=2)
                Q   = BitArray(y[i] == j for i = 1:n, j=1:c)
                g  .= reshape(X'ΛM .- X'Q, p * c)
                g .+= λ .* θ
            end
            f === nothing || begin
                ms = maximum(P, dims=2)
                ss = sum(M ./ exp.(ms), dims=2)
                @inbounds ps = [P[i, y[i]] for i in eachindex(y)]
                return sum(log.(ss) .+ ms .- ps) + λ * norm(θ)^2/2
            end
        end
    end
end
