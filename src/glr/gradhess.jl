# fg! -- objective function and gradient (avoiding recomputations)
# Hv! -- application of the Hessian

# ------------------------ #
#  -- Linear Regression -- #
# ------------------------ #
# -> ∇f(θ)  = X'(Xθ - y)   #
# -> ∇²f(θ) = X'X          #
# ------------------------ #

function fg!(glr::GLR{L2Loss,<:L2R}, X, y)
    J    = obj(glr) # GLR objective (loss+penalty)
    n, p = size(X)
    λ    = getscale(glr.penalty)
    if glr.fit_intercept
        (f, g, θ) -> begin
            β = θ[end]
            v = X * θ[1:end-1] .+ β .* ones(n)
            g !== nothing && (mul!(g, X', v .- y); g .+= λ .* θ)
            f !== nothing && return J(y, v, θ)
        end
    else
        (f, g, θ) -> begin
            v = X * θ
            g !== nothing && (mul!(g, X', v .- y); g .+= λ .* θ)
            f !== nothing && return J(y, v, θ)
        end
    end
end

function Hv!(glr::GLR{L2Loss,<:L2R}, X, y)
    n, p = size(X)
    λ    = getscale(glr.penalty)
    if glr.fit_intercept
        # H = [X 1]'[X 1] + λ I
        # rows a 1:p = [X'X + λI | X'1]
        # row  e end = [X'1      | n+λ]
        (Hv, v) -> begin
            # view on the first p rows
            a   = 1:p
            Hva = view(Hv, a)
            va  = view(v,  a)
            Xt1 = vec(sum(X, dims=1))
            ve  = v[end]
            # update for the first p rows -- (X'X + λI)v[1:p] + (X'1)v[end]
            mul!(Hva, X', X * va)
            Hva .+= λ .* va .+ Xt1 .* ve
            # update for the last row -- (X'1)'v + n v[end]
            Hv[end] = dot(Xt1, va) + (n+λ) * ve
        end
    else
        (Hv, v) -> (mul!(Hv, X', X * v); Hv .+= λ .* v)
    end
end
