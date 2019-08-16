export obj, objfun

"""
$SIGNATURES

Return the objective function (sum of loss + penalty) of a Generalized Linear Model.
"""
obj(glr::GLR) = glr.loss + glr.penalty

"""
$SIGNATURES

Return `X*θ` if `c=0` (default) or `X*P` where `P=reshape(θ, size(X, 2), p)` in the multi-class
case.
"""
apply_X(X, θ, c) = iszero(c) ? X * θ : X * reshape(θ, size(X, 2), c)

"""
$SIGNATURES

Return a function computing the objective at a given point `θ`.
"""
function objfun(glr::GLR, X, y; c::Int=0)
    if glr.fit_intercept
        θ -> obj(glr)(y, apply_X(X, θ[1:end-1], c) .+ θ[end], θ)
    else
        θ -> obj(glr)(y, apply_X(X, θ, c), θ)
    end
end
