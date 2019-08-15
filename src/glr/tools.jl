export obj, objfun

"""
$SIGNATURES

Return the objective function (sum of loss + penalty) of a Generalized Linear Model.
"""
obj(glr::GLR) = glr.loss + glr.penalty

"""
$SIGNATURES

Return a function computing the objective at a given point `θ`.
"""
function objfun(glr::GLR, X, y)
    if glr.fit_intercept
        θ -> obj(glr)(y, (X * θ[1:end-1]) .+ θ[end], θ)
    else
        θ -> obj(glr)(y, X * θ, θ)
    end
end
