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
objfun(glr::GLR, X, y; c::Int=0) = θ -> obj(glr)(y, apply_X(X, θ, c), θ)
