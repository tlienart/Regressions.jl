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
objfun(glr::GLR, X, y) = θ -> obj(glr)(y, X*θ, θ)
