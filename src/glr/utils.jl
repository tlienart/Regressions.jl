export objective

"""
$SIGNATURES

Return the objective function (sum of loss + penalty) of a Generalized Linear Model.
"""
objective(glr::GLR) = glr.loss + glr.penalty

"""
$SIGNATURES

Return a function computing the objective at a given point `θ`.
"""
objective(glr::GLR, X, y; c::Int=0) = θ -> objective(glr)(y, apply_X(X, θ, c), θ)
