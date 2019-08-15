# Newton and Quasi Newton solvers

"""
$SIGNATURES

Fit a logistic regression either with no penalty (OLS) or with a L2 penalty (Ridge) using Newton's
method.

## Complexity

Assuming `n` dominates `p`, O(κnp²), dominated by the construction of the Hessian at each step with
κ the number of Newton steps.
"""
function _fit(glr::GLR{LogisticLoss,<:L2R}, solver::Newton, X, y)
    p = size(X, 2) + Int(glr.fit_intercept)
    opt = Optim.only_fgh!(fgh!(glr, X, y))
    res = Optim.optimize(opt, zeros(p), Optim.Newton())
    return Optim.minimizer(res)
end

"""
$SIGNATURES

Fit a logistic regression either with no penalty (OLS) or with a L2 penalty (Ridge) using Newton's
method but using an iterative solver (conjugate gradient) to solve the problems (∇²f)⁻¹∇f.

## Complexity

Assuming `n` dominates `p`, O(κ₁κ₂np), dominated by the application of the Hessian at each step
where κ₁ is the number of Newton steps and κ₂ is the average number of CG steps per Newton step.
"""
function _fit(glr::GLR{LogisticLoss,<:L2R}, solver::NewtonCG, X, y)
    p = size(X, 2) + Int(glr.fit_intercept)
    f   = objfun(glr, X, y)
    fg! = (g, θ) -> fgh!(glr, X, y)(0.0, g, nothing, θ)
    θ₀  = zeros(p)
    opt = Optim.TwiceDifferentiableHV(f, fg!, Hv!(glr, X, y), θ₀)
    res = Optim.optimize(opt, θ₀, Optim.KrylovTrustRegion())
    return Optim.minimizer(res)
end

"""
$SIGNATURES

Fit a logistic regression either with no penalty (OLS) or with a L2 penalty (Ridge) using LBFGS.

## Complexity

Assuming `n` dominates `p`, O(κnp), dominated by the application of the Hessian at each step with
κ the number of LBFGS steps.
"""
function _fit(glr::GLR{LogisticLoss,<:L2R}, solver::LBFGS, X, y)
    p = size(X, 2) + Int(glr.fit_intercept)
    fg! = (f, g, θ) -> fgh!(glr, X, y)(f, g, nothing, θ)
    opt = Optim.only_fg!(fg!)
    res = Optim.optimize(opt, zeros(p), Optim.LBFGS())
    return Optim.minimizer(res)
end
