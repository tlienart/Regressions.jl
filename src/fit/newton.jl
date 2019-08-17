# Newton and Quasi Newton solvers

## LOGISTIC ==============

"""
$SIGNATURES

Fit a logistic regression either with no penalty or with a L2 penalty using Newton's method.

## Complexity

Assuming `n` dominates `p`, O(κnp²), dominated by the construction of the Hessian at each step with
κ the number of Newton steps.
"""
function _fit(glr::GLR{LogisticLoss,<:L2R}, solver::Newton, X, y)
    p     = size(X, 2) + Int(glr.fit_intercept)
    θ₀    = zeros(p)
    _fgh! = fgh!(glr, X, y)
    opt   = Optim.only_fgh!(_fgh!)
    res   = Optim.optimize(opt, θ₀, Optim.Newton())
    return Optim.minimizer(res)
end

"""
$SIGNATURES

Fit a logistic regression either with no penalty (OLS) or with a L2 penalty (Ridge) using Newton's
method but using an iterative solver (conjugate gradient) to solve the problems (∇²f)⁻¹∇f.

## Complexity

Assuming `n` dominates `p`, O(κ₁κ₂np), dominated by the application of the Hessian at each step
where κ₁ is the number of Newton steps and κ₂ is the average number of CG steps per Newton step
(which is at most p).
"""
function _fit(glr::GLR{LogisticLoss,<:L2R}, solver::NewtonCG, X, y)
    p    = size(X, 2) + Int(glr.fit_intercept)
    θ₀   = zeros(p)
    _f   = objfun(glr, X, y)
    _fg! = (g, θ) -> fgh!(glr, X, y)(0.0, g, nothing, θ) # XXX: Optim.jl/issues/738
    _Hv! = Hv!(glr, X, y)
    opt  = Optim.TwiceDifferentiableHV(_f, _fg!, _Hv!, θ₀)
    res  = Optim.optimize(opt, θ₀, Optim.KrylovTrustRegion())
    return Optim.minimizer(res)
end

"""
$SIGNATURES

Fit a logistic regression either with no penalty or with a L2 penalty using LBFGS.

## Complexity

Assuming `n` dominates `p`, O(κnp), dominated by the computation of the gradient at each step with
κ the number of LBFGS steps.
"""
function _fit(glr::GLR{LogisticLoss,<:L2R}, solver::LBFGS, X, y)
    p    = size(X, 2) + Int(glr.fit_intercept)
    θ₀   = zeros(p)
    _fg! = (f, g, θ) -> fgh!(glr, X, y)(f, g, nothing, θ)
    opt  = Optim.only_fg!(_fg!)
    res  = Optim.optimize(opt, θ₀, Optim.LBFGS())
    return Optim.minimizer(res)
end


## MULTINOMIAL ==============

"""
$SIGNATURES

Fit a multinomial regression either with no penalty or with a L2 penalty using Newton's method with
an iterative solver (conjugate gradient).

## Complexity

Assuming `n` dominates `p`, O(κ₁κ₂npc), where `c` is the number of classes. The computations are
dominated by the application of the Hessian at each step with κ₁ the number of Newton steps and κ₂
the average number of CG steps per Newton step.
"""
function _fit(glr::GLR{MultinomialLoss,<:L2R}, solver::NewtonCG, X, y)
    p    = size(X, 2) + Int(glr.fit_intercept)
    c    = maximum(y)
    θ₀   = zeros(p * c)
    _f   = objfun(glr, X, y; c=c)
    _fg! = (g, θ) -> fg!(glr, X, y)(0.0, g, θ) # XXX: Optim.jl/issues/738
    _Hv! = Hv!(glr, X, y)
    opt  = Optim.TwiceDifferentiableHV(_f, _fg!, _Hv!, θ₀)
    res  = Optim.optimize(opt, θ₀, Optim.KrylovTrustRegion())
    return Optim.minimizer(res)
end

"""
$SIGNATURES

Fit a multinomial regression either with no penalty or with a L2 penalty using LBFGS.

## Complexity

Assuming `n` dominates `p`, O(κnpc), with `c` the number of classes, dominated by the computation
of the gradient at each step with κ the number of LBFGS steps.
"""
function _fit(glr::GLR{MultinomialLoss,<:L2R}, solver::LBFGS, X, y)
    p    = size(X, 2) + Int(glr.fit_intercept)
    c    = maximum(y)
    θ₀   = zeros(p * c)
    _fg! = fg!(glr, X, y)
    opt  = Optim.only_fg!(_fg!)
    res  = Optim.optimize(opt, θ₀, Optim.LBFGS())
    return Optim.minimizer(res)
end
