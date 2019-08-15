export GeneralizedLinearRegression, GLR,
        LinearRegression, RidgeRegression, LassoRegression,
        LogisticRegression

"""
GeneralizedLinearRegression{L<:Loss, P<:Penalty}

Generalized Linear Regression (GLR) model with objective function:

``L(y, Xθ) + P(θ)``

where `L` is a loss function, `P` a penalty, `y` is the vector of observed response, `X` is
the feature matrix and `θ` the vector of parameters.

Special cases include:

* **OLS regression**:      L2 loss, no penalty.
* **Ridge regression**:    L2 loss, L2 penalty.
* **Lasso regression**:    L2 loss, L1 penalty.
* **Logistic regression**: Logit loss, [no,L1,L2] penalty.
"""
@with_kw mutable struct GeneralizedLinearRegression{L<:Loss, P<:Penalty}
    # Parameters that can be tuned
    loss::L             = L2Loss()    # L(y, ŷ=Xθ)
    penalty::P          = NoPenalty() # P(θ)
    fit_intercept::Bool = true        # add intercept ? def=true
end

const GLR = GeneralizedLinearRegression


"""
$SIGNATURES

Objective function: ``|y-Xθ|₂²/2``.
"""
LinearRegression(; fit_intercept::Bool=true) = GLR(fit_intercept=fit_intercept)

"""
$SIGNATURES

Objective function: ``|y-Xθ|₂²/2 + λ|θ|₂²/2``.
"""
function RidgeRegression(λ::Real=1.0; lambda::Real=λ, fit_intercept::Bool=true)
    check_pos(λ)
    GLR(fit_intercept=fit_intercept, penalty=lambda*L2Penalty())
end


"""
$SIGNATURES

Objective function: ``|y - Xθ|₂²/2 + λ|θ|₁``
"""
function LassoRegression(λ::Real=1.0; lambda::Real=λ, fit_intercept::Bool=true)
    check_pos(λ)
    GLR(fit_intercept=fit_intercept, penalty=lambda*L1Penalty())
end


"""
$SIGNATURES

Objective function: ``ll(y, Xθ) + λ|θ|₂²/2 + γ|θ|₁`` where `ll` is the logistic loss.
"""
function LogisticRegression(λ::Real=1.0, γ::Real=0.0; lambda::Real=λ,
                            penalty::Symbol=iszero(γ) ? :l2 : :en,
                            fit_intercept::Bool=true, gamma::Real=γ)
    check_pos.((λ, γ))
    penalty ∈ (:l1, :l2, :en, :none) ||
        throw(ArgumentError("Unrecognised penalty for a logistic regression: '$penalty' " *
                            "(expected none/l1/l2/en)"))

    penalty = if penalty == :none
       NoPenalty()
    elseif penalty == :l1
        λ * L1Penalty()
    elseif penalty == :l2
        λ * L2Penalty()
    else
        λ * L2Penalty() + γ * L1Penalty()
    end
    GeneralizedLinearRegression(loss=LogisticLoss(), penalty=penalty, fit_intercept=fit_intercept)
end
