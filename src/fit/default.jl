export fit

# Default solvers

# OLS
_solver(::GLR{L2Loss,<:L2R}, np::NTuple{2,Int}) = Analytical()
# LASSO
# XXX _solver(::GLR{L2Loss,L1R}, np::NTuple{2,Int}) = ISTA()
# Logistic with L2
# XXX _solver(::GLR{LogisticLoss,<:L2R}, np::NTuple{2,Int}) = LBFGS()

# Fallback NOTE: should revisit bc with non-smooth, wouldn't work probably PGD/PSGD
# depending on how much data there is
_solver(::GLR, n) = @error "Not yet implemented"


"""
$SIGNATURES

Fit a generalised linear regression model using an appropriate solver based on
the loss and penalty of the model. A method can, in some cases, be specified.
"""
function fit(glr::GLR, X::AbstractMatrix{<:Real}, y::AVR;
			 solver::Solver=_solver(glr, size(X)))
    check_nrows(X, y)
    _fit(glr, solver, X, y)
end