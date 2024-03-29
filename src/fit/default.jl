export fit

# Default solvers

# TODO: in the future, have cases where if the things are too big, take another default.
# also should check if p > n in which case should do dual stuff (or other appropriate alternative)

_solver(::GLR{L2Loss,<:L2R},	 	  np::NTuple{2,Int}) = Analytical()
_solver(::GLR{LogisticLoss,<:L2R}, 	  np::NTuple{2,Int}) = LBFGS()
_solver(::GLR{MultinomialLoss,<:L2R}, np::NTuple{2,Int}) = LBFGS()

_solver(::GLR{RobustLoss,<:L2R}, np::NTuple{2,Int}) = LBFGS()

function _solver(glr::GLR{<:SMOOTH_LOSS,<:ENR},
			     np::NTuple{2,Int})
	(is_l1(glr.penalty) || is_elnet(glr.penalty)) && return FISTA()
	@error "Not yet implemented"
end

# Fallback NOTE: should revisit bc with non-smooth, wouldn't work probably PGD/PSGD
# depending on how much data there is
_solver(::GLR, np::NTuple{2,Int}) = @error "Not yet implemented"


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
