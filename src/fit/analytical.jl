"""
$SIGNATURES

Fit a least square regression either with no penalty (OLS) or with a L2 penalty (Ridge).

## Complexity

Assuming `n` dominates `p`,

* non-iterative (full solve):     O(np²) - dominated by the construction of the Hessian X'X.
* iterative (conjugate gradient): O(κnp) - with κ the number of CG steps (κ ≤ p).
"""
function _fit(glr::GLR{L2Loss,<:L2R}, solver::Analytical, X, y)
	# augment X if appropriate
	X_ = augment_X(X, glr.fit_intercept)
	λ  = getscale(glr.penalty)
	# full solve
	if !solver.iterative
		iszero(getscale(glr.penalty)) && return X_ \ y
		# form the Hat Matrix
		H = Hermitian(X_'X_) + λ * I
		return cholesky!(H) \ X'y
	end
	# Conjugate Gradient
	p = size(X_, 2)
	max_cg_steps = min(solver.max_inner, p)
	# O(np) application of the hessian r = H * v
	Hm = LinearMap(Hv!(glr, X_, y), p; ismutating=true, isposdef=true, issymmetric=true)
	return cg(Hm, X_'y; maxiter=max_cg_steps)
end
