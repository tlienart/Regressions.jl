# f   -- objective function
# fg! -- objective function and gradient (avoiding recomputations)
# Hv! -- application of the Hessian

f(glr::GLR, X, y) = objfun(glr, X, y)

# ------------------------ #
#  -- Linear Regression -- #
# ------------------------ #

function fg!(glr::LinReg, X, y)
    J = obj(glr)
    (f, g, θ) -> begin
        # common computations
        v = X*θ
        if g !== nothing
            # update the gradient in place
            mul!(g, X', v .- y)
        end
        if f !== nothing
            # return the value of the objective function
            return J(y, v, θ)
        end
    end
end

Hv!(::LinReg, X, y) = (Hv, v) -> mul!(Hv, X', X * v)

# ----------------------- #
#  -- Ridge Regression -- #
# ----------------------- #

function fg!(glr::Ridge, X, y)
    J = obj(glr)
    λ = getscale(glr.penalty)
    (f, g, θ) -> begin
        # common computations
        v = X*θ
        if g !== nothing
            # update the gradient in place
            mul!(g, X', v .- y) # l2 loss
            g .+= λ .* θ        # l2 penalty
        end
        if f !== nothing
            # return the value of the objective function
            return J(y, v, θ)
        end
    end
end

Hv!(m::Ridge, X, y) = (Hv, v) -> (mul!(Hv, X', X * v); Hv .+= getscale(m.penalty) .* v)
