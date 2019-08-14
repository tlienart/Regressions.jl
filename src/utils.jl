"""
$SIGNATURES

Return nothing if the number of rows of `X` and `y` match and throws a
`DimensionMismatch` error otherwise.
"""
function check_nrows(X::Matrix, y::VecOrMat)::Nothing
	size(X, 1) == size(y, 1) && return nothing
	throw(DimensionMismatch("`X` and `y` must have the same number of rows."))
end

"""
$SIGNATURES

Throws an error if the argument (e.g. penalty scaling) is negative.
"""
check_pos(λ::Real) = λ ≥ 0 || throw(ArgumentError("Penalty scaling should be positive."))

"""
$SIGNATURES

Given a matrix `X`, append a column of ones if `fit_intercept` is true.
"""
function augment_X(X::Matrix{<:Real}, fit_intercept::Bool)
	fit_intercept || return X
	return hcat(X, ones(eltype(X), size(X, 1)))
end

# Sigmoid and log-sigmoid

const SIGMOID_64 = log(Float64(1)/eps(Float64) - Float64(1))
const SIGMOID_32 = log(Float32(1)/eps(Float32) - Float32(1))

"""
$SIGNATURES

Return the sigmoid computed in a numerically stable way:

``σ(x) = 1/(1+exp(-x))``
"""
function sigmoid(x::Float64)
	x > SIGMOID_64  && return one(x)
	x < -SIGMOID_64 && return zero(x)
	return one(x) / (one(x) + exp(-x))
end
function sigmoid(x::Float32)
	x > SIGMOID_32  && return one(x)
	x < -SIGMOID_32 && return zero(x)
	return one(x) / (one(x) + exp(-x))
end
sigmoid(x) = sigmoid(float(x))


"""
$SIGNATURES

Return the log sigmoid computed in a numerically stable way:

``logσ(x) = -log(1+exp(-x)) = log(exp(x)/(exp(x) + 1)) = x - log(1+exp(x))``
"""
function logsigmoid(x::Float64)
	x > SIGMOID_64  && return zero(x)
	x < -SIGMOID_64 && return x
	return -log1p(exp(-x))
end
function logsigmoid(x::Float32)
	x > SIGMOID_32  && return zero(x)
	x < -SIGMOID_32 && return x
	return -log1p(exp(-x))
end
logsigmoid(x) = logsigmoid(float(x))
