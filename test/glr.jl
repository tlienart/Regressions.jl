@testset "Constructors" begin
    glr     = GeneralizedLinearRegression()
    ols     = LinearRegression()
    ridge   = RidgeRegression()
    lasso   = LassoRegression()
    logreg0 = LogisticRegression(penalty=:none)
    logreg1 = LogisticRegression()
    logreg2 = LogisticRegression(1.0, 2.0)

    @test isa(glr.loss, L2Loss)
    @test isa(glr.penalty, NoPenalty)

    @test isa(ols.loss, L2Loss)
    @test isa(ols.penalty, NoPenalty)

    @test isa(ridge.loss, L2Loss)
    @test isa(ridge.penalty, ScaledPenalty{L2Penalty})

    @test isa(lasso.loss, L2Loss)
    @test isa(lasso.penalty, ScaledPenalty{L1Penalty})

    @test isa(logreg0.loss, LogisticLoss)
    @test isa(logreg0.penalty, NoPenalty)
    @test isa(logreg1.loss, LogisticLoss)
    @test isa(logreg1.penalty, ScaledPenalty{L2Penalty})
    @test isa(logreg2.loss, LogisticLoss)
    @test isa(logreg2.penalty, CompositePenalty)
    @test isa(logreg2.penalty.penalties[1], ScaledPenalty{L2Penalty})
    @test isa(logreg2.penalty.penalties[2], ScaledPenalty{L1Penalty})
end

Random.seed!(1234)
n, p = 50, 5
X = randn(n, p)
X_ = R.augment_X(X, true)
θ  = randn(p)
θ1 = randn(p+1)
y = rand(n)

@testset "Tools" begin
    lr = LogisticRegression(1.0, 2.0; fit_intercept=false)
    obj = R.objfun(lr, X, y)
    J = LogisticLoss() + L2Penalty() + 2L1Penalty()
    @test obj(θ) ≈ J(y, X*θ, θ)
    lr = LogisticRegression(1.0, 2.0; fit_intercept=true)
    obj = R.objfun(lr, X, y)
    J = LogisticLoss() + L2Penalty() + 2L1Penalty()
    @test obj(θ1) ≈ J(y, X_*θ1, θ1)
end

@testset "GH> Ridge" begin
    # with fit_intercept
    λ = 0.5
    r = RidgeRegression(λ)
    hv! = R.Hv!(r, X, y)
    v = randn(p + 1)
    hv = similar(v)
    hv!(hv, v)
    @test hv ≈ X_'*(X_*v) .+ λ * v
    # without fit_intercept
    r = RidgeRegression(λ; fit_intercept=false)
    hv! = R.Hv!(r, X, y)
    v = randn(p)
    hv = similar(v)
    hv!(hv, v)
    @test hv ≈ X'*(X*v) .+ λ * v
end

@testset "GH> LogitL2" begin
    # fgh! without fit_intercept
    λ = 0.5
    lr = LogisticRegression(λ; fit_intercept=false)
    fgh! = R.fgh!(lr, X, y)
    θ = randn(p)
    J = objfun(lr, X, y)
    f = 0.0
    g = similar(θ)
    H = zeros(p, p)
    f = fgh!(f, g, H, θ)
    @test f == J(θ)
    @test g ≈ -X' * (y .* R.σ.(-y .* (X * θ))) .+ λ .* θ
    @test H ≈ X' * (Diagonal(R.σ.(y .* (X * θ))) * X) + λ * I

    # fgh! with fit_intercept
    λ = 0.5
    lr1 = LogisticRegression(λ)
    fgh! = R.fgh!(lr1, X, y)
    θ1 = randn(p+1)
    J  = objfun(lr1, X, y)
    f1 = 0.0
    g1 = similar(θ1)
    H1 = zeros(p+1, p+1)
    f1 = fgh!(f1, g1, H1, θ1)
    @test f1 == J(θ1)
    @test g1 ≈ -X_' * (y .* R.σ.(-y .* (X_ * θ1))) .+ λ .* θ1
    @test H1 ≈ X_' * (Diagonal(R.σ.(y .* (X_ * θ1))) * X_) + λ * I

    # Hv! without  fit_intercept
    Hv! = R.Hv!(lr, X, y)
    v   = randn(p)
    Hv  = similar(v)
    Hv!(Hv, θ, v)
    @test Hv ≈ H * v

    # Hv! with fit_intercept
    Hv! = R.Hv!(lr1, X, y)
    v   = randn(p+1)
    Hv  = similar(v)
    Hv!(Hv, θ1, v)
    @test Hv ≈ H1 * v
end
