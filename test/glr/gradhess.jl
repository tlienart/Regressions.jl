
Random.seed!(1234)
n, p = 50, 5
X = randn(n, p)
X_ = R.augment_X(X, true)
θ  = randn(p)
θ1 = randn(p+1)
y = rand(n)

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

@testset "GH> MultinL2" begin
    # comparison sklearn // no intercept
    θ = [-0.04843, 0.99519, -0.67237, 1.08812, 0.13362, 0.77136]
    X = [ 0.78843 -0.28336;
         -0.75568  0.22546;
         -0.09012  0.68069;
         -0.34437 -0.98773;
          1.09285 -0.37161 ]
    y = [1, 2, 3, 1, 3]
    mnr = MultinomialRegression(0.0; fit_intercept=false)
    fg! = R.fg!(mnr, X, y)
    f = fg!(0.0, nothing, θ)
    mnl = MultinomialLoss()
    @test f ≈ mnl(y, X*reshape(θ, 2, 3))
    g_sk = [-0.12941349639677957,
             1.033822503077806,
             0.6025709048825946,
            -0.3233237353163467,
            -0.47315740848581506,
            -0.7104987677614594]
    g = similar(θ)
    fg!(nothing, g, θ)
    @test g ≈ g_sk
end
