
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

@testset "GH> EN/Lassso" begin
    λ = 3.4
    γ = 2.7
    r = LassoRegression(λ)
    fg! = R.smooth_fg!(r, X, y)
    g = similar(θ1)
    f = fg!(g, θ1)
    @test f ≈ sum(abs2.(X_*θ1 .- y))/2
    @test g ≈ X_'*(X_*θ1 .- y)

    r = ElasticNetRegression(λ, γ)
    fg! = R.smooth_fg!(r, X, y)
    g = similar(θ1)
    f = fg!(g, θ1)
    @test f ≈ sum(abs2.(X_*θ1 .- y))/2 + λ * norm(θ1)^2/2
    @test g ≈ X_' * (X_*θ1 .- y) .+ λ * θ1
end


@testset "GH> LogitL2" begin
    # fgh! without fit_intercept
    λ = 0.5
    lr = LogisticRegression(λ; fit_intercept=false)
    fgh! = R.fgh!(lr, X, y)
    θ = randn(p)
    J = objective(lr, X, y)
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
    J  = objective(lr1, X, y)
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
    X = [ 0.78843 -0.28336;
         -0.75568  0.22546;
         -0.09012  0.68069;
         -0.34437 -0.98773;
          1.09285 -0.37161 ]
    y = [1, 2, 3, 1, 3]
    # comparison sklearn // no intercept
    θ = [-0.04843, 0.99519, -0.67237, 1.08812, 0.13362, 0.77136]
    g_sk = [-0.12941349639677957, 1.033822503077806, 0.6025709048825946, -0.3233237353163467,
            -0.47315740848581506, -0.7104987677614594]
    v = [-0.28802, -0.90018, 1.48613, 1.77976, -1.06333, 1.36275]
    Hv_sk = [0.18500344954627015, -0.9109175305006267, 0.8560420396655112, 0.4034408910070668,
            -1.0410454892117813, 0.5074766394935599]
    mnr = MultinomialRegression(0.0; fit_intercept=false)
    fg! = R.fg!(mnr, X, y)
    f = fg!(0.0, nothing, θ)
    mnl = MultinomialLoss()
    @test f ≈ mnl(y, X*reshape(θ, 2, 3))
    g = similar(θ)
    fg!(nothing, g, θ)
    @test g ≈ g_sk
    hv! = R.Hv!(mnr, X, y)
    Hv = similar(θ)
    hv!(Hv, θ, v)
    @test Hv ≈ Hv_sk

    # -- with intercept
    θ1 = [0.17905, 1.91598, 1.30329, -1.03438, -1.26994, -0.38288, -0.96238, -0.47912, 0.70813]
    g_sk1 = [0.26979349788581053, 1.5036100824512861, 0.7042460191768531, 0.7282930655963799,
            -0.6529621156985143, -0.2504598040370489, -0.9980865634821908, -0.8506479667527724,
            -0.45378621513980455]
    v1 = [-1.05534, -0.12016, 0.28984, 0.81068, 0.1617, -0.82832, -0.60992, 0.74291, -0.14735]
    Hv1_sk = [-0.20815994906492608, -0.03987039776759468, 0.38689109859835424, 0.3516398288195569,
               0.0703034889451111, -0.3054162498173413, -0.14347987975463106, -0.03043309117751647,
               -0.08147484878101276]
    mnr1 = MultinomialRegression(0.0)
    fg! = R.fg!(mnr1, X, y)
    f = fg!(0.0, nothing, θ1)
    @test  f ≈ mnl(y, R.apply_X(X, θ1, 3))
    g = similar(θ1)
    fg!(nothing, g, θ1)
    @test g ≈ g_sk1
    hv! = R.Hv!(mnr1, X, y)
    Hv1 = similar(θ1)
    hv!(Hv1, θ1, v1)
    @test Hv1 ≈ Hv1_sk
end