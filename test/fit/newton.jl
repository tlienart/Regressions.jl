Random.seed!(52)
n = 500
p = 5
X = randn(n, p)
X_= R.augment_X(X, true)
θ = randn(p)
θ1= randn(p+1)
y = rand(n) .< R.σ.(X*θ)
y = y .* ones(Int, n) .- .!y .* ones(Int, n)
y1 = rand(n) .< R.σ.(X_*θ1)
y1 = y1 .* ones(Int, n) .- .!y1 .* ones(Int, n)

@testset "Logreg" begin
    # No intercept
    λ = 5.0
    lr = LogisticRegression(λ; fit_intercept=false)
    J  = objfun(lr, X, y)
    o  = LogisticLoss() + λ * L2Penalty()
    @test J(θ) == o(y, X*θ, θ)
    @test J(θ) ≤ 282.1

    θ_newton = fit(lr, X, y, solver=Newton())
    @test J(θ_newton) ≤ 280.4

    θ_newtoncg = fit(lr, X, y, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 280.4

    θ_lbfgs = fit(lr, X, y, solver=R.LBFGS())
    @test J(θ_lbfgs) ≤ 280.4
end
