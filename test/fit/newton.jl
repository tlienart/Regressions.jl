Random.seed!(52)
n  = 500
p  = 5
X  = randn(n, p)
X_ = R.augment_X(X, true)
θ  = randn(p)
θ1 = randn(p+1)
y  = rand(n) .< R.σ.(X*θ)
y  = y .* ones(Int, n) .- .!y .* ones(Int, n)
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

Random.seed!(525)
n  = 500
p  = 5
c  = 3
X  = randn(n, p)
X_ = R.augment_X(X, true)
θ  = randn(p * c)
θ1 = randn((p+1) * c)

y = zeros(Int, n)
y1 = zeros(Int, n)

P = R.apply_X(X, θ, c)
M = exp.(P)
Mn = M ./ sum(M, dims=2)
P1 = R.apply_X(X, θ1, c)
M1 = exp.(P1)
Mn1 = M1 ./ sum(M1, dims=2)

begin
    k1 = rand(n) .< Mn[:, 1] # case 1?
    k2 = rand(n) .< Mn[:, 2] ./ (Mn[:, 2] .+ Mn[:, 3]) # case 2?

    y = zeros(Int, n)
    y[k1] .= 1
    y[.!k1 .& k2] .= 2
    y[y .== 0] .= 3

    k11 = rand(n) .< Mn1[:, 1]
    k21 = rand(n) .< Mn1[:, 2] ./ (Mn1[:, 2] .+ Mn1[:, 3])

    y1 = zeros(Int, n)
    y1[k11] .= 1
    y1[.!k11 .& k21] .= 2
    y1[y1 .== 0] .= 3
end

@testset "Multinomial" begin
    # No intercept
    λ = 5.0
    mnr = MultinomialRegression(λ; fit_intercept=false)
    J  = objfun(mnr, X, y; c=c)
    @test J(θ) ≤ 370.5

    θ_newtoncg = fit(mnr, X, y, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 334.3

    θ_lbfgs = fit(mnr, X, y, solver=R.LBFGS())
    @test J(θ_lbfgs) ≤ 334.3

    #  With intercept
    λ = 5.0
    mnr = MultinomialRegression(λ)
    J  = objfun(mnr, X, y1; c=c)
    @test  J(θ1) ≤ 315.7

    θ_newtoncg = fit(mnr, X, y1, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 300.7

    θ_lbfgs = fit(mnr, X, y1, solver=R.LBFGS())
    @test J(θ_lbfgs) ≤ 300.7
end
