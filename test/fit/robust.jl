n, p = 500, 5
((X, y, θ), (X_, y1, θ1)) = generate_continuous(n, p; seed=525)

@testset "HuberReg" begin
    # No intercept
    δ = 0.01
    λ = 3.0
    hr = HuberRegression(δ, λ, fit_intercept=false)
    J = objective(hr, X, y)
    o = RobustLoss(HuberRho(δ)) + λ * L2Penalty()
    @test J(θ) == o(y, X*θ, θ)
    @test J(θ)          ≤ 10.61
    θ_newton = fit(hr, X, y, solver=Newton())
    @test J(θ_newton)   ≤ 7.71
    θ_newtoncg = fit(hr, X, y, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 7.71
    θ_lbfgs = fit(hr, X, y, solver=LBFGS())
    @test J(θ_lbfgs)    ≤ 7.71
    θ_iwls  = fit(hr, X, y, solver=IWLS())
    @test J(θ_iwls)     ≤ 7.71

    δ = 0.01
    λ = 3.0
    hr = HuberRegression(δ, λ)
    J = objective(hr, X, y1)
    o = RobustLoss(HuberRho(δ)) + λ * L2Penalty()
    @test J(θ1) == o(y1, X_*θ1, θ1)
    @test J(θ1)         ≤ 16.37
    θ_newton = fit(hr, X, y1, solver=Newton())
    @test J(θ_newton)   ≤ 10.52
    θ_newtoncg = fit(hr, X, y1, solver=NewtonCG())
    @test J(θ_newtoncg) ≤ 10.52
    θ_lbfgs = fit(hr, X, y1, solver=LBFGS())
    @test J(θ_lbfgs)    ≤ 10.52
    θ_iwls  = fit(hr, X, y1, solver=IWLS())
    @test J(θ_iwls)     ≤ 10.52
end
