Random.seed!(52)
n = 500
p = 3
X = randn(n, p)
y = randn(n)

@testset "linreg" begin
    lr = LinearRegression(fit_intercept=false)
    lr1 = LinearRegression()
    β_ref = X \ y
    @test β_ref == fit(lr, X, y)

    # fit_intercept
    β_ref = hcat(X, ones(n, 1)) \ y
    @test β_ref == fit(lr1, X, y)

    # == iterative solvers
    β_cg = fit(lr1, X, y; solver=CG())
    @test norm(β_cg - β_ref) / norm(β_ref) ≤ 1e-12
end

@testset "ridgereg" begin
    λ = 1.0
    rr  = RidgeRegression(lambda=λ, fit_intercept=false)
    rr1 = RidgeRegression(λ)
    X_  = R.augment_X(X, true)

    β_ref  = (X'X + λ*I) \ (X'y)
    β_ref1 = (X_'X_ + λ*I) \ (X_'y)
    @test β_ref ≈ fit(rr, X, y)
    @test β_ref1 ≈ fit(rr1, X, y)

    β_cg = fit(rr, X, y; solver=CG())
    β_cg1 = fit(rr1, X, y; solver=CG())

    @test norm(β_cg - β_ref) / norm(β_ref) ≤ 1e-12
    @test norm(β_cg1 - β_ref1) / norm(β_ref) ≤ 1e-12
end
