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
    rr = RidgeRegression(lambda=λ, fit_intercept=false)
    β_ref = (X'X + λ*I) \ (X'y)
    @test β_ref ≈ fit(rr, X, y)

    β_cg = fit(rr, X, y; solver=CG())

    @test norm(β_cg - β_ref) / norm(β_ref) ≤ 1e-12
end
