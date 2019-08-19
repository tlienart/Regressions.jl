Random.seed!(512)
n  = 500
p  = 100
X  = randn(n, p)
X_ = R.augment_X(X, true)
θ  = randn(p) .* (rand(p) .< 0.1)
θ1 = randn(p+1) .* (rand(p+1) .< 0.1)
y  = R.apply_X(X, θ) .+ 0.1 .* randn(n)
y1 = R.apply_X(X, θ1) .+ 0.1 .* randn(n)

@testset "lasso" begin
    λ   = 50
    lr  = LassoRegression(λ; fit_intercept=false)
    J   = objective(lr, X, y)
    # no intercept
    θ_ref = X \ y
    @test J(θ_ref)         ≤ 675.2
    θ_lasso = fit(lr, X, y)
    @test J(θ_lasso)       ≤ 626.11

    θ_lasso_ista = fit(lr, X, y, solver=ISTA())
    @test J(θ_lasso_ista)  ≤ 626.11
    # with intercept
    lr1 = LassoRegression(λ)
    J1  = objective(lr1, X, y1)
    θ_ref = X_ \ y1
    @test J1(θ_ref)        ≤ 286.5
    θ_lasso = fit(lr1, X, y1)
    @test J1(θ_lasso)      ≤ 244.1
    θ_lasso_ista = fit(lr1, X, y1, solver=ISTA())
    @test J1(θ_lasso_ista) ≤ 244.1

    if SKLEARN
        lr_sk = SK_LM.Lasso(alpha=λ/n)
        lr_sk.fit(X, y1)
        θ1_sk = vcat(lr_sk.coef_[:], lr_sk.intercept_)
        @test J1(θ1_sk) ≤ 245.1
    end
end

@testset "elnet" begin
    # our elastic net doesn't require to attach λ and γ
    ρ = 0.3
    α = 0.1
    λ = α * (1 - ρ) * n
    γ = α * ρ * n
    enr = ElasticNetRegression(λ, γ)
    J = objective(enr, X, y)
    θ_ref = X \ y1
    @test J(θ_ref)     ≤ 5_803
    θ_en = fit(enr, X, y1)
    @test J(θ_en)      ≤ 5_586.76
    θ_en_ista = fit(enr, X, y1, solver=ISTA())
    @test J(θ_en_ista) ≤ 5_586.89

    if SKLEARN
        enr_sk = SK_LM.ElasticNet(alpha=α, l1_ratio=ρ)
        enr_sk.fit(X, y1)
        θ_sk = vcat(enr_sk.coef_[:], enr_sk.intercept_)
        J(θ_sk)    ≤ 5_588.23
    end
end
