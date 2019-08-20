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
    @test J1(θ_lasso)      ≤ 244.02
    θ_lasso_ista = fit(lr1, X, y1, solver=ISTA())
    @test J1(θ_lasso_ista) ≤ 244.02

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
    θ_ref = X_ \ y1
    @test J(θ_ref)     ≤ 5_803
    θ_en = fit(enr, X, y1)
    @test J(θ_en)      ≤ 5_586.83
    θ_en_ista = fit(enr, X, y1, solver=ISTA())
    @test J(θ_en_ista) ≤ 5_586.87

    if SKLEARN
        enr_sk = SK_LM.ElasticNet(alpha=α, l1_ratio=ρ)
        enr_sk.fit(X, y1)
        θ_sk = vcat(enr_sk.coef_[:], enr_sk.intercept_)
        J(θ_sk)    ≤ 5_588.23
    end
end

Random.seed!(52551)
n  = 500
p  = 100
X  = randn(n, p)
X_ = R.augment_X(X, true)
θ  = randn(p+1) .* (rand(p+1) .< 0.1)
y  = rand(n) .< R.σ.(X_*θ)
y  = y .* ones(Int, n) .- .!y .* ones(Int, n)

@testset "Logreg/EN" begin
    ρ = 0.8
    α = 0.3
    λ = α * (1 - ρ) * n
    γ = α * ρ * n

    enlr = LogisticRegression(λ, γ)
    J  = objective(enlr, X, y)
    @test J(θ)        ≤ 1_111.95
    θ_fista = fit(enlr, X, y)
    @test J(θ_fista)  ≤ 346.6
    θ_ista  = fit(enlr, X, y, solver=ISTA())
    @test J(θ_ista)   ≤ 346.6

    # pure l1 regularization (not clear how sklearn does the mixing of l1/l2 for logreg)
    enlr = LogisticRegression(γ; penalty=:l1)
    J    = objective(enlr, X, y)
    @test J(θ)        ≤ 1_002.33
    θ_fista = fit(enlr, X, y)
    @test J(θ_fista)  ≤ 346.6

    if SKLEARN
        # Note: this algorithm is stochastic
        PY_RND.seed(1531)
        enlr_sk = SK_LM.LogisticRegression(penalty="elasticnet", C=1.0/γ, l1_ratio=1, solver="saga")
        enlr_sk.fit(X, y)
        θ_sk = vcat(enlr_sk.coef_[:], enlr_sk.intercept_)
        @test J(θ_sk) ≤ 355.0 # sometimes will do better
    end
end
