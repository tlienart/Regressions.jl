@testset "Constructors" begin
    glr     = GeneralizedLinearRegression()
    ols     = LinearRegression()
    ridge   = RidgeRegression(fit_intercept=false)
    lasso   = LassoRegression()
    logreg0 = LogisticRegression(penalty=:none)
    logreg1 = LogisticRegression()
    logreg2 = LogisticRegression(1.0, 2.0)
    mnreg2  = MultinomialRegression(1.0, 2.0)
    hlreg   = HuberRegression(0.5, 2.0)

    @test isa(glr.loss, L2Loss)
    @test isa(glr.penalty, NoPenalty)
    @test glr.fit_intercept

    @test isa(ols.loss, L2Loss)
    @test isa(ols.penalty, NoPenalty)
    @test ols.fit_intercept

    @test isa(ridge.loss, L2Loss)
    @test isa(ridge.penalty, ScaledPenalty{L2Penalty})
    @test !ridge.fit_intercept

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

    @test isa(mnreg2.loss, MultinomialLoss)
    @test isa(mnreg2.penalty.penalties[2], ScaledPenalty{L1Penalty})

    @test isa(hlreg.loss, HuberLoss{0.5})
    @test isa(hlreg.penalty, ScaledPenalty{L2Penalty})
    @test hlreg.fit_intercept
end
