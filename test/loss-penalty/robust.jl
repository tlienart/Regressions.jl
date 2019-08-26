x = randn(10)
y = randn(10)
r = x .- y

@testset "Huber Loss" begin
    δ = 0.5
    hlδ = RobustLoss(HuberRho(δ))
    @test hlδ isa RobustLoss{HuberRho{δ}}
    @test hlδ(r) == hlδ(x, y) == sum(ifelse(abs(rᵢ)≤δ, rᵢ^2/2, δ*(abs(rᵢ)-δ/2)) for rᵢ in r)
end
