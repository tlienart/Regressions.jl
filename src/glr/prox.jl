function prox!(glr::GLR{<:Loss,<:Union{L1R,CompositePenalty}}, X, y)
    γ = getscale_l1(glr.penalty)
    # TODO
end
