function is_elnet(cp::CompositePenalty)
    length(cp.penalties) == 2 || return false
    isa(cp.penalties[1], L2R) && isa(cp.penalties[2], L1R) && return true
    isa(cp.penalties[1], L1R) && isa(cp.penalties[2], L2R) && return true
    return false
end

get_l1(cp::CompositePenalty) = cp.penalties[findfirst(e->isa(e, L1R), cp.penalties)]
get_l2(cp::CompositePenalty) = cp.penalties[findfirst(e->isa(e, L2R), cp.penalties)]

getscale_l1(cp::CompositePenalty) = cp |> get_l1 |> getscale
getscale_l2(cp::CompositePenalty) = cp |> get_l2 |> getscale
