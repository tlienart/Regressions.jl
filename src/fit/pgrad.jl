# Proximal gradient methods

# Assumption: loss has gradient; penalty has prox e.g.: Lasso
# function _fit(glr::GLR, pgd::ProxGD, X, y)
#
#     X_ = augment_X(X, glr.fit_intercept)
#     p  = size(X_, 2)
#     λ  = getscale(glr.penalty)
#     # TODO: have a hasgrad/hasprox for loss/penalty to ensure ok
#     # prox( θ - α ∇L(θ) )
#     ∇L   = grad(glr.loss, X_, y)
#     Π    = prox(glr.penalty)
#     η    = ifelse(pgd.backtrack, 1.0, pgd.stepsize)
#     θ0   = zeros(p)
#     θ    = zeros(p)
#     Δθ   = zeros(p)
#     θ̂    = zeros(p)
#     ∇Lθ  = zeros(p)
#     iter =  0
#     tol  = Inf
#     while iter < pgd.max_iter && tol > pgd.tol
#         Δθ .= -θ
#         # ---
#         if pgd.backtrack
#             ∇Lθ   .= ∇L(θ) # store the current gradient
#             cond   = true
#             inner  = 0
#             while cond && inner < pgd.max_inner
#                 # candidate prox step
#                 θ̂ .= Π(θ .- η * ∇Lθ, η)
#                 # form the left and right hand side of the backtracking condition
#                 # http://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/08-prox-grad-scribed.pdf 8.1.5
#                 lhs  = glr.loss(θ, θ̂)
#                 rhs  = glr.loss(θ, θ0) - dot(∇Lθ, θ̂) + sum(abs2.(θ̂))/(2η)
#                 cond = lhs > rhs
#                 (η *= pgd.stepshrink)
#                 inner += 1
#             end
#             # take the step
#             θ .= θ̂
#         else
#             θ .= Π(θ .- η * ∇L(θ), η)
#         end
#         # ---
#         Δθ .+= θ
#         tol   = norm(Δθ) / (norm(θ) + eps())
#         iter += 1
#     end
#     if tol > pgd.tol
#         @warn "Proximal GD did not converge in $(pgd.max_iter)."
#     end
#     return θ
# end
