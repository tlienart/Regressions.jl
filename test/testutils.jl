const R  = Regressions
const CI = get(ENV, "CI", "false") == "true"

SKLEARN = DO_SKLEARN && !CI
SKLEARN && using PyCall
SK_LM = SKLEARN ? pyimport("sklearn.linear_model") : nothing

m(s) = println("\n== $s ==\n")
