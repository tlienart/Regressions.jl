# Regressions.jl

| [MacOS/Linux] | Coverage | Documentation |
| :-----------: | :------: | :-----------: |
| [![Build Status](https://travis-ci.org/tlienart/Regressions.jl.svg?branch=master)](https://travis-ci.org/tlienart/Regressions.jl) | [![codecov.io](http://codecov.io/github/tlienart/Regressions.jl/coverage.svg?branch=master)](http://codecov.io/github/tlienart/Regressions.jl?branch=master) | TBA |

This is a convenience package gathering functionalities to solve a number of generalised linear regression problems of the form

```
L(y, Xθ) + P(θ)
```

where `L` is a loss function and `P`  is a penalty function (both of those can be scaled or composed).

The core aims of this package are:

- make these regressions models "easy to call" and callable in a unified way,
- interface with [`MLJ.jl`](https://github.com/alan-turing-institute/MLJ.jl),
- high performances including in "big data" settings exploiting packages such as [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), [`ProximalAlgorithms.jl`](https://github.com/kul-forbes/ProximalAlgorithms.jl) and [`IterativeSolvers.jl`](https://github.com/JuliaMath/IterativeSolvers.jl).

## Implemented

| Model              | Formulation                  | Available solvers        | Comments |
| :----------------: | :--------------------------: | :----------------------: | :------: |
| OLS & Ridge        | L2Loss + No/L2Penalty        | Analytical (†) or CG (‡) |          |
| Lasso & ElasticNet | L2Loss + No/L2 + L1          | ISTA, FISTA              |          |
| Logistic 0/L2      | LogisticLoss + No/L2         | Newton, Newton-CG, LBFGS |          |
| Logistic ElNet     | LogisticLoss + No/L2 + L1    | ⚠ TBA ⚠                  |  ⚠⚠⚠     |
| Multinomial 0/L2   | MultinomialLoss + No/L2      | Newton-CG, LBFGS         |          |
| Multinomial Elnet  | MultinomialLoss + No/L2 + L1 | ⚠ TBA ⚠                  |  ⚠⚠⚠     |


* (†) Analytical means the solution is computed in "one shot" using the `\` solver,
* (‡) CG = conjugate gradient

## What about other packages

While the functionalities in this package overlap with a number of existing packages, the hope is that this package will offer a general entry point for all of them in a way that won't require too much thinking from an end user (similar to how someone would use the tools from `sklearn.linear_model`).
If you're looking for specific functionalities/algorithms, it's probably a good idea to look at one of the packages below:

- (unmaintained) [Regression.jl](https://github.com/lindahua/Regression.jl) that was released by [Dahua Lin](https://github.com/lindahua) under the MIT license for Julia 0.4.
- [SparseRegression.jl](https://github.com/joshday/SparseRegression.jl)
- [LASSO.jl](https://github.com/JuliaStats/Lasso.jl)
- (unmaintained )[LARS.jl](https://github.com/simonster/LARS.jl)
- ...

There's also [GLM](https://github.com/JuliaStats/GLM.jl) which is more geared towards statistical analysis and does not (as far as I'm aware) lack a few key functionalities for ML such as penalised regressions or multinomial regression.

## Dev notes

* Probit Loss --> via StatsFuns // Φ(x) (normcdf); ϕ(x) (normpdf); -xϕ(x)
* Newton, LBFGS take linesearches, seems NewtonCG doesn't
* several ways of doing backtracking (e.g. https://archive.siam.org/books/mo25/mo25_ch10.pdf); for FISTA many though see http://www.seas.ucla.edu/~vandenbe/236C/lectures/fista.pdf; probably best to have "decent safe defaults"; also this for FISTA http://150.162.46.34:8080/icassp2017/pdfs/0004521.pdf ; https://github.com/tiepvupsu/FISTA#in-case-lf-is-hard-to-find ; https://hal.archives-ouvertes.fr/hal-01596103/document; not so great https://github.com/klkeys/FISTA.jl/blob/master/src/lasso.jl ;
* https://www.ljll.math.upmc.fr/~plc/prox.pdf
* proximal QN http://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/24-prox-newton.pdf; https://www.cs.utexas.edu/~inderjit/public_papers/Prox-QN_nips2014.pdf; https://github.com/yuekai/PNOPT; https://arxiv.org/pdf/1206.1623.pdf


---

* LBFGSB https://www.researchgate.net/profile/Jose_Morales23/publication/220493046_Remark_on_Algorithm_778_L-BFGS-B_Fortran_Subroutines_for_Large-Scale_Bound_Constrained_Optimization/links/546ec9920cf2b5fc17607f33/Remark-on-Algorithm-778-L-BFGS-B-Fortran-Subroutines-for-Large-Scale-Bound-Constrained-Optimization.pdf
