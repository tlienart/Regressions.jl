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

| Model            | Formulation                    | Available solvers        | Comments |
| :--------------: | :----------------------------: | :----------------------: | :------: |
| OLS & Ridge      | L2Loss + No/L2Penalty          | Analytical (†) or CG (‡) |          |
| Logistic 0/L2    | LogisticLoss + No/L2Penalty    | Newton, Newton-CG, LBFGS |          |
| Multinomial 0/L2 | MultinomialLoss + No/L2Penalty | Newton-CG, LBFGS         |          |

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
