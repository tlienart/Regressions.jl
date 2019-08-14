# Regressions.jl

| [MacOS/Linux] | Coverage | Documentation |
| :-----------: | :------: | :-----------: |
| [![Build Status](https://travis-ci.org/tlienart/Regressions.jl.svg?branch=master)](https://travis-ci.org/tlienart/Regressions.jl) | [![codecov.io](http://codecov.io/github/tlienart/Regressions.jl/coverage.svg?branch=master)](http://codecov.io/github/tlienart/Regressions.jl?branch=master) | TBA |

This is a convenience package aiming to gather functionalities relating to solving generalised linear regression problems of the form

```
L(y, Xθ) + P(θ)
```

where `L` is a loss function and `P`  is a penalty function (both of those can be scaled or composed).

The core aims of this package are:

- make these regressions models "easy to call" while exploiting great packages such as [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) and [`ProximalAlgorithms.jl`](https://github.com/kul-forbes/ProximalAlgorithms.jl),
- interface with [`MLJ.jl`](https://github.com/alan-turing-institute/MLJ.jl).

While there is overlap with a number of packages which each offer specific functionalities, the hope is that this package will offer a general entry point for all of them.
If you're looking for specific functionalities/algorithms, it's probably a good idea to look at one of the packages below:

- (unmaintained) [Regression.jl](https://github.com/lindahua/Regression.jl) that was released by [Dahua Lin](https://github.com/lindahua) under the MIT license for Julia 0.4.
- [SparseRegression.jl](https://github.com/joshday/SparseRegression.jl)
- [LASSO.jl](https://github.com/JuliaStats/Lasso.jl)
- (unmaintained )[LARS.jl](https://github.com/simonster/LARS.jl)
- ...

There's also [GLM](https://github.com/JuliaStats/GLM.jl) which is more geared towards statistical analysis and does not (as far as I'm aware) offer penalised regressions.
