# Regressions.jl

| [MacOS/Linux] | Coverage | Documentation |
| :------------ | :------- | :------------ |
| [![Build Status](https://travis-ci.org/tlienart/Regressions.jl.svg?branch=master)](https://travis-ci.org/tlienart/Regressions.jl) | [![codecov.io](http://codecov.io/github/tlienart/Regressions.jl/coverage.svg?branch=master)](http://codecov.io/github/tlienart/Regressions.jl?branch=master) | TBA |

This is a convenience package gathering functionalities to solve a number of generalised linear regression problems of the form

```
L(y, Xθ) + P(θ)
```

where `L` is a loss function and `P`  is a penalty function (both of those can be scaled or composed).

The core aims of this package are:

- make these regressions models "easy to call" and callable in a unified way,
- interface with [`MLJ.jl`](https://github.com/alan-turing-institute/MLJ.jl),
- high performances including in "big data" settings exploiting packages such as [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), [`IterativeSolvers.jl`](https://github.com/JuliaMath/IterativeSolvers.jl) and maybe [`ProximalAlgorithms.jl`](https://github.com/kul-forbes/ProximalAlgorithms.jl) in the future.

## Implemented

| Model                     | Formulation (⭒)              | Available solvers        | Comments |
| :------------------------ | :--------------------------- | :----------------------- | :------- |
| OLS & Ridge               | L2Loss + No/L2Penalty        | Analytical (†) or CG (‡) |          |
| Lasso & ElasticNet        | L2Loss + No/L2 + L1          | (F)ISTA (⌂)              |          |
| Logistic 0/L2             | LogisticLoss + No/L2         | Newton, Newton-CG, LBFGS | `yᵢ∈{±1}`|
| Logistic L1/ElasticNet    | LogisticLoss + No/L2 + L1    | (F)ISTA                  | `yᵢ∈{±1}`|
| Multinomial 0/L2          | MultinomialLoss + No/L2      | Newton-CG, LBFGS         |          |
| Multinomial L1/ElasticNet | MultinomialLoss + No/L2 + L1 | ISTA, FISTA              |          |

* (⭒) "No" stands for no penalty
* (†) Analytical means the solution is computed in "one shot" using the `\` solver,
* (‡) CG = conjugate gradient
* (⌂) (Accelerated) Proximal Gradient Descent

Unless otherwise specified:

* Newton-like solvers use Hager - Zhang line search (default in [`Optim.jl`]((https://github.com/JuliaNLSolvers/Optim.jl)))
* ISTA, FISTA solvers use backtracking line search and a shrinkage factor of `β=0.8`

### Current limitations

* The models are built and tested assuming `n > p`; if this doesn't hold, tricks should be employed to speed up computations; these have not been implemented yet.
* Stochastic solvers that would be appropriate for huge models have not yet been implemented.

### Possible future models

| Model                     | Formulation (⭒)              | Comments |
| :------------------------ | :--------------------------- | :------- |
| Huber 0/L2                | HuberLosss + No/L2           |  ⭒       |
| Huber L1/ElasticNet       | HuberLosss + No/L2 + L1      |  ⭒       |
| Group Lasso               | L2Loss + ∑L1 over groups     |  ⭒       |
| Adaptive Lasso            | L2Loss + weighted L1         |  ⭒ [A](http://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf) |
| LAD                       | L1Loss                       | People seem to use a simplex algorithm (Barrodale and Roberts), prox like ADMM should be ok too [G](https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html), or [F](https://link.springer.com/content/pdf/10.1155/S1110865704401139.pdf) |
| SCAD                      | L2Loss + SCAD                |  A, [B](https://arxiv.org/abs/0903.5474), [C](https://orfe.princeton.edu/~jqfan/papers/01/penlike.pdf) |
| MCP                       | L2Loss + MCP                 |  A        |
| OMP                       | L2Loss + L0Loss              |  [D](https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf) |

* (⭒) should be added soon


#### Other regression models

There are a number of other regression models that may be included in this package in the longer term but may not directly correspond to the paradigm `Loss+Penalty` introduced earlier.

In some cases it will make more sense to just use [GLM.jl](https://github.com/JuliaStats/GLM.jl).

| Model                       | Note  | Link(s)                                            |
| :-------------------------- | :---- | :------------------------------------------------- |
| LARS                        | --    |                                                    |
| Quantile Regression         | --    | [Yang et al, 2013](https://www.stat.berkeley.edu/~mmahoney/pubs/quantile-icml13.pdf), [QuantileRegression.jl](https://github.com/pkofod/QuantileRegression.jl)
| Passive Agressive           | --    | [Crammer et al, 2006](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf) |
| Orthogonal Matching Pursuit | --    | [SkL](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit) |
| Least Median of Squares     | --    | [Rousseeuw, 1984](http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/LeastMedianOfSquares.pdf) |
| Ordinal regression          | _need to figure out how they work_ | [E](https://cran.r-project.org/web/packages/pscl/vignettes/countreg.pdf)|
| Count regression            | _need to figure out how they work_ | [R](https://cran.r-project.org/web/packages/pscl/vignettes/countreg.pdf) |
| Robust M estimators         |       | [F](https://arxiv.org/pdf/1508.01967.pdf) |


## What about other packages

While the functionalities in this package overlap with a number of existing packages, the hope is that this package will offer a general entry point for all of them in a way that won't require too much thinking from an end user (similar to how someone would use the tools from `sklearn.linear_model`).
If you're looking for specific functionalities/algorithms, it's probably a good idea to look at one of the packages below:

- [SparseRegression.jl](https://github.com/joshday/SparseRegression.jl)
- [Lasso.jl](https://github.com/JuliaStats/Lasso.jl)
- [QuantileRegression.jl](https://github.com/pkofod/QuantileRegression.jl)
- (unmaintained) [Regression.jl](https://github.com/lindahua/Regression.jl)
- (unmaintained) [LARS.jl](https://github.com/simonster/LARS.jl)
- (unmaintained) [FISTA.jl](https://github.com/klkeys/FISTA.jl)

There's also [GLM.jl](https://github.com/JuliaStats/GLM.jl) which is more geared towards statistical analysis for reasonably-sized datasets and does (as far as I'm aware) lack a few key functionalities for ML such as penalised regressions or multinomial regression.

## References

* **Minka**, [Algorithms for Maximum Likelihood Regression](https://tminka.github.io/papers/logreg/minka-logreg.pdf), 2003. For a review of numerical methods for the binary Logistic Regression.
* **Beck** and **Teboulle**, [A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems](https://tinyurl.com/beck-teboulle-fista), 2009. For the ISTA and FISTA algorithms.
* **Raman** et al, [DS-MLR: Exploiting Double Separability for Scaling up DistributedMultinomial Logistic Regression](https://arxiv.org/pdf/1604.04706.pdf), 2018. For a discussion of multinomial regression.

## Dev notes

* Probit Loss --> via StatsFuns // Φ(x) (normcdf); ϕ(x) (normpdf); -xϕ(x)
* Newton, LBFGS take linesearches, seems NewtonCG doesn't
* several ways of doing backtracking (e.g. https://archive.siam.org/books/mo25/mo25_ch10.pdf); for FISTA many though see http://www.seas.ucla.edu/~vandenbe/236C/lectures/fista.pdf; probably best to have "decent safe defaults"; also this for FISTA http://150.162.46.34:8080/icassp2017/pdfs/0004521.pdf ; https://github.com/tiepvupsu/FISTA#in-case-lf-is-hard-to-find ; https://hal.archives-ouvertes.fr/hal-01596103/document; not so great https://github.com/klkeys/FISTA.jl/blob/master/src/lasso.jl ;
* https://www.ljll.math.upmc.fr/~plc/prox.pdf
* proximal QN http://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/24-prox-newton.pdf; https://www.cs.utexas.edu/~inderjit/public_papers/Prox-QN_nips2014.pdf; https://github.com/yuekai/PNOPT; https://arxiv.org/pdf/1206.1623.pdf
* group lasso http://myweb.uiowa.edu/pbreheny/7600/s16/notes/4-27.pdf

---

* LBFGSB https://www.researchgate.net/profile/Jose_Morales23/publication/220493046_Remark_on_Algorithm_778_L-BFGS-B_Fortran_Subroutines_for_Large-Scale_Bound_Constrained_Optimization/links/546ec9920cf2b5fc17607f33/Remark-on-Algorithm-778-L-BFGS-B-Fortran-Subroutines-for-Large-Scale-Bound-Constrained-Optimization.pdf
