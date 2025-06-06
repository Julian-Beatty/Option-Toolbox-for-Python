# Non-Parametric and Universal Option Implied Densities

## Overview
This repository provides an easy to use toolbox for various option-related tasks, geared especially towards estimating quantities such as the risk neutral density from many options.

## Features
- KRisk Neutral Option Implied Density extraction using Shimko (1993) smoothened-implied-volatility approach and Breeden (1973) second derivative method.
  - Supports multiple curve fitting options that are found in the literature and academia: SABR, SVI, local polynomial kernel regression, Kernel Ridge Regression and more
  - Easy extrapolation and arbitrage-free cleaning using kernel-density estimation.
- Pricing Kernels
  -Computes pricing kernel by dividing Risk Neutral Density by the real density. The real density in this case is the density from simulating returns from a GARCH model.
-Risk Neutral Moments, obtained by taking the moments from the risk neutral density (in log-return space)

## Upcoming Features
-Bakshi Moment estimators
-Volatility Surface estimation
-Risk Neutral Density estimation using mixture of log-normals

```


# Sample Plots

## Kernel Ridge Regression
Kernel Ridge Regress to estimate the IV surface.
<img width="720" alt="Local Linear Regression" src="Images/GLD KRR.png" />

## Local Linear Regression
Locally Linear Regression to estimate the IV surface.
<img width="720" alt="Local Linear Regression" src="Images/Local Linear GLD.png" />

## Quadratic Polynomial
Fitting a quadratic polynomial for the IV surface
<img width="720" alt="Local Linear Regression" src="Images/Quadratic GLD.png" />

