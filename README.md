# Option Toolbox for Python

## Overview
This repository provides an easy to use toolbox for various option-related tasks, geared especially towards estimating quantities such as the risk neutral density from many options. Supports data extracted from OptionMetrics or Bitcoin Options from OptionsDX

## Features
- **Risk-Neutral Density Estimation**
  - Implements the Shimko (1993) smoothened implied volatility approach and the Breeden-Litzenberger (1973) second derivative method.
  - Supports a variety of curve-fitting methods from academic literature:
    - SABR
    - SVI
    - Local polynomial kernel regression
    - Kernel Ridge Regression
  - Includes arbitrage-free cleaning and extrapolation using kernel density estimation.

- **Pricing Kernels**
  - Computes the pricing kernel by dividing the risk-neutral density by the physical (real-world) density.
  - The physical density is obtained via GARCH-simulated return distributions.

- **Risk-Neutral Moments**
  - Extracts statistical moments (mean, variance, skewness, etc.) from the risk-neutral density in log-return space.

## Upcoming Features
- **Bakshi et al. Moment Estimators**
  - Implementation of higher-order moment estimation based on Bakshi, Kapadia, and Madan (2003).

- **Volatility Surface Estimation**
  - Tools for constructing and visualizing arbitrage-free implied volatility surfaces.

- **Risk-Neutral Density via Mixture of Log-Normals**
  - Estimation of RND using flexible mixtures of log-normal distributions for improved fit across strikes.


# Sample Plots

## Kernel Ridge Regression
Kernel Ridge Regression is used to estimate the implied volatility (IV) surface.
<img width="720" alt="Kernel Ridge Regression" src="Images/GLD KRR.png" />

## Local Linear Regression
Locally linear regression provides a nonparametric fit to the IV surface.
<img width="720" alt="Local Linear Regression" src="Images/Local Linear GLD.png" />

## Quadratic Polynomial Fit
A quadratic polynomial is used to approximate the shape of the IV surface.
<img width="720" alt="Quadratic Polynomial Fit" src="Images/Quadratic GLD.png" />


