# Non-Parametric and Universal Option Implied Densities

## Overview
This repository provides a non parametric, and robust procedure to estimate option implied distributions. Interpolating the IV surface with Kernel Ridge Regression, and using a kernel density estimate provides well behaved, smooth, arbitrage free and model free estimates of the option implied densities. The procedure is compared to other non parametric methods such as local polynomial regression. The technical paper is also in the github.

## Features
- Kernel density estimation for implied density extraction
- Support for various smoothing techniques
- Visualization tools for density comparison
- Efficient numerical methods for calibration

## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/Non-parametric-and-universal-option-implied-densities.git
cd Non-parametric-and-universal-option-implied-densities
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

