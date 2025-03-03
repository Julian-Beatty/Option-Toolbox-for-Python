# Non-Parametric and Universal Option Implied Densities

## Overview
This repository provides a non-parametric approach to estimating risk-neutral densities from option prices. It implements smoothing techniques to extract implied distributions from observed option market data.

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


## Sample Plots
## Kernel Ridge Regression
Kernel Ridge Regress to estimate the IV surface.
<img width="720" alt="Local Linear Regression" src="Images/GLD KRR.png" />
## Local Linear Regression
Locally Linear Regression to estimate the IV surface.
<img width="720" alt="Local Linear Regression" src="Images/Local Linear GLD.png" />
## Quadratic Polynomial
Fitting a quadratic polynomial for the IV surface
<img width="720" alt="Local Linear Regression" src="Images/Quadratic GLD.png" />
## Lowess
Lowess to estimate the surface
<img width="720" alt="Local Linear Regression" src="Images/Lowess GLD.png" />
