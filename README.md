# Romberg

Numerical integration experiments for 2D functions on `[0,1] x [0,1]` using Romberg-style extrapolation.

This repository compares two grid strategies:
- rectangular subdivision with a 2D trapezoidal rule
- triangular subdivision with midpoint integration

It also includes benchmarking across multiple test functions and visualization output for convergence and mesh refinement.

## What is included

- `romberg.py`: 2D trapezoidal Romberg integration with saved 2D/3D plots.
- `romberg2.py`: triangular subdivision + midpoint integration with Romberg extrapolation.
- `romberg3.py`: side-by-side accuracy and runtime comparison on several test functions.
- `visual.py`: 3D visualization of the test functions.
- `output/`: generated figures (mesh/refinement/3D/convergence visuals).
- `report.pdf`: project report.

## Requirements

Install dependencies:

```bash
pip install numpy scipy matplotlib pandas
```

## Run

```bash
python romberg.py
python romberg2.py
python romberg3.py
python visual.py
```

Generated images are written to `output/` (for scripts that save plots).

## Project focus

- compare numerical behavior of trapezoidal vs triangular partitioning
- inspect error convergence under refinement
- evaluate runtime/accuracy trade-offs on smooth and oscillatory surfaces

## Notes

- Current scripts are configured for the unit square integration domain.
- Exact reference values are computed with SciPy (`dblquad`) for comparison.
