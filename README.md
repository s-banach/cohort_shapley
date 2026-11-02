# cohort-shapley

`cohort-shapley` decomposes the change in a model's weighted mean prediction from a cohort $C_1$ to a cohort $C_2$.

The mathematical "novelty", relative to what data scientists commonly do with SHAP, is explained in [docs/math.md](docs/math.md).

Usage and API reference are provided in [docs/usage.md](docs/usage.md).

Python 3.12+ and NumPy are required.
The `polars` extra supports mixed-type DataFrames.

## Development

Run `scripts/CI.sh` to lint, type-check, and test the package and documentation examples.
