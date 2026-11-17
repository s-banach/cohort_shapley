# cohort-shapley

`cohort-shapley` decomposes the change in a model's weighted mean prediction from a cohort $C_1$ to a cohort $C_2$.

The [math guide](docs/math.md) uses car insurance premiums to show why pairing each driver's records across years can give different feature contributions from subtracting the cohorts' mean SHAP vectors for a fixed background.

See the [usage guide](docs/usage.md) for worked examples and the API reference.

Python 3.12+ and NumPy are required.
The `polars` extra supports mixed-type DataFrames.

## Development

Run `scripts/CI.sh` to lint, type-check, and test the package and documentation examples.
