# Using cohort-shapley

Your prediction function should return the quantity you want to explain, such as a premium in dollars.
The [math guide](math.md) explains why the choice of pairs and prediction scale matters.

The decomposition functions perform the Shapley calculation.
An adapter replaces selected features between paired observations, calls your prediction function, and detects unchanged feature groups.
`NumpyAdapter` and `PolarsAdapter` are included.

`NumpyAdapter` works with any prediction function that accepts NumPy batches and returns NumPy predictions.
To perform feature replacement in another data format, implement the [adapter interface](#adapter-interface) for that format; the decomposition functions stay the same.

## Decompose paired changes

This example fits a Poisson gradient boosting model, then compares cohorts $C_1$ and $C_2$ containing the same two entities in the same row order.
`model.predict` applies the inverse link and returns predictions in response units.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cohort_shapley import NumpyAdapter, decompose_paired

training_features = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
training_response = np.array([1.0, 2.0, 2.0, 4.0, 5.0, 8.0])
model = HistGradientBoostingRegressor(
    loss="poisson", min_samples_leaf=1, max_iter=20, early_stopping=False, random_state=0
).fit(training_features, training_response)

cohort_1 = np.array([[0.0, 0.0], [1.0, 0.0]])
cohort_2 = np.array([[1.0, 1.0], [2.0, 1.0]])
adapter = NumpyAdapter(cohort_1, cohort_2, model.predict)
result = decompose_paired(adapter, tol=1e-3)
print(result.contributions.round(1))
print(round(result.total, 1))
```

```text
[2.2 1.8]
4.0
```

For a fitted model with preprocessing, include the fitted transformations in `predict`.

## Choose the comparison

Use paired decomposition for cohorts with aligned rows, independent pairs when no pairing is specified, or the decomposition with cohort turnover to separate matched changes from entry and exit.
Cohort turnover means entry into and exit from the cohort.
The [operations table](#operations) gives the corresponding functions.

## API reference

Import the three operations, `Adapter`, `NumpyAdapter`, `Attribution`, `Component`, and `Decomposition` from `cohort_shapley`.
Array and callable type aliases are available there for annotations.
The optional Polars module also exports `PredictFrame`.

### Operations

| Function | Pair selection and weights | Result |
|---|---|---|
| `decompose_paired` | Row `i` of $C_1$ paired with row `i` of $C_2$; equal row counts; optional weights per pair | `Attribution` for the weighted mean change over the aligned rows |
| `decompose_independent_pairs` | All combinations of an observation of $C_1$ with one of $C_2$, weighted by the product of normalized cohort weights | `Attribution` for the full cohort mean change |
| `decompose_with_turnover` | Row `i` of $C_1$ paired with row `i` of $C_2$ for `i < first_n_matched`; later rows are exits and entrants; weights per observation in each cohort | `Decomposition` into matched, entry, exit, and weight change |

All operations take an `adapter` that implements the `Adapter` protocol as the first argument.
The functions use its `n_rows`, `n_feature_groups`, `evaluate`, and `differs`.
The other parameters are keyword-only.

| Parameter | Meaning |
|---|---|
| `tol` | Required largest estimated Monte Carlo standard error, in prediction units, with which a result reports convergence; compared with each group's `stderr`, or `stderr_total` for turnover |
| `first_n_matched` | Turnover only, required: number of leading rows of each cohort that form matched pairs |
| `weights=None` | One weight per aligned pair for paired decomposition; tuple of cohort weights for independent pairs and turnover |
| `batch_rows=8192` | Maximum observations passed to `evaluate` or `differs` in one call |
| `max_evals=50_000_000` | Maximum number of evaluated observations, including endpoints |
| `seed=0` | Random seed; `None` uses fresh entropy |

`max_evals` must cover the used endpoint observations and initial work for sampled comparisons; an insufficient budget raises `ValueError`.
Choose `tol` for the scale of your predictions: for a premium in dollars, `tol=0.01` asks for standard errors of at most one cent.
`batch_rows` limits prediction batches, not total memory, and results do not depend on it beyond floating-point rounding.
Retained sampling draws can require roughly 500 MB at the default evaluation budget with 25 feature groups, in addition to input storage and working arrays.

### Results

`Attribution.contributions` and `Attribution.stderr` contain one entry per feature group.
`total` is the weighted mean prediction change for the selected comparison, and the contributions sum to it up to floating-point error.

`Decomposition.matched`, `Decomposition.entry`, and `Decomposition.exit` are `Component` values.
Each has `contributions` and `stderr` with one entry per feature group, and a `total` that its contributions sum to.
`Decomposition.weight_change` is a number with no feature allocation.
The three component totals and `weight_change` sum to `Decomposition.total`.
`stderr_total` combines the independent sampling variances of the three components for each group.

Both result types report `n_evals` and `converged`.
`n_evals` counts evaluated observations, not calls to the model.
Repeated observations supplied as separate rows count as separate endpoint rows.
`converged` is true when the convergence checks pass and false when the remaining budget cannot fund another step first.
Inspect these fields and the standard errors before using estimated contributions; see [numerical accuracy](#numerical-accuracy).

### Prepare observations, weights, and feature groups

Pass the two cohorts separately to the adapter.
For `decompose_paired`, put corresponding observations in the same row positions and supply equal row counts.
Select or repeat rows before constructing the adapter when the comparison requires a subset or repeated observations.
For turnover, put the matched observations first in both cohorts, in the same order, and pass their count as `first_n_matched`.
Row `i` of $C_1$ then pairs with row `i` of $C_2$ for `i < first_n_matched`; the remaining rows of $C_1$ are exits and the remaining rows of $C_2$ are entrants.
Match each observation at most once: a key that appears twice in either cohort repeats or shifts matched rows without an error.

Weights change the average being decomposed.
For paired decomposition, `weights` contains one weight per aligned pair.
For independent pairs and turnover, `weights=(weights_1, weights_2)` contains one weight per observation in each cohort.
Omitting weights gives equal weights.
Weights must be finite and non-negative, with positive total weight for every population whose mean enters the calculation.
The functions normalize weights internally, so multiplying all weights of a population by the same positive constant leaves its mean unchanged.

Use `feature_group_of_column` on either adapter to replace related columns together.
Pass a sequence of integers, with one group index per column in column order.
For three columns, `feature_group_of_column=[0, 0, 1]` makes the first two columns group 0 and the third column group 1.
Group indices must cover consecutive integers starting at zero.
The output follows that group order.
A group's contribution can differ from the sum of its features' separate contributions because grouping changes the allowed replacement orders.
Apply the [Shapley formula](math.md#appendix-a-the-shapley-formula-for-a-change-in-mean) to the groups, with the number of groups in place of the number of features.

### Supply the prediction function

`predict` must return one finite float32 or float64 value per input observation.
It must depend only on the values in that observation, give the same prediction on repeated calls, and leave its input unchanged.
Batch-dependent normalization would change the function being explained as the library changes batch sizes.

Apply fitted preprocessing inside `predict` so that derived features use the hybrid observation's values.
For example, if the model uses an interaction between two input columns, compute that interaction after replacement.
Keep fitted preprocessing parameters fixed throughout the calculation.
A fitted scikit-learn pipeline's `predict` method can perform both transformation and prediction.

### Choose an input adapter

`NumpyAdapter(cohort_1, cohort_2, predict, feature_group_of_column=...)` accepts two-dimensional NumPy arrays with the same dtype and column count.
Each array must have at least one column and a fixed-width dtype without object references.
It preserves the dtype and stored feature bytes when constructing hybrid observations.
It compares bytes when detecting unchanged features, including signed zero and NaN payloads.
C-order, F-order, and strided source arrays are supported; hybrid arrays passed to `predict` have C order.
Keep both source arrays unchanged during the calculation.

`PolarsAdapter(cohort_1, cohort_2, predict, feature_group_of_column=...)` accepts two Polars DataFrames with the same column names, order, and dtypes.
Import it from `cohort_shapley.polars`; the top-level package works without installing Polars.
Object columns are unsupported.
Its difference detection uses Polars equality: null equals null, NaN equals NaN, and signed zeros compare equal.
The prediction function must respect those equalities.
Keep both source frames unchanged during the calculation.

This example uses a numeric and a categorical feature together as one feature group.

```python
import numpy as np
import polars as pl

from cohort_shapley import FloatArray1D, decompose_paired
from cohort_shapley.polars import PolarsAdapter


def predict(rows: pl.DataFrame) -> FloatArray1D:
    return rows.select(
        pl.col("amount") * pl.when(pl.col("category") == "B").then(2.0).otherwise(1.0)
    ).to_series().to_numpy()


cohort_1 = pl.DataFrame({"amount": [10.0], "category": ["A"]})
cohort_2 = pl.DataFrame({"amount": [15.0], "category": ["B"]})
adapter = PolarsAdapter(cohort_1, cohort_2, predict, feature_group_of_column=[0, 0])
result = decompose_paired(adapter, tol=1e-3)
np.testing.assert_allclose(result.contributions, [20.0])
np.testing.assert_allclose(result.total, 30.0 - 10.0)
```

### Include cohort turnover

`decompose_with_turnover` returns feature contributions for matched entities, entry, and exit, plus a separate total for weight change.
It requires at least one matched pair; for cohorts without matched observations, use `decompose_independent_pairs`.
The [turnover formulas](#turnover-formulas) define the weights and direction of each comparison.

Here one entity changes its predicted value from 10 to 12, an entity with prediction 20 exits, and an entity with prediction 30 enters.
The mean prediction is 15 over $C_1$ and 21 over $C_2$.

```python
import numpy as np

from cohort_shapley import (
    FloatArray1D,
    FloatArray2D,
    NumpyAdapter,
    decompose_with_turnover,
)


def predict(rows: FloatArray2D) -> FloatArray1D:
    return rows[:, 0]


cohort_1 = np.array([[10.0], [20.0]])
cohort_2 = np.array([[12.0], [30.0]])
adapter = NumpyAdapter(cohort_1, cohort_2, predict)
result = decompose_with_turnover(adapter, first_n_matched=1, tol=1e-3)
np.testing.assert_allclose(result.matched.contributions, [2.0])
np.testing.assert_allclose(result.entry.contributions, [9.0])
np.testing.assert_allclose(result.exit.contributions, [-5.0])
assert result.weight_change == 0.0
np.testing.assert_allclose(result.total, 21.0 - 15.0)
```

#### Turnover formulas

Let $f$ denote the supplied prediction function.
Write $\Delta$ for the change in weighted mean prediction from $C_1$ to $C_2$.
Unpaired observations of $C_1$ are exits, and unpaired observations of $C_2$ are entrants.

Both sets of matched observations must have positive total weight.
Index the matched entities by $i$, writing $p_i$ and $q_i$ for their observations in $C_1$ and $C_2$.
Let $\alpha_i$ and $\beta_i$ be the weights of $p_i$ and $q_i$, normalized within the matched observations of their respective cohorts.
Let $\mu_{M,1}$ and $\mu_{M,2}$ be the weighted mean predictions of those matched observations.
Let $\mu_X$ be the weighted mean prediction of exits and $\mu_E$ that of entrants.
Finally, let $s_X$ be the exits' share of the total weight of $C_1$ and $s_E$ the entrants' share of the total weight of $C_2$.

An entry or exit component with zero total weight contributes zero; omit its term below.
Subtracting the two full cohort means gives

$$
\Delta=\mu_{M,2}-\mu_{M,1}
+s_E(\mu_E-\mu_{M,2})
+s_X(\mu_{M,1}-\mu_X).
$$

The entry and exit terms compare the unmatched observations with the matched observations in their own cohort.
The remaining difference, $\mu_{M,2}-\mu_{M,1}$, includes changes in both predictions and weights.
The library splits this difference into `matched` and `weight_change`, treating the two cohorts symmetrically:

| Field | Total | Feature contributions |
|---|---|---|
| `matched` | $\sum_i(\alpha_i+\beta_i)[f(q_i)-f(p_i)]/2$ | Matched pairs weighted by $(\alpha_i+\beta_i)/2$ |
| `entry` | $s_E(\mu_E-\mu_{M,2})$ | Independent pairs from matched observations of $C_2$ to entrants, multiplied by $s_E$ |
| `exit` | $s_X(\mu_{M,1}-\mu_X)$ | Independent pairs from exits to matched observations of $C_1$, multiplied by $s_X$ |
| `weight_change` | $\sum_i(\beta_i-\alpha_i)[f(p_i)+f(q_i)]/2$ | No allocation to features |

Adding the matched and weight-change totals gives $\sum_i[\beta_i f(q_i)-\alpha_i f(p_i)]=\mu_{M,2}-\mu_{M,1}$.
The three component totals and `weight_change` therefore sum to $\Delta$.

Weight change can affect the mean even when every matched entity keeps the same features and prediction.
For example, predictions of 10 and 20 with weights (1, 1) in $C_1$ and (1, 3) in $C_2$ produce a mean change from 15 to 17.5.
The matched feature contributions are zero; `weight_change` accounts for the entire increase of 2.5.

### Adapter interface

`Adapter` is a protocol: any object with the following properties and methods satisfies it, without inheritance.
`n_rows` gives the row counts `(n_1, n_2)` of the two cohorts and stays fixed throughout decomposition.
`n_feature_groups` gives the positive number of feature groups and stays fixed throughout decomposition.

`evaluate(pairs, selections)` receives an integer array of shape `(n, 2)` and a Boolean array of shape `(n, n_feature_groups)`.
Adapter methods use shared row numbering: rows of $C_1$ come first, so row zero of $C_2$ has index `n_1`.
Either side of a pair can refer to either cohort; turnover calculations compare observations within each cohort as well as between cohorts.
For each pair, it must construct the hybrid observation using the features of the pair's second row wherever a selection is true and of its first row elsewhere, then return one deterministic finite prediction.
The return array has shape `(n,)`.
This interface permits data sources other than NumPy arrays and Polars DataFrames.

`differs(pairs)` returns a Boolean array of shape `(n, n_feature_groups)`.
A false entry promises that the feature group's values in the two rows are interchangeable without changing the prediction for every choice of values of the other feature groups.
Prediction equality for a particular pair does not establish this promise when the model has interactions.
False positives cost computation; false negatives can produce incorrect contributions.
An adapter without difference detection can return `np.ones((len(pairs), self.n_feature_groups), dtype=bool)` from `differs`.
Both methods receive read-only arrays whose contents are valid only during that call.
Keep the adapter's observations and feature groups unchanged throughout decomposition.

The supplied adapters detect unchanged feature groups using their documented equality rules.
For aligned or matched pairs, a group is unchanged if all its columns match within every positive-weight pair.
For independent comparisons, its columns must be constant across all positive-weight rows in both populations; equal distributions alone do not qualify.
Detected unchanged groups receive exactly zero contribution and standard error.

### Numerical accuracy

The library evaluates endpoint predictions to obtain the total, then computes feature contributions by exact enumeration or sampling.
It uses exact enumeration when a comparison has at most $2^{20}$ coalitions of active feature groups and its cost and the evaluation budget permit it; a comparison with at most one active feature group needs only its total.
The calculation treats a group as active unless difference detection establishes that its replacement cannot change a hybrid prediction.
For larger comparisons, the library samples pairs and sets of feature groups and fits contributions constrained to sum to the known total.

`stderr` estimates the Monte Carlo standard error of each contribution, conditional on the supplied cohorts and prediction function.
It does not measure predictive uncertainty, uncertainty from model fitting, or uncertainty from sampling the cohorts themselves.
Exact calculations have zero Monte Carlo standard error, subject to floating-point arithmetic.

A calculation reports convergence only if every feature group's estimated standard error is at most `tol`, in prediction units.
For turnover, this condition applies to the combined standard error across matched, entry, and exit.
Each sampled comparison must also have at least 960 draws, a full-rank regression, and agreement between its sampled and known totals.
A full-rank regression has enough independent equations to determine the fitted contributions.
Fewer than two draws or a rank-deficient fit produce infinite standard errors.

Even when a calculation converges, `tol` does not bound the error of a returned contribution.
Rare pairs with large offsetting contributions can make estimated standard errors too small.
Use the returned convergence status and standard errors when interpreting results; contributions summing to the known total does not establish their individual accuracy.
