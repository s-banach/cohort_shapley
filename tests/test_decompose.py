import numpy as np
import polars as pl
import pytest

from cohort_shapley._accounting import ATTRIBUTED_COMPONENTS
from cohort_shapley._arrays import FloatArray1D
from cohort_shapley.polars import PredictFrame
from tests.cohorts import cohorts, features, nonlinear
from tests.oracle import oracle_contributions
from tests.preparation import decompose_frames


def within_se(error: np.ndarray, stderr: np.ndarray) -> bool:
    """Whether every error is within 3 standard errors, allowing for rounding when SE is 0."""
    return bool(np.all(np.abs(error) <= 3 * stderr + 1e-12))


def test_component_totals_and_contributions_sum_exactly() -> None:
    d_1, d_2 = cohorts(0, n_matched=2000, n_exits=300, n_entrants=400, n_features=5)
    result = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, id_column="id", weight_column="w", tol=2e-3
    )
    assert sum(result.component_totals.values()) == pytest.approx(result.total, rel=1e-12)
    for k, component in enumerate(ATTRIBUTED_COMPONENTS):
        assert result.contributions[:, k].sum() == pytest.approx(
            result.component_totals[component], rel=1e-9
        )
    assert result.converged
    assert result.stderr_total.max() <= 2e-3


def test_exhaustive_pairs_and_coalitions_match_brute_force() -> None:
    """The matched game enumerates from the start; the entry and exit games switch to enumeration once sampling them would cost more."""
    d_1, d_2 = cohorts(2, n_matched=40, n_exits=5, n_entrants=4, n_features=4)
    result = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, id_column="id", weight_column="w", tol=0.0
    )
    expected = oracle_contributions(
        nonlinear, d_1, d_2, features(4), id_column="id", weight_column="w"
    )
    np.testing.assert_allclose(result.contributions, expected, rtol=0, atol=1e-10)
    assert result.stderr_total.max() == 0
    assert result.converged


def test_null_feature_group_gets_zero_in_matched_component() -> None:
    d_1, d_2 = cohorts(4, n_matched=1500, n_exits=20, n_entrants=20, n_features=10)
    d_2 = d_2.with_columns(x9=pl.concat([d_1["x9"].head(1500), d_2["x9"].tail(20)]))
    result = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, id_column="id", weight_column="w", tol=0.02
    )
    assert result.contributions[9, 0] == 0.0
    assert result.stderr[9, 0] == 0.0
    assert np.all(result.stderr[:9, 0] > 0)


def test_unit_weights_equal_in_both_periods_give_zero_weight_change() -> None:
    """Weights vary by unit; each matched unit keeps its weight, and only entrants and exits change the cohort weight sums.
    The last exit holds the largest weight of either cohort.
    """
    d_1, d_2 = cohorts(5, n_matched=500, n_exits=50, n_entrants=50, n_features=3, weighted=False)
    weights = np.append(np.random.default_rng(5).uniform(1, 3, 549), 7.0)
    unit_weight = pl.DataFrame({"id": d_1["id"], "w": weights})
    d_1 = d_1.join(unit_weight, on="id")
    d_2 = d_2.join(unit_weight, on="id", how="left").with_columns(pl.col("w").fill_null(1.0))
    result = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, id_column="id", weight_column="w"
    )
    assert result.component_totals["weight_change"] == 0.0


def test_integer_weights_match_duplicated_rows() -> None:
    d_1, d_2 = cohorts(6, n_matched=400, n_exits=60, n_entrants=80, n_features=4, weighted=False)
    rng = np.random.default_rng(6)
    unit_weight = pl.DataFrame({"id": d_1["id"], "w": rng.integers(1, 4, d_1.height)})
    d_1 = d_1.join(unit_weight, on="id")
    entry_weight = pl.Series("w", rng.integers(1, 4, d_2.height))
    d_2 = d_2.join(unit_weight, on="id", how="left").with_columns(
        pl.col("w").fill_null(entry_weight)
    )

    def duplicate(d: pl.DataFrame) -> pl.DataFrame:
        copies = d.select(pl.all().repeat_by("w").explode())
        return copies.with_columns(id=pl.col("id") * 10 + pl.int_range(pl.len()).over("id"))

    weighted = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, id_column="id", weight_column="w", tol=3e-3
    )
    duplicated = decompose_frames(
        nonlinear,
        duplicate(d_1),
        duplicate(d_2),
        batch_rows=8192,
        id_column="id",
        features=features(4),
        tol=3e-3,
        seed=1,
    )
    assert duplicated.total == pytest.approx(weighted.total, rel=1e-12)
    for component in ("matched", "entry", "exit", "weight_change"):
        assert duplicated.component_totals[component] == pytest.approx(
            weighted.component_totals[component], rel=1e-9, abs=1e-12
        )
    assert within_se(
        weighted.contributions - duplicated.contributions,
        np.hypot(weighted.stderr, duplicated.stderr),
    )


def test_no_matches_agree_with_interventional_shap_on_d_1_background() -> None:
    d_1, d_2 = cohorts(7, n_matched=0, n_exits=300, n_entrants=300, n_features=4)
    result = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, weight_column="w", features=features(4), tol=0.01
    )
    expected = oracle_contributions(nonlinear, d_1, d_2, features(4), weight_column="w")
    assert result.component_totals == {
        "matched": 0.0,
        "entry": result.total,
        "exit": 0.0,
        "weight_change": 0.0,
    }
    np.testing.assert_array_equal(result.contributions[:, [0, 2]], 0.0)
    assert np.all(result.stderr[:, 1] > 0)
    assert within_se(result.contributions[:, 1] - expected[:, 1], result.stderr[:, 1])


def test_rare_changes_are_attributed_exactly() -> None:
    """Only 5 of 3000 matched units change; the matched game holds only those 5 and enumerates."""
    d_1, d_2 = cohorts(9, n_matched=3000, n_exits=0, n_entrants=0, n_features=4)
    d_2 = d_2.with_columns(
        pl.when(pl.col("id") < 5).then(pl.col(c)).otherwise(d_1[c]) for c in features(4)
    )
    result = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, id_column="id", weight_column="w", tol=1e-9
    )
    expected = oracle_contributions(
        nonlinear, d_1, d_2, features(4), id_column="id", weight_column="w"
    )
    np.testing.assert_allclose(result.contributions, expected, rtol=0, atol=1e-12)
    assert result.component_totals["entry"] == result.component_totals["exit"] == 0.0
    assert result.stderr_total.max() == 0
    assert result.converged


def test_rare_large_changes_among_small_ones_are_drawn() -> None:
    """5 of 3000 units change three features by about 3, and every unit changes x2 by 0.01."""
    rng = np.random.default_rng(3)
    x_1 = rng.normal(size=(3000, 3))
    x_2 = x_1.copy()
    x_2[rng.choice(3000, 5, replace=False)] += rng.normal(3, 1, size=(5, 3))
    x_2[:, 2] += 0.01
    d_1, d_2 = (
        pl.DataFrame({"id": np.arange(3000), **{f"x{k}": x[:, k] for k in range(3)}})
        for x in (x_1, x_2)
    )

    def predict(frame: pl.DataFrame) -> FloatArray1D:
        x = frame.to_numpy()
        return np.exp(0.3 * x[:, 0] + 0.2 * x[:, 1] + 0.1 * x[:, 2] + 0.1 * x[:, 0] * x[:, 1])

    exact = decompose_frames(predict, d_1, d_2, batch_rows=8192, id_column="id", tol=0.0)
    result = decompose_frames(predict, d_1, d_2, batch_rows=8192, id_column="id", tol=2e-3)
    assert exact.stderr_total.max() == 0
    assert result.converged
    assert np.abs(result.contributions.sum(axis=1) - exact.contributions.sum(axis=1)).max() <= 2e-3


def _one_entrant_differs(n: int, offset: float) -> tuple[PredictFrame, pl.DataFrame, pl.DataFrame]:
    """n matched units and n entrants with x = y = 0, except one entrant with x = y = 1.
    A small sample of entry pairs then sees totals of exactly zero, while the entry total is 1000 / n.
    """
    d_1 = pl.DataFrame({"id": np.arange(n), "x": np.zeros(n), "y": np.zeros(n)})
    moved = (np.arange(2 * n) == 2 * n - 1).astype(np.float64)
    d_2 = pl.DataFrame({"id": np.arange(2 * n), "x": moved, "y": moved})

    def predict(frame: pl.DataFrame) -> FloatArray1D:
        return offset + 1e3 * (frame["x"] + frame["y"]).to_numpy()

    return predict, d_1, d_2


@pytest.mark.parametrize("offset", [0.0, 1e6])
def test_draws_that_miss_the_total_do_not_converge(offset: float) -> None:
    n = 50_000
    predict, d_1, d_2 = _one_entrant_differs(n, offset)
    result = decompose_frames(
        predict, d_1, d_2, batch_rows=8192, id_column="id", max_evals=3 * n + 2_000, seed=1
    )
    assert not result.converged
    assert result.stop_reason == "max_evals"
    assert result.contributions[:, 1].sum() == pytest.approx(
        result.component_totals["entry"], rel=1e-9
    )


def test_game_with_one_non_null_feature_group_is_exact() -> None:
    """Among the entry pairs only x differs, in one rare row; y is null, so x gets the total."""
    predict, d_1, d_2 = _one_entrant_differs(50_000, 0.0)
    d_2 = d_2.with_columns(y=pl.lit(0.0))
    full_rows = d_1.height + d_2.height
    result = decompose_frames(
        predict, d_1, d_2, batch_rows=8192, id_column="id", max_evals=full_rows
    )
    assert result.contributions[:, 1].tolist() == [result.component_totals["entry"], 0.0]
    assert result.stderr_total.max() == 0
    assert result.n_evals == full_rows
    assert result.converged


def test_draws_that_miss_the_total_grow_until_they_see_the_change() -> None:
    predict, d_1, d_2 = _one_entrant_differs(2_000, 0.0)
    result = decompose_frames(predict, d_1, d_2, batch_rows=8192, id_column="id", seed=1)
    assert result.converged
    assert result.contributions[:, 1].sum() == pytest.approx(
        result.component_totals["entry"], rel=1e-9
    )


def test_constant_predictions_give_exactly_zero() -> None:
    d_1, d_2 = cohorts(12, n_matched=1000, n_exits=50, n_entrants=50, n_features=3)
    result = decompose_frames(
        lambda frame: np.full(frame.height, 1e12),
        d_1,
        d_2,
        batch_rows=8192,
        id_column="id",
        weight_column="w",
    )
    assert result.total == 0.0
    assert set(result.component_totals.values()) == {0.0}
    np.testing.assert_array_equal(result.contributions, 0.0)


def test_heavy_entrants_are_attributed_against_the_matched_rows() -> None:
    """Entrants weigh 10^4 times the matched rows.
    Pairing entrants with all of d_2 would make almost every pair an entrant paired with an entrant, whose game has Shapley values of zero but adds noise.
    """
    d_1, d_2 = cohorts(13, n_matched=50, n_exits=0, n_entrants=500, n_features=3, weighted=False)
    d_1 = d_1.with_columns(w=pl.lit(1.0))
    d_2 = d_2.with_columns(w=pl.Series(np.r_[np.ones(50), np.full(500, 1e4)]))
    result = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, id_column="id", weight_column="w", tol=0.01, seed=1
    )
    expected = oracle_contributions(
        nonlinear, d_1, d_2, features(3), id_column="id", weight_column="w"
    )
    assert result.converged
    assert result.n_evals <= 100_000
    assert within_se(result.contributions[:, 1] - expected[:, 1], result.stderr[:, 1])


@pytest.mark.parametrize(
    ("n_1", "extra_row_2"),
    [(1, None), (200_000, None), (1, (1e15, 0.0, 0.0))],
    ids=["plain", "large_d_1", "zero_weight_outlier"],
)
def test_a_missed_total_below_tol_still_blocks_convergence(
    n_1: int, extra_row_2: tuple[float, float, float] | None
) -> None:
    """One entrant with weight 1e-7 moves x by 1e12 and y by -1e12 + 1e6; the first round misses it, and the gap of about 1e-4 between the sampled and the full-data total, below tol, is not rounding.
    Neither many identical d_1 rows nor a zero-weight row with a large prediction may make it count as rounding.
    """
    d_1 = pl.DataFrame({"x": np.zeros(n_1), "y": np.zeros(n_1), "w": np.ones(n_1)})
    d_2 = pl.DataFrame(
        {
            "x": np.r_[np.zeros(1000), 1e12],
            "y": np.r_[np.ones(1000), -1e12 + 1e6],
            "w": np.r_[np.ones(1000), 1e-7],
        }
    )
    if extra_row_2 is not None:
        d_2 = pl.concat([d_2, pl.DataFrame([extra_row_2], schema=d_2.schema, orient="row")])

    def predict(frame: pl.DataFrame) -> FloatArray1D:
        return (frame["x"] + frame["y"]).to_numpy()

    max_evals = d_1.height + d_2.height + 1920
    result = decompose_frames(
        predict, d_1, d_2, batch_rows=8192, weight_column="w", tol=1e-3, max_evals=max_evals
    )
    assert not result.converged


def test_a_large_unchanged_prediction_keeps_the_precision_of_small_changes() -> None:
    """Unit 0 predicts 1e17 in both cohorts; the other 100 units change their prediction from 0 to 1."""
    n = 101
    d_1 = pl.DataFrame({"id": range(n), "x": np.r_[1e17, np.zeros(n - 1)], "y": np.zeros(n)})
    d_2 = pl.DataFrame({"id": range(n), "x": np.r_[1e17, np.ones(n - 1)], "y": np.zeros(n)})

    def predict(frame: pl.DataFrame) -> FloatArray1D:
        return (frame["x"] + frame["y"]).to_numpy()

    result = decompose_frames(predict, d_1, d_2, batch_rows=8192, id_column="id")
    assert result.component_totals["matched"] == pytest.approx(100 / 101, rel=1e-15, abs=0)
    assert result.contributions[:, 0] == pytest.approx([100 / 101, 0.0], rel=1e-15, abs=0)


def test_rows_passed_to_predict_stay_within_max_evals() -> None:
    d_1, d_2 = cohorts(0, n_matched=20000, n_exits=300, n_entrants=300, n_features=4)
    rows = 0

    def counting(frame: pl.DataFrame) -> FloatArray1D:
        nonlocal rows
        rows += frame.height
        return nonlinear(frame)

    max_evals = d_1.height + d_2.height + 50_000
    result = decompose_frames(
        counting,
        d_1,
        d_2,
        batch_rows=8192,
        id_column="id",
        weight_column="w",
        tol=1e-6,
        max_evals=max_evals,
    )
    assert rows == result.n_evals <= max_evals
    assert result.stop_reason == "max_evals"


def test_small_budget_shrinks_the_first_round_and_stops() -> None:
    """1,000 rows fund 500 draws, fewer than the 960 a sampled game needs to converge."""
    d_1, d_2 = cohorts(14, n_matched=2000, n_exits=0, n_entrants=0, n_features=3, weighted=False)
    max_evals = d_1.height + d_2.height + 1000
    result = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, id_column="id", max_evals=max_evals
    )
    assert result.stop_reason == "max_evals"
    assert result.n_evals == max_evals
    assert np.all(np.isfinite(result.stderr_total) & (result.stderr_total > 0))


def test_a_singular_fit_gives_infinite_standard_errors_to_its_active_feature_groups() -> None:
    """One draw cannot separate 3 active feature groups; the null feature group x3 keeps a zero."""
    d_1, d_2 = cohorts(15, n_matched=2000, n_exits=0, n_entrants=0, n_features=4)
    d_2 = d_2.with_columns(x3=d_1["x3"])
    max_evals = d_1.height + d_2.height + 2
    result = decompose_frames(
        nonlinear, d_1, d_2, batch_rows=8192, id_column="id", weight_column="w", max_evals=max_evals
    )
    assert result.stop_reason == "max_evals"
    assert result.stderr[:, 0].tolist() == [np.inf, np.inf, np.inf, 0.0]
    assert result.stderr_total.tolist() == [np.inf, np.inf, np.inf, 0.0]
    assert result.contributions[3, 0] == 0.0
    assert result.contributions[:, 0].sum() == pytest.approx(
        result.component_totals["matched"], rel=1e-9
    )


def test_budget_without_room_for_one_draw_per_game_raises() -> None:
    d_1, d_2 = cohorts(0, n_matched=2000, n_exits=0, n_entrants=0, n_features=4)
    with pytest.raises(ValueError, match="the first estimation round needs at least 2"):
        decompose_frames(
            nonlinear,
            d_1,
            d_2,
            batch_rows=8192,
            id_column="id",
            weight_column="w",
            max_evals=d_1.height + d_2.height + 1,
        )


def _all_units_move(n: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """n matched units whose x and y both move from 0 to 1."""
    d_1 = pl.DataFrame({"id": np.arange(n), "x": np.zeros(n), "y": np.zeros(n)})
    return d_1, d_1.with_columns(x=pl.lit(1.0), y=pl.lit(1.0))


def _sum(frame: pl.DataFrame) -> FloatArray1D:
    return frame.to_numpy().sum(axis=1)


def test_identical_rows_without_matches_cost_only_the_full_data_predictions() -> None:
    """Every row has x = y = 0, so every feature group is null in the one game, whose total is 0."""
    d_1 = pl.DataFrame({"x": np.zeros(3), "y": np.zeros(3), "w": [1.0, 2.0, 3.0]})
    d_2 = pl.DataFrame({"x": np.zeros(4), "y": np.zeros(4), "w": [1.0, 1.0, 5.0, 1.0]})
    result = decompose_frames(_sum, d_1, d_2, batch_rows=8192, weight_column="w", max_evals=7)
    assert result.converged
    assert result.n_evals == 7
    np.testing.assert_array_equal(result.contributions, 0.0)
    np.testing.assert_array_equal(result.stderr, 0.0)


def test_unchanged_matched_units_without_other_games_give_zero_attributions() -> None:
    d_1 = pl.DataFrame({"id": np.arange(5), "x": np.arange(5.0), "y": np.zeros(5)})
    d_2 = d_1.with_columns(w=pl.Series([1.0, 1.0, 1.0, 1.0, 9.0]))
    result = decompose_frames(
        _sum,
        d_1.with_columns(w=pl.lit(1.0)),
        d_2,
        batch_rows=8192,
        id_column="id",
        weight_column="w",
    )
    assert result.converged
    assert result.n_evals == 10
    assert result.component_totals["weight_change"] != 0
    np.testing.assert_array_equal(result.contributions, 0.0)
    np.testing.assert_array_equal(result.stderr_total, 0.0)


def test_first_round_reserves_one_draw_per_game_before_enumerating() -> None:
    """The matched, entry and exit games have 2 feature groups each and enumeration costs of 8, 16 and 16 rows.
    Of the 10 rows left after the full-data predictions, enumerating the matched game would leave 2 for the other two games' first draws, which need 4.
    """
    d_1 = pl.DataFrame({"id": np.arange(6), "x": np.arange(6.0), "y": 2 * np.arange(6.0)})
    d_2 = pl.DataFrame(
        {"id": [0, 1, 2, 3, 6, 7], "x": np.arange(6.0) + 1, "y": 2 * np.arange(6.0) + 1}
    )
    result = decompose_frames(_sum, d_1, d_2, batch_rows=8192, id_column="id", max_evals=22)
    assert result.stop_reason == "max_evals"
    assert result.n_evals <= 22
    for k, component in enumerate(ATTRIBUTED_COMPONENTS):
        assert result.contributions[:, k].sum() == pytest.approx(
            result.component_totals[component], rel=1e-9
        )
