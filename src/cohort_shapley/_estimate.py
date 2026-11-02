"""Shapley values of each game from independent draws, with draw counts chosen to meet `tol`.

A draw pairs one row pair (first, second) from the game's pair distribution with one Shapley-kernel coalition S of the game's active feature groups, and costs two model rows: h(first, second, S) and h(first, second, Sᶜ), where Sᶜ is the complement of S among the active feature groups.
Games solve exactly when they have at most one active feature group, or when they have at most MAX_ENUMERATED_COALITIONS coalitions and enumeration costs at most the evaluations already spent sampling that game plus those of the next step.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import reduce
from typing import Self

import numpy as np

from cohort_shapley._arrays import (
    BoolArray1D,
    BoolArray2D,
    FloatArray1D,
    FloatArray2D,
    UInt8Array2D,
)
from cohort_shapley._games import EPS, Game, pair_gains
from cohort_shapley._model import Model, coalition_sums, contrasts
from cohort_shapley._shapley import (
    kernel_coalitions,
    shapley_values,
    zero_sum_basis,
)

# Sampled games converge only with at least this many draws.
MIN_DRAWS = 960
# The sampled mean total must agree with the full-data total within this many standard errors.
TOTAL_AGREEMENT_SE = 4.0
# Draw counts are planned for a total standard error of PLANNED_MARGIN * tol.
PLANNED_MARGIN = 0.9
# Bounded growth allows re-estimation of standard errors before a large increase in model rows.
MIN_GROWTH = 1.25
MAX_GROWTH = 4.0
# Eigenvalues of the regression's Gram matrix below this fraction of the largest count as zero.
SINGULAR_RTOL = 1e-10
# Draws per DrawBlock.
DRAW_BLOCK = 8192
# Enumeration holds a coalition mask table of one byte per coalition and active feature group, and a few arrays of 8 bytes per coalition; this bound keeps them to tens of megabytes.
MAX_ENUMERATED_COALITIONS = 2**20


@dataclass(frozen=True, slots=True)
class DrawBlock:
    """At most DRAW_BLOCK draws of one game, sampled together.
    Draw k's coalition S_k has bit j of row packed[k] (np.packbits order) set for active feature group j.
    With coef_k = scale * likelihood ratio, contrasts[k] = coef_k * (f(h(first_k, second_k, S_k)) - f(h(first_k, second_k, S_kᶜ))) and deviations[k] = coef_k * (f(second_k) - f(first_k)) - T for the game's total T.
    """

    packed: UInt8Array2D
    contrasts: FloatArray1D
    deviations: FloatArray1D


@dataclass(frozen=True)
class DrawSums:
    """Sums and a maximum over draws of a game with total T and n active feature groups, where d_k[j] = +1 for j in S_k and -1 otherwise.

    `count` is the number of draws, `gram` is sum_k d_k d_k', and `sign_residual` is sum_k d_k (contrasts[k] - T sum_j d_k[j] / n).
    `deviation_sum` adds, over blocks, the correctly rounded sum of deviations[k].
    `abs_total_sum` and `abs_total_max` are the sum and maximum of |coef_k * (f(second_k) - f(first_k))|, the absolute draw totals.
    """

    count: int
    gram: FloatArray2D
    sign_residual: FloatArray1D
    deviation_sum: float
    abs_total_sum: float
    abs_total_max: float

    def __add__(self, other: Self) -> Self:
        return type(self)(
            self.count + other.count,
            self.gram + other.gram,
            self.sign_residual + other.sign_residual,
            self.deviation_sum + other.deviation_sum,
            self.abs_total_sum + other.abs_total_sum,
            max(self.abs_total_max, other.abs_total_max),
        )


@dataclass(frozen=True)
class Draws:
    """The draws of one game and their sums."""

    blocks: tuple[DrawBlock, ...]
    sums: DrawSums

    def __add__(self, other: Self) -> Self:
        return type(self)(self.blocks + other.blocks, self.sums + other.sums)


def no_draws(n_active: int) -> Draws:
    """No blocks, with zero sums."""
    return Draws((), DrawSums(0, np.zeros((n_active, n_active)), np.zeros(n_active), 0.0, 0.0, 0.0))


def block_draws(
    coalitions: BoolArray2D, contrasts: FloatArray1D, totals: FloatArray1D, total: float
) -> Draws:
    """One nonempty block of draws of a game with total `total`; `coalitions` has one column per active feature group."""
    n_active = coalitions.shape[1]
    signs = 2 * coalitions.astype(np.float64) - 1
    deviations = totals - total
    return Draws(
        (DrawBlock(np.packbits(coalitions, axis=1), contrasts, deviations),),
        DrawSums(
            len(contrasts),
            signs.T @ signs,
            signs.T @ (contrasts - total * signs.sum(axis=1) / n_active),
            math.fsum(deviations.tolist()),
            float(np.abs(totals).sum()),
            float(np.abs(totals).max()),
        ),
    )


@dataclass(frozen=True)
class GameFit:
    """Shapley value estimates of a game and their standard errors, one entry per feature group.

    `checks_pass` says whether the fit meets the conditions for convergence other than `tol`: sampled fits need at least MIN_DRAWS draws, a Gram matrix of full rank, and a sampled mean total that agrees with the full-data total.
    Exact fits pass.
    """

    contributions: FloatArray1D
    stderr: FloatArray1D
    checks_pass: bool


def fit_draws(draws: Draws, n_active: int, total: float, total_rounding_bound: float) -> GameFit:
    """The constrained KernelSHAP fit of a game's draws, with the known-total control variate; one entry per active feature group.

    The fit minimizes sum_k (contrast_k - d_k . contributions)^2 subject to sum(contributions) = total, where d_k[j] = +1 for j in S_k and -1 otherwise.
    Writing contributions = total / n + B y, with B an orthonormal basis of the vectors summing to zero, removes the constraint symmetrically in the feature groups.
    A singular Gram matrix gets the minimum-norm y and infinite standard errors.

    Draw k's influence on contributions is psi_k = H d_k e_k, with H = N B G^+ B' for N draws, Gram matrix G = B' (sum_k d_k d_k') B and residual e_k = contrast_k - d_k . contributions.
    The control variate subtracts slope (mean of u), where u_k = deviations[k], the draw total minus `total`, has expectation zero and slope is the least-squares slope of psi on u over the same draws, which biases the estimate by O(1 / N).
    When the standard deviation of the draw totals is within 8 EPS max |draw total| + `total_rounding_bound`, the draw totals differ only by rounding and slope is 0.
    The standard error is the norm of the adjusted influences psi_k - slope (u_k - mean u) over sqrt(N (N - 1)).

    The sampled draw totals agree with the full-data total when |mean u| is at most TOTAL_AGREEMENT_SE standard errors of mean u, plus a rounding allowance of 8 EPS mean |draw total| + `total_rounding_bound`.

    The contributions come from `draws.sums`.
    The influences depend on the contributions, and u_k - mean u depends on every draw, so the sum of squared u_k - mean u, the slope and the standard errors each take a pass over the blocks; the slope pass is skipped when slope is 0.
    Expanding the squared adjusted influences into sums over draws would cancel catastrophically when u nearly explains psi.
    """
    sums = draws.sums
    n = sums.count
    if n == 0:
        return _no_draws_fit(n_active, total)
    basis = zero_sum_basis(n_active)
    eigenvalues, eigenvectors = np.linalg.eigh(basis.T @ sums.gram @ basis)
    kept = eigenvalues > SINGULAR_RTOL * eigenvalues[-1]
    gram_pinv = (eigenvectors[:, kept] / eigenvalues[kept]) @ eigenvectors[:, kept].T
    contributions = total / n_active + basis @ (gram_pinv @ (basis.T @ sums.sign_residual))
    if n == 1 or not kept.all():
        return GameFit(contributions, np.full(n_active, np.inf), checks_pass=False)
    influence = n * (basis @ gram_pinv @ basis.T)
    u_mean = sums.deviation_sum / n
    s_vv = sum(_squared_norm(block.deviations - u_mean) for block in draws.blocks)
    rounding = 8 * EPS * sums.abs_total_max + total_rounding_bound
    if math.sqrt(s_vv / n) <= rounding:
        slope = np.zeros(n_active)
    else:
        sign_covariance = sum(
            (_sign_covariance(block, contributions, u_mean=u_mean) for block in draws.blocks),
            start=np.zeros(n_active),
        )
        slope = influence @ sign_covariance / s_vv
    squares = sum(
        (
            _adjusted_squares(block, contributions, influence, slope, u_mean=u_mean)
            for block in draws.blocks
        ),
        start=np.zeros(n_active),
    )
    stderr = np.sqrt(squares / (n * (n - 1)))
    allowance = 8 * EPS * sums.abs_total_sum / n + total_rounding_bound
    u_se = math.sqrt(s_vv / (n * (n - 1)))
    agrees = abs(u_mean) <= TOTAL_AGREEMENT_SE * u_se + allowance
    return GameFit(contributions - slope * u_mean, stderr, checks_pass=n >= MIN_DRAWS and agrees)


def _no_draws_fit(n_active: int, total: float) -> GameFit:
    """Equal shares of the total, infinite standard errors and failing checks."""
    return GameFit(
        np.full(n_active, total / n_active), np.full(n_active, np.inf), checks_pass=False
    )


def _squared_norm(values: FloatArray1D) -> float:
    return float(values @ values)


def _signs(block: DrawBlock, n_active: int) -> FloatArray2D:
    """d_k for each draw of the block: +1 in S_k, -1 outside."""
    return 2 * np.unpackbits(block.packed, axis=1, count=n_active).astype(np.float64) - 1


def _sign_covariance(
    block: DrawBlock, contributions: FloatArray1D, *, u_mean: float
) -> FloatArray1D:
    """sum_k d_k e_k (u_k - mean u) over the block; H times its sum over all draws is sum_k psi_k (u_k - mean u)."""
    signs = _signs(block, len(contributions))
    return signs.T @ ((block.contrasts - signs @ contributions) * (block.deviations - u_mean))


def _adjusted_squares(
    block: DrawBlock,
    contributions: FloatArray1D,
    influence: FloatArray2D,
    slope: FloatArray1D,
    *,
    u_mean: float,
) -> FloatArray1D:
    """sum_k (psi_k - slope (u_k - mean u))^2 over the block, for each feature group.

    psi_k - slope (u_k - mean u) is the product of the row (d_k e_k, u_k - mean u) and the matrix (H; -slope), since H is symmetric.
    """
    signs = _signs(block, len(contributions))
    rows = np.column_stack(
        (
            signs * (block.contrasts - signs @ contributions)[:, None],
            block.deviations - u_mean,
        )
    )
    adjusted = rows @ np.vstack((influence, -slope))
    return np.einsum("kj,kj->j", adjusted, adjusted)


@dataclass(frozen=True)
class GameState:
    """A game, its draws, its fit and the model rows spent on it.

    An exact game has no draws.
    A sampled game without draws has the fit of no draws.
    """

    game: Game
    draws: Draws
    fit: GameFit
    exact: bool
    n_evals: int

    @property
    def n_draws(self) -> int:
        return self.draws.sums.count


@dataclass(frozen=True)
class Step:
    """Per game, whether to enumerate it and how many draws to add."""

    enumerate: tuple[bool, ...]
    new_draws: tuple[int, ...]


@dataclass(frozen=True)
class Component:
    """Contributions and Monte Carlo standard errors of one comparison, one entry per feature group.

    `contributions` sums to `total`.
    """

    contributions: FloatArray1D
    stderr: FloatArray1D
    total: float


@dataclass(frozen=True)
class Estimate:
    """One component per game; `n_evals` counts the model rows spent by `estimate`.

    `converged` says whether every fit passes its checks with total standard errors at most `tol`.
    """

    components: list[Component]
    n_evals: int
    converged: bool


def stderr_total(stderrs: Sequence[FloatArray1D]) -> FloatArray1D:
    """The standard error of each feature group's contribution summed over independently sampled comparisons."""
    return np.hypot.reduce(stderrs)


def estimate(
    games: list[Game],
    model: Model,
    *,
    tol: float,
    budget: int,
    rng: np.random.Generator,
) -> Estimate:
    """Draw until every feature group's total standard error is at most `tol` and every fit passes its checks, spending at most `budget` model rows.

    Games draw independently, so a feature group's total variance is the sum of its games' variances.
    Raises ValueError when the allowance cannot fund one draw of every sampled game.
    """
    states = [_initial_state(g) for g in games]
    while True:
        n_evals = sum(s.n_evals for s in states)
        fits = [s.fit for s in states]
        if (
            all(f.checks_pass for f in fits)
            and stderr_total([f.stderr for f in fits]).max(initial=0.0) <= tol
        ):
            return Estimate(_components(states), n_evals, converged=True)
        step = _step(states, _wanted_draws(states, tol), budget=budget - n_evals)
        if step is None:
            return Estimate(_components(states), n_evals, converged=False)
        states = [
            _advance(model, s, enumerate_game=enumerate_game, new_draws=new, rng=rng)
            for s, enumerate_game, new in zip(states, step.enumerate, step.new_draws, strict=True)
        ]


def _initial_state(game: Game) -> GameState:
    """A game with at most one active feature group is exact: that feature group gets the whole total, and a game without active feature groups has a total of zero."""
    if game.n_active <= 1:
        contributions = np.where(game.active, game.total, 0.0)
        fit = GameFit(contributions, np.zeros_like(contributions), checks_pass=True)
        return GameState(game, no_draws(game.n_active), fit, exact=True, n_evals=0)
    fit = _expanded_fit(_no_draws_fit(game.n_active, game.total), game.active)
    return GameState(game, no_draws(game.n_active), fit, exact=False, n_evals=0)


def _components(states: list[GameState]) -> list[Component]:
    return [Component(s.fit.contributions, s.fit.stderr, s.game.total) for s in states]


def _wanted_draws(states: list[GameState], tol: float) -> list[int]:
    """Each game's wanted draw count.

    Let s_g be a game's largest standard error times sqrt(draws), over games that pass their checks, and S their sum.
    Game g wants s_g S / tau^2 draws for tau = PLANNED_MARGIN * tol.
    This minimizes total draws under the conservative constraint sum_g s_g^2 / draws_g <= tau^2, which bounds each feature group's total variance.
    A sampled game whose fit fails its checks wants MAX_GROWTH times its draws, and at least MIN_DRAWS.
    """
    scales = [
        float(s.fit.stderr.max()) * math.sqrt(s.n_draws) if s.fit.checks_pass else 0.0
        for s in states
    ]
    scale_sum = math.fsum(scales)
    return [
        _wanted(state, _neyman_draws(scale, scale_sum, PLANNED_MARGIN * tol))
        for state, scale in zip(states, scales, strict=True)
    ]


def _wanted(state: GameState, planned: float) -> int:
    """The draw count wanted by a game, given the `planned` draws from `_neyman_draws` when its fit passes."""
    n = state.n_draws
    if state.exact:
        return 0
    if not state.fit.checks_pass:
        return max(MIN_DRAWS, math.ceil(MAX_GROWTH * n))
    if planned <= n:
        return n
    return math.ceil(min(max(planned, MIN_GROWTH * n), MAX_GROWTH * n))


def _neyman_draws(scale: float, scale_sum: float, tau: float) -> float:
    if scale == 0:
        return 0.0
    if tau == 0:
        return math.inf
    return (scale_sum / tau) * (scale / tau)


def _step(states: list[GameState], wanted: list[int], *, budget: int) -> Step | None:
    """Choose enumeration or additional draws within the evaluation allowance, or None when the allowance funds neither.

    The first round reserves one draw for every sampled game.
    Every game without draws wants MIN_DRAWS, so scaling gives each sampled game at least one draw in the first round.
    Enumeration requires at most MAX_ENUMERATED_COALITIONS coalitions and costs at most the evaluations spent sampling the game plus those requested for the next step.
    Games are considered for enumeration in order.
    """
    first_round = any(not s.exact and s.n_draws == 0 for s in states)
    new = [max(0, n - s.n_draws) for s, n in zip(states, wanted, strict=True)]
    reserves = [2 if first_round and n else 0 for n in new]
    costs = [_enumeration_cost(s.game) for s in states]
    eligible = [
        n > 0 and 2**s.game.n_active <= MAX_ENUMERATED_COALITIONS and cost <= 2 * w
        for s, n, w, cost in zip(states, new, wanted, costs, strict=True)
    ]
    enumerated = _enumerated(
        eligible,
        [cost - reserve for cost, reserve in zip(costs, reserves, strict=True)],
        budget - sum(reserves),
    )
    remaining = budget - sum(cost for cost, e in zip(costs, enumerated, strict=True) if e)
    sampled = [0 if e else n for n, e in zip(new, enumerated, strict=True)]
    n_sampled = sum(n > 0 for n in sampled)
    if first_round and remaining < 2 * n_sampled:
        raise ValueError(
            f"max_evals leaves {remaining} rows after endpoint predictions and exact games; "
            f"the first estimation round needs at least {2 * n_sampled}"
        )
    counts = _scaled(sampled, remaining)
    if not any(enumerated) and not any(counts):
        return None
    return Step(enumerated, counts)


def _enumerated(eligible: Sequence[bool], extra: Sequence[int], budget: int) -> tuple[bool, ...]:
    """Enumerate each eligible game in order whose `extra` evaluations, added to those of the games already enumerated, fit within `budget`.

    A game's `extra` evaluations are its enumeration cost minus the evaluations reserved for its first-round draw.
    """

    def choose(
        chosen: tuple[tuple[bool, ...], int], game: tuple[bool, int]
    ) -> tuple[tuple[bool, ...], int]:
        flags, spent = chosen
        is_eligible, game_extra = game
        take = is_eligible and spent + game_extra <= budget
        return (*flags, take), spent + (game_extra if take else 0)

    return reduce(choose, zip(eligible, extra, strict=True), ((), 0))[0]


def _scaled(sampled: Sequence[int], budget: int) -> tuple[int, ...]:
    """The draw counts `sampled` scaled down in proportion, rounding down, so that their two rows per draw fit within `budget`."""
    cost = 2 * sum(sampled)
    return tuple(n * min(budget, cost) // max(cost, 1) for n in sampled)


def _enumeration_cost(game: Game) -> int:
    """Model rows to evaluate every pair on every coalition strictly between no active feature groups and all of them."""
    return game.pair_distribution.n_pairs * (2**game.n_active - 2)


def _advance(
    model: Model,
    state: GameState,
    *,
    enumerate_game: bool,
    new_draws: int,
    rng: np.random.Generator,
) -> GameState:
    """`state` after enumerating it or adding `new_draws` draws."""
    game = state.game
    if enumerate_game:
        return replace(
            state,
            draws=no_draws(game.n_active),
            fit=_enumerated_fit(model, game),
            exact=True,
            n_evals=state.n_evals + _enumeration_cost(game),
        )
    if new_draws == 0:
        return state
    draws = state.draws + _draw(model, game, new_draws, rng)
    fitted = fit_draws(draws, game.n_active, game.total, game.total_rounding_bound)
    return replace(
        state,
        draws=draws,
        fit=_expanded_fit(fitted, game.active),
        n_evals=state.n_evals + 2 * new_draws,
    )


def _expanded[A: (FloatArray1D, BoolArray2D)](values: A, active: BoolArray1D) -> A:
    """`values`, whose last axis runs over the active feature groups, with zeros inserted for the null feature groups."""
    return np.insert(values, np.cumsum(active)[~active], 0, axis=-1)


def _expanded_fit(fit: GameFit, active: BoolArray1D) -> GameFit:
    return GameFit(
        _expanded(fit.contributions, active), _expanded(fit.stderr, active), fit.checks_pass
    )


def _draw(model: Model, game: Game, n: int, rng: np.random.Generator) -> Draws:
    """`n` new draws of `game`, DRAW_BLOCK at a time."""
    return sum(
        (
            _draw_block(model, game, min(DRAW_BLOCK, n - start), rng)
            for start in range(0, n, DRAW_BLOCK)
        ),
        start=no_draws(game.n_active),
    )


def _draw_block(model: Model, game: Game, n: int, rng: np.random.Generator) -> Draws:
    pairs, ratio = game.pair_distribution.draw(n, rng)
    coef = game.scale * ratio
    coalitions = kernel_coalitions(n, game.n_active, rng)
    return block_draws(
        coalitions,
        coef * contrasts(model, pairs, _expanded(coalitions, game.active)),
        coef * pair_gains(game.predictions, pairs),
        game.total,
    )


def _enumerated_fit(model: Model, game: Game) -> GameFit:
    """The exact Shapley values of `game` from every pair on every coalition of its active feature groups."""
    gains = _coalition_gains(model, game)
    contributions = shapley_values(np.concatenate(([0.0], gains, [game.total])))
    return _expanded_fit(
        GameFit(contributions, np.zeros_like(contributions), checks_pass=True), game.active
    )


def _coalition_gains(model: Model, game: Game) -> FloatArray1D:
    """The gain of every coalition strictly between no active feature groups and all of them, indexed by code minus one, with pairs taken max(1, batch_rows // 2) at a time."""
    codes = np.arange(1, 2**game.n_active - 1)
    masks = ((codes[:, None] >> np.arange(game.n_active)) & 1).astype(bool)
    pair_distribution = game.pair_distribution
    n_pairs, weight_total = pair_distribution.n_pairs, pair_distribution.weight_total
    span = max(1, model.batch_rows // 2)
    chunks = (
        pair_distribution.chunk(start, min(start + span, n_pairs))
        for start in range(0, n_pairs, span)
    )
    return sum(
        (
            coalition_sums(
                model, game, pairs, coef=game.scale * weights / weight_total, masks=masks
            )
            for pairs, weights in chunks
        ),
        start=np.zeros(len(codes)),
    )
