"""Shapley values of each game from independent draws, with draw counts chosen to meet `tol`.

A draw pairs one row pair (first, second) from the game's pair distribution with one Shapley-kernel coalition S of the game's active feature groups, and costs two model rows: h(first, second, S) and h(first, second, Sᶜ), where Sᶜ is the complement of S among the active feature groups.
Games solve exactly when they have at most one active feature group, or when they have at most MAX_ENUMERATED_COALITIONS coalitions and enumeration costs at most the evaluations already spent sampling that game plus those of the next step.
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np

from cohort_shapley._arrays import BoolArray1D, FloatArray1D, FloatArray2D, UInt8Array2D
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
# Draws per block of sampling and of the fitting passes.
FIT_BLOCK = 8192
# Enumeration holds a coalition mask table of one byte per coalition and active feature group, and a few arrays of 8 bytes per coalition; this bound keeps them to tens of megabytes.
MAX_ENUMERATED_COALITIONS = 2**20


@dataclass(frozen=True, slots=True)
class DrawChunk:
    """Draws of one game.
    Draw k's coalition S_k has bit j of row packed[k] (np.packbits order) set for active feature group j.
    With coef_k = scale * likelihood ratio, contrasts[k] = coef_k * (f(h(first_k, second_k, S_k)) - f(h(first_k, second_k, S_kᶜ))) and totals[k] = coef_k * (f(second_k) - f(first_k)).
    """

    packed: UInt8Array2D
    contrasts: FloatArray1D
    totals: FloatArray1D


@dataclass(frozen=True)
class GameFit:
    """Shapley value estimates of a game and their standard errors, one entry per feature group.

    `checks_pass` says whether the fit meets the conditions for convergence other than `tol`: sampled fits need at least MIN_DRAWS draws, a Gram matrix of full rank, and a sampled mean total that agrees with the full-data total.
    Exact fits pass.
    """

    contributions: FloatArray1D
    stderr: FloatArray1D
    checks_pass: bool


def fit_draws(
    draws: Sequence[DrawChunk],
    n_active: int,
    total: float,
    total_rounding_bound: float,
    block: int,
) -> GameFit:
    """The constrained KernelSHAP fit of a game's draws, with the known-total control variate; one entry per active feature group.

    The fit minimizes sum_k (contrast_k - d_k . contributions)^2 subject to sum(contributions) = total, where d_k[j] = +1 for j in S_k and -1 otherwise.
    Writing contributions = total / n + B y, with B an orthonormal basis of the vectors summing to zero, removes the constraint symmetrically in the feature groups.
    A singular Gram matrix gets the minimum-norm y and infinite standard errors.

    Draw k's influence on contributions is psi_k = H d_k e_k, with H = N B G^+ B' for N draws, Gram matrix G = B' (sum_k d_k d_k') B and residual e_k = contrast_k - d_k . contributions.
    The control variate subtracts beta (mean of u), where u_k = totals[k] - total has expectation zero and beta is the least-squares slope of psi on u over the same draws, which biases the estimate by O(1 / N).
    When the standard deviation of the totals is within 8 EPS max |totals| + `total_rounding_bound`, the totals differ only by rounding and beta is 0.
    The standard error is the norm of the adjusted influences psi_k - beta (u_k - mean u) over sqrt(N (N - 1)).

    The sampled totals agree with the full-data total when |mean u| is at most TOTAL_AGREEMENT_SE standard errors of mean u, plus a rounding allowance of 8 EPS mean |totals| + `total_rounding_bound`.

    The design passes over the draws three times, `block` draws at a time; a separate streaming reduction computes the mean total deviation.
    These passes never form an array with one row per draw.
    """
    n = sum(len(chunk.contrasts) for chunk in draws)
    if n == 0:
        return _no_draws_fit(n_active, total)
    sign_gram = np.zeros((n_active, n_active))
    sign_residual = np.zeros(n_active)
    abs_total_sum = 0.0
    abs_total_max = 0.0
    for signs, values, totals in _blocks(draws, n_active, block):
        sign_gram += signs.T @ signs
        sign_residual += signs.T @ (values - total * signs.sum(axis=1) / n_active)
        abs_total_sum += float(np.abs(totals).sum())
        abs_total_max = max(abs_total_max, float(np.abs(totals).max()))
    basis = zero_sum_basis(n_active)
    eigenvalues, eigenvectors = np.linalg.eigh(basis.T @ sign_gram @ basis)
    kept = eigenvalues > SINGULAR_RTOL * eigenvalues[-1]
    gram_pinv = (eigenvectors[:, kept] / eigenvalues[kept]) @ eigenvectors[:, kept].T
    contributions = total / n_active + basis @ (gram_pinv @ (basis.T @ sign_residual))
    if n == 1 or not kept.all():
        return GameFit(contributions, np.full(n_active, np.inf), checks_pass=False)
    influence = n * (basis @ gram_pinv @ basis.T)
    u_mean = (
        math.fsum(
            math.fsum(chunk.totals[start : start + block] - total)
            for chunk in draws
            for start in range(0, len(chunk.totals), block)
        )
        / n
    )
    s_pv = np.zeros(n_active)
    s_vv = 0.0
    for signs, values, totals in _blocks(draws, n_active, block):
        v = totals - total - u_mean
        s_pv += _influences(signs, values, contributions, influence).T @ v
        s_vv += float(v @ v)
    rounding = 8 * EPS * abs_total_max + total_rounding_bound
    slope = np.zeros(n_active) if math.sqrt(s_vv / n) <= rounding else s_pv / s_vv
    squares = np.zeros(n_active)
    for signs, values, totals in _blocks(draws, n_active, block):
        v = totals - total - u_mean
        adjusted = _influences(signs, values, contributions, influence) - np.outer(v, slope)
        squares += (adjusted**2).sum(axis=0)
    stderr = np.sqrt(squares / (n * (n - 1)))
    allowance = 8 * EPS * abs_total_sum / n + total_rounding_bound
    u_se = math.sqrt(s_vv / (n * (n - 1)))
    agrees = abs(u_mean) <= TOTAL_AGREEMENT_SE * u_se + allowance
    return GameFit(contributions - slope * u_mean, stderr, checks_pass=n >= MIN_DRAWS and agrees)


def _no_draws_fit(n_active: int, total: float) -> GameFit:
    """Equal shares of the total, infinite standard errors and failing checks."""
    return GameFit(
        np.full(n_active, total / n_active), np.full(n_active, np.inf), checks_pass=False
    )


def _influences(
    signs: FloatArray2D, values: FloatArray1D, contributions: FloatArray1D, influence: FloatArray2D
) -> FloatArray2D:
    """psi_k = H d_k e_k for each draw of a block, with e_k = contrast_k - d_k . contributions."""
    return (signs @ influence) * (values - signs @ contributions)[:, None]


def _blocks(
    draws: Sequence[DrawChunk], n_active: int, block: int
) -> Iterator[tuple[FloatArray2D, FloatArray1D, FloatArray1D]]:
    """The draws `block` at a time: signs d (+1 in S, -1 outside), contrasts and totals."""
    for chunk in draws:
        for start in range(0, len(chunk.contrasts), block):
            rows = slice(start, start + block)
            bits = np.unpackbits(chunk.packed[rows], axis=1, count=n_active)
            yield 2 * bits.astype(np.float64) - 1, chunk.contrasts[rows], chunk.totals[rows]


@dataclass(frozen=True)
class GameState:
    """A game, its draws, its fit and the model rows spent on it.

    An exact game has no draws.
    A sampled game without draws has the fit of no draws.
    """

    game: Game
    draws: tuple[DrawChunk, ...]
    fit: GameFit
    exact: bool
    n_evals: int

    @property
    def n_draws(self) -> int:
        return sum(len(chunk.contrasts) for chunk in self.draws)


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
        return GameState(game, (), fit, exact=True, n_evals=0)
    fit = _expanded(_no_draws_fit(game.n_active, game.total), game.active)
    return GameState(game, (), fit, exact=False, n_evals=0)


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
    wanted = []
    for state, scale in zip(states, scales, strict=True):
        n = state.n_draws
        if state.exact:
            wanted.append(0)
        elif not state.fit.checks_pass:
            wanted.append(max(MIN_DRAWS, math.ceil(MAX_GROWTH * n)))
        else:
            planned = _neyman_draws(scale, scale_sum, PLANNED_MARGIN * tol)
            grown = min(max(planned, MIN_GROWTH * n), MAX_GROWTH * n)
            wanted.append(n if planned <= n else math.ceil(grown))
    return wanted


def _neyman_draws(scale: float, scale_sum: float, tau: float) -> float:
    if scale == 0:
        return 0.0
    if tau == 0:
        return math.inf
    return (scale_sum / tau) * (scale / tau)


def _step(states: list[GameState], wanted: list[int], *, budget: int) -> Step | None:
    """Choose enumeration or additional draws within the evaluation allowance, or None when the allowance funds neither.

    The first round reserves one draw for every sampled game.
    Enumeration requires at most MAX_ENUMERATED_COALITIONS coalitions and costs at most the evaluations spent sampling the game plus those requested for the next step.
    """
    first_round = any(not s.exact and s.n_draws == 0 for s in states)
    new = [max(0, n - s.n_draws) if not s.exact else 0 for s, n in zip(states, wanted, strict=True)]
    reserve = [2 if first_round and n else 0 for n in new]
    enumerated = []
    for i, state in enumerate(states):
        cost = _enumeration_cost(state.game)
        chosen = (
            new[i] > 0
            and 2**state.game.n_active <= MAX_ENUMERATED_COALITIONS
            and cost <= 2 * wanted[i]
            and cost <= budget - (sum(reserve) - reserve[i])
        )
        enumerated.append(chosen)
        if chosen:
            budget -= cost
            reserve[i] = 0
            new[i] = 0
    if first_round and any(new):
        n_sampled = sum(n > 0 for n in new)
        if budget < 2 * n_sampled:
            raise ValueError(
                f"max_evals leaves {budget} rows after endpoint predictions and exact games; "
                f"the first estimation round needs at least {2 * n_sampled}"
            )
        count = min(MIN_DRAWS, budget // (2 * n_sampled))
        return Step(tuple(enumerated), tuple(count if n else 0 for n in new))
    fraction = min(1.0, budget / (2 * sum(new))) if any(new) else 0.0
    counts = tuple(math.floor(fraction * n) for n in new)
    if not any(enumerated) and not any(counts):
        return None
    return Step(tuple(enumerated), counts)


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
            draws=(),
            fit=_enumerated_fit(model, game),
            exact=True,
            n_evals=state.n_evals + _enumeration_cost(game),
        )
    if new_draws == 0:
        return state
    draws = state.draws + _draw(model, game, new_draws, FIT_BLOCK, rng)
    fitted = fit_draws(
        draws,
        game.n_active,
        game.total,
        game.total_rounding_bound,
        FIT_BLOCK,
    )
    return replace(
        state,
        draws=draws,
        fit=_expanded(fitted, game.active),
        n_evals=state.n_evals + 2 * new_draws,
    )


def _expanded(fit: GameFit, active: BoolArray1D) -> GameFit:
    """`fit` over the active feature groups extended to every feature group, with zeros for the null feature groups."""
    contributions = np.zeros(len(active))
    stderr = np.zeros(len(active))
    contributions[active] = fit.contributions
    stderr[active] = fit.stderr
    return GameFit(contributions, stderr, checks_pass=fit.checks_pass)


def _draw(
    model: Model, game: Game, n: int, block: int, rng: np.random.Generator
) -> tuple[DrawChunk, ...]:
    """`n` new draws of `game`, in chunks of `block` draws."""
    chunks = []
    masks = np.zeros((min(n, block), len(game.active)), dtype=bool)
    for start in range(0, n, block):
        size = min(block, n - start)
        pairs, ratio = game.pair_distribution.draw(size, rng)
        coef = game.scale * ratio
        coalitions = kernel_coalitions(size, game.n_active, rng)
        masks[:size, game.active] = coalitions
        chunks.append(
            DrawChunk(
                np.packbits(coalitions, axis=1),
                coef * contrasts(model, pairs, masks[:size]),
                coef * pair_gains(game.predictions, pairs),
            )
        )
    return tuple(chunks)


def _enumerated_fit(model: Model, game: Game) -> GameFit:
    """The exact Shapley values of `game` from every pair on every coalition of its active feature groups, with pairs taken max(1, batch_rows // 2) at a time."""
    n_active = game.n_active
    codes = np.arange(1, 2**n_active - 1)
    masks = np.zeros((len(codes), n_active), dtype=bool)
    for bit in range(n_active):
        masks[:, bit] = (codes >> bit) & 1
    table = np.zeros(2**n_active)
    table[-1] = game.total
    pair_distribution = game.pair_distribution
    n_pairs, weight_total = pair_distribution.n_pairs, pair_distribution.weight_total
    span = max(1, model.batch_rows // 2)
    for start in range(0, n_pairs, span):
        pairs, weights = pair_distribution.chunk(start, min(start + span, n_pairs))
        coef = game.scale * weights / weight_total
        table[1:-1] += coalition_sums(model, game, pairs, coef=coef, masks=masks)
    del codes, masks
    contributions = np.zeros(len(game.active))
    contributions[game.active] = shapley_values(table)
    return GameFit(contributions, np.zeros_like(contributions), checks_pass=True)
