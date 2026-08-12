"""Schedules and finite FSL profiles for the transfer experiment."""

from __future__ import annotations

import dataclasses
import itertools
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class Problem:
    modes: int
    steps: int
    paths: int
    checkpoints: int
    depth: int
    beta: float
    chi: float
    sigma: float
    eta_base: float
    c_init: float
    seed: int

    @property
    def alpha(self) -> float:
        return 2.0 - 2.0 / self.depth

    @property
    def p(self) -> float:
        return self.beta + self.alpha * self.chi

    @property
    def s(self) -> float:
        return self.beta + 2.0 * self.chi - 1.0

    @property
    def reference_time(self) -> float:
        """Intrinsic time of the middle constant-rate schedule."""

        return self.steps * self.eta_base

    def validate(self) -> None:
        integer_positive = (self.modes, self.steps, self.paths, self.checkpoints)
        if any(value <= 0 for value in integer_positive):
            raise ValueError("modes, steps, paths, and checkpoints must be positive")
        if self.depth < 2:
            raise ValueError("depth must be at least two")
        if self.beta <= 1.0 or self.s <= 0.0 or self.p <= 1.0:
            raise ValueError("require beta > 1, s > 0, and p > 1")
        if self.sigma < 0.0 or self.eta_base <= 0.0 or self.c_init <= 0.0:
            raise ValueError("require sigma >= 0, eta_base > 0, and c_init > 0")


def schedule_shapes(steps: int) -> Dict[str, np.ndarray]:
    """Return unit-mean transfer shapes and three constant calibration shapes."""

    if steps < 4:
        raise ValueError("at least four steps are required")
    x = np.arange(steps, dtype=np.float64) / float(steps - 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * x))

    stable_fraction = 0.8
    wsd = np.ones(steps, dtype=np.float64)
    decay = x >= stable_fraction
    wsd[decay] = np.power(
        np.maximum((1.0 - x[decay]) / (1.0 - stable_fraction), 0.0), 2.0
    )

    cyclic = 0.35 + 1.3 * np.abs(2.0 * np.mod(3.0 * x + 0.5, 1.0) - 1.0)
    late_drop = np.where(x < 0.8, 1.0, 0.1)

    def unit_mean(values: np.ndarray) -> np.ndarray:
        mean = float(np.mean(values))
        if mean <= 0.0:
            raise ValueError("schedule shape must have positive mean")
        return values / mean

    return {
        "constant_low": np.full(steps, 0.8, dtype=np.float64),
        "constant_mid": np.ones(steps, dtype=np.float64),
        "constant_high": np.full(steps, 1.2, dtype=np.float64),
        "cosine": unit_mean(cosine),
        "wsd": unit_mean(wsd),
        "cyclic": unit_mean(cyclic),
        "late_drop": unit_mean(late_drop),
    }


def schedules(problem: Problem) -> Dict[str, np.ndarray]:
    problem.validate()
    return {
        name: (problem.eta_base * shape).astype(np.float32)
        for name, shape in schedule_shapes(problem.steps).items()
    }


def checkpoint_indices(steps: int, count: int) -> np.ndarray:
    """Use dense early checkpoints and a uniform grid thereafter."""

    count = min(max(count, 8), steps + 1)
    early_count = max(6, count // 3)
    early_stop = max(2, steps // 20)
    early = np.unique(
        np.rint(np.geomspace(1.0, float(early_stop), early_count)).astype(np.int64)
    )
    linear = np.rint(
        np.linspace(float(early_stop), float(steps), count - early_count)
    ).astype(np.int64)
    return np.unique(np.concatenate(([0], early, linear, [steps]))).astype(np.int64)


def spectral_arrays(problem: Problem) -> Tuple[np.ndarray, np.ndarray]:
    j = np.arange(1, problem.modes + 1, dtype=np.float64)
    lambdas = np.power(j, -problem.beta)
    target = np.power(j, -problem.chi)
    return lambdas, target


def initialization(problem: Problem, eta: np.ndarray) -> np.ndarray:
    """Common slope-independent initialization for one transfer study.

    The theorem permits constant-factor changes in total intrinsic time.  We
    therefore use the middle constant schedule's time for every trajectory.
    This keeps the initial parameters fixed while low and high constant rates
    bracket the transfer schedules.
    """

    if eta.shape != (problem.steps,):
        raise ValueError("eta has the wrong length")
    j = np.arange(1, problem.modes + 1, dtype=np.float64)
    return (
        problem.c_init
        * problem.reference_time ** (-1.0 / problem.alpha)
        * np.power(j, problem.beta / problem.alpha)
    )


def initialization_is_admissible(
    problem: Problem, eta: np.ndarray, *, tolerance: float = 1.0e-12
) -> bool:
    _, target = spectral_arrays(problem)
    return bool(np.all(initialization(problem, eta) <= target * (1.0 + tolerance)))


def intrinsic_times(eta: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(eta, dtype=np.float64)))


def _h(values: np.ndarray, alpha: float) -> np.ndarray:
    if abs(alpha - 1.0) < 1.0e-12:
        return np.log(values)
    return np.power(values, 1.0 - alpha) / (1.0 - alpha)


def barrier_weights(problem: Problem, u0: np.ndarray) -> np.ndarray:
    """Compute Theta_{j,0}=Psi(L_{j,0}) from the general signal profile."""

    _, target = spectral_arrays(problem)
    numerator = _h(0.75 * target, problem.alpha) - _h(u0, problem.alpha)
    denominator = _h(0.75 * target, problem.alpha) - _h(
        0.5 * target, problem.alpha
    )
    transform = numerator / denominator
    result = np.zeros_like(transform)
    positive = transform > 0.0
    x = transform[positive]
    result[positive] = np.exp(np.clip(-1.0 / x + x, -745.0, 700.0))
    return result


def signal_components(
    problem: Problem,
    eta: np.ndarray,
    times: np.ndarray,
    contraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the initial-error and barrier parts of the finite signal profile."""

    if contraction <= 0.0:
        raise ValueError("contraction must be positive")
    lambdas, target = spectral_arrays(problem)
    u0 = initialization(problem, eta)
    theta = barrier_weights(problem, u0)
    j = np.arange(1, problem.modes + 1, dtype=np.float64)
    rates = np.power(j, -problem.p)
    first_weight = lambdas * np.square(u0 - target)
    second_weight = lambdas * np.square(target) * theta
    initial_error = np.zeros_like(times, dtype=np.float64)
    barrier = np.zeros_like(times, dtype=np.float64)
    for index in range(problem.modes):
        scaled = times * rates[index]
        decay = np.exp(-contraction * scaled)
        initial_error += first_weight[index] * decay
        barrier += second_weight[index] * scaled * decay
    return initial_error, barrier


def signal_profile(
    problem: Problem,
    eta: np.ndarray,
    times: np.ndarray,
    contraction: float,
    *,
    include_barrier: bool = False,
) -> np.ndarray:
    """Evaluate the finite signal basis at arbitrary intrinsic times.

    The initial-error term is the empirical basis.  The optional barrier term
    belongs to the theorem's upper bound and is retained for diagnostics.
    """

    initial_error, barrier = signal_components(problem, eta, times, contraction)
    return initial_error + barrier if include_barrier else initial_error


def kernel_convolution(
    problem: Problem,
    eta: np.ndarray,
    forcing: np.ndarray,
    contraction: float,
) -> np.ndarray:
    """Compute sum eta_n^2 K(t_q-t_{n+1}) forcing_n in O(Nd)."""

    if eta.ndim != 1 or forcing.shape != eta.shape:
        raise ValueError("eta and forcing must be one-dimensional with equal length")
    if contraction <= 0.0:
        raise ValueError("contraction must be positive")
    j = np.arange(1, problem.modes + 1, dtype=np.float64)
    rates = np.power(j, -problem.p)
    weights = np.power(j, -2.0 * problem.p)
    state = np.zeros(problem.modes, dtype=np.float64)
    output = np.zeros(problem.steps + 1, dtype=np.float64)
    for n in range(problem.steps):
        state *= np.exp(-contraction * rates * float(eta[n]))
        state += float(eta[n]) ** 2 * float(forcing[n])
        output[n + 1] = float(weights @ state)
    return output


def nonnegative_least_squares(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Exact NNLS for the two- or three-column designs used here."""

    if design.ndim != 2 or design.shape[0] != target.size:
        raise ValueError("invalid design shape")
    columns = design.shape[1]
    if not 1 <= columns <= 6:
        raise ValueError("active-set enumeration is intended for small designs")
    root_w = np.sqrt(weights)
    xw = design * root_w[:, None]
    yw = target * root_w
    best_coef = np.zeros(columns, dtype=np.float64)
    best_loss = float(np.sum(np.square(yw)))
    for size in range(1, columns + 1):
        for active in itertools.combinations(range(columns), size):
            sub = xw[:, active]
            coef, *_ = np.linalg.lstsq(sub, yw, rcond=None)
            if np.any(coef < -1.0e-12):
                continue
            full = np.zeros(columns, dtype=np.float64)
            full[list(active)] = np.maximum(coef, 0.0)
            loss = float(np.sum(np.square(xw @ full - yw)))
            if loss < best_loss:
                best_loss = loss
                best_coef = full
    return best_coef, best_loss
