"""Unit tests for schedules, profiles, convolution, and constrained fitting."""

from __future__ import annotations

import unittest

import numpy as np

from core import (
    Problem,
    initialization,
    initialization_is_admissible,
    kernel_convolution,
    nonnegative_least_squares,
    schedule_shapes,
    schedules,
    signal_components,
    spectral_arrays,
)


def problem() -> Problem:
    return Problem(
        modes=8,
        steps=32,
        paths=4,
        checkpoints=12,
        depth=5,
        beta=2.0,
        chi=-0.2,
        sigma=0.3,
        eta_base=1.5,
        c_init=0.2,
        seed=1,
    )


class CoreTests(unittest.TestCase):
    def test_transfer_shapes_have_unit_mean(self) -> None:
        shapes = schedule_shapes(101)
        for name in ("cosine", "wsd", "cyclic", "late_drop"):
            self.assertAlmostEqual(float(np.mean(shapes[name])), 1.0, places=12)
            self.assertTrue(np.all(shapes[name] >= 0.0))

    def test_slope_independent_initialization_is_below_target(self) -> None:
        case = problem()
        all_schedules = schedules(case)
        reference = initialization(case, all_schedules["constant_mid"])
        for eta in all_schedules.values():
            self.assertTrue(initialization_is_admissible(case, eta))
            np.testing.assert_array_equal(initialization(case, eta), reference)

    def test_fast_convolution_matches_definition(self) -> None:
        case = problem()
        eta = schedules(case)["cyclic"]
        forcing = np.linspace(0.3, 1.1, case.steps)
        contraction = 1.7
        fast = kernel_convolution(case, eta, forcing, contraction)
        times = np.concatenate(([0.0], np.cumsum(eta, dtype=np.float64)))
        j = np.arange(1, case.modes + 1, dtype=np.float64)
        direct = np.zeros(case.steps + 1)
        for q in range(1, case.steps + 1):
            for n in range(q):
                remaining = times[q] - times[n + 1]
                kernel = np.sum(
                    np.power(j, -2.0 * case.p)
                    * np.exp(-contraction * remaining * np.power(j, -case.p))
                )
                # ``eta`` is stored as float32 for the simulator.  Convert each
                # entry before squaring so that this direct reference remains a
                # float64 calculation.
                direct[q] += float(eta[n]) ** 2 * kernel * forcing[n]
        np.testing.assert_allclose(fast, direct, rtol=2.0e-12, atol=2.0e-12)

    def test_small_nnls_recovers_nonnegative_coefficients(self) -> None:
        rng = np.random.default_rng(3)
        design = rng.uniform(0.1, 2.0, size=(100, 3))
        expected = np.array([0.7, 0.0, 1.4])
        target = design @ expected
        actual, loss = nonnegative_least_squares(
            design, target, np.ones(target.size)
        )
        np.testing.assert_allclose(actual, expected, atol=1.0e-10)
        self.assertLess(loss, 1.0e-18)

    def test_signal_starts_at_exact_initial_risk(self) -> None:
        case = problem()
        eta = schedules(case)["constant_mid"]
        initial_error, barrier = signal_components(
            case, eta, np.array([0.0]), contraction=3.0
        )
        lambdas, target = spectral_arrays(case)
        u0 = initialization(case, eta)
        expected = np.sum(lambdas * np.square(u0 - target))
        self.assertAlmostEqual(float(initial_error[0]), float(expected), places=13)
        self.assertEqual(float(barrier[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
