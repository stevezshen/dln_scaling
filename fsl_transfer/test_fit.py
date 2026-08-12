"""Regression tests for the fitted discrete response models."""

from __future__ import annotations

import unittest

import numpy as np

from core import Problem, checkpoint_indices, signal_profile
from fit import Trajectory, trajectory_metrics, volterra_basis


class FitTests(unittest.TestCase):
    def test_zero_feedback_volterra_is_the_signal(self) -> None:
        problem = Problem(
            modes=8,
            steps=32,
            paths=3,
            checkpoints=12,
            depth=5,
            beta=2.0,
            chi=-0.2,
            sigma=0.3,
            eta_base=1.5,
            c_init=0.2,
            seed=2,
        )
        eta = np.full(problem.steps, problem.eta_base, dtype=np.float64)
        indices = checkpoint_indices(problem.steps, problem.checkpoints)
        trajectory = Trajectory(
            name="constant_mid",
            calibration=True,
            eta=eta,
            checkpoint_steps=indices,
            risk_paths=np.ones((indices.size, problem.paths)),
            risk_mean=np.ones(indices.size),
            initial_u=np.ones(problem.modes),
            metadata=problem.__dict__,
        )
        contraction = 3.0
        signal_part, noise_part = volterra_basis(
            trajectory,
            problem,
            contraction,
            kernel_contraction=5.0,
            feedback=0.0,
        )
        times = np.concatenate(([0.0], np.cumsum(eta, dtype=np.float64)))
        expected = signal_profile(problem, eta, times, contraction)[indices]
        np.testing.assert_allclose(signal_part, expected, rtol=2.0e-13)
        np.testing.assert_array_equal(noise_part, np.zeros_like(noise_part))

    def test_log_metrics_are_zero_for_exact_prediction(self) -> None:
        values = np.geomspace(2.0, 0.01, 20)
        metrics = trajectory_metrics(values, values.copy())
        self.assertEqual(metrics["log_rmse"], 0.0)
        self.assertEqual(metrics["terminal_relative_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
