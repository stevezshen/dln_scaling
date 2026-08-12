"""Fit finite-profile FSLs on constant schedules and evaluate transfer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from core import (
    Problem,
    kernel_convolution,
    nonnegative_least_squares,
    schedules,
    signal_profile,
)


@dataclass
class Trajectory:
    name: str
    calibration: bool
    eta: np.ndarray
    checkpoint_steps: np.ndarray
    risk_paths: np.ndarray
    risk_mean: np.ndarray
    initial_u: np.ndarray
    metadata: Mapping[str, object]


@dataclass
class FitResult:
    model: str
    signal_contraction: float
    kernel_contraction: float
    coefficients: np.ndarray
    calibration_loss: float


def volterra_basis(
    trajectory: Trajectory,
    problem: Problem,
    signal_contraction: float,
    kernel_contraction: float,
    feedback: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the finite discrete Volterra recurrence in two affine parts."""

    if feedback < 0.0:
        raise ValueError("feedback must be nonnegative")
    times = np.concatenate(([0.0], np.cumsum(trajectory.eta, dtype=np.float64)))
    signal = signal_profile(problem, trajectory.eta, times, signal_contraction)
    j = np.arange(1, problem.modes + 1, dtype=np.float64)
    rates = np.power(j, -problem.p)
    weights = np.power(j, -2.0 * problem.p)
    signal_state = np.zeros(problem.modes, dtype=np.float64)
    noise_state = np.zeros(problem.modes, dtype=np.float64)
    signal_response = np.zeros(problem.steps + 1, dtype=np.float64)
    noise_response = np.zeros(problem.steps + 1, dtype=np.float64)
    signal_response[0] = signal[0]
    for n in range(problem.steps):
        decay = np.exp(
            -kernel_contraction * rates * float(trajectory.eta[n])
        )
        signal_state *= decay
        noise_state *= decay
        eta_squared = float(trajectory.eta[n]) ** 2
        signal_state += eta_squared * signal_response[n]
        noise_state += eta_squared * (noise_response[n] + problem.sigma**2)
        signal_response[n + 1] = signal[n + 1] + feedback * float(
            weights @ signal_state
        )
        noise_response[n + 1] = feedback * float(weights @ noise_state)
        if not (
            np.isfinite(signal_response[n + 1])
            and np.isfinite(noise_response[n + 1])
        ):
            raise FloatingPointError("Volterra recurrence diverged")
    indices = trajectory.checkpoint_steps
    return signal_response[indices], noise_response[indices]


def load_trajectory(path: Path) -> Trajectory:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        return Trajectory(
            name=str(metadata["name"]),
            calibration=bool(metadata["calibration"]),
            eta=np.asarray(data["eta"], dtype=np.float64),
            checkpoint_steps=np.asarray(data["checkpoint_steps"], dtype=np.int64),
            risk_paths=np.asarray(data["risk_paths"], dtype=np.float64),
            risk_mean=np.asarray(data["risk_mean"], dtype=np.float64),
            initial_u=np.asarray(data["initial_u"], dtype=np.float64),
            metadata=metadata,
        )


def problem_from_metadata(metadata: Mapping[str, object]) -> Problem:
    return Problem(
        modes=int(metadata["modes"]),
        steps=int(metadata["steps"]),
        paths=int(metadata["paths"]),
        checkpoints=int(metadata["checkpoints"]),
        depth=int(metadata["depth"]),
        beta=float(metadata["beta"]),
        chi=float(metadata["chi"]),
        sigma=float(metadata["sigma"]),
        eta_base=float(metadata["eta_base"]),
        c_init=float(metadata["c_init"]),
        seed=int(metadata["seed"]),
    )


def validate_trajectories(trajectories: Sequence[Trajectory]) -> Problem:
    """Reject incomplete, stale, or incompatible collections before fitting."""

    names = [trajectory.name for trajectory in trajectories]
    if len(names) != len(set(names)):
        raise ValueError("trajectory names must be unique")
    problem = problem_from_metadata(trajectories[0].metadata)
    expected_schedules = schedules(problem)
    if set(names) != set(expected_schedules):
        missing = sorted(set(expected_schedules) - set(names))
        extra = sorted(set(names) - set(expected_schedules))
        raise ValueError(f"incomplete schedule set; missing={missing}, extra={extra}")
    fields = (
        "modes",
        "steps",
        "paths",
        "checkpoints",
        "depth",
        "beta",
        "chi",
        "sigma",
        "eta_base",
        "c_init",
        "seed",
    )
    reference_metadata = tuple(trajectories[0].metadata[field] for field in fields)
    reference_backend = trajectories[0].metadata["backend"]
    reference_u = trajectories[0].initial_u
    reference_initial_risk = trajectories[0].risk_paths[0]
    for trajectory in trajectories:
        metadata = tuple(trajectory.metadata[field] for field in fields)
        if metadata != reference_metadata:
            raise ValueError(f"incompatible metadata in {trajectory.name}")
        if trajectory.metadata["backend"] != reference_backend:
            raise ValueError("all trajectories must use one backend")
        if trajectory.eta.shape != (problem.steps,):
            raise ValueError(f"invalid schedule length in {trajectory.name}")
        np.testing.assert_allclose(
            trajectory.eta,
            expected_schedules[trajectory.name],
            rtol=0.0,
            atol=0.0,
            err_msg=f"stale schedule in {trajectory.name}",
        )
        np.testing.assert_allclose(
            trajectory.initial_u,
            reference_u,
            rtol=2.0e-7,
            atol=2.0e-9,
            err_msg=f"initialization differs in {trajectory.name}",
        )
        if trajectory.risk_paths.shape != (
            trajectory.checkpoint_steps.size,
            problem.paths,
        ):
            raise ValueError(f"invalid risk array in {trajectory.name}")
        np.testing.assert_allclose(
            trajectory.risk_paths[0],
            reference_initial_risk,
            rtol=2.0e-7,
            atol=2.0e-9,
            err_msg=f"initial risk differs in {trajectory.name}",
        )
        if not np.all(np.isfinite(trajectory.risk_paths)):
            raise ValueError(f"nonfinite risk in {trajectory.name}")
    return problem


def design_matrix(
    trajectory: Trajectory,
    problem: Problem,
    signal_contraction: float,
    kernel_contraction: float,
    model: str,
) -> np.ndarray:
    times = np.asarray(
        np.concatenate(([0.0], np.cumsum(trajectory.eta, dtype=np.float64)))
    )
    signal = signal_profile(
        problem, trajectory.eta, times, signal_contraction
    )
    if model == "signal":
        full = signal[:, None]
    elif model in ("shared", "anchored"):
        forcing = signal[:-1] + problem.sigma**2
        convolution = kernel_convolution(
            problem, trajectory.eta, forcing, kernel_contraction
        )
        full = np.column_stack((signal, convolution))
    elif model == "split":
        state = kernel_convolution(
            problem, trajectory.eta, signal[:-1], kernel_contraction
        )
        noise = kernel_convolution(
            problem,
            trajectory.eta,
            np.full(problem.steps, problem.sigma**2),
            kernel_contraction,
        )
        full = np.column_stack((signal, state, noise))
    else:
        raise ValueError(f"unknown model {model}")
    return full[trajectory.checkpoint_steps]


def _relative_weights(target: np.ndarray) -> np.ndarray:
    positive = target[target > 0.0]
    if positive.size == 0:
        raise ValueError("risk target must be positive")
    floor = max(float(np.quantile(positive, 0.05)) * 0.1, 1.0e-14)
    return 1.0 / np.square(np.maximum(target, floor))


def fit_model(
    trajectories: Sequence[Trajectory],
    model: str,
    contractions: Sequence[float],
) -> FitResult:
    calibration = [trajectory for trajectory in trajectories if trajectory.calibration]
    if not calibration:
        raise ValueError("at least one calibration trajectory is required")
    problem = problem_from_metadata(calibration[0].metadata)
    for trajectory in calibration[1:]:
        other = problem_from_metadata(trajectory.metadata)
        comparable = (other.modes, other.steps, other.depth, other.beta, other.chi)
        reference = (problem.modes, problem.steps, problem.depth, problem.beta, problem.chi)
        if comparable != reference:
            raise ValueError("calibration trajectories do not share one problem")

    best: FitResult | None = None
    kernel_values = (1.0,) if model == "signal" else contractions
    for signal_contraction in contractions:
        for kernel_contraction in kernel_values:
            designs = []
            targets = []
            weights = []
            for trajectory in calibration:
                design = design_matrix(
                    trajectory,
                    problem,
                    signal_contraction,
                    kernel_contraction,
                    model,
                )
                target = trajectory.risk_mean
                designs.append(design)
                targets.append(target)
                weights.append(_relative_weights(target) / target.size)
            x = np.concatenate(designs, axis=0)
            y = np.concatenate(targets, axis=0)
            w = np.concatenate(weights, axis=0)
            if model == "anchored":
                remaining, loss = nonnegative_least_squares(
                    x[:, 1:], y - x[:, 0], w
                )
                coefficients = np.concatenate(([1.0], remaining))
                loss = float(np.sum(w * np.square(x @ coefficients - y)))
            else:
                coefficients, loss = nonnegative_least_squares(x, y, w)
            candidate = FitResult(
                model=model,
                signal_contraction=float(signal_contraction),
                kernel_contraction=float(kernel_contraction),
                coefficients=coefficients,
                calibration_loss=loss,
            )
            if best is None or candidate.calibration_loss < best.calibration_loss:
                best = candidate
    assert best is not None
    return best


def fit_volterra(
    trajectories: Sequence[Trajectory], template: FitResult
) -> FitResult:
    """Fit the recursive FSL after selecting its decay constants on calibration data."""

    calibration = [trajectory for trajectory in trajectories if trajectory.calibration]
    problem = problem_from_metadata(calibration[0].metadata)
    center = max(float(template.coefficients[-1]), 1.0e-6)
    feedback_grid = center * np.geomspace(1.0 / 16.0, 4.0, 17)
    best: FitResult | None = None
    for feedback in feedback_grid:
        signal_parts = []
        noise_parts = []
        targets = []
        weights = []
        try:
            for trajectory in calibration:
                signal, noise = volterra_basis(
                    trajectory,
                    problem,
                    template.signal_contraction,
                    template.kernel_contraction,
                    float(feedback),
                )
                target = trajectory.risk_mean
                signal_parts.append(signal)
                noise_parts.append(noise)
                targets.append(target)
                weights.append(_relative_weights(target) / target.size)
        except FloatingPointError:
            continue
        signal_all = np.concatenate(signal_parts)
        noise_all = np.concatenate(noise_parts)
        target_all = np.concatenate(targets)
        weight_all = np.concatenate(weights)
        denominator = float(np.sum(weight_all * np.square(signal_all)))
        if denominator <= 0.0:
            continue
        amplitude = max(
            float(
                np.sum(weight_all * signal_all * (target_all - noise_all))
                / denominator
            ),
            0.0,
        )
        loss = float(
            np.sum(
                weight_all
                * np.square(amplitude * signal_all + noise_all - target_all)
            )
        )
        candidate = FitResult(
            model="volterra",
            signal_contraction=template.signal_contraction,
            kernel_contraction=template.kernel_contraction,
            coefficients=np.asarray([amplitude, feedback], dtype=np.float64),
            calibration_loss=loss,
        )
        if best is None or candidate.calibration_loss < best.calibration_loss:
            best = candidate
    if best is None:
        raise FloatingPointError("every Volterra feedback candidate diverged")
    return best


def predict(trajectory: Trajectory, fit: FitResult) -> np.ndarray:
    problem = problem_from_metadata(trajectory.metadata)
    if fit.model == "volterra":
        signal, noise = volterra_basis(
            trajectory,
            problem,
            fit.signal_contraction,
            fit.kernel_contraction,
            float(fit.coefficients[1]),
        )
        return float(fit.coefficients[0]) * signal + noise
    design = design_matrix(
        trajectory,
        problem,
        fit.signal_contraction,
        fit.kernel_contraction,
        fit.model,
    )
    return design @ fit.coefficients


def checkpoint_times(trajectory: Trajectory) -> np.ndarray:
    times = np.concatenate(([0.0], np.cumsum(trajectory.eta, dtype=np.float64)))
    return times[trajectory.checkpoint_steps]


def time_only_prediction(
    reference: Trajectory,
    trajectory: Trajectory,
    *,
    reference_mean: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate the middle constant trajectory using intrinsic time alone."""

    values = reference.risk_mean if reference_mean is None else reference_mean
    floor = max(float(np.min(values[values > 0.0])) * 1.0e-6, 1.0e-30)
    return np.exp(
        np.interp(
            checkpoint_times(trajectory),
            checkpoint_times(reference),
            np.log(np.maximum(values, floor)),
        )
    )


def trajectory_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    floor = max(float(np.min(observed[observed > 0.0])) * 1.0e-6, 1.0e-30)
    observed_safe = np.maximum(observed, floor)
    predicted_safe = np.maximum(predicted, floor)
    errors = np.log(predicted_safe) - np.log(observed_safe)
    count = observed.size
    cuts = [0, count // 3, 2 * count // 3, count]
    metrics = {
        "log_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "terminal_relative_error": float(
            abs(predicted[-1] - observed[-1]) / observed_safe[-1]
        ),
    }
    for label, left, right in zip(("early", "middle", "terminal"), cuts[:-1], cuts[1:]):
        section = errors[left:right]
        metrics[f"{label}_log_rmse"] = float(np.sqrt(np.mean(np.square(section))))
    return metrics


def _quantile_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "q95": float(np.quantile(array, 0.95)),
    }


def bootstrap_primary_fit(
    trajectories: Sequence[Trajectory],
    fit: FitResult,
    *,
    repetitions: int,
    seed: int,
) -> Dict[str, object]:
    """Paired path bootstrap with the selected contractions held fixed."""

    if repetitions <= 0:
        return {}
    path_counts = {trajectory.risk_paths.shape[1] for trajectory in trajectories}
    if len(path_counts) != 1:
        raise ValueError("paired bootstrap requires equal path counts")
    path_count = path_counts.pop()
    calibration = [trajectory for trajectory in trajectories if trajectory.calibration]
    held_out = [trajectory for trajectory in trajectories if not trajectory.calibration]
    middle = next(
        trajectory for trajectory in trajectories if trajectory.name == "constant_mid"
    )
    problem = problem_from_metadata(calibration[0].metadata)
    designs = {
        trajectory.name: design_matrix(
            trajectory,
            problem,
            fit.signal_contraction,
            fit.kernel_contraction,
            fit.model,
        )
        for trajectory in trajectories
    }
    coefficient_samples = [[] for _ in range(fit.coefficients.size)]
    metric_samples: Dict[str, Dict[str, list[float]]] = {
        trajectory.name: {
            "fsl_log_rmse": [],
            "time_only_log_rmse": [],
            "fsl_terminal_relative_error": [],
        }
        for trajectory in held_out
    }
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        indices = rng.integers(0, path_count, size=path_count)
        means = {
            trajectory.name: trajectory.risk_paths[:, indices].mean(axis=1)
            for trajectory in trajectories
        }
        x_parts = []
        y_parts = []
        w_parts = []
        for trajectory in calibration:
            target = means[trajectory.name]
            x_parts.append(designs[trajectory.name])
            y_parts.append(target)
            w_parts.append(_relative_weights(target) / target.size)
        coefficients, _ = nonnegative_least_squares(
            np.concatenate(x_parts),
            np.concatenate(y_parts),
            np.concatenate(w_parts),
        )
        for index, value in enumerate(coefficients):
            coefficient_samples[index].append(float(value))
        for trajectory in held_out:
            observed = means[trajectory.name]
            fsl_metrics = trajectory_metrics(
                observed, designs[trajectory.name] @ coefficients
            )
            time_metrics = trajectory_metrics(
                observed,
                time_only_prediction(
                    middle, trajectory, reference_mean=means[middle.name]
                ),
            )
            samples = metric_samples[trajectory.name]
            samples["fsl_log_rmse"].append(fsl_metrics["log_rmse"])
            samples["time_only_log_rmse"].append(time_metrics["log_rmse"])
            samples["fsl_terminal_relative_error"].append(
                fsl_metrics["terminal_relative_error"]
            )
    return {
        "repetitions": repetitions,
        "paired_paths": True,
        "contractions_held_fixed": True,
        "coefficient_intervals": [
            _quantile_summary(samples) for samples in coefficient_samples
        ],
        "held_out_metric_intervals": {
            name: {
                metric: _quantile_summary(samples)
                for metric, samples in values.items()
            }
            for name, values in metric_samples.items()
        },
    }


def write_transfer_plot(
    trajectories: Sequence[Trajectory],
    fits: Mapping[str, FitResult],
    output_dir: Path,
) -> Path:
    import os

    cache_dir = output_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    middle = next(
        trajectory for trajectory in trajectories if trajectory.name == "constant_mid"
    )
    held_out = [trajectory for trajectory in trajectories if not trajectory.calibration]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True)
    for axis, trajectory in zip(axes.flat, held_out):
        fraction = trajectory.checkpoint_steps / float(trajectory.eta.size)
        axis.semilogy(fraction, trajectory.risk_mean, color="black", label="SGD mean")
        axis.fill_between(
            fraction,
            np.quantile(trajectory.risk_paths, 0.05, axis=1),
            np.quantile(trajectory.risk_paths, 0.95, axis=1),
            color="black",
            alpha=0.12,
            linewidth=0,
        )
        primary_model = "volterra" if "volterra" in fits else "shared"
        axis.semilogy(
            fraction,
            predict(trajectory, fits[primary_model]),
            label="frozen recursive FSL" if primary_model == "volterra" else "frozen FSL",
        )
        axis.semilogy(
            fraction,
            time_only_prediction(middle, trajectory),
            linestyle="--",
            label="intrinsic-time baseline",
        )
        axis.set_title(trajectory.name.replace("_", " "))
        axis.grid(alpha=0.22)
    for axis in axes[-1, :]:
        axis.set_xlabel("fraction of examples processed")
    for axis in axes[:, 0]:
        axis.set_ylabel("population excess risk")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "held_out_transfer.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def fit_and_report(
    data_dir: Path,
    output_dir: Path,
    *,
    contractions: Sequence[float] | None = None,
    bootstrap_repetitions: int = 0,
) -> Dict[str, object]:
    trajectories = [load_trajectory(path) for path in sorted(data_dir.glob("*.npz"))]
    if not trajectories:
        raise ValueError(f"no trajectories found in {data_dir}")
    problem = validate_trajectories(trajectories)
    grid = (
        np.geomspace(0.25, 128.0, 13)
        if contractions is None
        else np.asarray(contractions, dtype=np.float64)
    )
    fits = {
        model: fit_model(trajectories, model, grid)
        for model in ("signal", "shared", "anchored", "split")
    }
    fits["volterra"] = fit_volterra(trajectories, fits["shared"])
    middle = next(
        trajectory for trajectory in trajectories if trajectory.name == "constant_mid"
    )
    report: Dict[str, object] = {"fits": {}, "trajectories": {}}
    for model, fit in fits.items():
        report["fits"][model] = {
            "signal_contraction": fit.signal_contraction,
            "kernel_contraction": fit.kernel_contraction,
            "coefficients": fit.coefficients.tolist(),
            "calibration_loss": fit.calibration_loss,
        }
    for trajectory in trajectories:
        row: Dict[str, object] = {
            "calibration": trajectory.calibration,
            "metrics": {},
        }
        for model, fit in fits.items():
            row["metrics"][model] = trajectory_metrics(
                trajectory.risk_mean, predict(trajectory, fit)
            )
        row["metrics"]["time_only"] = trajectory_metrics(
            trajectory.risk_mean, time_only_prediction(middle, trajectory)
        )
        shared = row["metrics"]["shared"]["log_rmse"]
        volterra = row["metrics"]["volterra"]["log_rmse"]
        signal = row["metrics"]["signal"]["log_rmse"]
        time_only = row["metrics"]["time_only"]["log_rmse"]
        row["shared_improvement_over_signal"] = float(signal - shared)
        row["shared_improvement_over_time_only"] = float(time_only - shared)
        row["volterra_improvement_over_time_only"] = float(time_only - volterra)
        report["trajectories"][trajectory.name] = row

    report["bootstrap"] = bootstrap_primary_fit(
        trajectories,
        fits["shared"],
        repetitions=bootstrap_repetitions,
        seed=problem.seed + 9187,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_transfer_plot(trajectories, fits, output_dir)
    return report
