"""Fresh-sample one-pass SGD simulation on MLX Metal or explicit NumPy."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from core import (
    Problem,
    checkpoint_indices,
    initialization,
    initialization_is_admissible,
    spectral_arrays,
)


def _risk_paths(
    a: np.ndarray, depth: int, lambdas: np.ndarray, target: np.ndarray
) -> np.ndarray:
    u = np.power(a, depth)
    return np.sum(lambdas[None, :] * np.square(u - target[None, :]), axis=1)


def simulate_numpy(problem: Problem, eta: np.ndarray) -> Dict[str, np.ndarray]:
    """Small reference implementation; selected only with --backend numpy."""

    problem.validate()
    if not initialization_is_admissible(problem, eta):
        raise ValueError("initialization exceeds the target in at least one coordinate")
    lambdas, target = spectral_arrays(problem)
    sqrt_lambdas = np.sqrt(lambdas)
    u0 = initialization(problem, eta)
    a = np.broadcast_to(
        np.power(u0, 1.0 / problem.depth), (problem.paths, problem.modes)
    ).copy()
    rng = np.random.default_rng(problem.seed)
    checkpoints = checkpoint_indices(problem.steps, problem.checkpoints)
    risks = [_risk_paths(a, problem.depth, lambdas, target)]
    min_a = float(np.min(a))
    max_relative = 0.0
    checkpoint_cursor = 1
    for n in range(problem.steps):
        try:
            with np.errstate(over="raise", invalid="raise"):
                features = (
                    rng.standard_normal((problem.paths, problem.modes))
                    * sqrt_lambdas
                )
                label_noise = problem.sigma * rng.standard_normal(problem.paths)
                u = np.power(a, problem.depth)
                residual = (
                    np.sum((u - target[None, :]) * features, axis=1)
                    - label_noise
                )
                gradient_factor = features * residual[:, None]
                relative = (
                    problem.depth
                    * float(eta[n])
                    * np.power(a, problem.depth - 2)
                    * gradient_factor
                )
                candidate = a - (
                    problem.depth
                    * float(eta[n])
                    * np.power(a, problem.depth - 1)
                    * gradient_factor
                )
        except FloatingPointError as exc:
            raise FloatingPointError(f"unstable SGD update at step {n}") from exc
        if not np.all(np.isfinite(candidate)) or float(np.min(candidate)) <= 0.0:
            raise FloatingPointError(
                f"layer coordinate became nonpositive or nonfinite at step {n}"
            )
        max_relative = max(max_relative, float(np.max(np.abs(relative))))
        a = candidate
        min_a = min(min_a, float(np.min(a)))
        if checkpoint_cursor < len(checkpoints) and n + 1 == checkpoints[checkpoint_cursor]:
            risks.append(_risk_paths(a, problem.depth, lambdas, target))
            checkpoint_cursor += 1
    return {
        "checkpoint_steps": checkpoints,
        "risk_paths": np.stack(risks, axis=0),
        "minimum_layer_coordinate": np.asarray(min_a),
        "maximum_relative_update": np.asarray(max_relative),
        "initial_u": u0,
    }


def _configure_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("MLX is unavailable; install fsl_transfer/requirements.txt") from exc
    mx.set_default_device(mx.gpu)
    if "gpu" not in str(mx.default_device()).lower():
        raise RuntimeError(f"refusing non-GPU MLX device {mx.default_device()}")
    return mx


def simulate_mlx(
    problem: Problem, eta: np.ndarray, *, block_size: int = 128
) -> Dict[str, np.ndarray]:
    problem.validate()
    if not initialization_is_admissible(problem, eta):
        raise ValueError("initialization exceeds the target in at least one coordinate")
    mx = _configure_mlx()
    lambdas_np, target_np = spectral_arrays(problem)
    u0_np = initialization(problem, eta)
    lambdas = mx.array(lambdas_np.astype(np.float32))
    sqrt_lambdas = mx.sqrt(lambdas)
    target = mx.array(target_np.astype(np.float32))
    etas = mx.array(eta.astype(np.float32))
    a = mx.broadcast_to(
        mx.array(np.power(u0_np, 1.0 / problem.depth).astype(np.float32))[None, :],
        (problem.paths, problem.modes),
    )
    key = mx.random.key(problem.seed)
    step = mx.array(0, dtype=mx.int32)
    min_seen = mx.min(a)
    max_relative = mx.array(0.0, dtype=mx.float32)

    def block(
        state: Any, k: Any, random_key: Any, requested: Any,
        minimum: Any, maximum_relative: Any,
    ) -> tuple[Any, ...]:
        for _ in range(block_size):
            active = k < requested
            safe_k = mx.minimum(k, problem.steps - 1)
            keys = mx.random.split(random_key, num=3)
            next_key, feature_key, noise_key = keys[0], keys[1], keys[2]
            features = mx.random.normal(
                shape=(problem.paths, problem.modes),
                dtype=mx.float32,
                key=feature_key,
            ) * sqrt_lambdas[None, :]
            noise = problem.sigma * mx.random.normal(
                shape=(problem.paths,), dtype=mx.float32, key=noise_key
            )
            u = mx.power(state, problem.depth)
            residual = mx.sum((u - target[None, :]) * features, axis=1) - noise
            gradient_factor = features * residual[:, None]
            eta_k = etas[safe_k]
            relative = (
                float(problem.depth)
                * eta_k
                * mx.power(state, problem.depth - 2)
                * gradient_factor
            )
            candidate = state - (
                float(problem.depth)
                * eta_k
                * mx.power(state, problem.depth - 1)
                * gradient_factor
            )
            state = mx.where(active, candidate, state)
            minimum = mx.where(active, mx.minimum(minimum, mx.min(candidate)), minimum)
            maximum_relative = mx.where(
                active,
                mx.maximum(maximum_relative, mx.max(mx.abs(relative))),
                maximum_relative,
            )
            k = k + active.astype(mx.int32)
            random_key = mx.where(active, next_key, random_key)
        return state, k, random_key, minimum, maximum_relative

    compiled_block = mx.compile(block)

    def risk() -> np.ndarray:
        u = mx.power(a, problem.depth)
        value = mx.sum(
            lambdas[None, :] * mx.square(u - target[None, :]), axis=1
        )
        mx.eval(value)
        return np.asarray(value).astype(np.float64)

    checkpoints = checkpoint_indices(problem.steps, problem.checkpoints)
    risks = [risk()]
    current = 0
    for requested in checkpoints[1:]:
        while current < int(requested):
            a, step, key, min_seen, max_relative = compiled_block(
                a,
                step,
                key,
                mx.array(int(requested), dtype=mx.int32),
                min_seen,
                max_relative,
            )
            mx.eval(a, step, key, min_seen, max_relative)
            new_current = int(np.asarray(step))
            if new_current <= current:
                raise RuntimeError("compiled MLX block made no progress")
            current = new_current
        values = risk()
        minimum_value = float(np.asarray(min_seen))
        maximum_value = float(np.asarray(max_relative))
        if (
            not np.all(np.isfinite(values))
            or not np.isfinite(minimum_value)
            or minimum_value <= 0.0
        ):
            raise FloatingPointError(
                f"nonpositive coordinate or nonfinite value at step {current}; "
                f"minimum={minimum_value:.6g}, maximum relative update={maximum_value:.6g}"
            )
        risks.append(values)
    return {
        "checkpoint_steps": checkpoints,
        "risk_paths": np.stack(risks, axis=0),
        "minimum_layer_coordinate": np.asarray(min_seen).astype(np.float64),
        "maximum_relative_update": np.asarray(max_relative).astype(np.float64),
        "initial_u": u0_np,
    }


def save_trajectory(
    output_dir: Path,
    name: str,
    problem: Problem,
    eta: np.ndarray,
    result: Mapping[str, np.ndarray],
    *,
    backend: str,
) -> Path:
    risk_paths = np.asarray(result["risk_paths"], dtype=np.float64)
    minimum = float(np.asarray(result["minimum_layer_coordinate"]))
    if not np.all(np.isfinite(risk_paths)) or not np.isfinite(minimum) or minimum <= 0.0:
        raise FloatingPointError("refusing to save an unstable trajectory")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.npz"
    checkpoint_steps = np.asarray(result["checkpoint_steps"], dtype=np.int64)
    times = np.concatenate(([0.0], np.cumsum(eta, dtype=np.float64)))
    metadata = {
        "name": name,
        "backend": backend,
        "calibration": name.startswith("constant_"),
        **problem.__dict__,
        "alpha": problem.alpha,
        "p": problem.p,
        "s": problem.s,
        "total_intrinsic_time": float(times[-1]),
    }
    np.savez_compressed(
        path,
        eta=eta.astype(np.float32),
        intrinsic_times=times,
        checkpoint_steps=checkpoint_steps,
        checkpoint_times=times[checkpoint_steps],
        risk_paths=risk_paths,
        risk_mean=risk_paths.mean(axis=1),
        risk_q05=np.quantile(risk_paths, 0.05, axis=1),
        risk_q95=np.quantile(risk_paths, 0.95, axis=1),
        minimum_layer_coordinate=result["minimum_layer_coordinate"],
        maximum_relative_update=result["maximum_relative_update"],
        initial_u=result["initial_u"],
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path
