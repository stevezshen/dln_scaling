#!/usr/bin/env python3
"""Native-MLX reproduction of the SDE perturbation figure.

The Monte Carlo SDE and perturbative hierarchy both execute on MLX's native
Apple-silicon GPU backend.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from common import (
    CASES,
    CASE_ORDER,
    apply_preset,
    build_parser,
    check_deadline,
    checkpoint_grid,
    configure_mlx,
    host_array,
    host_float,
    initial_value,
    mlx_version,
    plot_results,
    serializable_args,
    validate_args,
)


def phi(mx: Any, t: Any, horizon: float, c1: float, c2: float, nu1: float, nu2: float) -> Any:
    warm = mx.power(mx.maximum(t / (c1 * horizon), 0.0), nu1)
    decay_fraction = mx.maximum((horizon - t) / ((1.0 - c2) * horizon), 0.0)
    decay = mx.power(decay_fraction, nu2)
    return mx.where(t <= c1 * horizon, warm, mx.where(t <= c2 * horizon, 1.0, decay))


def make_problem(mx: Any, case: Any, modes: int, horizon: float, chi: float) -> Tuple[Any, Any, float]:
    indices = mx.arange(1, modes + 1, dtype=mx.float32)
    lambdas = mx.power(indices, -case.beta)
    targets = mx.power(indices, -case.alpha)
    theta = initial_value(case, horizon, chi)
    mx.eval(lambdas, targets)
    return lambdas, targets, theta


def make_sde_step(
    mx: Any,
    *,
    lambdas: Any,
    targets: Any,
    horizon: float,
    depth: int,
    chi: float,
    eta_peak: float,
    sigma: float,
    c1: float,
    c2: float,
    nu1: float,
    nu2: float,
    dt_max: float,
    drift_limit: float,
    diffusion_limit: float,
    block_size: int,
    noise_model: str,
) -> Any:
    l2 = float(depth * depth)

    def step(
        z: Any,
        t: Any,
        key: Any,
        requested_t: Any,
        interval_start: Any,
    ) -> Tuple[Any, Any, Any, Any]:
        steps_taken = mx.zeros(t.shape, dtype=mx.int32)
        time_tolerance = 1.0e-7 * mx.maximum(requested_t, 1.0)
        for _ in range(block_size):
            active = requested_t - t > time_tolerance
            absolute_t = interval_start + t
            u = mx.exp(z)
            risk = 0.5 * mx.sum(
                lambdas[None, :] * mx.square(u - targets[None, :]), axis=1
            )
            residual = u - targets[None, :]
            if noise_model == "full":
                scalar_variance = 2.0 * risk + sigma * sigma
                shared_loading = lambdas[None, :] * residual
                marginal_variance = (
                    scalar_variance[:, None] * lambdas[None, :]
                    + mx.square(shared_loading)
                )
            else:
                scalar_variance = risk + sigma * sigma
                shared_loading = mx.zeros_like(u)
                marginal_variance = scalar_variance[:, None] * lambdas[None, :]
            eta = eta_peak * phi(mx, absolute_t, horizon, c1, c2, nu1, nu2)
            u_chi_minus_1 = mx.power(u, chi - 1.0)
            drift = -l2 * lambdas[None, :] * u_chi_minus_1 * residual
            drift = drift - (
                0.5
                * float(depth * depth * depth)
                * eta[:, None]
                * marginal_variance
                * mx.power(u, 2.0 * chi - 2.0)
            )
            diffusion_norm = (
                l2
                * u_chi_minus_1
                * mx.sqrt(
                    mx.maximum(eta[:, None] * marginal_variance, 0.0)
                )
            )
            max_drift = mx.max(mx.abs(drift), axis=1)
            max_diffusion = mx.max(mx.abs(diffusion_norm), axis=1)
            drift_dt = drift_limit / mx.maximum(max_drift, 1.0e-20)
            diffusion_dt = mx.square(
                diffusion_limit / mx.maximum(max_diffusion, 1.0e-20)
            )
            candidate_dt = mx.minimum(
                mx.maximum(requested_t - t, 0.0),
                mx.minimum(dt_max, mx.minimum(drift_dt, diffusion_dt)),
            )
            dt = mx.where(active, candidate_dt, 0.0)
            if noise_model == "full":
                keys = mx.random.split(key, num=3)
                candidate_key, independent_key, shared_key = keys[0], keys[1], keys[2]
                independent_noise = mx.random.normal(
                    shape=z.shape, dtype=mx.float32, key=independent_key
                )
                shared_noise = mx.random.normal(
                    shape=(z.shape[0], 1), dtype=mx.float32, key=shared_key
                )
                raw_noise = (
                    mx.sqrt(
                        mx.maximum(
                            scalar_variance[:, None] * lambdas[None, :], 0.0
                        )
                    )
                    * independent_noise
                    + shared_loading * shared_noise
                )
                noise_increment = (
                    l2
                    * u_chi_minus_1
                    * mx.sqrt(eta[:, None] * dt[:, None])
                    * raw_noise
                )
            else:
                keys = mx.random.split(key, num=2)
                candidate_key, noise_key = keys[0], keys[1]
                noise = mx.random.normal(
                    shape=z.shape, dtype=mx.float32, key=noise_key
                )
                noise_increment = diffusion_norm * mx.sqrt(dt[:, None]) * noise
            z = z + drift * dt[:, None] + noise_increment
            t = t + dt
            key = mx.where(mx.any(active), candidate_key, key)
            steps_taken = steps_taken + active.astype(mx.int32)
        return z, t, key, steps_taken

    return mx.compile(step)


def make_risk_function(mx: Any, lambdas: Any, targets: Any) -> Any:
    def risk(z: Any) -> Any:
        u = mx.exp(z)
        return 0.5 * mx.sum(
            lambdas[None, :] * mx.square(u - targets[None, :]), axis=1
        )

    return mx.compile(risk)


def make_moment_rhs(
    mx: Any,
    *,
    lambdas: Any,
    targets: Any,
    horizon: float,
    depth: int,
    chi: float,
    sigma: float,
    c1: float,
    c2: float,
    nu1: float,
    nu2: float,
    noise_model: str,
) -> Any:
    l2 = float(depth * depth)
    l4 = l2 * l2

    def rhs(t: Any, y: Any) -> Any:
        u0, p, mean_u2 = y[0], y[1], y[2]
        u = mx.maximum(u0, 1.0e-30)
        residual = u - targets
        risk0 = 0.5 * mx.sum(lambdas * mx.square(residual))
        if noise_model == "full":
            marginal_variance = (
                (2.0 * risk0 + sigma * sigma) * lambdas
                + mx.square(lambdas * residual)
            )
        else:
            marginal_variance = (risk0 + sigma * sigma) * lambdas
        schedule = phi(mx, t, horizon, c1, c2, nu1, nu2)
        a = -l2 * lambdas * (
            mx.power(u, chi) + chi * mx.power(u, chi - 1.0) * residual
        )
        c = -l2 * lambdas * (
            chi * mx.power(u, chi - 1.0)
            + 0.5
            * chi
            * (chi - 1.0)
            * mx.power(u, chi - 2.0)
            * residual
        )
        d = (
            0.25
            * chi
            * l4
            * schedule
            * mx.power(u, 2.0 * chi - 1.0)
            * marginal_variance
        )
        du0 = -l2 * lambdas * mx.power(u, chi) * residual
        dp = (
            2.0 * a * p
            + l4
            * schedule
            * mx.power(u, 2.0 * chi)
            * marginal_variance
        )
        dmean_u2 = a * mean_u2 + c * p + d
        return mx.stack((du0, dp, dmean_u2))

    return rhs


def make_dopri_proposal(mx: Any, rhs: Any, atol: float, rtol: float) -> Any:
    def proposal(t: Any, y: Any, dt: Any) -> Tuple[Any, Any]:
        k1 = rhs(t, y)
        k2 = rhs(t + dt * (1.0 / 5.0), y + dt * ((1.0 / 5.0) * k1))
        k3 = rhs(
            t + dt * (3.0 / 10.0),
            y + dt * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2),
        )
        k4 = rhs(
            t + dt * (4.0 / 5.0),
            y
            + dt
            * (
                (44.0 / 45.0) * k1
                + (-56.0 / 15.0) * k2
                + (32.0 / 9.0) * k3
            ),
        )
        k5 = rhs(
            t + dt * (8.0 / 9.0),
            y
            + dt
            * (
                (19372.0 / 6561.0) * k1
                + (-25360.0 / 2187.0) * k2
                + (64448.0 / 6561.0) * k3
                + (-212.0 / 729.0) * k4
            ),
        )
        k6 = rhs(
            t + dt,
            y
            + dt
            * (
                (9017.0 / 3168.0) * k1
                + (-355.0 / 33.0) * k2
                + (46732.0 / 5247.0) * k3
                + (49.0 / 176.0) * k4
                + (-5103.0 / 18656.0) * k5
            ),
        )
        y5 = y + dt * (
            (35.0 / 384.0) * k1
            + (500.0 / 1113.0) * k3
            + (125.0 / 192.0) * k4
            + (-2187.0 / 6784.0) * k5
            + (11.0 / 84.0) * k6
        )
        k7 = rhs(t + dt, y5)
        y4 = y + dt * (
            (5179.0 / 57600.0) * k1
            + (7571.0 / 16695.0) * k3
            + (393.0 / 640.0) * k4
            + (-92097.0 / 339200.0) * k5
            + (187.0 / 2100.0) * k6
            + (1.0 / 40.0) * k7
        )
        scale = atol + rtol * mx.maximum(mx.abs(y), mx.abs(y5))
        error_ratio = mx.max(mx.abs(y5 - y4) / scale)
        return y5, error_ratio

    return proposal


def make_dopri_block(
    mx: Any,
    proposal: Any,
    *,
    block_size: int,
    horizon: float,
) -> Any:
    """Compile several adaptive ODE trials into one Metal dispatch."""

    time_tolerance = 1.0e-7 * max(horizon, 1.0)

    def block(y: Any, t: Any, dt: Any, target: Any) -> Tuple[Any, ...]:
        trials_taken = mx.array(0, dtype=mx.int32)
        all_finite = mx.array(True)
        for _ in range(block_size):
            active = target - t > time_tolerance
            step_dt = mx.minimum(dt, mx.maximum(target - t, 0.0))
            proposed_y, ratio = proposal(t, y, step_dt)
            finite = mx.isfinite(ratio) & mx.all(mx.isfinite(proposed_y))
            all_finite = all_finite & mx.where(active, finite, True)
            accepted = active & finite & (ratio <= 1.0)
            safe_ratio = mx.where(finite, mx.maximum(ratio, 1.0e-12), mx.inf)
            factor = mx.minimum(
                5.0,
                mx.maximum(0.2, 0.9 * mx.power(safe_ratio, -0.2)),
            )
            proposed_dt = mx.maximum(step_dt * factor, 1.0e-10)
            clipped_by_checkpoint = step_dt < 0.999999 * dt
            candidate_dt = mx.where(
                accepted & clipped_by_checkpoint,
                mx.maximum(dt, proposed_dt),
                proposed_dt,
            )
            candidate_y = mx.concatenate(
                (mx.maximum(proposed_y[0:1], 1.0e-30), proposed_y[1:]), axis=0
            )
            y = mx.where(accepted, candidate_y, y)
            t = t + mx.where(accepted, step_dt, 0.0)
            dt = mx.where(active, candidate_dt, dt)
            trials_taken = trials_taken + active.astype(mx.int32)
        return y, t, dt, trials_taken, all_finite

    return mx.compile(block)


def advance_moments(
    mx: Any,
    block: Any,
    y: Any,
    t: float,
    dt: float,
    trials: int,
    target: float,
    max_trials: int,
    started: float,
    max_wall_minutes: float,
    context: str,
    time_tolerance: float,
) -> Tuple[Any, float, float, int]:
    while target - t > time_tolerance:
        if trials >= max_trials:
            raise RuntimeError(f"Moment ODE exceeded {max_trials} trials at t={t:g}.")
        previous_t = t
        y, device_t, device_dt, block_trials, all_finite = block(
            y,
            mx.array(t, dtype=mx.float32),
            mx.array(dt, dtype=mx.float32),
            mx.array(target, dtype=mx.float32),
        )
        mx.eval(y, device_t, device_dt, block_trials, all_finite)
        if not bool(host_float(mx, all_finite)):
            raise FloatingPointError(f"Non-finite moment proposal at t={t:g}.")
        t = host_float(mx, device_t)
        dt = host_float(mx, device_dt)
        trials_taken = int(host_float(mx, block_trials))
        trials += trials_taken
        if trials_taken <= 0 or not math.isfinite(t) or t < previous_t:
            raise FloatingPointError(
                f"Invalid MLX moment block at t={previous_t}; new_t={t}."
            )
        check_deadline(started, max_wall_minutes, context)
    return y, t, dt, trials


def simulate_case(
    args: Any,
    case: Any,
    *,
    mx: Any,
    started: float,
    case_index: int,
) -> Dict[str, Any]:
    import numpy as np

    chi = 2.0 - 2.0 / args.depth
    lambdas, targets, theta = make_problem(mx, case, args.modes, args.horizon, chi)
    times = checkpoint_grid(args.horizon, args.checkpoints)
    risk_fn = make_risk_function(mx, lambdas, targets)
    sde_step = make_sde_step(
        mx,
        lambdas=lambdas,
        targets=targets,
        horizon=args.horizon,
        depth=args.depth,
        chi=chi,
        eta_peak=args.eta_peak,
        sigma=args.sigma,
        c1=args.c1,
        c2=args.c2,
        nu1=args.nu1,
        nu2=args.nu2,
        dt_max=args.sde_dt_max,
        drift_limit=args.drift_increment_limit,
        diffusion_limit=args.diffusion_std_limit,
        block_size=args.sde_block_size,
        noise_model=args.noise_model,
    )
    rhs = make_moment_rhs(
        mx,
        lambdas=lambdas,
        targets=targets,
        horizon=args.horizon,
        depth=args.depth,
        chi=chi,
        sigma=args.sigma,
        c1=args.c1,
        c2=args.c2,
        nu1=args.nu1,
        nu2=args.nu2,
        noise_model=args.noise_model,
    )
    proposal = make_dopri_proposal(mx, rhs, args.ode_atol, args.ode_rtol)
    ode_block = make_dopri_block(
        mx,
        proposal,
        block_size=args.ode_block_size,
        horizon=args.horizon,
    )

    z = mx.full((args.paths, args.modes), math.log(theta), dtype=mx.float32)
    key = mx.random.key(args.seed + 1009 * case_index)
    mx.eval(z, key)
    sde_steps_by_path = np.zeros(args.paths, dtype=np.int64)

    u0 = mx.full((args.modes,), theta, dtype=mx.float32)
    ode_y = mx.stack((u0, mx.zeros_like(u0), mx.zeros_like(u0)))
    mx.eval(ode_y)
    ode_t = 0.0
    ode_dt = args.ode_dt_initial
    ode_trials = 0

    empirical_paths: List[Any] = []
    gf_values: List[float] = []
    perturb_values: List[float] = []

    initial_risks = host_array(mx, risk_fn(z)).astype(np.float64)
    empirical_paths.append(initial_risks)
    initial_gf = 0.5 * mx.sum(lambdas * mx.square(u0 - targets))
    gf_values.append(host_float(mx, initial_gf))
    perturb_values.append(host_float(mx, initial_gf))
    next_progress = (
        started + 60.0 * args.progress_minutes if args.progress_minutes > 0 else math.inf
    )

    for sample_index, target_time_raw in enumerate(times[1:], start=1):
        target_time = float(target_time_raw)
        interval_start = float(times[sample_index - 1])
        interval_length = target_time - interval_start
        interval_tolerance = 1.0e-7 * max(interval_length, 1.0)
        sde_t = mx.zeros((args.paths,), dtype=mx.float32)
        mx.eval(sde_t)
        interval_elapsed = 0.0
        while interval_elapsed + interval_tolerance < interval_length:
            if int(sde_steps_by_path.max()) >= args.max_sde_steps:
                raise RuntimeError(
                    f"SDE exceeded {args.max_sde_steps} steps at t={target_time:g}."
                )
            relative_target = interval_length - interval_elapsed
            z, sde_t, key, block_steps = sde_step(
                z,
                sde_t,
                key,
                mx.array(relative_target, dtype=mx.float32),
                mx.array(interval_start + interval_elapsed, dtype=mx.float32),
            )
            mx.eval(z, sde_t, key, block_steps)
            sde_times_host = host_array(mx, sde_t).astype(np.float64)
            clock_advance = float(sde_times_host.min())
            steps_host = host_array(mx, block_steps).astype(np.int64)
            if (
                int(steps_host.max()) <= 0
                or not np.all(np.isfinite(sde_times_host))
                or clock_advance <= 0.0
            ):
                raise FloatingPointError(
                    f"Invalid MLX SDE block at t={interval_elapsed}; "
                    f"clock advance={clock_advance}."
                )
            # Keep the device clocks close to zero.  Without this rebase, a
            # valid adaptive dt can fall below the float32 ulp of a clock that
            # has already advanced several intrinsic-time units.
            sde_t = mx.maximum(
                sde_t - mx.array(clock_advance, dtype=mx.float32), 0.0
            )
            mx.eval(sde_t)
            interval_elapsed += clock_advance
            sde_steps_by_path += steps_host
            check_deadline(started, args.max_wall_minutes, f"{case.name} SDE")

        ode_y, ode_t, ode_dt, ode_trials = advance_moments(
            mx,
            ode_block,
            ode_y,
            ode_t,
            ode_dt,
            ode_trials,
            target_time,
            args.max_ode_trials,
            started,
            args.max_wall_minutes,
            f"{case.name} moment ODE",
            1.0e-7 * max(args.horizon, 1.0),
        )
        risks = host_array(mx, risk_fn(z)).astype(np.float64)
        if not np.all(np.isfinite(risks)):
            raise FloatingPointError(f"Non-finite SDE risk in {case.name}.")
        current_u0, current_p, current_mean_u2 = ode_y[0], ode_y[1], ode_y[2]
        residual = current_u0 - targets
        gf = 0.5 * mx.sum(lambdas * mx.square(residual))
        correction = 0.5 * args.eta_peak * mx.sum(
            lambdas * (current_p + 2.0 * residual * current_mean_u2)
        )
        perturb = gf + correction
        mx.eval(gf, perturb)
        perturb_host = host_float(mx, perturb)
        if not math.isfinite(perturb_host):
            raise FloatingPointError(f"Non-finite perturbative risk in {case.name}.")
        empirical_paths.append(risks)
        gf_values.append(host_float(mx, gf))
        perturb_values.append(perturb_host)
        check_deadline(
            started,
            args.max_wall_minutes,
            f"{case.name}, checkpoint {sample_index}/{len(times) - 1}",
        )
        now = time.monotonic()
        if now >= next_progress:
            print(
                f"progress {case.name}: {100.0 * target_time / args.horizon:.1f}%; "
                f"SDE steps={int(sde_steps_by_path.max()):,}; "
                f"elapsed={(now - started) / 60.0:.1f} min",
                flush=True,
            )
            next_progress = now + 60.0 * args.progress_minutes

    paths_by_time = np.stack(empirical_paths, axis=0)
    q05, q95 = np.quantile(paths_by_time, (0.05, 0.95), axis=1)
    return {
        "times": times,
        "sde_paths": paths_by_time,
        "sde_mean": paths_by_time.mean(axis=1),
        "q05": q05,
        "q95": q95,
        "gradient_flow": np.asarray(gf_values, dtype=np.float64),
        "perturbation": np.asarray(perturb_values, dtype=np.float64),
        "theta": theta,
        "alpha": case.alpha,
        "chi": chi,
        "sde_steps": int(sde_steps_by_path.max()),
        "sde_steps_mean": float(sde_steps_by_path.mean()),
        "ode_trials": ode_trials,
    }


def save_case(output_dir: Path, case: Any, result: Mapping[str, Any]) -> Path:
    import numpy as np

    output_path = output_dir / f"{case.name}.npz"
    np.savez_compressed(
        output_path,
        times=result["times"],
        sde_paths=result["sde_paths"],
        sde_mean=result["sde_mean"],
        q05=result["q05"],
        q95=result["q95"],
        gradient_flow=result["gradient_flow"],
        perturbation=result["perturbation"],
    )
    return output_path


def load_case(output_dir: Path, case_name: str) -> Dict[str, Any]:
    import numpy as np

    path = output_dir / f"{case_name}.npz"
    if not path.exists():
        raise SystemExit(f"Missing saved case for --plot-only: {path}")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = apply_preset(build_parser().parse_args(argv))
    validate_args(args)
    args.output_root = Path(args.output_root)
    data_dir = args.output_root / "data" / "sde"
    figure_dir = args.output_root / "figures"
    metadata_dir = args.output_root / "metadata"
    for directory in (data_dir, figure_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        saved_results = {
            case_name: load_case(data_dir, case_name)
            for case_name in args.cases
        }
        png_path, pdf_path = plot_results(figure_dir, saved_results, args)
        print(f"rebuilt {png_path} and {pdf_path}", flush=True)
        return 0

    mx, device = configure_mlx()
    started = time.monotonic()
    print(
        f"backend=mlx; device={device}; preset={args.preset}; modes={args.modes}; "
        f"paths={args.paths}; T={args.horizon:g}",
        flush=True,
    )
    results: Dict[str, Dict[str, Any]] = {}
    case_files: List[str] = []
    for case_name in args.cases:
        check_deadline(started, args.max_wall_minutes, f"before {case_name}")
        print(f"starting {case_name}", flush=True)
        result = simulate_case(
            args,
            CASES[case_name],
            mx=mx,
            started=started,
            case_index=CASE_ORDER.index(case_name),
        )
        results[case_name] = result
        case_files.append(str(save_case(data_dir, CASES[case_name], result)))
        print(
            f"finished {case_name}: SDE steps={result['sde_steps']}, "
            f"ODE trials={result['ode_trials']}, final mean={result['sde_mean'][-1]:.6g}, "
            f"final second-order={result['perturbation'][-1]:.6g}",
            flush=True,
        )

    figure_files: List[str] = []
    if not args.no_plot:
        png_path, pdf_path = plot_results(figure_dir, results, args)
        figure_files.extend((str(png_path), str(pdf_path)))
    elapsed = time.monotonic() - started
    metadata = {
        "arguments": serializable_args(args),
        "backend": "mlx",
        "device": device,
        "mlx_version": mlx_version(),
        "elapsed_seconds": elapsed,
        "case_files": case_files,
        "figure_files": figure_files,
        "notes": [
            "depth=10 is inferred from the adjacent numerical experiments and source figure scale; draft_new.tex omits L in this caption",
            "MLX uses float32 on its native Apple-silicon GPU backend",
            "independent paths use independent adaptive clocks within each Metal batch",
        ],
        "cases": {
            name: {
                "alpha": result["alpha"],
                "chi": result["chi"],
                "theta": result["theta"],
                "sde_steps": result["sde_steps"],
                "sde_steps_mean": result["sde_steps_mean"],
                "ode_trials": result["ode_trials"],
                "final_sde_mean": float(result["sde_mean"][-1]),
                "final_gradient_flow": float(result["gradient_flow"][-1]),
                "final_perturbation": float(result["perturbation"][-1]),
            }
            for name, result in results.items()
        },
    }
    metadata_path = metadata_dir / "sde.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"completed in {elapsed:.2f}s; metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
