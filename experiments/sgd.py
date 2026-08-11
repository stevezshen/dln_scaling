#!/usr/bin/env python3
"""Compare one-pass SGD with the saved SDE and perturbation trajectories.

The SGD simulation uses fresh Gaussian features and label noise at every
update.  It therefore retains the sample-gradient correlations discarded by
the diagonal-noise SDE approximation in draft_new.tex.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from common import (
    CASES,
    CASE_ORDER,
    check_deadline,
    configure_mlx,
    host_array,
    host_float,
    initial_value,
    mlx_version,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native-MLX one-pass SGD comparison for the SDE figure."
    )
    parser.add_argument(
        "--cases", nargs="+", choices=CASE_ORDER, default=["easy-horizon"]
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--modes", type=int, default=512)
    parser.add_argument("--paths", type=int, default=8)
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Fresh samples per optimizer update; batch 1 is exact one-pass SGD.",
    )
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--horizon", type=float, default=10_000.0)
    parser.add_argument("--eta-peak", type=float, default=1.0e-3)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--c1", type=float, default=0.3)
    parser.add_argument("--c2", type=float, default=0.7)
    parser.add_argument("--nu1", type=float, default=0.8)
    parser.add_argument("--nu2", type=float, default=0.8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-wall-minutes", type=float, default=28.0)
    parser.add_argument(
        "--progress-minutes",
        type=float,
        default=0.0,
        help="Print checkpoint progress at this wall-clock interval; zero disables it.",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--confirm-long-run", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild plots from existing SGD and SDE NPZ files.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "modes",
        "paths",
        "batch_size",
        "depth",
        "horizon",
        "eta_peak",
        "sigma",
        "block_size",
        "max_wall_minutes",
    )
    bad = [name for name in positive if getattr(args, name) <= 0]
    if bad:
        raise SystemExit("Expected positive values for: " + ", ".join(bad))
    if args.depth < 2:
        raise SystemExit("Require depth >= 2.")
    if not 0.0 < args.c1 < args.c2 < 1.0:
        raise SystemExit("Require 0 < c1 < c2 < 1.")
    if not 0.0 < args.nu1 < 1.0 or not 0.0 < args.nu2 < 1.0:
        raise SystemExit("Require 0 < nu1, nu2 < 1.")
    if args.progress_minutes < 0:
        raise SystemExit("Require progress-minutes >= 0.")


def physical_wsd_parameters(args: argparse.Namespace) -> Dict[str, float]:
    """Invert Eq. (WSD-intrinsic) to obtain the physical-step schedule."""

    gamma1 = args.nu1 / (1.0 - args.nu1)
    gamma2 = args.nu2 / (1.0 - args.nu2)
    denominator = (
        (gamma1 + 1.0) * args.c1
        + (args.c2 - args.c1)
        + (gamma2 + 1.0) * (1.0 - args.c2)
    )
    z_wsd = 1.0 / denominator
    b1 = (gamma1 + 1.0) * args.c1 * z_wsd
    b2 = b1 + (args.c2 - args.c1) * z_wsd
    step_eta_peak = args.batch_size * args.eta_peak
    steps = int(round(args.horizon / (step_eta_peak * z_wsd)))
    return {
        "gamma1": gamma1,
        "gamma2": gamma2,
        "z_wsd": z_wsd,
        "b1": b1,
        "b2": b2,
        "steps": steps,
        "fresh_samples": steps * args.batch_size,
        "batch_size": args.batch_size,
        "step_eta_peak": step_eta_peak,
    }


def intrinsic_times_to_steps(
    times: Any,
    *,
    horizon: float,
    steps: int,
    c1: float,
    c2: float,
    gamma1: float,
    gamma2: float,
    b1: float,
    b2: float,
    z_wsd: float,
) -> Any:
    import numpy as np

    r = np.asarray(times, dtype=np.float64) / horizon
    physical_fraction = np.empty_like(r)
    warm = r <= c1
    stable = (r > c1) & (r <= c2)
    decay = r > c2
    physical_fraction[warm] = b1 * np.power(
        np.maximum(r[warm] / c1, 0.0), 1.0 / (gamma1 + 1.0)
    )
    physical_fraction[stable] = b1 + z_wsd * (r[stable] - c1)
    physical_fraction[decay] = 1.0 - (1.0 - b2) * np.power(
        np.maximum((1.0 - r[decay]) / (1.0 - c2), 0.0),
        1.0 / (gamma2 + 1.0),
    )
    targets = np.rint(steps * physical_fraction).astype(np.int64)
    targets[0] = 0
    targets[-1] = steps
    return np.maximum.accumulate(targets)


def load_sde_case(sde_dir: Path, case_name: str) -> Dict[str, Any]:
    import numpy as np

    path = sde_dir / f"{case_name}.npz"
    if not path.exists():
        raise SystemExit(f"Missing SDE result: {path}")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def make_sgd_block(
    mx: Any,
    *,
    lambdas: Any,
    targets: Any,
    depth: int,
    sigma: float,
    step_eta_peak: float,
    batch_size: int,
    total_steps: int,
    b1: float,
    b2: float,
    gamma1: float,
    gamma2: float,
    block_size: int,
) -> Any:
    sqrt_lambdas = mx.sqrt(lambdas)
    total_steps_f = float(total_steps)

    def physical_eta(k: Any) -> Any:
        fraction = k.astype(mx.float32) / total_steps_f
        warm = step_eta_peak * mx.power(
            mx.maximum(fraction / b1, 0.0), gamma1
        )
        decay = step_eta_peak * mx.power(
            mx.maximum((1.0 - fraction) / (1.0 - b2), 0.0), gamma2
        )
        return mx.where(
            fraction < b1,
            warm,
            mx.where(fraction < b2, step_eta_peak, decay),
        )

    def block(a: Any, k: Any, key: Any, requested_k: Any) -> Tuple[Any, ...]:
        updates = mx.array(0, dtype=mx.int32)
        for _ in range(block_size):
            active = k < requested_k
            keys = mx.random.split(key, num=3)
            candidate_key, feature_key, noise_key = keys[0], keys[1], keys[2]
            features = (
                mx.random.normal(
                    shape=(a.shape[0], batch_size, a.shape[1]),
                    dtype=mx.float32,
                    key=feature_key,
                )
                * sqrt_lambdas[None, None, :]
            )
            label_noise = sigma * mx.random.normal(
                shape=(a.shape[0], batch_size), dtype=mx.float32, key=noise_key
            )
            u = mx.power(a, depth)
            residual = mx.sum(
                (u[:, None, :] - targets[None, None, :]) * features, axis=2
            ) - label_noise
            sample_gradient = (
                float(depth)
                * mx.power(a, depth - 1)[:, None, :]
                * features
                * residual[:, :, None]
            )
            gradient = mx.mean(sample_gradient, axis=1)
            candidate = a - physical_eta(k) * gradient
            a = mx.where(active, candidate, a)
            k = k + active.astype(mx.int32)
            key = mx.where(active, candidate_key, key)
            updates = updates + active.astype(mx.int32)
        return a, k, key, updates

    return mx.compile(block)


def risk_paths(mx: Any, a: Any, depth: int, lambdas: Any, targets: Any) -> Any:
    u = mx.power(a, depth)
    return 0.5 * mx.sum(
        lambdas[None, :] * mx.square(u - targets[None, :]), axis=1
    )


def simulate_sgd_case(
    args: argparse.Namespace,
    case_name: str,
    sde: Mapping[str, Any],
    schedule: Mapping[str, float],
    *,
    mx: Any,
    started: float,
) -> Dict[str, Any]:
    import numpy as np

    case = CASES[case_name]
    chi = 2.0 - 2.0 / args.depth
    theta = initial_value(case, args.horizon, chi)
    indices = mx.arange(1, args.modes + 1, dtype=mx.float32)
    lambdas = mx.power(indices, -case.beta)
    targets = mx.power(indices, -case.alpha)
    a = mx.full(
        (args.paths, args.modes), theta ** (1.0 / args.depth), dtype=mx.float32
    )
    key = mx.random.key(args.seed + 7919 * CASE_ORDER.index(case_name))
    step = mx.array(0, dtype=mx.int32)
    mx.eval(lambdas, targets, a, key, step)

    times = np.asarray(sde["times"], dtype=np.float32)
    target_steps = intrinsic_times_to_steps(
        times,
        horizon=args.horizon,
        steps=int(schedule["steps"]),
        c1=args.c1,
        c2=args.c2,
        gamma1=schedule["gamma1"],
        gamma2=schedule["gamma2"],
        b1=schedule["b1"],
        b2=schedule["b2"],
        z_wsd=schedule["z_wsd"],
    )
    advance = make_sgd_block(
        mx,
        lambdas=lambdas,
        targets=targets,
        depth=args.depth,
        sigma=args.sigma,
        step_eta_peak=schedule["step_eta_peak"],
        batch_size=args.batch_size,
        total_steps=int(schedule["steps"]),
        b1=schedule["b1"],
        b2=schedule["b2"],
        gamma1=schedule["gamma1"],
        gamma2=schedule["gamma2"],
        block_size=args.block_size,
    )

    path_values = []
    initial = host_array(mx, risk_paths(mx, a, args.depth, lambdas, targets))
    path_values.append(initial.astype(np.float64))
    step_host = 0
    next_progress = (
        started + 60.0 * args.progress_minutes if args.progress_minutes > 0 else math.inf
    )
    for checkpoint, requested in enumerate(target_steps[1:], start=1):
        requested_int = int(requested)
        while step_host < requested_int:
            previous = step_host
            a, step, key, updates = advance(
                a, step, key, mx.array(requested_int, dtype=mx.int32)
            )
            mx.eval(a, step, key, updates)
            step_host = int(host_float(mx, step))
            updates_host = int(host_float(mx, updates))
            if updates_host <= 0 or step_host <= previous:
                raise FloatingPointError(
                    f"Invalid SGD block at step {previous}; new step={step_host}."
                )
            if not np.all(np.isfinite(host_array(mx, a))):
                raise FloatingPointError(
                    f"Non-finite SGD state in {case_name} at step {step_host}."
                )
            check_deadline(
                started,
                args.max_wall_minutes,
                f"{case_name} SGD checkpoint {checkpoint}/{len(times) - 1}",
            )
        risks = host_array(
            mx, risk_paths(mx, a, args.depth, lambdas, targets)
        ).astype(np.float64)
        if not np.all(np.isfinite(risks)):
            raise FloatingPointError(
                f"Non-finite SGD risk in {case_name} at step {step_host}."
            )
        path_values.append(risks)
        now = time.monotonic()
        if now >= next_progress:
            print(
                f"progress SGD {case_name}: {100.0 * checkpoint / (len(times) - 1):.1f}%; "
                f"updates={step_host:,}; elapsed={(now - started) / 60.0:.1f} min",
                flush=True,
            )
            next_progress = now + 60.0 * args.progress_minutes

    paths = np.stack(path_values, axis=0)
    q05, q95 = np.quantile(paths, (0.05, 0.95), axis=1)
    return {
        "times": times,
        "target_steps": target_steps,
        "sgd_paths": paths,
        "sgd_mean": paths.mean(axis=1),
        "sgd_median": np.median(paths, axis=1),
        "q05": q05,
        "q95": q95,
        "theta": theta,
        "final_step": step_host,
    }


def save_sgd_case(data_dir: Path, case_name: str, result: Mapping[str, Any]) -> Path:
    import numpy as np

    path = data_dir / f"{case_name}.npz"
    np.savez_compressed(
        path,
        times=result["times"],
        target_steps=result["target_steps"],
        sgd_paths=result["sgd_paths"],
        sgd_mean=result["sgd_mean"],
        sgd_median=result["sgd_median"],
        q05=result["q05"],
        q95=result["q95"],
    )
    return path


def load_sgd_case(data_dir: Path, case_name: str) -> Dict[str, Any]:
    import numpy as np

    path = data_dir / f"{case_name}.npz"
    if not path.exists():
        raise SystemExit(f"Missing SGD result: {path}")
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def plot_comparison(
    args: argparse.Namespace,
    sde_results: Mapping[str, Mapping[str, Any]],
    sgd_results: Mapping[str, Mapping[str, Any]],
    figure_dir: Path,
) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [name for name in CASE_ORDER if name in sgd_results]
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.0 * len(names), 3.2), squeeze=False
    )
    source_limits = {
        "hard-horizon": (5.0e-3, 5.0e3),
        "hard-constant": (7.0e-3, 1.0),
        "easy-horizon": (5.0e-5, 1.0),
        "easy-constant": (5.0e-5, 4.0e-1),
    }
    for axis, name in zip(axes[0], names):
        case = CASES[name]
        sde = sde_results[name]
        sgd = sgd_results[name]
        axis.fill_between(
            sde["times"], sde["q05"], sde["q95"],
            color="#4C78A8", alpha=0.13, linewidth=0,
        )
        axis.fill_between(
            sgd["times"], sgd["q05"], sgd["q95"],
            color="#7A5195", alpha=0.13, linewidth=0,
        )
        axis.plot(
            sde["times"], sde["gradient_flow"], "--",
            color="#333333", linewidth=1.15, label="Gradient flow",
        )
        axis.plot(
            sde["times"], sde["perturbation"], "-.",
            color="#C76B1D", linewidth=1.25, label="Second order",
        )
        axis.plot(
            sde["times"], sde["sde_mean"],
            color="#2F5FA7", linewidth=1.15, label="SDE mean",
        )
        axis.plot(
            sde["times"], np.median(sde["sde_paths"], axis=1), ":",
            color="#2F5FA7", linewidth=1.0, label="SDE median",
        )
        axis.plot(
            sgd["times"], sgd["sgd_mean"],
            color="#7A5195", linewidth=1.15,
            label=f"SGD mean (batch {args.batch_size})",
        )
        axis.plot(
            sgd["times"], sgd["sgd_median"], ":",
            color="#7A5195", linewidth=1.0,
            label=f"SGD median (batch {args.batch_size})",
        )
        init_title = (
            "Horizon-dependent init" if case.init == "horizon"
            else "Constant-order init"
        )
        axis.set_title(
            rf"{case.regime.capitalize()} regime, $(s,\beta)=({case.s:g},{case.beta:g})$"
            + "\n" + init_title,
            fontsize=9,
        )
        axis.set_yscale("log")
        axis.set_xlim(0.0, args.horizon)
        axis.set_ylim(*source_limits[name])
        axis.axvspan(
            args.c2 * args.horizon, args.horizon,
            color="#777777", alpha=0.035, linewidth=0,
        )
        axis.axvline(
            args.c2 * args.horizon, color="#777777", alpha=0.25,
            linewidth=0.7, linestyle="--",
        )
        axis.grid(alpha=0.12, linewidth=0.5)
        axis.tick_params(labelsize=8)
    axes[0][0].set_ylabel("Risk")
    fig.supxlabel(r"Intrinsic time $t$", y=0.01, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.86))
    png = figure_dir / "comparison.png"
    pdf = figure_dir / "comparison.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_decay_zoom(
    args: argparse.Namespace,
    sde_results: Mapping[str, Mapping[str, Any]],
    sgd_results: Mapping[str, Mapping[str, Any]],
    figure_dir: Path,
) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [name for name in CASE_ORDER if name in sgd_results]
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.0 * len(names), 3.2), squeeze=False
    )
    decay_start = args.c2 * args.horizon
    for axis, name in zip(axes[0], names):
        case = CASES[name]
        sde = sde_results[name]
        sgd = sgd_results[name]
        sde_mask = sde["times"] >= decay_start
        sgd_mask = sgd["times"] >= decay_start
        axis.fill_between(
            sde["times"][sde_mask], sde["q05"][sde_mask], sde["q95"][sde_mask],
            color="#4C78A8", alpha=0.13, linewidth=0,
        )
        axis.fill_between(
            sgd["times"][sgd_mask], sgd["q05"][sgd_mask], sgd["q95"][sgd_mask],
            color="#7A5195", alpha=0.13, linewidth=0,
        )
        axis.plot(
            sde["times"][sde_mask], sde["perturbation"][sde_mask], "-.",
            color="#C76B1D", linewidth=1.3, label="Second order",
        )
        axis.plot(
            sde["times"][sde_mask], sde["sde_mean"][sde_mask],
            color="#2F5FA7", linewidth=1.15, label="SDE mean",
        )
        axis.plot(
            sde["times"][sde_mask],
            np.median(sde["sde_paths"], axis=1)[sde_mask], ":",
            color="#2F5FA7", linewidth=1.0, label="SDE median",
        )
        axis.plot(
            sgd["times"][sgd_mask], sgd["sgd_mean"][sgd_mask],
            color="#7A5195", linewidth=1.15,
            label=f"SGD mean (batch {args.batch_size})",
        )
        axis.plot(
            sgd["times"][sgd_mask], sgd["sgd_median"][sgd_mask], ":",
            color="#7A5195", linewidth=1.0,
            label=f"SGD median (batch {args.batch_size})",
        )
        init_title = (
            "Horizon-dependent init" if case.init == "horizon"
            else "Constant-order init"
        )
        axis.set_title(
            rf"{case.regime.capitalize()} regime, $(s,\beta)=({case.s:g},{case.beta:g})$"
            + "\n" + init_title,
            fontsize=9,
        )
        axis.set_yscale("log")
        axis.set_xlim(decay_start, args.horizon)
        axis.grid(alpha=0.12, linewidth=0.5)
        axis.tick_params(labelsize=8)
        schedule_axis = axis.twinx()
        schedule_t = sde["times"][sde_mask]
        relative_eta = np.power(
            np.maximum(
                (args.horizon - schedule_t)
                / ((1.0 - args.c2) * args.horizon),
                0.0,
            ),
            args.nu2,
        )
        schedule_axis.plot(
            schedule_t, relative_eta, color="#888888", alpha=0.22,
            linewidth=0.9,
        )
        schedule_axis.set_ylim(0.0, 1.05)
        schedule_axis.set_yticks([])
    axes[0][0].set_ylabel("Risk")
    fig.supxlabel(r"Decay-phase intrinsic time $t$", y=0.01, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.text(
        0.995, 0.5, r"gray: $\eta(t)/\eta_{\rm peak}$",
        ha="right", va="center", rotation=90, fontsize=7, color="#777777",
    )
    fig.tight_layout(rect=(0.0, 0.06, 0.985, 0.86))
    png = figure_dir / "decay.png"
    pdf = figure_dir / "decay.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def decay_summary(
    args: argparse.Namespace,
    sde_results: Mapping[str, Mapping[str, Any]],
    sgd_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    import numpy as np

    summary: Dict[str, Any] = {}
    decay_start = args.c2 * args.horizon
    for name in [case for case in CASE_ORDER if case in sgd_results]:
        sde = sde_results[name]
        sgd = sgd_results[name]
        index = int(np.argmin(np.abs(sde["times"] - decay_start)))
        sde_median = np.median(sde["sde_paths"], axis=1)
        summary[name] = {
            "decay_start_time": float(sde["times"][index]),
            "sde_mean_at_decay_start": float(sde["sde_mean"][index]),
            "sde_final_mean": float(sde["sde_mean"][-1]),
            "sde_mean_decay_factor": float(
                sde["sde_mean"][index] / sde["sde_mean"][-1]
            ),
            "sde_final_median": float(sde_median[-1]),
            "sde_median_decay_factor": float(sde_median[index] / sde_median[-1]),
            "sgd_mean_at_decay_start": float(sgd["sgd_mean"][index]),
            "sgd_final_mean": float(sgd["sgd_mean"][-1]),
            "sgd_mean_decay_factor": float(
                sgd["sgd_mean"][index] / sgd["sgd_mean"][-1]
            ),
            "sgd_final_median": float(sgd["sgd_median"][-1]),
            "sgd_median_decay_factor": float(
                sgd["sgd_median"][index] / sgd["sgd_median"][-1]
            ),
            "final_second_order": float(sde["perturbation"][-1]),
            "final_gradient_flow": float(sde["gradient_flow"][-1]),
        }
    return summary


def write_comparison_metadata(
    args: argparse.Namespace,
    sde_results: Mapping[str, Mapping[str, Any]],
    sgd_results: Mapping[str, Mapping[str, Any]],
    figure_files: Iterable[Path],
    metadata_dir: Path,
) -> Path:
    metadata = {
        "output_root": str(args.output_root),
        "cases": list(args.cases),
        "modes": args.modes,
        "sgd_paths": args.paths,
        "sgd_batch_size": args.batch_size,
        "depth": args.depth,
        "horizon": args.horizon,
        "eta_peak": args.eta_peak,
        "figure_files": [str(path) for path in figure_files],
        "decay": decay_summary(args, sde_results, sgd_results),
    }
    path = metadata_dir / "results.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    args.output_root = Path(args.output_root)
    sde_dir = args.output_root / "data" / "sde"
    sgd_dir = args.output_root / "data" / "sgd"
    figure_dir = args.output_root / "figures"
    metadata_dir = args.output_root / "metadata"
    for directory in (sgd_dir, figure_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    schedule = physical_wsd_parameters(args)
    if (
        schedule["fresh_samples"] > 5_000_000
        and not args.confirm_long_run
        and not args.plot_only
    ):
        raise SystemExit(
            f"Refusing {schedule['steps']:,} SGD updates over "
            f"{schedule['fresh_samples']:,} fresh samples without "
            "--confirm-long-run. The wall-clock guard remains active."
        )

    sde_results = {name: load_sde_case(sde_dir, name) for name in args.cases}
    if args.plot_only:
        sgd_results = {name: load_sgd_case(sgd_dir, name) for name in args.cases}
        png, pdf = plot_comparison(args, sde_results, sgd_results, figure_dir)
        zoom_png, zoom_pdf = plot_decay_zoom(
            args, sde_results, sgd_results, figure_dir
        )
        figures = (
            figure_dir / "sde.png",
            figure_dir / "sde.pdf",
            png,
            pdf,
            zoom_png,
            zoom_pdf,
        )
        metadata_path = write_comparison_metadata(
            args, sde_results, sgd_results, figures, metadata_dir,
        )
        print(
            f"rebuilt {png}, {pdf}, {zoom_png}, and {zoom_pdf}; "
            f"metadata={metadata_path}",
            flush=True,
        )
        return 0

    mx, device = configure_mlx()
    started = time.monotonic()
    print(
        f"backend=mlx; device={device}; modes={args.modes}; paths={args.paths}; "
        f"SGD updates={schedule['steps']:,}; batch={args.batch_size}; "
        f"fresh samples={schedule['fresh_samples']:,}; Z_WSD={schedule['z_wsd']:.9g}",
        flush=True,
    )
    results: Dict[str, Dict[str, Any]] = {}
    files = []
    for name in args.cases:
        check_deadline(started, args.max_wall_minutes, f"before {name}")
        print(f"starting SGD {name}", flush=True)
        result = simulate_sgd_case(
            args, name, sde_results[name], schedule, mx=mx, started=started
        )
        results[name] = result
        files.append(str(save_sgd_case(sgd_dir, name, result)))
        print(
            f"finished SGD {name}: steps={result['final_step']:,}; "
            f"final mean={result['sgd_mean'][-1]:.6g}; "
            f"median={result['sgd_median'][-1]:.6g}",
            flush=True,
        )
    png, pdf = plot_comparison(args, sde_results, results, figure_dir)
    zoom_png, zoom_pdf = plot_decay_zoom(args, sde_results, results, figure_dir)
    figures = (
        figure_dir / "sde.png",
        figure_dir / "sde.pdf",
        png,
        pdf,
        zoom_png,
        zoom_pdf,
    )
    comparison_metadata = write_comparison_metadata(
        args, sde_results, results, figures, metadata_dir
    )
    elapsed = time.monotonic() - started
    metadata = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": device,
        "mlx_version": mlx_version(),
        "elapsed_seconds": elapsed,
        "physical_schedule": schedule,
        "case_files": files,
        "figure_files": [str(path) for path in figures],
        "comparison_metadata": str(comparison_metadata),
        "notes": [
            "SGD uses fresh Gaussian features and label noise; no sample is reused",
            "batch-size scaling keeps step_eta_peak / batch_size equal to the SDE eta_peak",
            "the tied-coordinate update uses the learning-rate convention matching the draft SDE drift",
            "SGD retains cross-coordinate gradient-noise correlations omitted by the diagonal SDE",
        ],
    }
    metadata_path = metadata_dir / "sgd.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"completed in {elapsed:.2f}s; metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
