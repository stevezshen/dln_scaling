"""Shared configuration, Metal helpers, and plotting for the experiments."""

from __future__ import annotations

import argparse
import dataclasses
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Tuple


class Preset(NamedTuple):
    modes: int
    paths: int
    horizon: float
    checkpoints: int
    max_wall_minutes: float
    sde_dt_max: float
    ode_dt_initial: float
    max_sde_steps: int
    max_ode_trials: int


PRESETS: Mapping[str, Preset] = {
    "preview": Preset(
        modes=256,
        paths=8,
        horizon=100.0,
        checkpoints=100,
        max_wall_minutes=8.0,
        sde_dt_max=1.0,
        ode_dt_initial=1.0e-4,
        max_sde_steps=200_000,
        max_ode_trials=200_000,
    ),
    "paper": Preset(
        modes=10_000,
        paths=100,
        horizon=10_000.0,
        checkpoints=500,
        max_wall_minutes=28.0,
        sde_dt_max=5.0,
        ode_dt_initial=1.0e-4,
        max_sde_steps=6_000_000,
        max_ode_trials=1_000_000,
    ),
}

CASE_ORDER = (
    "hard-horizon",
    "hard-constant",
    "easy-horizon",
    "easy-constant",
)


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    regime: str
    init: str
    s: float
    beta: float

    @property
    def alpha(self) -> float:
        return 0.5 * (1.0 + self.beta * (self.s - 1.0))


CASES: Mapping[str, Case] = {
    "hard-horizon": Case("hard-horizon", "hard", "horizon", 0.2, 5.0),
    "hard-constant": Case("hard-constant", "hard", "constant", 0.2, 5.0),
    "easy-horizon": Case("easy-horizon", "easy", "horizon", 1.5, 2.0),
    "easy-constant": Case("easy-constant", "easy", "constant", 1.5, 2.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native-MLX reproduction of the draft's SDE perturbation figure."
    )
    parser.add_argument("--preset", choices=PRESETS, default="preview")
    parser.add_argument("--cases", nargs="+", choices=CASE_ORDER, default=list(CASE_ORDER))
    parser.add_argument("--confirm-paper-run", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the figure from existing per-case NPZ files without using Metal.",
    )
    parser.add_argument("--modes", type=int)
    parser.add_argument("--paths", type=int)
    parser.add_argument("--horizon", type=float)
    parser.add_argument("--checkpoints", type=int)
    parser.add_argument("--max-wall-minutes", type=float)
    parser.add_argument(
        "--progress-minutes",
        type=float,
        default=0.0,
        help="Print checkpoint progress at this wall-clock interval; zero disables it.",
    )
    parser.add_argument("--max-sde-steps", type=int)
    parser.add_argument("--max-ode-trials", type=int)
    parser.add_argument("--sde-dt-max", type=float)
    parser.add_argument(
        "--sde-block-size", type=int, default=64,
        help="Adaptive Euler steps executed per compiled MLX call.",
    )
    parser.add_argument("--ode-dt-initial", type=float)
    parser.add_argument(
        "--ode-block-size", type=int, default=8,
        help="Adaptive Dormand--Prince trials executed per compiled MLX call.",
    )
    parser.add_argument("--ode-atol", type=float, default=1.0e-6)
    parser.add_argument("--ode-rtol", type=float, default=1.0e-4)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--eta-peak", type=float, default=1.0e-3)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument(
        "--noise-model",
        choices=("diagonal", "full"),
        default="diagonal",
        help="Diagonal draft covariance or exact Gaussian-gradient covariance.",
    )
    parser.add_argument("--c1", type=float, default=0.3)
    parser.add_argument("--c2", type=float, default=0.7)
    parser.add_argument("--nu1", type=float, default=0.8)
    parser.add_argument("--nu2", type=float, default=0.8)
    parser.add_argument("--drift-increment-limit", type=float, default=0.05)
    parser.add_argument("--diffusion-std-limit", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--no-plot", action="store_true")
    return parser


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    for field in Preset._fields:
        if getattr(args, field, None) is None:
            setattr(args, field, getattr(preset, field))
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        name: getattr(args, name)
        for name in (
            "modes",
            "paths",
            "horizon",
            "checkpoints",
            "max_wall_minutes",
            "max_sde_steps",
            "max_ode_trials",
            "sde_dt_max",
            "sde_block_size",
            "ode_dt_initial",
            "ode_block_size",
            "ode_atol",
            "ode_rtol",
            "depth",
            "eta_peak",
            "sigma",
        )
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise SystemExit("Expected positive values for: " + ", ".join(bad))
    if args.progress_minutes < 0:
        raise SystemExit("Require progress-minutes >= 0.")
    if not 0.0 < args.c1 < args.c2 < 1.0:
        raise SystemExit("Require 0 < c1 < c2 < 1.")
    if not 0.0 < args.nu1 <= 1.0 or not 0.0 < args.nu2 <= 1.0:
        raise SystemExit("Require 0 < nu1, nu2 <= 1.")
    if args.depth < 2:
        raise SystemExit("This DLN experiment requires depth >= 2.")
    if args.preset == "paper" and not args.confirm_paper_run and not args.plot_only:
        raise SystemExit(
            "Refusing the paper-sized run without --confirm-paper-run. "
            "The preset uses 100 paths, 10,000 modes, and T=10,000."
        )


def checkpoint_grid(horizon: float, count: int) -> Any:
    import numpy as np

    n_log = max(12, count // 3)
    n_linear = max(12, count - n_log)
    first = max(1.0e-6 * horizon, 1.0e-6)
    early = np.geomspace(first, horizon, n_log, dtype=np.float32)
    linear = np.linspace(0.0, horizon, n_linear, dtype=np.float32)
    return np.unique(np.concatenate(([0.0], early, linear))).astype(np.float32)


def initial_value(case: Case, horizon: float, chi: float) -> float:
    if case.init == "constant":
        return 1.0
    denominator = case.beta + case.alpha * chi
    if denominator <= 0.0:
        raise ValueError("beta + alpha*chi must be positive")
    return horizon ** (-case.alpha / denominator)


def check_deadline(started: float, max_wall_minutes: float, context: str) -> None:
    elapsed = time.monotonic() - started
    if elapsed > 60.0 * max_wall_minutes:
        raise TimeoutError(
            f"Wall-clock guard reached after {elapsed / 60.0:.2f} minutes ({context})."
        )


def configure_mlx() -> Tuple[Any, str]:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise SystemExit(
            "MLX is not installed. Follow experiments/README.md to create "
            "the pinned environment."
        ) from exc
    mx.set_default_device(mx.gpu)
    device = str(mx.default_device())
    if "gpu" not in device.lower():
        raise SystemExit(f"Refusing to run without the MLX GPU device; got {device}.")
    return mx, device


def mlx_version() -> str:
    try:
        return version("mlx")
    except PackageNotFoundError:
        return "unknown"


def host_array(mx: Any, value: Any) -> Any:
    import numpy as np

    mx.eval(value)
    return np.asarray(value)


def host_float(mx: Any, value: Any) -> float:
    return float(host_array(mx, value))


def plot_results(
    figure_dir: Path,
    results: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required unless --no-plot is used.") from exc

    available = [name for name in CASE_ORDER if name in results]
    source_limits = {
        "hard-horizon": (5.0e-3, 5.0e3),
        "hard-constant": (7.0e-3, 1.0),
        "easy-horizon": (5.0e-5, 1.0),
        "easy-constant": (5.0e-5, 4.0e-1),
    }
    fig, axes = plt.subplots(
        1, len(available), figsize=(3.35 * len(available), 3.0), squeeze=False
    )
    for axis, name in zip(axes[0], available):
        case = CASES[name]
        result = results[name]
        t = result["times"]
        axis.fill_between(
            t, result["q05"], result["q95"], color="#4C78A8", alpha=0.20, linewidth=0
        )
        axis.plot(
            t, result["gradient_flow"], color="#333333", linestyle="--",
            linewidth=1.25, label=r"$\mathcal{E}^{(0)}_t$ (GF)",
        )
        axis.plot(
            t, result["perturbation"], color="#C76B1D", linestyle="-.",
            linewidth=1.25, label=r"$\mathcal{E}^{(0)}_t+\eta_{\rm peak}\mathbb{E}[\mathcal{E}^{(2)}_t]$",
        )
        axis.plot(
            t, result["sde_mean"], color="#2F5FA7", linewidth=1.25,
            label="Averaged empirical excess risk",
        )
        init_title = (
            "Horizon-dependent init" if case.init == "horizon" else "Constant-order init"
        )
        axis.set_title(
            rf"{case.regime.capitalize()} regime, $(s,\beta)=({case.s:g},{case.beta:g})$"
            + "\n" + init_title,
            fontsize=8,
        )
        axis.set_yscale("log")
        axis.set_xlim(0.0, args.horizon)
        axis.set_ylim(*source_limits[name])
        axis.grid(alpha=0.12, linewidth=0.5)
        axis.tick_params(labelsize=7)
    axes[0][0].set_ylabel("Risk")
    fig.supxlabel(r"Intrinsic time $t$", y=0.01, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.88))
    png_path = figure_dir / "sde.png"
    pdf_path = figure_dir / "sde.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
