#!/usr/bin/env python3
"""Run an isolated one-pass FSL calibration and schedule-transfer study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from core import Problem, schedules
from fit import fit_and_report
from simulate import save_trajectory, simulate_mlx, simulate_numpy


PRESETS = {
    "smoke": dict(
        modes=8,
        steps=8192,
        paths=8,
        checkpoints=72,
        eta_base=0.003,
    ),
    "pilot": dict(
        modes=16,
        steps=65536,
        paths=128,
        checkpoints=180,
        eta_base=0.0016,
    ),
    "study": dict(
        modes=32,
        steps=262144,
        paths=128,
        checkpoints=300,
        eta_base=0.0013,
    ),
}

REGIMES = {
    "hard": -0.2,
    "boundary": 0.0,
    "easy": 0.2,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Fit an FSL on constant-rate DLN trajectories and predict unseen schedules."
    )
    result.add_argument("--preset", choices=PRESETS, default="smoke")
    result.add_argument("--backend", choices=("mlx", "numpy"), default="mlx")
    result.add_argument("--regime", choices=REGIMES, default="hard")
    result.add_argument("--depth", type=int, default=5)
    result.add_argument("--beta", type=float, default=2.0)
    result.add_argument("--sigma", type=float, default=0.3)
    result.add_argument(
        "--eta-base",
        type=float,
        default=None,
        help="override the preset middle constant learning rate",
    )
    result.add_argument("--c-init", type=float, default=0.2)
    result.add_argument("--seed", type=int, default=20260812)
    result.add_argument("--block-size", type=int, default=128)
    result.add_argument("--bootstrap", type=int, default=100)
    result.add_argument("--output-root", type=Path, default=Path("fsl_transfer/results"))
    result.add_argument("--force", action="store_true")
    result.add_argument("--fit-only", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    settings = dict(PRESETS[args.preset])
    if args.eta_base is not None:
        settings["eta_base"] = args.eta_base
    problem = Problem(
        **settings,
        depth=args.depth,
        beta=args.beta,
        chi=REGIMES[args.regime],
        sigma=args.sigma,
        c_init=args.c_init,
        seed=args.seed,
    )
    problem.validate()
    run_dir = args.output_root / f"{args.preset}-{args.regime}-L{args.depth}"
    data_dir = run_dir / "data"
    if data_dir.exists() and any(data_dir.glob("*.npz")) and not (args.force or args.fit_only):
        raise SystemExit(f"results already exist in {data_dir}; pass --force or --fit-only")
    started = time.monotonic()
    if not args.fit_only:
        data_dir.mkdir(parents=True, exist_ok=True)
        for name, eta in schedules(problem).items():
            schedule_problem = Problem(**{**problem.__dict__, "seed": problem.seed})
            print(
                f"simulate {name}: steps={problem.steps}, paths={problem.paths}, "
                f"T={float(np.sum(eta)):.6g}, backend={args.backend}",
                flush=True,
            )
            if args.backend == "mlx":
                result = simulate_mlx(schedule_problem, eta, block_size=args.block_size)
            else:
                result = simulate_numpy(schedule_problem, eta)
            path = save_trajectory(
                data_dir, name, schedule_problem, eta, result, backend=args.backend
            )
            print(
                f"  saved {path}; min(a)={float(result['minimum_layer_coordinate']):.4g}; "
                f"max relative update={float(result['maximum_relative_update']):.4g}",
                flush=True,
            )
    report = fit_and_report(
        data_dir, run_dir, bootstrap_repetitions=args.bootstrap
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"elapsed minutes={(time.monotonic() - started) / 60.0:.2f}")


if __name__ == "__main__":
    main()
