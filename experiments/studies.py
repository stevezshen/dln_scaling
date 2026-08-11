#!/usr/bin/env python3
"""Restartable large-scale validation studies with aggregate final artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

import sde
import sgd
from common import CASE_ORDER, CASES, checkpoint_grid, configure_mlx


SCRATCH = Path("/private/tmp/scaling-studies")
FINAL_DATA = Path("output/data/studies.npz")
FINAL_METADATA = Path("output/metadata/studies.json")
FIGURE_DIR = Path("output/figures")


@dataclasses.dataclass(frozen=True)
class Job:
    key: str
    study: str
    method: str
    case: str
    modes: int = 512
    paths: int = 128
    horizon: float = 10_000.0
    checkpoints: int = 500
    depth: int = 10
    eta_peak: float = 1.0e-3
    batch_size: int = 8
    drift_limit: float = 0.0125
    diffusion_limit: float = 0.02
    noise_model: str = "diagonal"
    max_wall_minutes: float = 360.0
    version: int = 2
    seed: int = 20260810


def experiment_jobs() -> List[Job]:
    jobs: List[Job] = []

    # Numerical convergence at progressively tighter adaptive-step controls.
    bounds = (
        ("base", 0.05, 0.08),
        ("half", 0.025, 0.04),
        ("quarter", 0.0125, 0.02),
    )
    for case in ("easy-horizon", "easy-constant"):
        for label, drift, diffusion in bounds:
            jobs.append(
                Job(
                    f"convergence-{case}-{label}",
                    "convergence",
                    "sde",
                    case,
                    paths=64,
                    drift_limit=drift,
                    diffusion_limit=diffusion,
                )
            )
    jobs.append(
        Job(
            "convergence-easy-horizon-eighth",
            "convergence",
            "sde",
            "easy-horizon",
            paths=64,
            drift_limit=0.00625,
            diffusion_limit=0.01,
        )
    )
    # One selected empirical reference per panel.  The two easy references are
    # the refined runs above; the hard references are inexpensive enough to add.
    for case in ("hard-horizon", "hard-constant"):
        jobs.append(
            Job(
                f"convergence-{case}-paths",
                "convergence",
                "sde",
                case,
                paths=64,
            )
        )
    # Target the unexpected small-eta discrepancy in the hard-horizon panel
    # with an independent refinement of both adaptive increment bounds.
    jobs.append(
        Job(
            "convergence-hard-horizon-0250-half",
            "convergence",
            "sde",
            "hard-horizon",
            paths=32,
            eta_peak=2.5e-4,
            drift_limit=0.00625,
            diffusion_limit=0.01,
        )
    )

    # Retain the two completed hard-horizon SDE points as numerical
    # diagnostics.  Their strong step-size dependence makes them unsuitable
    # for estimating the perturbative remainder.
    for label, eta in (("0500", 5.0e-4), ("0250", 2.5e-4)):
        jobs.append(
            Job(
                f"learning-hard-horizon-{label}",
                "learning",
                "sde",
                "hard-horizon",
                paths=32,
                eta_peak=eta,
            )
        )

    # Estimate learning-rate error scaling with direct fresh-sample SGD.  The
    # eta=1e-3 references are the four higher-confidence SGD runs below.
    for case in CASE_ORDER:
        for label, eta in (("0500", 5.0e-4), ("0250", 2.5e-4)):
            jobs.append(
                Job(
                    f"learning-sgd-{case}-{label}",
                    "learning",
                    "sgd",
                    case,
                    paths=16,
                    batch_size=8,
                    eta_peak=eta,
                )
            )

    # Higher-confidence SGD for every panel and one representative batch sweep.
    for case in CASE_ORDER:
        jobs.append(
            Job(
                f"sgd-{case}-paths",
                "sgd",
                "sgd",
                case,
                paths=16,
                batch_size=8,
            )
        )
    jobs.append(
        Job(
            "covariance-hard-horizon-full",
            "sgd",
            "sde",
            "hard-horizon",
            paths=64,
            noise_model="full",
        )
    )
    for batch in (1, 2, 4):
        jobs.append(
            Job(
                f"sgd-easy-constant-batch{batch}",
                "sgd",
                "sgd",
                "easy-constant",
                paths=4,
                batch_size=batch,
            )
        )

    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Experiment job keys are not unique.")
    return jobs


JOBS = {job.key: job for job in experiment_jobs()}


def result_path(job: Job) -> Path:
    return SCRATCH / f"{job.key}.npz"


def valid_result(job: Job) -> bool:
    path = result_path(job)
    if not path.exists():
        return False
    try:
        with np.load(path) as saved:
            config = json.loads(str(saved["config"]))
            expected = dataclasses.asdict(job)
            for name, value in expected.items():
                if name not in config:
                    if name == "seed" and value == 20260810:
                        continue
                    return False
                if config[name] != value:
                    return False
            return "times" in saved.files
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def sde_arguments(
    job: Job, max_wall_minutes: float | None = None
) -> argparse.Namespace:
    argv = [
        "--preset", "paper",
        "--confirm-paper-run",
        "--cases", job.case,
        "--modes", str(job.modes),
        "--paths", str(job.paths),
        "--horizon", str(job.horizon),
        "--checkpoints", str(job.checkpoints),
        "--depth", str(job.depth),
        "--eta-peak", str(job.eta_peak),
        "--drift-increment-limit", str(job.drift_limit),
        "--diffusion-std-limit", str(job.diffusion_limit),
        "--noise-model", job.noise_model,
        "--seed", str(job.seed),
        "--max-wall-minutes", str(
            job.max_wall_minutes if max_wall_minutes is None else max_wall_minutes
        ),
        "--progress-minutes", "5",
        "--max-sde-steps", "500000000",
        "--max-ode-trials", "5000000",
        "--no-plot",
    ]
    args = sde.apply_preset(sde.build_parser().parse_args(argv))
    sde.validate_args(args)
    return args


def sgd_arguments(
    job: Job, max_wall_minutes: float | None = None
) -> argparse.Namespace:
    argv = [
        "--cases", job.case,
        "--modes", str(job.modes),
        "--paths", str(job.paths),
        "--batch-size", str(job.batch_size),
        "--depth", str(job.depth),
        "--horizon", str(job.horizon),
        "--eta-peak", str(job.eta_peak),
        "--seed", str(job.seed),
        "--max-wall-minutes", str(
            job.max_wall_minutes if max_wall_minutes is None else max_wall_minutes
        ),
        "--progress-minutes", "5",
        "--confirm-long-run",
    ]
    args = sgd.build_parser().parse_args(argv)
    sgd.validate_args(args)
    return args


def run_sde(
    job: Job, max_wall_minutes: float | None = None
) -> Mapping[str, Any]:
    args = sde_arguments(job, max_wall_minutes)
    mx, device = configure_mlx()
    started = time.monotonic()
    result = sde.simulate_case(
        args,
        CASES[job.case],
        mx=mx,
        started=started,
        case_index=CASE_ORDER.index(job.case),
    )
    return {
        **result,
        "elapsed_seconds": time.monotonic() - started,
        "device": np.asarray(device),
    }


def run_sgd(
    job: Job, max_wall_minutes: float | None = None
) -> Mapping[str, Any]:
    args = sgd_arguments(job, max_wall_minutes)
    schedule = sgd.physical_wsd_parameters(args)
    mx, device = configure_mlx()
    started = time.monotonic()
    result = sgd.simulate_sgd_case(
        args,
        job.case,
        {"times": checkpoint_grid(job.horizon, job.checkpoints)},
        schedule,
        mx=mx,
        started=started,
    )
    combined: Dict[str, Any] = {
        **result,
        "elapsed_seconds": time.monotonic() - started,
        "device": np.asarray(device),
        "updates": np.asarray(schedule["steps"]),
        "fresh_samples": np.asarray(schedule["fresh_samples"]),
    }
    if job.study == "learning":
        # The normalized P,Q hierarchy is independent of eta_peak.  Reuse the
        # eta=1e-3 reference and rescale only its correction, avoiding an
        # identical expensive ODE solve for every SGD learning-rate point.
        reference = reference_result(job.case)
        gradient_flow = np.asarray(reference["gradient_flow"], dtype=np.float64)
        correction = (
            np.asarray(reference["perturbation"], dtype=np.float64) - gradient_flow
        )
        combined["gradient_flow"] = gradient_flow
        combined["perturbation"] = (
            gradient_flow + (job.eta_peak / 1.0e-3) * correction
        )
        combined["ode_trials"] = np.asarray(reference["ode_trials"])
    return combined


def save_result(job: Job, result: Mapping[str, Any]) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    final = result_path(job)
    partial = final.with_suffix(".partial.npz")
    payload: Dict[str, Any] = {
        key: value
        for key, value in result.items()
        if isinstance(value, (np.ndarray, np.generic, int, float, str))
    }
    payload["config"] = np.asarray(json.dumps(dataclasses.asdict(job), sort_keys=True))
    np.savez_compressed(partial, **payload)
    os.replace(partial, final)
    return final


def run_job(job: Job, deadline_epoch: float | None = None) -> None:
    if valid_result(job):
        print(f"already complete: {job.key}", flush=True)
        return
    print(
        f"starting {job.key}: method={job.method}, case={job.case}, "
        f"J={job.modes}, paths={job.paths}, T={job.horizon:g}, "
        f"eta={job.eta_peak:g}",
        flush=True,
    )
    max_wall_minutes = job.max_wall_minutes
    if deadline_epoch is not None:
        # Keep two hours for aggregation, plotting, interpretation, and cleanup.
        available = (deadline_epoch - time.time() - 2.0 * 3600.0) / 60.0
        if available <= 0.0:
            raise TimeoutError("Campaign compute window has closed.")
        max_wall_minutes = min(max_wall_minutes, available)
    if job.method == "sde":
        result = run_sde(job, max_wall_minutes)
    else:
        result = run_sgd(job, max_wall_minutes)
    path = save_result(job, result)
    final_key = {
        "sde": "sde_mean",
        "sgd": "sgd_mean",
    }[job.method]
    print(
        f"completed {job.key}: final={float(result[final_key][-1]):.8g}; "
        f"elapsed={float(result['elapsed_seconds']) / 60.0:.2f} min; {path}",
        flush=True,
    )


def selected_jobs(study: str) -> List[Job]:
    if study == "all":
        return list(JOBS.values())
    return [job for job in JOBS.values() if job.study == study]


def run_study(study: str, deadline_epoch: float | None = None) -> None:
    for job in selected_jobs(study):
        if valid_result(job):
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "--job", job.key]
        if deadline_epoch is not None:
            if time.time() >= deadline_epoch - 2.0 * 3600.0:
                print("stopping: reserved reporting window has begun", flush=True)
                return
            command.extend(("--deadline-epoch", str(deadline_epoch)))
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            print(f"stopping study after failed job: {error}", flush=True)
            return


def process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_campaign(deadline_epoch: float, wait_pid: int | None) -> None:
    """Run the prioritized 24-hour queue and preserve a reporting window."""

    if wait_pid is not None:
        print(f"waiting for active job process {wait_pid}", flush=True)
        while process_running(wait_pid) and time.time() < deadline_epoch:
            time.sleep(30.0)
    for study in ("convergence", "learning", "sgd"):
        if time.time() >= deadline_epoch - 2.0 * 3600.0:
            break
        run_study(study, deadline_epoch)
    collect_results(require_complete=False)
    missing = [job.key for job in JOBS.values() if not valid_result(job)]
    if missing:
        print(
            f"campaign compute ended with {len(missing)} jobs incomplete; "
            "aggregate partial results were preserved",
            flush=True,
        )
    else:
        generate_report()


def collect_results(require_complete: bool) -> None:
    missing = [job.key for job in JOBS.values() if not valid_result(job)]
    if missing and require_complete:
        raise SystemExit(f"Cannot collect: {len(missing)} jobs remain.")

    archive: Dict[str, Any] = {}
    records = []
    for job in JOBS.values():
        if not valid_result(job):
            continue
        prefix = job.key.replace("-", "_")
        with np.load(result_path(job)) as saved:
            for name in saved.files:
                if name != "config":
                    archive[f"{prefix}__{name}"] = saved[name]
            records.append(
                {
                    **dataclasses.asdict(job),
                    "elapsed_seconds": float(saved["elapsed_seconds"]),
                    "final_mean": float(
                        saved[
                            {
                                "sde": "sde_mean",
                                "sgd": "sgd_mean",
                            }[job.method]
                        ][-1]
                    ),
                    "final_median": float(
                        np.median(saved["sde_paths"][-1])
                        if job.method == "sde"
                        else np.median(saved["sgd_paths"][-1])
                    ),
                }
            )

    FINAL_DATA.parent.mkdir(parents=True, exist_ok=True)
    FINAL_METADATA.parent.mkdir(parents=True, exist_ok=True)
    partial = FINAL_DATA.with_suffix(".partial.npz")
    np.savez_compressed(partial, **archive)
    os.replace(partial, FINAL_DATA)
    metadata = {
        "completed": len(records),
        "total": len(JOBS),
        "missing": missing,
        "archive": str(FINAL_DATA),
        "jobs": records,
    }
    FINAL_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"collected {len(records)}/{len(JOBS)} jobs into {FINAL_DATA}", flush=True
    )


def load_result(key: str) -> Dict[str, Any]:
    job = JOBS[key]
    if not valid_result(job):
        raise SystemExit(f"Missing or invalid result: {key}")
    with np.load(result_path(job)) as saved:
        return {name: saved[name] for name in saved.files if name != "config"}


def endpoint(result: Mapping[str, Any], method: str = "sde") -> Dict[str, float]:
    paths = np.asarray(result[f"{method}_paths"][-1], dtype=np.float64)
    return {
        "mean": float(paths.mean()),
        "median": float(np.median(paths)),
        "se": float(paths.std(ddof=1) / np.sqrt(len(paths))) if len(paths) > 1 else 0.0,
    }


def combine_sde(keys: Iterable[str]) -> Dict[str, Any]:
    key_list = list(keys)
    results = [load_result(key) for key in key_list]
    combined = dict(results[0])
    paths = np.concatenate([result["sde_paths"] for result in results], axis=1)
    combined["sde_paths"] = paths
    combined["sde_mean"] = paths.mean(axis=1)
    combined["q05"], combined["q95"] = np.quantile(paths, (0.05, 0.95), axis=1)
    combined["elapsed_seconds"] = np.asarray(
        sum(float(result["elapsed_seconds"]) for result in results)
    )
    return combined


def reference_keys(case: str) -> List[str]:
    if case == "easy-horizon":
        return ["convergence-easy-horizon-eighth"]
    if case == "easy-constant":
        return ["convergence-easy-constant-quarter"]
    return [f"convergence-{case}-paths"]


def reference_result(case: str) -> Dict[str, Any]:
    return combine_sde(reference_keys(case))


def learning_result(case: str, eta: float) -> Dict[str, Any]:
    labels = {5.0e-4: "0500", 2.5e-4: "0250"}
    if eta == 1.0e-3:
        result = load_result(f"sgd-{case}-paths")
        theory = reference_result(case)
        result["gradient_flow"] = theory["gradient_flow"]
        result["perturbation"] = theory["perturbation"]
        return result
    return load_result(f"learning-sgd-{case}-{labels[eta]}")


def covariance_result(case: str) -> Dict[str, Any]:
    return load_result(f"covariance-{case}-full")


def plot_convergence() -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cases = ("easy-horizon", "easy-constant")
    colors = ("#7A5195", "#EF8354", "#2F5FA7", "#009E73")
    summary: Dict[str, Any] = {}
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6), squeeze=False)
    for column, case in enumerate(cases):
        labels = (
            ("base", "half", "quarter", "eighth")
            if case == "easy-horizon"
            else ("base", "half", "quarter")
        )
        case_summary = {}
        for label, color in zip(labels, colors):
            result = load_result(f"convergence-{case}-{label}")
            axes[0, column].plot(
                result["times"], result["sde_mean"], color=color, linewidth=1.0,
                label=label,
            )
            case_summary[label] = endpoint(result)
        high_keys = reference_keys(case)
        high = combine_sde(high_keys)
        high_paths = sum(JOBS[key].paths for key in high_keys)
        axes[0, column].plot(
            high["times"], high["sde_mean"], color="#111111", linewidth=1.1,
            label=f"selected bounds, {high_paths} paths",
        )
        axes[0, column].plot(
            high["times"], high["perturbation"], "--", color="#C76B1D",
            linewidth=1.1, label="second order",
        )
        axes[0, column].set_yscale("log")
        axes[0, column].set_title(case.replace("-", " ").title())
        axes[0, column].set_xlabel(r"Intrinsic time $t$")
        axes[0, column].grid(alpha=0.12)

        means = [case_summary[label]["mean"] for label in labels]
        errors = [1.96 * case_summary[label]["se"] for label in labels]
        positions = np.arange(len(labels))
        axes[1, column].errorbar(
            positions, means, yerr=errors, marker="o", capsize=3,
            color="#2F5FA7", label="64 paths",
        )
        high_endpoint = endpoint(high)
        axes[1, column].errorbar(
            [positions[-1] + 0.08], [high_endpoint["mean"]],
            yerr=[1.96 * high_endpoint["se"]],
            marker="D", capsize=3, color="#111111", label="high path count",
        )
        axes[1, column].axhline(
            float(high["perturbation"][-1]), linestyle="--", color="#C76B1D",
            linewidth=1.0,
        )
        axes[1, column].set_xticks(positions, labels)
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylabel("Final risk")
        axes[1, column].grid(alpha=0.12)
        case_summary["high_paths"] = high_endpoint
        case_summary["second_order"] = float(high["perturbation"][-1])
        summary[case] = case_summary
    axes[0, 0].set_ylabel("Mean risk")
    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(FIGURE_DIR / "convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_learning() -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    eta_by_case = {
        case: np.asarray((2.5e-4, 5.0e-4, 1.0e-3)) for case in CASE_ORDER
    }
    summary: Dict[str, Any] = {}
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.2), squeeze=False)
    for axis, case in zip(axes[0], CASE_ORDER):
        eta_values = eta_by_case[case]
        zeroth = []
        corrected = []
        for eta in eta_values:
            result = learning_result(case, float(eta))
            mean = endpoint(result, method="sgd")["mean"]
            zeroth.append(abs(mean - float(result["gradient_flow"][-1])))
            corrected.append(abs(mean - float(result["perturbation"][-1])))
        zeroth_array = np.maximum(np.asarray(zeroth), 1.0e-30)
        corrected_array = np.maximum(np.asarray(corrected), 1.0e-30)
        slope_zero = float(np.polyfit(np.log(eta_values), np.log(zeroth_array), 1)[0])
        slope_corrected = float(
            np.polyfit(np.log(eta_values), np.log(corrected_array), 1)[0]
        )
        axis.loglog(eta_values, zeroth_array, "o-", label="gradient flow")
        axis.loglog(eta_values, corrected_array, "s-", label="second order")
        axis.set_title(
            case.replace("-", " ").title()
            + "\n"
            + rf"slopes: ${slope_zero:.2f}$, ${slope_corrected:.2f}$",
            fontsize=10,
        )
        axis.set_xlabel(r"$\eta_{\rm peak}$")
        axis.grid(alpha=0.12)
        summary[case] = {
            "eta": eta_values.tolist(),
            "gradient_flow_error": zeroth_array.tolist(),
            "second_order_error": corrected_array.tolist(),
            "gradient_flow_slope": slope_zero,
            "second_order_slope": slope_corrected,
        }
    axes[0, 0].set_ylabel("Final absolute error")
    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(FIGURE_DIR / "learning.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_batches() -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    cases = ("easy-constant",)
    summary: Dict[str, Any] = {}
    fig, axes = plt.subplots(1, 1, figsize=(5.8, 3.5), squeeze=False)
    for axis, case in zip(axes[0], cases):
        case_summary = {}
        for batch in (1, 2, 4):
            result = load_result(f"sgd-{case}-batch{batch}")
            axis.plot(result["times"], result["sgd_mean"], label=f"batch {batch}")
            case_summary[str(batch)] = endpoint(result, "sgd")
        batch8 = load_result(f"sgd-{case}-paths")
        axis.plot(batch8["times"], batch8["sgd_mean"], label="batch 8")
        reference = reference_result(case)
        axis.plot(
            reference["times"], reference["perturbation"], "--",
            color="#222222", label="second order",
        )
        case_summary["8"] = endpoint(batch8, "sgd")
        axis.set_yscale("log")
        axis.set_title(case.replace("-", " ").title())
        axis.set_xlabel(r"Intrinsic time $t$")
        axis.grid(alpha=0.12)
        summary[case] = case_summary
    axes[0, 0].set_ylabel("Mean risk")
    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    fig.savefig(FIGURE_DIR / "batches.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_covariance() -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    cases = ("hard-horizon",)
    summary: Dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), squeeze=False)
    for axis, case in zip(axes[0], cases):
        diagonal = reference_result(case)
        full = covariance_result(case)
        discrete = load_result(f"sgd-{case}-paths")
        visible = np.asarray(diagonal["times"]) >= 100.0
        axis.plot(
            diagonal["times"][visible], diagonal["sde_mean"][visible],
            label="diagonal SDE",
        )
        axis.plot(
            full["times"][visible], full["sde_mean"][visible], label="full SDE",
        )
        axis.plot(
            discrete["times"][visible], discrete["sgd_mean"][visible], label="SGD",
        )
        axis.plot(
            diagonal["times"][visible], diagonal["perturbation"][visible], "--",
            color="#C76B1D", label="diagonal second order",
        )
        axis.plot(
            full["times"][visible], full["perturbation"][visible], ":",
            color="#111111", label="full second order",
        )
        axis.set_yscale("log")
        axis.set_title(case.replace("-", " ").title())
        axis.set_xlabel(r"Intrinsic time $t$")
        axis.grid(alpha=0.12)
        summary[case] = {
            "diagonal": endpoint(diagonal),
            "full": endpoint(full),
            "sgd": endpoint(discrete, "sgd"),
        }
    coarse = load_result("learning-hard-horizon-0250")
    refined = load_result("convergence-hard-horizon-0250-half")
    axis = axes[0, 1]
    visible = np.asarray(coarse["times"]) >= 100.0
    axis.plot(
        coarse["times"][visible], coarse["sde_mean"][visible],
        label="selected bounds",
    )
    axis.plot(
        refined["times"][visible], refined["sde_mean"][visible],
        label="halved bounds",
    )
    axis.plot(
        refined["times"][visible], refined["perturbation"][visible], "--",
        color="#C76B1D", label="second order",
    )
    axis.set_yscale("log")
    axis.set_title(r"Step refinement, $\eta=2.5\times10^{-4}$")
    axis.set_xlabel(r"Intrinsic time $t$")
    axis.grid(alpha=0.12)
    summary["step_refinement"] = {
        "selected": endpoint(coarse),
        "halved": endpoint(refined),
    }
    axes[0, 0].set_ylabel("Mean risk")
    handles, labels_out = [], []
    for axis in axes[0]:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels_out:
                handles.append(handle)
                labels_out.append(label)
    fig.legend(handles, labels_out, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    fig.savefig(FIGURE_DIR / "covariance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def generate_report() -> None:
    missing = [job.key for job in JOBS.values() if not valid_result(job)]
    if missing:
        raise SystemExit(f"Cannot report: {len(missing)} jobs remain.")
    summaries = {
        "convergence": plot_convergence(),
        "learning": plot_learning(),
        "batches": plot_batches(),
        "covariance": plot_covariance(),
    }
    metadata = json.loads(FINAL_METADATA.read_text(encoding="utf-8"))
    metadata["summaries"] = summaries
    metadata["figures"] = [
        str(FIGURE_DIR / f"{name}.png")
        for name in ("convergence", "learning", "batches", "covariance")
    ]
    metadata["notes"] = [
        "The hard-horizon diagonal SDE endpoint is strongly step-size sensitive; "
        "treat its apparent learning-rate slope as unresolved.",
        "Replacing the diagonal diffusion by the exact Gaussian-gradient "
        "covariance does not reconcile the hard-horizon SDE with direct SGD.",
        "A 10000-mode deterministic resolution run was not retained: one compiled "
        "MLX integration block exceeded the 45-minute per-job guard before it "
        "could return control to Python.",
    ]
    FINAL_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("generated four aggregate study figures", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", choices=JOBS)
    group.add_argument(
        "--study", choices=("convergence", "learning", "sgd", "all")
    )
    group.add_argument(
        "--list", dest="list_study",
        choices=("convergence", "learning", "sgd", "all"),
    )
    group.add_argument("--progress", action="store_true")
    group.add_argument("--collect", action="store_true")
    group.add_argument("--report", action="store_true")
    group.add_argument("--campaign", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--wait-pid",
        type=int,
        help="For --campaign, wait for this already-running job before resuming.",
    )
    parser.add_argument(
        "--deadline-epoch",
        type=float,
        help="Absolute Unix deadline; the final two hours are reserved for reporting.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.campaign:
        if args.deadline_epoch is None:
            raise SystemExit("--campaign requires --deadline-epoch")
        run_campaign(args.deadline_epoch, args.wait_pid)
    elif args.job:
        run_job(JOBS[args.job], args.deadline_epoch)
    elif args.study:
        run_study(args.study, args.deadline_epoch)
    elif args.list_study:
        for job in selected_jobs(args.list_study):
            print(job.key)
    elif args.progress:
        complete = sum(valid_result(job) for job in JOBS.values())
        print(f"{complete}/{len(JOBS)} jobs complete")
        for study in ("convergence", "learning", "sgd"):
            jobs = selected_jobs(study)
            done = sum(valid_result(job) for job in jobs)
            print(f"{study}: {done}/{len(jobs)}")
    elif args.report:
        generate_report()
    else:
        collect_results(require_complete=not args.allow_partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
