# One-pass FSL schedule-transfer experiments

This directory tests whether an FSL calibrated on constant-learning-rate
one-pass SGD trajectories predicts unseen learning-rate schedules.  All code
and results for this study stay here.  The existing `experiments/` suite
remains unchanged.

The main claim is schedule transfer in a synthetic diagonal network.  MLP and
convolutional experiments come after the diagonal study.  Read `WORKLOG.md`
before changing the protocol.

## Environment

On Apple silicon:

```bash
python3.12 -m venv .venv-fsl-transfer
source .venv-fsl-transfer/bin/activate
python -m pip install -r fsl_transfer/requirements.txt
python -c "import mlx.core as mx; mx.set_default_device(mx.gpu); print(mx.default_device())"
```

The simulator refuses to run unless the MLX GPU device is available.

## Intended commands

The quick pilot will be:

```bash
.venv-mlx312/bin/python fsl_transfer/run.py --preset smoke
```

The retained pilot uses $128$ paired paths and $d=16$:

```bash
.venv-mlx312/bin/python fsl_transfer/run.py --preset pilot
```

The retained diagonal study will use:

```bash
.venv-mlx312/bin/python fsl_transfer/run.py --preset study
```

Results are written under `fsl_transfer/results/`, which is ignored by Git.
The report records calibration schedules, frozen fitted parameters, metrics on
each held-out schedule, uncertainty intervals, and positivity diagnostics.
`PILOT_REPORT.md` gives the retained Metal result and its interpretation.

## Interpretation

A diagonal result supports the schedule-dependent convolution for the tested
finite one-pass SGD systems.  The theorem remains a one-sided statement.
An MLP or convolutional result would support the empirical formula for that
network.
