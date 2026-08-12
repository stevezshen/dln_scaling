# One-pass FSL schedule-transfer experiments

This directory tests whether an FSL calibrated on constant-learning-rate
one-pass SGD trajectories predicts unseen learning-rate schedules.  It is
self-contained and does not modify or import the existing `experiments/`
suite.

The confirmatory claim is schedule transfer in a synthetic diagonal network.
MLP and convolutional experiments are a later external-validity test.  Read
`WORKLOG.md` before changing the protocol.

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

The retained diagonal study will use:

```bash
.venv-mlx312/bin/python fsl_transfer/run.py --preset study
```

Results are written under `fsl_transfer/results/`, which is ignored by Git.
The report records calibration schedules, frozen fitted parameters, metrics on
each held-out schedule, uncertainty intervals, and positivity diagnostics.

## Interpretation

Passing the diagonal experiment means that the schedule-dependent convolution
is a useful predictive surrogate for this finite one-pass SGD system.  It does
not turn the one-sided theorem into a two-sided equality.  Passing an MLP or
convolutional experiment supports an empirical ansatz outside the theorem.

