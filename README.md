# SDE perturbation and SGD comparison

This repository contains the Apple Metal reproduction of the four SDE panels
from the accompanying manuscript and the matching SGD experiment with fresh samples.
MLX is used because it was substantially faster than JAX on this Apple GPU.

## Files

- `experiments/sde.py`: Monte Carlo SDE, gradient flow, and the second-order perturbation.
- `experiments/sgd.py`: direct SGD and SDE/SGD comparison plots.
- `experiments/studies.py`: restartable validation campaign and aggregate reports.
- `experiments/common.py`: shared cases, safeguards, Metal setup, and SDE plotting.

All programs use one output root:

```text
output/
├── data/{sde,sgd}/
├── figures/
└── metadata/
```

The saved figures are `sde.{png,pdf}`, `comparison.{png,pdf}`,
`decay.{png,pdf}`, `convergence.png`, `learning.png`, `batches.png`, and
`covariance.png`. The retained arrays and run metadata are included under
`output/data/` and `output/metadata/`.

## Environment

On Apple silicon with Python 3.12:

```bash
python3.12 -m venv .venv-mlx312
source .venv-mlx312/bin/activate
python -m pip install -r experiments/requirements.txt
python -c "import mlx.core as mx; mx.set_default_device(mx.gpu); print(mx.default_device())"
```

The last command must print an MLX GPU device.

## SDE

The default preview has an eight-minute wall-clock guard:

```bash
python experiments/sde.py
```

The retained result used 512 modes, 32 paths, depth 10, and the conservative
SDE increment bounds from the manuscript. Run two cases per process:

```bash
python experiments/sde.py --preset paper --confirm-paper-run \
  --cases hard-horizon hard-constant --modes 512 --paths 32

python experiments/sde.py --preset paper --confirm-paper-run \
  --cases easy-horizon easy-constant --modes 512 --paths 32
```

Rebuild `output/figures/sde.{png,pdf}` without starting a simulation:

```bash
python experiments/sde.py --preset paper --plot-only
```

For depth $L$, the simulation uses

$$
\chi=2-\frac{2}{L},\qquad
\mathcal E_t=\frac12\sum_j\lambda_j(u_j-u_j^\star)^2.
$$

Writing $z_j=\log u_j$, the simulated SDE is

$$
\begin{aligned}
\mathrm dz_j={}&\left[-L^2\lambda_j u_j^{\chi-1}(u_j-u_j^\star)
+\frac{(\chi-2)L^4}{4}\lambda_j\eta(t)
(\mathcal E_t+\sigma^2)u_j^{2\chi-2}\right]\mathrm dt\\
&+L^2u_j^{\chi-1}
\sqrt{\eta(t)(\mathcal E_t+\sigma^2)\lambda_j}\,\mathrm dB_{j,t}.
\end{aligned}
$$

The logarithmic Euler--Maruyama solver preserves positivity. The moment
solver uses embedded Dormand--Prince 5(4) and plots

$$
\mathbb E[\mathcal E_t]\approx
\mathcal E_t^{\rm GF}
+\frac{\eta_{\rm peak}}2\sum_j\lambda_j(P_j+2Q_j).
$$

## SGD

The reported comparison uses batch 8, 4.25 million optimizer updates, and 34
million fresh samples. Its peak optimizer step is $0.008$, so
$0.008/8=0.001$, matching the SDE diffusion scale. This is a long run and
requires explicit confirmation:

```bash
python experiments/sgd.py --cases easy-horizon easy-constant \
  --confirm-long-run

python experiments/sgd.py --cases hard-horizon hard-constant \
  --confirm-long-run
```

Rebuild `comparison.{png,pdf}` and `decay.{png,pdf}` from the saved arrays:

```bash
python experiments/sgd.py \
  --cases hard-horizon hard-constant easy-horizon easy-constant --plot-only
```

Each SGD update draws fresh Gaussian features $x$ and label noise $\varepsilon$.
The covariance of the stochastic gradient contains off-diagonal entries.

## Validation studies

`studies.py` defines 28 retained convergence, learning-rate, direct-SGD,
batch-size, and covariance jobs.  The learning-rate figure compares direct SGD
at three effective diffusion scales; the two smaller-rate theoretical curves
reuse the normalized perturbative hierarchy and rescale its correction exactly.
The hard-horizon diagnostics also retain a halved-step SDE refinement, which
exposes the strong step-size sensitivity of the Monte Carlo SDE curve.

Each job writes one restartable checkpoint under
`/private/tmp/scaling-studies`. These temporary files are collected into
`output/data/studies.npz` and `output/metadata/studies.json`. An absolute Unix
deadline can be passed to any job or study. The scheduler reserves its final two
hours for aggregation, plotting, interpretation, and cleanup.

```bash
python experiments/studies.py --progress
python experiments/studies.py --study convergence
python experiments/studies.py --study learning --deadline-epoch UNIX_TIME
python experiments/studies.py --study sgd
python experiments/studies.py --collect
python experiments/studies.py --report
```

The covariance of the Gaussian stochastic gradient is

$$
\operatorname{Cov}[x(x^\top\delta-\varepsilon)]
=\left(\delta^\top\Lambda\delta+\sigma^2\right)\Lambda
+\Lambda\delta\delta^\top\Lambda.
$$

The second term has rank one. The implementation applies its square root without
storing a dense covariance matrix.
