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

## Figures

The PNG and PDF versions of a figure contain the same curves. The retained
arrays and run metadata are under `output/data/` and `output/metadata/`.

### [SDE](output/figures/sde.png) ([PDF](output/figures/sde.pdf))

This is the four-panel reproduction of the manuscript figure. The panels cross
the hard and easy spectra with horizon-dependent and constant-order initial
conditions. The dashed curve is gradient flow $\mathcal E_t^{(0)}$. The
dash-dotted curve adds the first correction,
$\mathcal E_t^{(0)}+\eta_{\rm peak}\mathbb E[\mathcal E_t^{(2)}]$. The blue
curve and band show the mean and the 5th--95th percentiles of the simulated SDE
paths. The SDE mean follows the corrected curve through most of training, while
its final value depends strongly on the numerical step bounds.

### [Comparison](output/figures/comparison.png) ([PDF](output/figures/comparison.pdf))

This figure adds direct SGD with fresh Gaussian samples to the four SDE panels.
Solid lines are means and dotted lines are medians. The vertical line at
$t\simeq7000$ marks the start of learning-rate decay. Direct SGD learns in all
four regimes and finishes close to the second-order prediction. At
$\eta_{\rm peak}=10^{-3}$, the four relative endpoint errors are $0.002\%$,
$0.124\%$, $1.04\%$, and $0.119\%$. The SDE endpoint is visibly higher in the
easy panels and in the hard horizon-dependent panel.

### [Decay](output/figures/decay.png) ([PDF](output/figures/decay.pdf))

This is a magnified view of $7000\leq t\leq10000$. The faint gray curve is the
normalized learning rate $\eta(t)/\eta_{\rm peak}$. From the start of decay to
the endpoint, the SGD mean risk falls by factors of $10.1$, $2.66$,
$837$, and $354$ in panel order. The SGD curves in the easy regimes therefore learn
strongly during decay, despite the less convincing pathwise SDE curves in the
original reproduction.

### [Convergence](output/figures/convergence.png)

The upper panels compare SDE means after successively halving the allowed drift
and diffusion increments. The lower panels show final means and Monte Carlo
standard errors. The black diamonds use 64 paths at the selected bounds, and
the dashed line is the second-order prediction. In the easy horizon-dependent
case, the quarter and eighth bounds agree within sampling error, around
$8\times10^{-5}$, while the prediction is $1.95\times10^{-5}$. The easy
constant-order case retains a visible step dependence. These checks show that
the pathwise Euler--Maruyama result has a substantial numerical bias at the
endpoint.

### [Learning](output/figures/learning.png)

Each panel plots the direct SGD endpoint error against $\eta_{\rm peak}$ for
gradient flow and the second-order approximation. The displayed slopes are
least-squares fits on the three log--log points. The easy horizon-dependent
case gives slopes $1.12$ and $2.91$: the correction removes the leading error
and leaves a much smaller residual. The three points in each hard regime leave
the small-$\eta$ scaling unresolved, although their endpoint errors at
$\eta_{\rm peak}=10^{-3}$ remain small.

### [Batches](output/figures/batches.png)

This figure compares direct SGD with batches $1$, $2$, $4$, and $8$ in the easy
constant-order case. The optimizer step scales with the batch so that
$\text{step}/\text{batch}=\eta_{\rm peak}$ remains fixed. All four endpoints
lie within about two Monte Carlo standard errors of the second-order
prediction. Any systematic batch dependence lies below the Monte Carlo
resolution of this experiment.

### [Covariance](output/figures/covariance.png)

The left panel compares the diagonal-noise SDE, the SDE with the exact Gaussian
gradient covariance, direct SGD, and their perturbative curves in the hard
horizon-dependent regime. The diagonal SDE, full SDE, and direct SGD endpoints
are $0.002134$, $0.002155$, and $0.001145$. Including the off-diagonal
covariance therefore leaves the SDE--SGD gap essentially unchanged. The right
panel repeats the SDE at $\eta_{\rm peak}=2.5\times10^{-4}$ with halved step
bounds. The endpoint moves from $0.001666$ to $0.002103$, a $26\%$ change. This
step sensitivity makes the pathwise SDE curves unsuitable as a numerical test
of the perturbative prediction.

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
