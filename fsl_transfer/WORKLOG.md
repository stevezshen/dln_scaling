# Working log: schedule transfer for the one-pass SGD FSL

Last updated: 2026-08-12.

This file records decisions that must survive context compression.  The
existing `experiments/` directory is outside the scope of this task and must
remain unchanged.  All new work belongs in `fsl_transfer/`.

## Question

Can a functional scaling law calibrated only on constant-learning-rate
one-pass SGD trajectories predict the population excess-risk trajectory under
an unseen learning-rate schedule?

The important word is **predict**.  A schedule-transfer test fits one set of
parameters on constant rates and freezes them before evaluating new schedules.
A separate five-parameter fit to each schedule only measures fit error.

## Evidence inspected

1. `one_pass_sgd.tex` proves a discrete one-sided upper bound.  For a bounded
   deterministic schedule, the stopped expected risk is controlled by a
   discrete Volterra equation.  Its kernel is a finite power-law sum and its
   schedule dependence enters through `eta_n^2` and remaining intrinsic time.
2. `draft_new.tex`, Section "Numerical Form of the FSL" and Appendix
   "FSL for DLNs", fits five parameters separately to post-warmup trajectories.
   The fit discards warmup memory.  It lacks held-out schedules, an
   intrinsic-time baseline, and uncertainty intervals.
3. Li et al., *Functional Scaling Laws in Kernel Regression* (arXiv:2509.19189,
   version 4), fit three linear coefficients to averaged kernel-SGD curves.
   Their LLM experiment has the stronger protocol: fit the 8-1-1 trajectory
   and predict unseen cosine and WSD trajectories.  The paper's Appendix B.1
   uses 200 independent runs, batch size one, and 10,000 steps for each PLK
   configuration.  Sources:
   - https://arxiv.org/abs/2509.19189
   - https://arxiv.org/html/2509.19189v4

## Claim ladder

The experiments should distinguish four claims.

1. **Descriptive fit:** an FSL can fit one observed trajectory.
2. **Seed generalization:** parameters fitted to some constant-rate seeds
   predict new seeds at the same constant rate.
3. **Schedule transfer:** parameters fitted only to constant-rate runs predict
   new schedule shapes without refitting.
4. **Network transfer:** the same formula predicts MLP or convolutional-network
   losses.  This claim concerns those experiments; the theorem concerns
   diagonal networks.

Claim 3 is the primary test.  Claim 1 is too weak.  Claim 4 should be attempted
only after Claim 3 succeeds in the diagonal model.

## Discrete formula from the proof

For depth $L$, put

$$
\alpha=2-\frac{2}{L},\qquad
p=\beta+\alpha\chi,\qquad
s=\beta+2\chi-1.
$$

For schedule $\eta_0,\ldots,\eta_{N-1}$, define

$$
t_q=\sum_{n<q}\eta_n.
$$

The finite signal profile from the note is

$$
S_{d,0}^{(c_s)}(t)
=\sum_{j\le d}\lambda_j(u_{j,0}-u_j^*)^2e^{-c_stj^{-p}}
+\sum_{j\le d}\lambda_j(u_j^*)^2(tj^{-p})
 e^{-c_stj^{-p}}\Theta_{j,0},
$$

and the finite forgetting kernel is

$$
K_d^{(c_k)}(t)=\sum_{j\le d}j^{-2p}e^{-c_ktj^{-p}}.
$$

The first Picard iterate used for the primary fit is

$$
\widehat E_q
=a_S S_{d,0}^{(c_s)}(t_q)
+a_R\sum_{n<q}\eta_n^2 K_d^{(c_k)}(t_q-t_{n+1})
 S_{d,0}^{(c_s)}(t_n)
+a_\sigma\sigma^2\sum_{n<q}\eta_n^2
 K_d^{(c_k)}(t_q-t_{n+1}).
$$

The coefficients are constrained to be nonnegative.  The primary model shares
one coefficient between the two convolution terms; the relaxed diagnostic
allows $a_R$ and $a_\sigma$ to differ.  The decay constants $c_s,c_k$ are
selected on the calibration trajectories only.  The exponents $p,s$ remain
fixed at their known synthetic values in the main analysis.

For the slope-independent initialization, the barrier part containing
$\Theta_{j,0}$ can be much larger than the observed risk.  It enters the proof
as part of a one-sided bound.  The empirical equality therefore uses the first
sum in $S_{d,0}^{(c_s)}$ as its signal basis.  A separate diagnostic restores
the barrier part and records the resulting error.  For proportional
initialization with $u_{j,0}=\rho u_j^*$ and $\rho\in[3/4,1)$, the barrier
part vanishes.

This formula has four advantages over a free five-parameter power curve:

- it retains the complete schedule from step zero;
- it uses the finite-mode profiles actually appearing in the proof;
- positivity is automatic;
- the amplitude fit is a small nonnegative least-squares problem;
- every fitted quantity is frozen before evaluating a new schedule.

The full discrete Volterra recurrence is a secondary model.  The theorem gives
an upper bound.  Successful transfer of the first Picard iterate supports the
schedule dependence in that bound.

## Synthetic protocol

### Data and recursion

- Independent fresh Gaussian features at every update:
  $X_{n,j}=j^{-\beta/2}Z_{n,j}$.
- Labels:
  $Y_n=\sum_j u_j^*X_{n,j}+\xi_n$, with
  $u_j^*=j^{-\chi}$ and $\xi_n\sim N(0,\sigma^2)$.
- Tied positive depth-$L$ diagonal network, updated with the exact sample SGD
  recursion used in `one_pass_sgd.tex`.
- Population excess risk is evaluated exactly from the parameters; it is not
  estimated from a finite validation set.
- Record individual paths as well as their mean.  Never fit to a smoothed path
  without retaining seed-level uncertainty.

### Initialization

The main experiment uses the slope-independent initialization

$$
u_{j,0}=c_{\rm init}T_{\rm ref}^{-1/\alpha}j^{\beta/\alpha},
\qquad T_{\rm ref}=N\eta_{\rm mid}.
$$

Every schedule in a transfer study starts from this same initialization.  The
low and high constant schedules have total times $0.8T_{\rm ref}$ and
$1.2T_{\rm ref}$, which remain comparable to $d^p$ under the theorem's
assumption.  The finite signal profile uses the stored initial condition.  A
proportional initialization gives the cleanest signal profile and is included
as a secondary experiment.

### Calibration and test split

- Calibration: three constant learning rates, jointly fitted.  Their rates
  bracket the mean rate of every transfer schedule.
- Held-out schedules: cosine, WSD, cyclic, and late-drop.
- Keep $d,N,L,\beta,\chi,\sigma$ fixed within a transfer study.
- Normalize transfer schedules so their total intrinsic time matches the
  middle constant-rate run.  Every calibration and transfer trajectory uses
  the same initialization.
- Repeat hard ($\chi<0$), boundary ($\chi=0$), and easy ($\chi>0$) regimes.
- Repeat at $L=2,5,10$ after the pilot.

### Controls and ablations

1. Signal-only model, which sees intrinsic time but no schedule convolution.
2. FSL with the theoretical exponents fixed.
3. FSL with exponents estimated jointly from calibration trajectories.
4. Per-schedule refit, reported only as an optimistic descriptive ceiling.
5. Zero-label-noise and at least two positive values of $\sigma$.
6. At least three truncations $d$ or an analytic omitted-tail correction.
7. At least three peak-rate scales to expose discretization failure.
8. Full Volterra recurrence versus its first Picard iterate.

### Metrics and uncertainty

- held-out log RMSE over the full recorded trajectory;
- held-out log RMSE in early, middle, and terminal thirds;
- relative terminal-risk error;
- improvement over the signal-only baseline;
- calibration-to-transfer error ratio;
- bootstrap 90% intervals obtained by resampling independent SGD paths;
- positivity failures and maximal relative layer update;
- sensitivity to $d$, number of paths, checkpoint density, and fitted window.

A compelling result beats the signal-only and intrinsic-time baselines on all
held-out schedules, with uncertainty intervals and frozen parameters.  Results
are reported separately for every schedule.

## MLP and convolutional networks

These experiments make a separate claim about conventional networks.  Their
kernel and signal exponents are learned from the constant-rate trajectories.
The fitted expression is called an empirical FSL.

- Use validation negative log likelihood as the primary loss and show training
  loss separately.  Accuracy is too discontinuous for trajectory fitting.
- Fit only constant-rate trajectories and predict unseen schedules.
- Fix batch size, data order, augmentation, optimizer, initialization family,
  weight decay, normalization, and total examples.
- Pair seeds across schedules when possible.
- Report parameter-matched and FLOP-matched depth comparisons; fixed width is
  insufficient because parameters and compute grow with depth.
- Compare a floor-inclusive signal model and simpler spline/exponential
  baselines.  Compare their held-out errors.
- For a literal one-pass study, use each base example once.  Report augmentation
  and every reuse of a base image.
- Start with an MLP on Fashion-MNIST or MNIST.  Move to a small convolutional
  network on CIFAR-10 only after the diagonal schedule-transfer gate passes.

The network FSL may use learned effective exponents.  They are shared across
schedules and fitted only on calibration runs.  Batch
size enters the noise feature as $\eta_n^2/B$.

## MLX decision

MLX is appropriate for the diagonal simulation because paths and coordinates
vectorize into dense Metal arrays.  The implementation must select `mx.gpu`
explicitly and refuse silent CPU fallback.  Compilation should cover a fixed
block of SGD updates.  Host synchronization is restricted to checkpoints.

The repository already contains an MLX 0.32.0 environment.  In the sandbox,
MLX imports but Metal is unavailable; a GPU run must be executed with host
permission.  A NumPy implementation checks the fitter and schedule algebra on
small tests.  Retained simulations use MLX Metal.

## Current implementation status

- [x] Experimental question and pass criteria fixed.
- [x] Problems in the draft experiment identified.
- [x] Original FSL paper's fitting and schedule-transfer experiments reviewed.
- [x] MLX diagonal simulator with explicit Metal selection and stability checks.
- [x] Finite-profile first Picard iterate and discrete Volterra fits.
- [x] Intrinsic-time and signal-only baselines.
- [x] Automated Metal pilot, paired bootstrap, plot, and JSON report.
- [ ] Real-network trajectory adapter.
- [x] Real-network protocol.
- [x] Metal pilot results summarized in `PILOT_REPORT.md`.

## Current conclusion

The retained hard-regime pilot supports held-out transfer for cosine, WSD, and
late-drop schedules.  It fails against the intrinsic-time baseline for the
cyclic schedule.  The discrete Volterra recurrence improves final-risk
prediction over its first Picard iterate.  In the boundary regime, separate
coefficients for state-dependent and label-noise forcing are required for good
transfer.  The next experiments should test dimension scaling and finite-step
error before moving to an MLP.
