# Working log: schedule transfer for the one-pass SGD FSL

Last updated: 2026-08-12.

This file records decisions that must survive context compression.  The
existing `experiments/` directory is outside the scope of this task and must
remain unchanged.  All new work belongs in `fsl_transfer/`.

## Question

Can a functional scaling law calibrated only on constant-learning-rate
one-pass SGD trajectories predict the population excess-risk trajectory under
an unseen learning-rate schedule?

The important word is **predict**.  Fitting a separate five-parameter curve to
each schedule tests interpolation capacity and does not establish schedule
transfer.

## Evidence inspected

1. `one_pass_sgd.tex` proves a discrete one-sided upper bound.  For a bounded
   deterministic schedule, the stopped expected risk is controlled by a
   discrete Volterra equation.  Its kernel is a finite power-law sum and its
   schedule dependence enters through `eta_n^2` and remaining intrinsic time.
2. `draft_new.tex`, Section "Numerical Form of the FSL" and Appendix
   "FSL for DLNs", fits five parameters separately to post-warmup trajectories.
   The fit discards warmup memory, does not report schedule-held-out error,
   does not compare with a schedule-blind baseline, and gives no parameter or
   prediction uncertainty.
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
4. **Architecture transfer:** the same ansatz predicts MLP or convolutional
   network losses.  This is an empirical extension and is not covered by the
   diagonal-network theorem.

Claim 3 is the primary test.  Claim 1 is too weak.  Claim 4 should be attempted
only after Claim 3 succeeds in the diagonal model.

## Theory-faithful discrete surrogate

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

The first-response FSL used for the primary fit is

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
fixed at their known synthetic values in the confirmatory analysis.

This surrogate is preferable to a free five-parameter power curve because:

- it retains the complete schedule from step zero;
- it uses the finite-mode profiles actually appearing in the proof;
- positivity is automatic;
- the amplitude fit is a small nonnegative least-squares problem;
- every fitted quantity is frozen before evaluating a new schedule.

The full discrete Volterra recurrence is a secondary model.  The theorem is an
upper bound rather than an equality, so a successful first-response transfer
test is evidence for the schedule-response mechanism, not a verification of
two-sided equality.

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
u_{j,0}=c_{\rm init}T_N^{-1/\alpha}j^{\beta/\alpha}.
$$

Because $T_N$ changes with the schedule, its value and the resulting initial
condition are stored with every trajectory.  The finite signal profile is
computed from that actual initialization.  A proportional initialization is a
mechanism check because it gives the cleanest signal profile, but it is not the
main result.

### Calibration and test split

- Calibration: three constant learning rates, jointly fitted.  Their rates
  bracket the mean rate of every transfer schedule.
- Held-out schedules: cosine, WSD, cyclic, and late-drop.
- Keep $d,N,L,\beta,\chi,\sigma$ fixed within a transfer study.
- Normalize transfer schedules so their total intrinsic time matches the
  middle constant-rate run.  This prevents total training time from revealing
  the answer and keeps the initialization identical for the central transfer
  comparison.
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
8. Full Volterra recurrence versus first response.

### Metrics and uncertainty

- held-out log RMSE over the full recorded trajectory;
- held-out log RMSE in early, middle, and terminal thirds;
- relative terminal-risk error;
- improvement over the signal-only baseline;
- calibration-to-transfer error ratio;
- bootstrap 90% intervals obtained by resampling independent SGD paths;
- positivity failures and maximal relative layer update;
- sensitivity to $d$, number of paths, checkpoint density, and fitted window.

A compelling result requires schedule transfer to beat the signal-only
baseline on all prespecified held-out schedules, with uncertainty intervals,
without changing parameters.  One favorable schedule is exploratory evidence.

## MLP and convolutional networks

These are worthwhile as external-validity tests, with a separate claim.
The theory does not identify their kernel or signal exponents during feature
learning.  The protocol should therefore call the fitted formula an FSL
ansatz.

- Use validation negative log likelihood as the primary loss and show training
  loss separately.  Accuracy is too discontinuous for trajectory fitting.
- Fit only constant-rate trajectories and predict unseen schedules.
- Fix batch size, data order, augmentation, optimizer, initialization family,
  weight decay, normalization, and total examples.
- Pair seeds across schedules when possible.
- Report parameter-matched and FLOP-matched depth comparisons; fixed width is
  insufficient because parameters and compute grow with depth.
- Compare a floor-inclusive signal model and simpler spline/exponential
  baselines.  A flexible FSL ansatz must win out of sample.
- For a literal one-pass study, use each base example once.  Random augmentation
  changes the sample but does not make reuse of the same image independent;
  this distinction must be reported.
- Start with an MLP on Fashion-MNIST or MNIST.  Move to a small convolutional
  network on CIFAR-10 only after the diagonal schedule-transfer gate passes.

The practical network ansatz may use learned effective exponents, but they
must be shared across schedules and fitted only on calibration runs.  Batch
size enters the noise feature as $\eta_n^2/B$.

## MLX decision

MLX is appropriate for the diagonal simulation because paths and coordinates
vectorize into dense Metal arrays.  The implementation must select `mx.gpu`
explicitly and refuse silent CPU fallback.  Compilation should cover a fixed
block of SGD updates.  Host synchronization is restricted to checkpoints.

The repository already contains an MLX 0.32.0 environment.  In the sandbox,
MLX imports but Metal is unavailable; a GPU run must be executed with host
permission.  A NumPy reference implementation is retained for unit tests of
the fitter and schedule algebra, not as the paper-scale simulation backend.

## Current implementation status

- [x] Experimental question and falsifiable protocol fixed.
- [x] Problems in the draft experiment identified.
- [x] Original FSL paper's fitting and schedule-transfer experiments reviewed.
- [ ] MLX diagonal simulator.
- [ ] Finite-profile FSL fitter and baselines.
- [ ] Automated pilot and report.
- [ ] Real-network trajectory adapter.
- [ ] Metal pilot results.

