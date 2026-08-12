# Pilot report: one-pass SGD schedule transfer

Date: 2026-08-12.

## Question

Three constant learning rates are used for calibration.  All fitted quantities
are then frozen.  The test asks whether the discretized FSL predicts population
excess-risk trajectories under cosine, WSD, cyclic, and late-drop schedules.
Every schedule processes fresh samples once, has the same number of updates,
starts from the same parameters, and uses paired random keys.

## Retained pilot

The retained Metal run uses

$$
d=16,\quad N=65{,}536,\quad L=5,\quad
\beta=2,\quad\chi=-0.2,\quad\sigma=0.3,
\quad\eta_{\rm mid}=0.0016.
$$

There are $128$ independent paths per schedule.  The low and high constant
rates are $0.8\eta_{\rm mid}$ and $1.2\eta_{\rm mid}$.  The four held-out
schedules have the same total intrinsic time as the middle constant schedule.
The largest relative layer update is $0.229$ and all layer coordinates remain
positive.  A previous run with twice the step size failed on cosine at step
$283$ when its largest relative update reached $1.14$.  That failed run was
discarded before fitting.

The primary first Picard iterate is

$$
\widehat E_q
=a_S S(t_q)
+a_K\sum_{n<q}\eta_n^2K(t_q-t_{n+1})
\bigl(S(t_n)+\sigma^2\bigr).
$$

The fit uses the known finite profiles.  The signal basis contains the
initial-error sum.  The proof's barrier contribution is
excluded because it belongs to a one-sided upper bound and is much larger than
the observed trajectory for this initialization.  The fitted values are

$$
a_S=1.5204,\qquad a_K=202.28,\qquad c_S=c_K=16.
$$

The recursive diagnostic solves

$$
F_q=a_SS(t_q)+a_K\sum_{n<q}\eta_n^2K(t_q-t_{n+1})
(F_n+\sigma^2)
$$

with $a_S=1.5500$, $a_K=170.10$, and the same decay constants.  The recursive
prediction and the first Picard iterate are close.

## Held-out results

The table reports log-RMSE over the full trajectory.  The intrinsic-time
baseline interpolates the middle constant-rate trajectory using $t_q$.  Its
input contains no information about the order of the learning rates.

| schedule | first Picard iterate | discrete recurrence | intrinsic-time baseline | recurrence minus baseline |
|---|---:|---:|---:|---:|
| cosine | $0.503$ | $0.486$ | $1.124$ | $-0.637$ |
| WSD | $0.504$ | $0.497$ | $0.632$ | $-0.135$ |
| late drop | $0.501$ | $0.489$ | $0.709$ | $-0.219$ |
| cyclic | $0.462$ | $0.467$ | $0.348$ | $+0.119$ |

For the first Picard iterate, a paired $200$-replicate path bootstrap gives
the following $90\%$ intervals for log-RMSE:

| schedule | FSL $90\%$ interval | baseline $90\%$ interval |
|---|---:|---:|
| cosine | $[0.499,0.512]$ | $[1.096,1.155]$ |
| WSD | $[0.503,0.511]$ | $[0.611,0.659]$ |
| late drop | $[0.499,0.507]$ | $[0.689,0.732]$ |
| cyclic | $[0.463,0.468]$ | $[0.346,0.362]$ |

The convolution carries predictive information for cosine, WSD, and late
drop.  The cyclic result fails the stated criterion.  Its risk responds to the
oscillations, while the fitted finite kernel has larger error than the
intrinsic-time interpolation.

Terminal relative error reveals another limitation.  The first Picard iterate
has errors $1.19$ on cosine, $1.21$ on WSD, $0.50$ on late drop, and $0.26$ on
cyclic.  The discrete recurrence reduces these to $0.86$, $0.89$, $0.29$, and
$0.075$.  The recursive formula is therefore more useful for final-risk
analysis even though its full-trajectory log-RMSE is similar.

## What this validates

The pilot supports an $\eta_n^2$ schedule response with forgetting by remaining
intrinsic time.  The result is strongest for schedules whose rate decays late
in training.  Its scope is finite-dimensional prediction for the four tested
schedules; equality with the upper bound and the asymptotic exponents require
separate tests.

The exact discrete recurrence is the main object for subsequent experiments.
For terminal risk it makes visibly better predictions than one Picard step.
The next diagonal study should vary $d$, $N$, $L$, $\chi$, $\sigma$, and the
peak step size.  The required checks are:

1. hold $T_N/d^p$ fixed while increasing $d$;
2. halve $\max_n\eta_n$ at fixed $T_N$ to measure discretization error;
3. compare smooth cyclic schedules with increasing frequency;
4. fit constants at one $d$ and predict another $d$;
5. test $\sigma=0$ and two positive noise levels;
6. report terminal error and full-trajectory error separately;
7. rerun the contraction search on a finer calibration grid after the design
   is frozen.

The anchored diagnostic fixes $a_S=1$, as required by the exact risk at step
zero.  Its fit is worse and its kernel contraction reaches the search boundary.
The early transient therefore needs study before any equality claim.  A useful
next comparison is the coordinatewise deterministic recursion obtained by
replacing the sample gradient with its conditional mean.

## Boundary regime

A second retained run changes only $\chi$ from $-0.2$ to $0$.  Its trajectories
remain stable, with largest relative update $0.155$.  The model with one
coefficient for state-dependent and label-noise forcing transfers poorly:

| schedule | shared coefficient | separate coefficients | intrinsic-time baseline |
|---|---:|---:|---:|
| cosine | $0.996$ | $0.518$ | $1.103$ |
| WSD | $0.763$ | $0.287$ | $0.622$ |
| late drop | $0.844$ | $0.294$ | $0.697$ |
| cyclic | $0.583$ | $0.372$ | $0.333$ |

The separate-coefficient fit has

$$
a_S=0.966,\qquad a_R=18{,}814.5,\qquad a_\sigma=65.19,
\qquad c_S=45.25,\qquad c_K=5.66.
$$

This model predicts cosine, WSD, and late drop well.  Cyclic remains worse than
the intrinsic-time baseline.  The large difference between $a_R$ and
$a_\sigma$ shows that one shared empirical coefficient is too restrictive in
the boundary regime.  In the proof, the same $C_R$ bounds both forcing terms.
As a bound constant, $C_R$ places no equality constraint on $a_R$ and
$a_\sigma$.

The boundary run strengthens two design choices.  Report the shared model as
the direct analogue of the bound and the separate-coefficient model as the
main predictive formula.  Include at least one oscillatory schedule among the
held-out schedules.

## MLP and convolutional networks

A real-network experiment can test the same schedule-transfer idea after the
diagonal study passes across dimensions.  Start with a one-pass MLP on MNIST or
Fashion-MNIST and use validation negative log likelihood.  Calibrate on three
constant rates, freeze every exponent and coefficient, and predict cosine,
WSD, and late-drop schedules with paired initialization and data-order seeds.
Batch size enters the schedule feature as $\eta_n^2/B$.

For a small CIFAR-10 convolutional network, fix augmentation, data order,
normalization, weight decay, batch size, and total examples.  Report both
parameter-matched and FLOP-matched comparisons.  Learned effective exponents
are allowed only when selected jointly on constant-rate trajectories.  These
experiments test an empirical formula outside the diagonal proof.

The complete real-network design is in `REAL_NETWORK_PROTOCOL.md`.
