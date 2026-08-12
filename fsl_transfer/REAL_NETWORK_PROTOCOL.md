# Protocol for MLP and convolutional extensions

This stage begins only after frozen-parameter schedule transfer works for the
diagonal model.

## Stage A: MLP

- Dataset: MNIST or Fashion-MNIST with a prespecified train/validation split.
- One shuffled pass through unique training examples.
- Loss: validation negative log likelihood at fixed example-count checkpoints;
  training negative log likelihood is reported separately.
- Optimizer: plain SGD, fixed batch size, no momentum for the first study.
- Calibration: three constant rates and at least five paired seeds per rate.
- Transfer: cosine, WSD, cyclic, and late-drop schedules with frozen FSL
  parameters.
- Architectures: a single fixed MLP first.  Depth comparisons require both
  parameter-matched and FLOP-matched variants.

## Stage B: convolutional network

- Dataset: CIFAR-10.
- Small convolutional network with explicit channel counts, stride, pooling,
  normalization, biases, augmentation, and parameter count.
- The first experiment omits normalization and momentum so the schedule is the
  main changing input.  Later experiments add modern training components one
  at a time.
- Every base image is used once in the literal one-pass result.  If repeated
  augmentation is used, report it as multi-pass training.

## Shared fit

For a fixed model and batch size, use

$$
t_k=\sum_{i<k}\eta_i,\qquad
v_k=\sum_{i<k}\frac{\eta_i^2}{B}
K(t_k-t_{i+1}).
$$

The signal exponent, kernel exponent, shifts, and amplitudes are fitted jointly
on constant-rate calibration runs and frozen.  The fit includes a positive
loss floor.  Report schedule-held-out log RMSE and terminal error against:

1. an intrinsic-time signal-only law;
2. an exponential-kernel convolution;
3. a monotone spline using intrinsic time and cumulative squared learning rate;
4. a per-schedule refit as a descriptive ceiling.

The network result supports the FSL ansatz only if its frozen transfer error is
consistently below these baselines across seeds and schedules.

