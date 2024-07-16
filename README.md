# KAN

Experiments with learnable KAN / univariate functions.

The actual main code for training networks are in tests across the various packages.

Status: Probably not going to work on this anymore as Bezier KANs seem to not train without residual functions.

## Ideas backlog

 - Some loss penality for non-convexity?
 - Squimble-lock: dynamically reduce learning rate if gradient at each step oscillates
   - What if we could characterize the oscillation + negate it?
 - What if we adjust the weights beyond the spline it contributes to?
 - What if we try and smooth out the spline every N steps?
