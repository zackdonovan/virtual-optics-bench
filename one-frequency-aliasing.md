# why one modulation frequency can hide a second heat path

In the small-signal example, remove the known drive amplitude, DC gain, and
cavity/detector transfer functions. The remaining normalized complex
measurement of a fast response plus one thermal pole is

```text
z = (1 - q) + q/(1 + i*x),       x = 2*pi*f*tau
1 - z = q*(x*x + i*x)/(1 + x*x).
```

Write `u = 1 - Re(z)` and `v = -Im(z)`. For nondegenerate positive
`u,v`, the mixed model with `x_eff = u/v` and
`q_eff = (u*u + v*v)/u` returns exactly this same complex number.
Every positive normalized mixture of thermal poles lies in the convex
lower half-disc bounded by the arc `1/(1+i*x)` and the real axis; its
half-disc condition `u*u + v*v <= u` ensures `q_eff <= 1`.
Thus a positive two-pole mechanism can exactly alias a one-pole-plus-Kerr
mechanism at one frequency even without noise, if the fit menu contains
the effective parameters. A discrete or restricted menu may miss this
exact point, but the observed difference can still be below measurement
noise. The result concerns this transfer-function model, not general
identifiability of real optical hardware.

Concrete example: equal heat components with time constants 0.25 ms and
5.0 ms. At 300 Hz their response matches a single heat component with
a smaller fraction and an apparent instantaneous component, despite the
truth containing no instantaneous component. Repeating the algebra at a
different frequency generally gives different effective parameters, so a
shared-parameter fit across frequencies can break the alias. It is not
guaranteed to do so at finite SNR, even with multiple measurements.

In that noiseless example the equivalent fit changes from `q_eff = 0.688`,
`tau_eff = 1.266 ms` at 300 Hz to `q_eff = 0.924`, `tau_eff = 0.562 ms`
at 1000 Hz. One set of shared parameters cannot reproduce both exactly.
