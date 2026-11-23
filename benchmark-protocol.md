# virtual optics bench: evaluation protocol v1

This is a synthetic experiment-design benchmark, not instrument validation.
The selection problem is deliberately narrow: one additional modulation
measurement after a shared slow/static calibration. "Adaptive" means
conditioning that one choice on the case's calibration, not running a
sequential or RL policy.

## frozen split and information boundaries

- Development: 72 cases per in-menu truth label (seeds rooted at 771410). Test:
  192 cases per truth label (seed root 976104). Balanced by construction;
  these are not prevalence-weighted estimates.
- In-menu truth labels: pure Kerr, pure thermal, mixed. Generate a separate
  omitted-mechanism label. When an omitted mechanism mimics a menu model,
  a false acceptance is still a false acceptance: do not discard hard cases.
- A simulator truth case exposes only its noisy calibration to either policy
  before experiment choice. Neither policy sees truth parameters, labels,
  any dynamic response from that case, or shared test outcomes.
- Compare one adaptive and one fixed-design measurement, both using the same
  instrument, measurement-noise model, classifier and cost per query.
  Reuse the same observation whenever the policies choose exactly the same
  design; otherwise use independent observation noise keyed by case and design.
  Measurement noise is separate from the case-generation/calibration seeds.
- Choose the fixed design on the development cases by maximum unconditional
  correct in-menu classification, ties broken by lower frequency. Report
  omitted-mechanism coverage but do not use it for baseline selection.
- "Unresolved" means the runner did not commit to an in-menu label. Keep
  out-of-family rejection separate from insufficient separation where the
  classifier can distinguish them. Report neither as a mechanism identification.
- If a query returns summarized lock-in features of fixed precision, that is
  a unit-cost *feature-query* abstraction. It does not establish equal
  acquisition time at different physical modulation frequencies.

## frozen comparison and reporting

- Primary: fraction of **all** known-label test cases labelled correctly;
  abstentions, rejects and wrong known-label decisions do not count.
  Report the paired adaptive-minus-fixed difference with a stratified
  case bootstrap (10,000 resamples, seed 67449, percentile 95% interval).
- Secondary: known-label coverage and wrong-label fraction of all known cases,
  correct fraction conditional on commitment, and omitted-mechanism rejection,
  ambiguity, and false acceptance. Show confusion counts by truth label as
  well as chosen-design counts. A strong recognition result does not establish
  that the system diagnoses a previously unseen physical mechanism.
- An adaptive win requires the primary interval lower bound > 0; separately
  report whether omitted-mechanism false acceptance goes up. If the selectors
  choose the same design almost everywhere, compare the identical outcomes
  honestly instead of weakening the fixed baseline or forcing diversity.
- Do not change designs, thresholds, generators, or score definition after
  looking at locked test outcomes; any subsequent method is a new version
  requiring a fresh sealed test seed. Calibration and correctness tests
  may use private synthetic fixtures or the development split only.

## secondary fidelity ablation

A third unprivileged policy ranks the same tests using idealized instantaneous
cavity/detector response, then uses the same full-model classifier on the actual
bandwidth-limited measurement. Report how often the recommended test differs and
its paired outcome. The primary comparison still uses the full bandwidth model.

The initial two-per-label integration smoke test used seed 976103. To keep
those already exercised cases out of the locked set without altering any
physics or scoring, final locked seed 976104 was recorded before its first run.
Nondefault exploratory CLI invocations use seed roots offset by 1,000,000.
