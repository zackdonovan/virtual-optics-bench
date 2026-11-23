# two-query adequacy probe: locked follow-up

V1 found no benefit from case-conditioned one-query ranking: the full policy
always picked 300 Hz, tied the strongest development-selected fixed protocol,
and did not reject the omitted mechanism. The question here is different:
at the same two-query budget, does a second frequency detect a missing
mechanism better than a second copy of the first measurement?

- Compare fixed 300 + 1,000 Hz against fixed 300 + 300 Hz. Both have the same
  prior, static calibration, and 300 Hz first measurement; independent I/Q
  noise on the second measurement, keyed by seed, case, frequency, replicate.
- The reason for choosing 1,000 Hz before the run is that the omitted fast
  thermal pole can resemble Kerr at 300 Hz but is dynamically attenuated by
  1,000 Hz. The instrument itself also attenuates the signal. This is a
  follow-up informed by V1 and the synthetic generator family, not an
  untouched research hypothesis.
- Run 192 fresh cases per label with NumPy SeedSequence root 8921438, unchanged
  V1 truth family and physics model, 0.8 class probability and nominal 0.01
  rejection threshold with four real observation coordinates. Cases are new,
  but neither the mechanism family nor instrument is independent of V1.
- Primary: 300+1,000 minus 300+300 omitted-mechanism out-of-class rejection.
  Report paired case bootstrap percentile 95% interval (10,000 resamples,
  root 513780). Meaningful lift requires at least +10 percentage points and
  interval lower bound above zero. Unresolved does not count as a rejection.
- Guardrails: unconditional known-label correct rate (stratify bootstrap by
  known mechanism, lower bound must exceed −5 points); known false rejection
  (upper bound must be below +5 points). Report wrong decisions, false
  acceptances and unresolved separately, with both confusion matrices.
  No threshold or design changes after inspecting this run.
- These are equal-precision lock-in I/Q feature queries, not equal-duration
  or hardware tests. The comparison is against the specified replication
  control, not a claim of optimal fixed design or learned experiment selection.

Run: `python -m opticsbench.two_query --output results/two-query.json`.
