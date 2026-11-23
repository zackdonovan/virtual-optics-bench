"""Locked two-query quality check; see TWO-QUERY.md."""
from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .core import (Bench, MODELS, DEFAULT_MIN_PROBABILITY,
                   DEFAULT_REJECTION_ALPHA, diagnose, fit_static)
from .evaluate import _summarize
from .truth import TRUTHS, CALIBRATION_NOISE, make_case, response, static_response

CASE_SEED = 8921438
BOOT_SEED = 513780
PER_LABEL = 192
REPLICATES = 10_000
DESIGNS = {"replicate_300": (300, 300), "add_1000": (300, 1000)}


def _rng(seed, *key):
    return np.random.default_rng(np.random.SeedSequence([seed, *key]))


def query(case, bench, label_i, ix, freq, replicate):
    rng = _rng(CASE_SEED, label_i, ix, 7, freq, replicate)
    noise = complex(*rng.normal(0., bench.noise_sd, 2))
    return response(case, freq, bench.drive_amplitude)+noise


def summarize_pair(rows, labels, scoring, key):
    rng = _rng(BOOT_SEED, key)
    observed, sample_sums = [], np.zeros(REPLICATES, dtype=int)
    for label in labels:
        deltas = np.array([int(scoring(r["add_1000"], label)) -
                           int(scoring(r["replicate_300"], label))
                           for r in rows if r["truth"] == label])
        observed.append(deltas)
        sample_sums += deltas[rng.integers(len(deltas), size=(REPLICATES,len(deltas)))].sum(axis=1)
    d = np.concatenate(observed)
    return {"difference": float(d.mean()), "paired_bootstrap_95": np.quantile(
        sample_sums / len(d), [.025,.975]).tolist(), "n": len(d),
        "add_only": int(np.sum(d>0)), "replicate_only": int(np.sum(d<0))}


def run():
    bench = Bench()
    rows = []
    for label_i, label in enumerate(TRUTHS):
        for ix in range(PER_LABEL):
            c = make_case(label, _rng(CASE_SEED,label_i,ix,0),
                          bench.cavity_tau,bench.detector_tau)
            x, y = static_response(c, _rng(CASE_SEED,label_i,ix,1))
            fit = fit_static(x, y, CALIBRATION_NOISE, bench)
            first = query(c, bench, label_i, ix, 300, 0)
            second_300 = query(c, bench, label_i, ix, 300, 1)
            second_1000 = query(c, bench, label_i, ix, 1000, 1)
            choices = {"replicate_300": (300, second_300), "add_1000": (1000, second_1000)}
            rows.append({"truth": label, "case_within_label": ix,
                         **{m: diagnose(fit, [(300, first), (f, z)]).label
                            for m, (f,z) in choices.items()}})
    ts = [r["truth"] for r in rows]
    metrics = {m: _summarize(ts, [r[m] for r in rows], [design[1]] * len(rows))
               for m, design in DESIGNS.items()}
    for s in metrics.values():
        s["second_frequencies_hz"] = s.pop("selected_frequencies_hz")
    comp = {
        "omitted_reject": summarize_pair(rows, ("omitted",), lambda x,t:x=="out_of_class", 0),
        "known_correct": summarize_pair(rows, MODELS, lambda x,t:x==t, 1),
        "known_false_reject": summarize_pair(rows, MODELS, lambda x,t:x=="out_of_class", 2),
        "omitted_false_accept": summarize_pair(rows, ("omitted",), lambda x,t:x in MODELS, 3),
    }
    reject, good, false = (comp[k] for k in ("omitted_reject", "known_correct", "known_false_reject"))
    passes = {
        "meaningful_omitted_reject_gain": reject["difference"] >= .10 and reject["paired_bootstrap_95"][0] > 0,
        "known_accuracy_guardrail": good["paired_bootstrap_95"][0] > -.05,
        "false_rejection_guardrail": false["paired_bootstrap_95"][1] < .05,
    }
    return {
        "protocol": "two-query-adequacy-probe-v2",
        "fresh_seed": CASE_SEED, "cases_per_label": PER_LABEL,
        "bootstrap_seed": BOOT_SEED, "bootstrap_replicates": REPLICATES,
        "bench": asdict(bench),
        "thresholds": {"min_probability": DEFAULT_MIN_PROBABILITY,
                       "nominal_rejection_alpha": DEFAULT_REJECTION_ALPHA},
        "design_hz": DESIGNS, "query_cost_per_case": 2,
        "summaries": metrics, "paired_comparisons": comp,
        "passes": passes, "all_criteria_met": all(passes.values()), "cases": rows,
    }


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/two-query.json"))
    args = p.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("summaries", "paired_comparisons", "passes")}, indent=2))


if __name__ == "__main__":
    main()
