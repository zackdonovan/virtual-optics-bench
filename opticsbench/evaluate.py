"""Locked unit-cost feature-query benchmark. No instrument/hardware claims."""
from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .core import (
    Bench, MODELS, DEFAULT_FREQUENCIES, DEFAULT_MIN_PROBABILITY,
    DEFAULT_REJECTION_ALPHA, diagnose, fit_static, select_frequency,
)
from .truth import (
    TRUTHS, CALIBRATION_NOISE, CALIBRATION_POWERS,
    make_case, response, static_response,
)

DEV_ROOT = 771410
TEST_ROOT = 976104
BOOTSTRAP_ROOT = 67449
DEV_N = 72
TEST_N = 192
N_BOOT = 10_000
DECISIONS = (*MODELS, "unresolved", "out_of_class")
METHODS = ("adaptive_full", "fixed_dev_selected", "adaptive_ideal_rank")


def _rng(*key):
    return np.random.default_rng(np.random.SeedSequence(key))


def _indices(root, count):
    keys = [(label, j, k) for j, label in enumerate(TRUTHS) for k in range(count)]
    return [keys[t] for t in _rng(root, 889).permutation(len(keys))]


def _case_and_fit(root, j, k, label, bench):
    case = make_case(label, _rng(root, j, k, 0), bench.cavity_tau, bench.detector_tau)
    x, y = static_response(case, _rng(root, j, k, 1))
    fit = fit_static(x, y, CALIBRATION_NOISE, bench)
    return case, fit


def _query(case, root, j, k, f, bench):
    # Domain separation by design gives both policies exactly the same
    # measured feature if they select the same protocol on the same case.
    real, imag = _rng(root, j, k, 2, int(f)).normal(0., bench.noise_sd, 2)
    return response(case, f, bench.drive_amplitude) + complex(real, imag)


def _at_frequency(case, fit, root, j, k, f):
    return diagnose(fit, [(f, _query(case, root, j, k, f, fit.bench))])


def _rate(x, n):
    return None if not n else float(x / n)


def _summarize(true, pred, frequencies):
    conf = {t: {p: 0 for p in DECISIONS} for t in TRUTHS}
    for t, p in zip(true, pred, strict=True):
        conf[t][p] += 1
    known = [t in MODELS for t in true]
    committed = [p in MODELS for p in pred]
    n = sum(known)
    m = sum(not x for x in known)
    both = sum(k and d for k, d in zip(known, committed))
    correct = sum(k and t == p for k, t, p in zip(known, true, pred))
    u = sum(conf[t]["unresolved"] for t in MODELS)
    r = sum(conf[t]["out_of_class"] for t in MODELS)
    assert n == both + u + r
    out = conf["omitted"]
    return {
        "in_menu_n": n,
        "out_of_menu_n": m,
        "in_menu_correct_count": correct,
        "in_menu_correct_fraction_all": _rate(correct, n),
        "in_menu_commit_fraction": _rate(both, n),
        "in_menu_correct_given_commitment": _rate(correct, both),
        "in_menu_wrong_label_fraction_all": _rate(both-correct, n),
        "in_menu_unresolved_fraction": _rate(u, n),
        "in_menu_false_reject_fraction": _rate(r, n),
        "omitted_false_acceptance_fraction": _rate(sum(out[t] for t in MODELS), m),
        "omitted_unresolved_fraction": _rate(out["unresolved"], m),
        "omitted_out_of_class_fraction": _rate(out["out_of_class"], m),
        "confusion_counts": conf,
        "selected_frequencies_hz": {str(k): v for k, v in sorted(Counter(frequencies).items())},
    }


def _paired_diff(truth, a, b, metric):
    if metric == "known_correct":
        labels = MODELS
        scores = {l: np.array([int(x == t)-int(y == t)
                              for t,x,y in zip(truth,a,b) if t == l]) for l in labels}
    elif metric == "omitted_false_accept":
        labels = ("omitted",)
        scores = {l: np.array([int(x in MODELS)-int(y in MODELS)
                               for t,x,y in zip(truth,a,b) if t == l]) for l in labels}
    else:
        raise ValueError(metric)
    rng = _rng(BOOTSTRAP_ROOT)
    counts = np.zeros(N_BOOT)
    n = sum(len(scores[l]) for l in labels)
    for l in labels:
        s = scores[l]
        counts += s[rng.integers(len(s), size=(N_BOOT, len(s)))].sum(axis=1)
    all_scores = np.concatenate([scores[l] for l in labels])
    return {
        "difference": float(all_scores.mean()),
        "stratified_paired_bootstrap_95": (np.quantile(counts/n, [.025,.975])).tolist(),
        "n": n,
        "a_only": int((all_scores == 1).sum()),
        "b_only": int((all_scores == -1).sum()),
    }


def _fixed_design(bench, dev_per_label, candidates, root):
    # Only development truth labels are unsealed for baseline selection.
    successes = {f: 0 for f in candidates}
    for label, j, k in _indices(root, dev_per_label):
        if label not in MODELS:
            continue
        case, fit = _case_and_fit(root, j, k, label, bench)
        for f in candidates:
            successes[f] += _at_frequency(case, fit, root, j, k, f).label == label
    fixed = min(candidates, key=lambda f: (-successes[f], f))
    return fixed, {str(f): successes[f] for f in candidates}


def run(dev_per_label=DEV_N, test_per_label=TEST_N):
    if min(dev_per_label, test_per_label) < 1:
        raise ValueError("positive case counts required")
    bench = Bench()
    candidates = DEFAULT_FREQUENCIES
    locked = (dev_per_label, test_per_label) == (DEV_N, TEST_N)
    offset = 0 if locked else 1_000_000
    dev_root, test_root = DEV_ROOT + offset, TEST_ROOT + offset
    fixed, dev_scores = _fixed_design(bench, dev_per_label, candidates, dev_root)
    rows = []
    for label, j, k in _indices(test_root, test_per_label):
        case, fit = _case_and_fit(test_root, j, k, label, bench)
        # Freeze every design for this case before any dynamic observation.
        frequencies = (
            select_frequency(fit, candidates, fidelity="full"),
            fixed,
            select_frequency(fit, candidates, fidelity="ideal"),
        )
        decisions = {f: _at_frequency(case, fit, test_root, j, k, f)
                     for f in sorted(set(frequencies))}
        rows.append({
            "truth": label, "case_within_label": k,
            "static_gain_fit": fit.gain_mean,
            "design_hz": dict(zip(METHODS, frequencies)),
            "predictions": {m: decisions[f].label for m,f in zip(METHODS, frequencies)},
        })

    labels = [row["truth"] for row in rows]
    pred = {m: [row["predictions"][m] for row in rows] for m in METHODS}
    summary = {m: _summarize(labels, pred[m], [row["design_hz"][m] for row in rows])
               for m in METHODS}
    pair = {f"full_minus_{other}":
             {metric: _paired_diff(labels, pred["adaptive_full"], pred[other], metric)
              for metric in ("known_correct","omitted_false_accept")}
            for other in METHODS[1:]}
    return {
        "protocol": "unit-cost-lock-in-feature-v1",
        "locked_default_counts": locked,
        "dev_seed_root": dev_root,
        "test_seed_root": test_root,
        "bootstrap_seed": BOOTSTRAP_ROOT,
        "bootstrap_replicates": N_BOOT,
        "dev_per_known_label": dev_per_label,
        "test_per_truth_label": test_per_label,
        "query_cost_per_case": 1,
        "calibration_powers": CALIBRATION_POWERS,
        "calibration_noise_sd": CALIBRATION_NOISE,
        "instrument_and_inference_parameters": asdict(bench),
        "inference_thresholds": {"min_probability": DEFAULT_MIN_PROBABILITY,
                                 "nominal_rejection_alpha": DEFAULT_REJECTION_ALPHA},
        "candidate_frequencies_hz": candidates,
        "fixed_choice_hz": fixed,
        "fixed_choice_dev_correct_count": dev_scores,
        "summaries": summary,
        "paired_comparisons": pair,
        "fidelity_rank_changed_count": sum(row["design_hz"]["adaptive_full"] !=
                                            row["design_hz"]["adaptive_ideal_rank"] for row in rows),
        "adaptive_differs_from_fixed_count": sum(row["design_hz"]["adaptive_full"] != fixed
                                                 for row in rows),
        "cases": rows,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--dev-per-label", type=int, default=DEV_N)
    parser.add_argument("--test-per-label", type=int, default=TEST_N)
    args = parser.parse_args()
    report = run(args.dev_per_label, args.test_per_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Wrote {args.output}; fixed protocol {report['fixed_choice_hz']:g} Hz")
    for name, metrics in report["summaries"].items():
        print(f"{name:24s} in-menu {metrics['in_menu_correct_count']}/{metrics['in_menu_n']}"
              f"; omitted false acceptance {metrics['omitted_false_acceptance_fraction']:.1%}")


if __name__ == "__main__":
    main()
