"""Inspect a calibration and rank candidate experiments without running them."""
import argparse
import json
from pathlib import Path

from .core import Bench, diagnose, fit_static, rank_frequencies


EXAMPLE = {
    "power": [0.1000, 0.2125, 0.3250, 0.4375, 0.5500, 0.6625, 0.7750, 0.8875, 1.0],
    "shift": [0.0640, 0.09375, 0.1325, 0.15925, 0.1970, 0.22775, 0.2665, 0.29325, 0.3310],
    "static_noise_sd": 0.012,
    "frequencies_hz": [10, 30, 100, 300, 1000, 3000],
    "observed_iq": [],
}


def inspect(calibration):
    bench = Bench(**calibration.get("bench", {}))
    fit = fit_static(calibration["power"], calibration["shift"],
                     calibration["static_noise_sd"], bench)
    observed = [(f, complex(i, q)) for f, i, q in calibration.get("observed_iq", [])]
    result = diagnose(fit, observed)
    report = {
        "static_gain": fit.gain_mean,
        "static_gain_sd": fit.gain_sd,
        "diagnosis": result.label,
        "model_probabilities": result.probabilities,
        "queries_paid": len(observed),
    }
    candidates = list(calibration.get("frequencies_hz", EXAMPLE["frequencies_hz"]))
    if set(candidates) - {f for f, _ in observed}:
        report["candidate_designs"] = {}
        for fidelity in ("full", "ideal"):
            ranks = rank_frequencies(fit, candidates, observed, fidelity=fidelity)
            report["candidate_designs"][fidelity] = [
                {"frequency_hz": frequency, "approx_information_nats": information}
                for frequency, information in ranks
            ]
    if observed:
        report["minimum_model_residual_chi2"] = result.min_chi2
        report["nominal_rejection_screen"] = result.rejection_threshold
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("calibration", type=Path, nargs="?",
                     help="JSON input; default is an illustrative static sweep")
    args = cli.parse_args()
    calibration = json.loads(args.calibration.read_text()) if args.calibration else EXAMPLE
    print(json.dumps(inspect(calibration), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
