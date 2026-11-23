"""Competing small-signal models and budgeted design for a virtual optics bench.

The policy only sees equilibrated calibration observations and purchased I/Q
measurements. It does not import the evaluator's truth model.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from scipy.stats import chi2


MODELS = ("kerr", "thermal", "mixed")
DEFAULT_FREQUENCIES = (10., 30., 100., 300., 1000., 3000.)
DEFAULT_MIN_PROBABILITY = .8
DEFAULT_REJECTION_ALPHA = .01
_MIXED_FRACTIONS = tuple(np.linspace(.1, .9, 17))
_GAIN_NODES, _GAIN_WEIGHTS = hermgauss(9)
_GAIN_WEIGHTS /= np.sqrt(np.pi)
_IQ_NODES1, _IQ_WEIGHTS1 = hermgauss(9)
_IQ_NODES = np.array(np.meshgrid(_IQ_NODES1, _IQ_NODES1)).reshape(2, -1).T
_IQ_WEIGHTS = np.outer(_IQ_WEIGHTS1, _IQ_WEIGHTS1).ravel() / np.pi


@dataclass(frozen=True)
class Bench:
    cavity_tau: float = .00004
    detector_tau: float = .00012
    thermal_tau_grid: tuple = (.0007, .0012, .002, .0035, .006, .01)
    noise_sd: float = .0075
    drive_amplitude: float = .2

    def __post_init__(self):
        vals = np.asarray([self.cavity_tau, self.detector_tau, self.noise_sd,
                           self.drive_amplitude, *self.thermal_tau_grid])
        if (not np.all(np.isfinite(vals)) or not self.thermal_tau_grid
                or min(self.cavity_tau, self.detector_tau) < 0
                or min(self.noise_sd, self.drive_amplitude, *self.thermal_tau_grid) <= 0):
            raise ValueError("invalid physical time constants, noise, or drive amplitude")


@dataclass(frozen=True)
class Fit:
    bench: Bench
    gain_mean: float
    gain_sd: float
    static_intercept: float
    gains: np.ndarray
    fractions: np.ndarray
    taus: np.ndarray
    model_index: np.ndarray
    log_weights: np.ndarray


@dataclass(frozen=True)
class Diagnosis:
    label: str
    probabilities: dict
    min_chi2: float
    rejection_threshold: float


def transfer_response(frequency, gain, thermal_fraction, thermal_tau,
                      bench=Bench(), fidelity="full"):
    """Peak complex amplitude, using Re(Y exp(i 2 pi f t)).

    Input is a sinusoidal incident-power perturbation. The linearized
    intracavity power and detector each behave as a first-order low-pass.
    Gain is total calibrated DC shift per incident-power unit. Thermal fraction
    and Kerr fraction partition this DC gain; the bounded inference menu assumes
    their signs agree. 'ideal' omits cavity/detector bandwidth only.
    """
    w = 2 * np.pi * np.asarray(frequency)
    h = (1. - thermal_fraction) + thermal_fraction / (1. + 1j*w*thermal_tau)
    if fidelity == "full":
        h = h / ((1. + 1j*w*bench.cavity_tau) * (1. + 1j*w*bench.detector_tau))
    elif fidelity != "ideal":
        raise ValueError("fidelity must be 'full' or 'ideal'")
    return bench.drive_amplitude * gain * h


@lru_cache(maxsize=64)
def _structural_prior(thermal_tau_grid):
    rows = [[] for _ in MODELS]
    rows[0].append((0., 0.))
    rows[1].extend((1., tau) for tau in thermal_tau_grid)
    rows[2].extend((q, tau) for q in _MIXED_FRACTIONS for tau in thermal_tau_grid)
    all_rows = []
    for k, group in enumerate(rows):
        for fraction, tau in group:
            for z, w in zip(_GAIN_NODES, _GAIN_WEIGHTS):
                all_rows.append((k, fraction, tau, z, math.log(w/(len(MODELS)*len(group)))))
    out = np.asarray(all_rows)
    return out


def _locked(array):
    array = np.array(array, copy=True)
    array.flags.writeable = False
    return array


def fit_static(powers, shifts, noise_sd, bench=Bench()):
    """Fit common gain/intercept from equilibrium data; mechanism priors stay equal.

    Assumes known per-point static noise standard deviation and uses the
    flat-coefficient prior for Gaussian linear regression. Nine Gauss–Hermite
    support points approximate its gain posterior; the common offset does not
    enter a sinusoidal response.
    """
    x, y = np.asarray(powers, dtype=float), np.asarray(shifts, dtype=float)
    if (x.ndim != 1 or x.shape != y.shape or x.size < 3 or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y)) or not np.isfinite(noise_sd) or noise_sd <= 0):
        raise ValueError("need 3+ finite, matching power/shift pairs and positive noise")
    centered = x - x.mean()
    ssx = centered @ centered
    if not ssx > 0:
        raise ValueError("cannot fit gain without power variation")
    gain = (centered @ y) / ssx
    gain_sd = noise_sd / np.sqrt(ssx)
    prior = _structural_prior(tuple(bench.thermal_tau_grid))
    ki, frac, tau, node, logw = prior.T
    return Fit(bench, float(gain), float(gain_sd), float(y.mean() - gain*x.mean()),
               _locked(gain + np.sqrt(2.)*gain_sd*node), _locked(frac), _locked(tau),
               _locked(ki.astype(int)), _locked(logw))


def _observation_array(observed):
    pairs = tuple(observed)
    if not pairs:
        return np.empty(0), np.empty(0, dtype=complex)
    frequencies = np.asarray([f for f, _ in pairs], dtype=float)
    y = np.asarray([y for _, y in pairs], dtype=complex)
    if (frequencies.ndim != 1 or y.ndim != 1 or not np.all(np.isfinite(frequencies))
            or not np.all(np.isfinite(y)) or np.any(frequencies <= 0)):
        raise ValueError("observations must be (positive Hz, finite scalar phasor) pairs")
    return frequencies, y


def _pred(fit, frequencies, fidelity="full"):
    return transfer_response(np.asarray(frequencies)[..., None], fit.gains, fit.fractions,
                             fit.taus, fit.bench, fidelity=fidelity)


def _posterior(fit, frequencies, y):
    pred = _pred(fit, frequencies)
    sq = np.sum(np.abs(pred - y[:, None])**2, axis=0) / fit.bench.noise_sd**2
    posterior = fit.log_weights - .5 * sq
    return np.exp(posterior - logsumexp(posterior)), sq


def diagnose(fit, observed, min_probability=DEFAULT_MIN_PROBABILITY,
             rejection_alpha=DEFAULT_REJECTION_ALPHA):
    """Choose three mechanisms or report unresolved/out_of_class.

    'out_of_class' means that none of this bounded predictive menu plus its
    declared noise explained the measurements, not discovery of a particular
    missing physical process. The fixed chi-squared screen is deliberately
    conservative for exact included parameter points and is only nominal for
    interpolated/perturbed truth. It is not a calibrated p-value.
    """
    if not 0 < rejection_alpha < 1 or not 1/len(MODELS) < min_probability < 1:
        raise ValueError("invalid diagnosis thresholds")
    frequencies, y = _observation_array(observed)
    weights, sq = _posterior(fit, frequencies, y)
    probs = np.bincount(fit.model_index, weights=weights, minlength=len(MODELS))
    threshold = float(chi2.ppf(1. - rejection_alpha, 2 * len(y))) if y.size else math.inf
    minimum = float(sq.min())
    best = int(probs.argmax())
    if minimum > threshold:
        label = "out_of_class"
    elif probs[best] < min_probability or not y.size:
        label = "unresolved"
    else:
        label = MODELS[best]
    return Diagnosis(label, dict(zip(MODELS, probs.tolist())), minimum, threshold)


def _model_information(pred, weights, model_index, noise_sd):
    """Approximate mutual information via moment-matched I/Q distributions.

    Keeping an entire class's predictive mean AND covariance penalizes
    experiments whose apparent signal is obscured by within-class uncertainty.
    Integrate the resulting Gaussian mixture by deterministic quadrature.
    """
    iq = np.column_stack((pred.real, pred.imag))
    means, covs, priors = [], [], []
    for k in range(len(MODELS)):
        take = model_index == k
        p = weights[take].sum()
        if p < 1e-12:
            continue
        w = weights[take] / p
        pts = iq[take]
        mu = w @ pts
        d = pts - mu
        means.append(mu)
        covs.append((d.T * w) @ d + noise_sd**2 * np.eye(2))
        priors.append(p)
    priors = np.asarray(priors)
    if priors.size <= 1:
        return 0.0
    priors /= priors.sum()
    means, covs = np.asarray(means), np.asarray(covs)
    chol = np.linalg.cholesky(covs)
    inv = np.linalg.inv(covs)
    norm = -np.log(2 * np.pi) - np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    value = 0.
    for j, p in enumerate(priors):
        ys = means[j] + np.sqrt(2.) * _IQ_NODES @ chol[j].T
        d = ys[:, None, :] - means[None, :, :]
        lps = norm - .5 * np.einsum("nki,kij,nkj->nk", d, inv, d)
        value += p * (_IQ_WEIGHTS @ (lps[:, j] - logsumexp(lps+np.log(priors), axis=1)))
    return float(np.clip(value, 0., -priors @ np.log(priors)))


def rank_frequencies(fit, candidates=DEFAULT_FREQUENCIES, observed=(), fidelity="full"):
    """Rank unused, equal-cost tests by estimated label-information gain."""
    frequencies, y = _observation_array(observed)
    candidate = tuple(sorted(set(float(f) for f in candidates) - set(frequencies)))
    if not candidate or not np.all(np.isfinite(candidate)) or min(candidate) <= 0.:
        raise ValueError("need at least one finite, positive, unused candidate")
    weights, _ = _posterior(fit, frequencies, y)
    preds = _pred(fit, candidate, fidelity=fidelity)
    return tuple(sorted(((f, _model_information(p, weights, fit.model_index, fit.bench.noise_sd))
                         for f, p in zip(candidate, preds)),
                        key=lambda item: (-item[1], item[0])))


def select_frequency(fit, candidates=DEFAULT_FREQUENCIES, observed=(), fidelity="full"):
    """Choose one unused frequency without access to its observation."""
    return rank_frequencies(fit, candidates, observed, fidelity=fidelity)[0][0]
