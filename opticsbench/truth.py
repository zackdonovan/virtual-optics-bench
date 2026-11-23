"""Independent, normalized, small-signal virtual bench used only by evaluator.

Propagates four real states with a periodic implicit trapezoid solver. Case
truth is not passed to fitted models or policies. This is distinct numerical
code, not independently observed physics or a full Maxwell/heat solver.
"""
from dataclasses import dataclass
import numpy as np

TRUTHS = ("kerr", "thermal", "mixed", "omitted")
CALIBRATION_POWERS = (0.0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.0)
CALIBRATION_NOISE = 0.010


@dataclass(frozen=True)
class TruthCase:
    label: str
    gain: float
    fast: float
    thermal_a: float
    thermal_b: float
    tau_a: float
    tau_b: float
    cavity_tau: float
    detector_tau: float
    offset: float

    @property
    def dc_coefficient(self):
        return self.fast + self.thermal_a + self.thermal_b


def make_case(label, rng, cavity_tau, detector_tau):
    """Sample positive DC shifts; both hidden thermal poles are unmodeled for omitted cases."""
    gain = float(np.exp(rng.uniform(np.log(0.4), np.log(1.35))))
    offset = float(rng.uniform(-.05, .05))
    tau_a = float(np.exp(rng.uniform(np.log(0.00075), np.log(.009))))
    tau_b = tau_a
    if label == "kerr":
        a, b, c = gain, 0.0, 0.0
    elif label == "thermal":
        a, b, c = 0.0, gain, 0.0
    elif label == "mixed":
        mix = rng.uniform(0.17, 0.83)
        a, b, c = gain * (1 - mix), gain * mix, 0.0
    elif label == "omitted":
        # Two physically positive heat modes with separated time constants.
        # No true instantaneous Kerr contribution. Both poles sum to same
        # DC shift as a simple single-pole or pure Kerr model.
        tau_a = float(np.exp(rng.uniform(np.log(.00016), np.log(.00038))))
        tau_b = float(np.exp(rng.uniform(np.log(.003), np.log(.009))))
        mix = rng.uniform(0.34, 0.66)
        a, b, c = 0.0, gain * mix, gain * (1 - mix)
    else:
        raise ValueError(f"unknown truth {label}")
    return TruthCase(label, gain, a, b, c, tau_a, tau_b, cavity_tau,
                     detector_tau, offset)


def static_response(case, rng, powers=CALIBRATION_POWERS, noise_sd=CALIBRATION_NOISE):
    p = np.asarray(powers, dtype=float)
    return p, case.offset + case.dc_coefficient * p + rng.normal(0.0, noise_sd, p.shape)


def response(case: TruthCase, frequency: float, amplitude: float, *, steps: int = 128) -> complex:
    """Steady-periodic detector lock-in; phasor convention Re(z exp(iwt))."""
    if frequency <= 0 or steps < 16:
        raise ValueError("frequency must be positive and steps >= 16")
    # States are cavity power modulation, thermal states A and B, detector.
    A = np.array([
        [-1/case.cavity_tau,           0,                          0,                           0],
        [1/case.tau_a,               -1/case.tau_a,                0,                           0],
        [1/case.tau_b,                 0,                         -1/case.tau_b,                0],
        [case.fast/case.detector_tau, case.thermal_a/case.detector_tau, case.thermal_b/case.detector_tau,
         -1/case.detector_tau],
    ])
    B = np.array([amplitude/case.cavity_tau, 0, 0, 0])
    dt = 1 / (steps * frequency)
    I = np.eye(4)
    # Implicit trapezoid on x' = Ax + B cos(wt).
    left = I - dt/2*A
    M = np.linalg.solve(left, I + dt/2*A)
    v = np.linalg.solve(left, dt/2*B)
    theta = 2*np.pi*np.arange(steps + 1)/steps
    C, S = np.cos(theta[1]), np.sin(theta[1])
    # Solve periodic initial conditions instead of simulating hundreds of
    # burn-in cycles when the hidden thermal and modulation times differ.
    D = C*I-M
    initial = np.linalg.solve(np.block([[D, S*I], [-S*I, D]]),
                              np.r_[v*(1+C), -v*S])[:4]
    x = initial.copy()
    cos = np.cos(theta)
    output = np.empty(steps)
    for i in range(steps):
        output[i] = x[-1]
        x = M @ x + v * (cos[i] + cos[i+1])
    if not np.allclose(x, initial, rtol=1e-9, atol=1e-11):
        raise ArithmeticError("periodic state did not close")
    return complex((2/steps) * np.dot(output, np.exp(-1j*theta[:-1])))
