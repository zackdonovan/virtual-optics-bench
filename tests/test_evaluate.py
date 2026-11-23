import numpy as np
import unittest

from opticsbench.core import Bench, MODELS, transfer_response
from opticsbench.evaluate import DEV_ROOT, TEST_ROOT, _rng, _query, run
from opticsbench.truth import TRUTHS, make_case, response, static_response


def _integrator_refines(label, f):
    bench = Bench()
    case = make_case(label, _rng(81), bench.cavity_tau, bench.detector_tau)
    steps = (128, 256, 512, 1024)
    y = [response(case, f, bench.drive_amplitude, steps=n) for n in steps]
    error = [abs(x-y[-1]) for x in y[:-1]]
    assert error[0] < bench.noise_sd / 70
    assert error[0] / error[1] > 3.5
    assert error[1] / error[2] > 3.5
    if label in MODELS:
        exact = transfer_response(f, case.gain, case.thermal_a / case.gain, case.tau_a, bench)
        assert abs(y[-1] - exact) < bench.noise_sd / 4000


def _equilibrium_has_no_structural_label_signal():
    b = Bench()
    traces = [static_response(make_case(label, _rng(81), b.cavity_tau, b.detector_tau), _rng(31))[1]
              for label in TRUTHS]
    for trace in traces[1:]:
        np.testing.assert_allclose(traces[0], trace, atol=1e-15, rtol=0)


def _measurement_noise_same_if_design_same():
    b = Bench()
    c = make_case('mixed', _rng(2), b.cavity_tau, b.detector_tau)
    assert _query(c, 10, 0, 0, 300., b) == _query(c, 10, 0, 0, 300., b)
    assert _query(c, 10, 0, 0, 300., b) != _query(c, 11, 0, 0, 300., b)


def _small_runner_deterministic_and_disjoint_from_locked():
    report = run(3, 2)
    assert report == run(3, 2)
    assert not report['locked_default_counts']
    assert report['dev_seed_root'] != DEV_ROOT
    assert report['test_seed_root'] != TEST_ROOT
    assert len(report['cases']) == 8
    assert report['query_cost_per_case'] == 1
    for m in report['summaries'].values():
        assert m['in_menu_n'] == 6
        assert m['out_of_menu_n'] == 2
        assert sum(m['selected_frequencies_hz'].values()) == 8
        assert sum(m['confusion_counts']['omitted'].values()) == 2
        assert np.isclose(m['omitted_false_acceptance_fraction'] +
                          m['omitted_unresolved_fraction'] +
                          m['omitted_out_of_class_fraction'], 1)


class TestEvaluation(unittest.TestCase):
    def test_periodic_refinement(self):
        for label in TRUTHS:
            for f in (10.0, 300.0, 3000.0):
                with self.subTest(label=label, frequency=f):
                    _integrator_refines(label, f)

    def test_equilibrium(self):
        _equilibrium_has_no_structural_label_signal()

    def test_measurement_noise(self):
        _measurement_noise_same_if_design_same()

    def test_small_runner(self):
        _small_runner_deterministic_and_disjoint_from_locked()
