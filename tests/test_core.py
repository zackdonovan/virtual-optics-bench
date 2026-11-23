import math
import unittest

import numpy as np

from opticsbench.core import (
    DEFAULT_FREQUENCIES,
    MODELS,
    Bench,
    diagnose,
    fit_static,
    rank_frequencies,
    select_frequency,
    transfer_response,
)


def static_fit(gain=1., bench=Bench()):
    power = np.linspace(.1, 1, 9)
    return fit_static(power, .031 + gain*power, .012, bench)


class CoreTests(unittest.TestCase):
    def test_static_identical_dc_but_gain_and_offset_recovered(self):
        fit = static_fit(1.1)
        self.assertAlmostEqual(fit.gain_mean, 1.1)
        self.assertAlmostEqual(fit.static_intercept, .031)
        self.assertGreater(fit.gain_sd, 0.)
        result = diagnose(fit, ())
        self.assertEqual(result.label, "unresolved")
        np.testing.assert_allclose(list(result.probabilities.values()), (1./3.,)*3)
        for q in (0., .5, 1.):
            self.assertAlmostEqual(transfer_response(0., 1.1, q, .002), .22)

    def test_linear_thermal_and_shared_poles(self):
        bench = Bench()
        f = 1/(2*math.pi*.002)
        kerr, thermal, mixed = [transfer_response(f, 1., q, .002, bench)
                                for q in (0., 1., .35)]
        self.assertAlmostEqual(thermal/kerr, (1.-1j)/2.)
        self.assertAlmostEqual(mixed, .65*kerr+.35*thermal)
        self.assertLess(abs(transfer_response(1.e8, 1., 0., .002)), 1.e-8)
        self.assertLess(kerr.imag, 0.)

    def test_fit_recovers_each_supported_mechanism(self):
        fit = static_fit()
        for expected, q in zip(MODELS, (0., 1., .5)):
            observed = [(f, transfer_response(f, 1., q, .002))
                        for f in (100., 300., 1000.)]
            result = diagnose(fit, observed)
            self.assertEqual(result.label, expected)
            self.assertAlmostEqual(sum(result.probabilities.values()), 1.)
            self.assertLess(result.min_chi2, 1.e-12)

    def test_quality_check_rejects_obvious_missing_mechanism(self):
        result = diagnose(static_fit(), ((300., complex(.25, .12)),
                                        (1000., complex(-.21, .04))))
        self.assertEqual(result.label, "out_of_class")
        self.assertGreater(result.min_chi2, result.rejection_threshold)

    def test_adaptive_choice_has_no_repeats_and_is_deterministic(self):
        fit = static_fit()
        first = select_frequency(fit)
        self.assertEqual(first, select_frequency(fit))
        self.assertIn(first, DEFAULT_FREQUENCIES)
        observed = [(first, transfer_response(first, 1., .35, .002))]
        order = rank_frequencies(fit, observed=observed)
        self.assertEqual(len(order), len(DEFAULT_FREQUENCIES)-1)
        self.assertNotIn(first, dict(order))
        self.assertTrue(all(0 <= score <= math.log(3) for _, score in order))
        self.assertGreaterEqual(order[0][1], order[-1][1])

    def test_instrument_bandwidth_can_change_selected_frequency(self):
        fit = static_fit(.3)
        self.assertEqual(select_frequency(fit, fidelity="full"), 300.)
        self.assertEqual(select_frequency(fit, fidelity="ideal"), 3000.)
        attenuated = abs(transfer_response(3000., 1., 0., .002, fidelity="full"))
        ideal = abs(transfer_response(3000., 1., 0., .002, fidelity="ideal"))
        self.assertLess(attenuated/ideal, .35)

    def test_reject_malformed_inputs(self):
        with self.assertRaises(ValueError):
            Bench(noise_sd=0.)
        with self.assertRaises(ValueError):
            Bench(thermal_tau_grid=())
        with self.assertRaises(ValueError):
            fit_static([1, 1, 1], [1, 2, 3], .01)
        with self.assertRaises(ValueError):
            fit_static([0, 1, 2], [1, 2, float('nan')], .01)
        with self.assertRaises(ValueError):
            diagnose(static_fit(), ((-100., 0j),))
        with self.assertRaises(ValueError):
            select_frequency(static_fit(), candidates=())
        with self.assertRaises(ValueError):
            select_frequency(static_fit(), fidelity="wrong")


if __name__ == "__main__":
    unittest.main()
