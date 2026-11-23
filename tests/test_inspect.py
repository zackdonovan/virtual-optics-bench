import json
import unittest
from opticsbench.__main__ import EXAMPLE, inspect


class InspectionTest(unittest.TestCase):
    def test_static_only_is_unresolved_and_serializable(self):
        result = inspect(EXAMPLE)
        self.assertEqual(result["diagnosis"], "unresolved")
        self.assertEqual(result["queries_paid"], 0)
        self.assertAlmostEqual(sum(result["model_probabilities"].values()), 1)
        self.assertEqual(len(result["candidate_designs"]["full"]), 6)
        json.dumps(result, allow_nan=False)

    def test_a_paid_frequency_is_not_recommended_twice(self):
        result = inspect(dict(EXAMPLE, observed_iq=[[300, 0.14, -0.04]]))
        self.assertEqual(result["queries_paid"], 1)
        self.assertNotIn(300, [p["frequency_hz"] for p in result["candidate_designs"]["full"]])
        self.assertIn("nominal_rejection_screen", result)
