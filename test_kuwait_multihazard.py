import unittest
import numpy as np
from kuwait_multihazard import (
    flood_guidance, lightning_environment, hail_environment,
    diagnostics, combined_guidance,
)


class MultiHazardTests(unittest.TestCase):
    def test_low_values(self):
        self.assertEqual(float(flood_guidance(0, 0, 0, 0)), 0)
        self.assertEqual(float(lightning_environment(0, 0, 0, 0)), 0)

    def test_strong_environment(self):
        self.assertGreater(float(lightning_environment(2000, -1, 95, 9)), 95)
        self.assertGreater(float(hail_environment(2000, 1800, 9, 30, -1)), 95)

    def test_missing_data_propagates(self):
        a = flood_guidance(np.array([5, np.nan]), 20, 3, 0.8)
        self.assertTrue(np.isnan(a[1]))
        self.assertTrue(np.isfinite(a[0]))

    def test_blank_plot_diagnostics(self):
        d = diagnostics(np.array([0, 4, 14, np.nan]))
        self.assertEqual(d["finite_cells"], 3)
        self.assertEqual(d["fraction_above_plot_floor"], 0)

    def test_combined_missing(self):
        v = combined_guidance(np.array([10, np.nan]), 20, 30)
        self.assertEqual(v[0], 30)
        self.assertTrue(np.isnan(v[1]))


if __name__ == "__main__":
    unittest.main()
