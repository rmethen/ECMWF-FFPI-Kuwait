"""Scientific checks for new (non-operational) FFPI diagnostics."""
import unittest
import numpy as np
from ffpi_diagnostics import classify_filter_reasons


class DiagnosticMaskTests(unittest.TestCase):
    def test_each_of_the_four_filter_outcomes(self):
        raw = np.array([[0.0, 12.0, 20.0, 60.0, 22.0]])
        passed = np.array([[False, True, False, True, True]])
        final = np.array([[0.0, 0.0, 0.0, 0.0, 22.0]])
        expected = np.array([[0, 1, 2, 3, 4]])
        np.testing.assert_array_equal(classify_filter_reasons(raw, passed, final), expected)

    def test_small_but_real_score_is_preserved_before_gate(self):
        raw = np.array([[4.0, 14.9, 15.0]])
        gate = np.array([[True, True, True]])
        published = np.array([[0.0, 0.0, 15.0]])
        np.testing.assert_array_equal(
            classify_filter_reasons(raw, gate, published),
            np.array([[1, 1, 4]]),
        )
        np.testing.assert_array_equal(raw, np.array([[4.0, 14.9, 15.0]]))

    def test_nan_does_not_become_a_risk_signal(self):
        self.assertEqual(
            classify_filter_reasons(
                np.array([[np.nan]]), np.array([[True]]),
                np.array([[0.0]]),
            )[0, 0], 0,
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            classify_filter_reasons(np.zeros((2, 2)), np.zeros((2, 2)),
                                    np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
