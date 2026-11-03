import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ffpi_cycle import is_newer_cycle, read_previous_cycle, utc_naive


class CycleFreshnessTests(unittest.TestCase):
    def test_initial_run_is_eligible(self):
        self.assertTrue(is_newer_cycle(datetime(2026, 11, 2, 18), None))

    def test_identical_and_older_runs_are_skipped(self):
        current = datetime(2026, 11, 2, 12)
        self.assertFalse(is_newer_cycle(current, current))
        self.assertFalse(is_newer_cycle(datetime(2026, 11, 2, 6), current))

    def test_later_cycle_is_accepted(self):
        self.assertTrue(is_newer_cycle(datetime(2026, 11, 2, 18), datetime(2026, 11, 2, 12)))
        self.assertTrue(is_newer_cycle(datetime(2026, 11, 3, 0), datetime(2026, 11, 2, 18)))

    def test_timezone_aware_cycle(self):
        self.assertEqual(utc_naive(datetime(2026, 11, 3, 3, tzinfo=timezone(timedelta(hours=3)))), datetime(2026, 11, 3, 0))

    def test_same_cycle_in_different_timezones_is_not_new(self):
        kuwait_time = datetime(2026, 11, 2, 21, tzinfo=timezone(timedelta(hours=3)))
        utc_time = datetime(2026, 11, 2, 18)
        self.assertFalse(is_newer_cycle(kuwait_time, utc_time))

    def test_read_marker_and_missing_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "last_run.txt"
            self.assertIsNone(read_previous_cycle(marker))
            marker.write_text("2026-11-02T18:00:00\n", encoding="utf-8")
            self.assertEqual(read_previous_cycle(marker), datetime(2026, 11, 2, 18))
            marker.write_text("not-a-date", encoding="utf-8")
            self.assertIsNone(read_previous_cycle(marker))


if __name__ == "__main__":
    unittest.main()
