"""Offline checks for the independent 10–11 November IFS/GFS rain comparison."""
import unittest
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import numpy as np

from compare_ifs_gfs_nov10_11 import (
    STARTS, END, allowed_steps, cycles_from_now, gfs_apcp_url,
    gfs_endpoint_steps, align_grid, nearest_cell,
)


class ForecastWindowTests(unittest.TestCase):
    def setUp(self):
        self.cycle = datetime(2026,11,2,12,tzinfo=timezone.utc)

    def test_ifs_steps_00_12_long_range(self):
        self.assertEqual(allowed_steps(self.cycle),(180,204,228))
        with self.assertRaises(ValueError):
            allowed_steps(datetime(2026,11,2,18,tzinfo=timezone.utc))

    def test_precisely_two_nonoverlapping_utc_days(self):
        self.assertEqual((STARTS[1]-STARTS[0]).total_seconds(),86400)
        self.assertEqual((END-STARTS[1]).total_seconds(),86400)
        self.assertEqual(STARTS[0].hour,0)
        self.assertEqual(END.hour,0)

    def test_gfs_steps_cover_full_48h_without_double_count(self):
        intervals=gfs_endpoint_steps(self.cycle)
        self.assertEqual(intervals[0],(180,186))
        self.assertEqual(intervals[-1],(222,228))
        self.assertEqual(len(intervals),8)
        for first, second in zip(intervals,intervals[1:]):
            self.assertEqual(first[1],second[0])
        self.assertEqual(sum(b-a for a,b in intervals),48)

    def test_3hour_apcp_near_event_date(self):
        c=datetime(2026,11,9,0,tzinfo=timezone.utc)
        steps=gfs_endpoint_steps(c)
        self.assertEqual(steps[0],(24,27))
        self.assertEqual(steps[-1],(69,72))
        self.assertEqual(len(steps),16)

    def test_gfs_real_nomads_url(self):
        u=urlsplit(gfs_apcp_url(self.cycle,186))
        p=parse_qs(u.query,keep_blank_values=True)
        self.assertEqual(u.hostname,"nomads.ncep.noaa.gov")
        self.assertEqual(p["file"],["gfs.t12z.pgrb2.0p25.f186"])
        self.assertEqual(p["var_APCP"],["on"])
        self.assertEqual(p["lev_surface"],["on"])

    def test_only_00_12_candidates(self):
        instant=datetime(2026,11,3,7,tzinfo=timezone.utc)
        options=list(cycles_from_now(instant))
        self.assertTrue(options)
        self.assertTrue(all(t.hour in (0,12) and t<=instant for t in options))

    def test_coordinate_sort_nearest_cell(self):
        lon=np.arange(54,42,-.25)
        lat=np.arange(33,24,-.25)
        sample=np.add.outer(lat,lon)
        x,y,values=align_grid(lon,lat,[sample,sample+2])
        self.assertGreater(np.diff(x).min(),0)
        self.assertGreater(np.diff(y).min(),0)
        self.assertEqual(len(values),2)
        self.assertAlmostEqual(nearest_cell(x,y,values[0],45.9666,28.4361),74.5,delta=0.4)

    def test_reject_inadequate_geography(self):
        with self.assertRaises(ValueError):
            align_grid(np.arange(10,12,.25),np.arange(10,12,.25),
                       [np.zeros((8,8))])


if __name__=="__main__":
    unittest.main()
