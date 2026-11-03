"""Offline guards for rolling IFS/GFS 10-day comparison."""
import unittest
import json
from datetime import datetime,timezone,timedelta
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rolling_ifs_gfs_rain import (
    DAYS,MAX_LEAD,PREFIX,valid_cycle,candidate_cycles,endpoints,
    gfs_intervals,same_window_details,build_status,main,
)


class RollingRainTests(unittest.TestCase):
    def setUp(self):
        self.cycle=datetime(2026,11,3,0,tzinfo=timezone.utc)

    def test_exact_10_days(self):
        self.assertEqual(DAYS,10)
        self.assertEqual(MAX_LEAD,240)
        self.assertEqual(endpoints(),tuple(range(0,241,24)))
        days=same_window_details(self.cycle)
        self.assertEqual(len(days),10)
        self.assertEqual(days[0]["start_kuwait"],"2026-11-03 03:00")
        self.assertEqual(days[-1]["end_kuwait"],"2026-11-13 03:00")
        self.assertEqual(days[5]["image"],f"{PREFIX}_D06.png")

    def test_12z_local_day_is_15_to_15_not_midnight(self):
        cycle=datetime(2026,11,3,12,tzinfo=timezone.utc)
        d=same_window_details(cycle)
        self.assertEqual(d[0]["start_kuwait"],"2026-11-03 15:00")
        self.assertEqual(d[0]["end_kuwait"],"2026-11-04 15:00")

    def test_rolling_endpoints_ifs_are_available_on_00_12z(self):
        self.assertEqual(endpoints()[-1],240)
        for cycle in (self.cycle,datetime(2026,11,3,12,tzinfo=timezone.utc)):
            self.assertEqual(valid_cycle(cycle),cycle)
        for h in endpoints()[1:]:
            self.assertTrue(h<=144 or h%6==0)
        with self.assertRaises(ValueError):
            valid_cycle(datetime(2026,11,3,6,tzinfo=timezone.utc))

    def test_exact_non_overlapping_gfs_windows(self):
        pieces=gfs_intervals()
        self.assertEqual(pieces[0],(0,3))
        self.assertEqual(pieces[39],(117,120))
        self.assertEqual(pieces[40],(120,126))
        self.assertEqual(pieces[-1],(234,240))
        self.assertEqual(len(pieces),60)
        self.assertEqual(sum(b-a for a,b in pieces),240)
        self.assertTrue(all(a//24==(b-1)//24 for a,b in pieces))
        self.assertTrue(all(x[1]==y[0] for x,y in zip(pieces,pieces[1:])))

    def test_latest_candidates_00_12_only(self):
        a=candidate_cycles(datetime(2026,11,3,7,tzinfo=timezone.utc))
        self.assertEqual(a[0],self.cycle)
        self.assertEqual(a[1],datetime(2026,11,2,12,tzinfo=timezone.utc))
        self.assertEqual(len(a),3)

    def test_status_uses_actual_cycle_and_city_totals(self):
        lon=np.arange(42,55,.25)
        lat=np.arange(24,34,.25)
        base=np.ones((len(lat),len(lon)),float)
        data=(lon,lat,[base*i for i in range(1,11)])
        status=build_status(self.cycle,self.cycle+timedelta(hours=3),data,data)
        self.assertEqual(len(status["daily_windows"]),10)
        self.assertEqual(status["cycle_utc"],"2026-11-03T00:00:00Z")
        self.assertEqual(status["cities"]["Al Qaysumah"]["ECMWF"]["daily_mm"][0],1.0)
        self.assertEqual(status["cities"]["Kuwait City"]["GFS"]["total_10d_mm"],55.0)
        self.assertEqual(status["model_count"],2)

    def test_no_newer_cycle_leaves_existing_status_and_images(self):
        with TemporaryDirectory() as d:
            folder=Path(d)
            content={"cycle_utc":"2026-11-03T00:00:00Z","cities":{}}
            (folder/f"{PREFIX}_STATUS.json").write_text(json.dumps(content))
            existing=folder/f"{PREFIX}_TOTAL.png"
            existing.write_bytes(b"existing")
            with patch("rolling_ifs_gfs_rain.urlopen") as fetch:
                result=main(now=self.cycle,outdir=folder)
            self.assertFalse(result)
            fetch.assert_not_called()
            self.assertEqual(existing.read_bytes(),b"existing")

    def test_no_data_does_not_fabricate_images(self):
        with TemporaryDirectory() as d:
            with patch("rolling_ifs_gfs_rain.urlopen",side_effect=OSError("No NOAA feed")):
                with self.assertRaises(RuntimeError):
                    main(now=self.cycle,outdir=Path(d))
            self.assertFalse(list(Path(d).iterdir()))

    def test_site_has_dynamic_controls_and_no_fake_values(self):
        html=Path("index.html").read_text(encoding="utf-8")
        workflow=Path(".github/workflows/ifs_gfs_rolling_10day.yml").read_text()
        self.assertIn(f"{PREFIX}_STATUS.json",html)
        self.assertIn("rolling-rain-image",html)
        self.assertIn("rolling-rain-days",html)
        self.assertIn("لا تتوفر",html)
        self.assertIn("rolling_ifs_gfs_rain.py",workflow)
        self.assertIn(f"{PREFIX}_D*.png",workflow)
        self.assertIn(f"{PREFIX}_TOTAL.png",workflow)


if __name__=="__main__":
    unittest.main()
