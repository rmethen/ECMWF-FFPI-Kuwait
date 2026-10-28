"""GFS-only experimental flood potential, using the ECMWF FFPI weights."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryFile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import eccodes
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label


OUT = Path("output")
OUT.mkdir(exist_ok=True)
BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
HOURS = range(6, 73, 6)
BOX = (20, 80, 5, 45)  # west, east, south, north


def url(cycle, hour):
    params = {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}",
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
        "var_APCP": "on", "var_PRATE": "on", "var_WATR": "on",
        "var_SOILW": "on", "lev_surface": "on",
        "lev_0-0.1_m_below_ground": "on", "subregion": "",
        "leftlon": BOX[0], "rightlon": BOX[1],
        "bottomlat": BOX[2], "toplat": BOX[3],
    }
    return BASE + "?" + urlencode(params)


def choose_cycle():
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0,
                         microsecond=0)
    for n in range(5):
        cycle = anchor - timedelta(hours=6 * n)
        try:
            data = download(cycle, 72)
            read_fields(data, 72)  # require a complete forecast endpoint
            return cycle
        except (OSError, ValueError) as exc:
            print(f"GFS {cycle:%Y-%m-%d %HZ} incomplete: {exc}")
    raise RuntimeError("No GFS cycle has the 72-hour endpoint")


def download(cycle, hour):
    with urlopen(Request(url(cycle, hour),
                         headers={"User-Agent": "ECMWF-FFPI-Kuwait/1.0"}),
                 timeout=60) as response:
        data = response.read()
    if not data.startswith(b"GRIB"):
        raise ValueError(f"No GRIB at forecast hour {hour}")
    return data


def read_fields(data, hour):
    """Select instantaneous rate and non-overlapping six-hour totals."""
    found = {}
    lats = lons = None
    with TemporaryFile() as stream:
        stream.write(data)
        stream.seek(0)
        while message := eccodes.codes_grib_new_from_file(stream):
            try:
                name = eccodes.codes_get(message, "shortName")
                step_type = eccodes.codes_get(message, "stepType")
                step_range = eccodes.codes_get(message, "stepRange")
                if name in ("tp", "watr"):
                    bounds = step_range.split("-")
                    if step_type != "accum" or len(bounds) != 2 or \
                            int(bounds[1]) != hour or int(bounds[0]) != hour - 6:
                        continue
                elif name == "prate":
                    if step_type != "instant":
                        continue
                elif name == "soilw":
                    if eccodes.codes_get(message, "typeOfLevel") != "depthBelowLandLayer" or \
                            eccodes.codes_get(message, "level") != 0:
                        continue
                else:
                    continue
                if name not in found:
                    ny = eccodes.codes_get(message, "Nj")
                    nx = eccodes.codes_get(message, "Ni")
                    values = eccodes.codes_get_array(message, "values").reshape(ny, nx)
                    missing_value = eccodes.codes_get(message, "missingValue")
                    found[name] = np.where(np.isclose(values, missing_value), np.nan, values)
                    lats = eccodes.codes_get_array(message, "distinctLatitudes")
                    lons = eccodes.codes_get_array(message, "distinctLongitudes")
            finally:
                eccodes.codes_release(message)
    missing = {"tp", "watr", "soilw", "prate"} - found.keys()
    if missing:
        raise ValueError(f"GFS f{hour:03d} missing {sorted(missing)}")
    return lons, lats, found


def scale(x, low, high):
    return np.clip((x - low) / (high - low), 0, 1)


def calculate(cycle, hours=HOURS):
    rain = runoff = peak_rate = soil = None
    peak_hour = None
    for hour in hours:
        lons, lats, f = read_fields(download(cycle, hour), hour)
        if rain is None:
            rain = np.zeros_like(f["tp"])
            runoff = np.zeros_like(rain)
            peak_rate = np.zeros_like(rain)
            soil = np.zeros_like(rain)
        # APCP and runoff are interval totals in kg/m², equivalent to mm.
        rain += np.maximum(f["tp"], 0)
        runoff += np.maximum(f["watr"], 0)
        rate = f["prate"] * 3600  # kg/m²/s to mm/h
        if peak_hour is None or np.nanmax(rate) > np.nanmax(peak_rate):
            peak_hour = hour
        peak_rate = np.maximum(peak_rate, rate)
        soil = np.maximum(soil, f["soilw"])  # volumetric fraction
        print(f"GFS +{hour}h loaded")
    ffpi = 100 * (0.40 * scale(peak_rate, 3, 25)
                  + 0.30 * scale(rain, 10, 60)
                  + 0.20 * scale(runoff, 1, 20)
                  + 0.10 * scale(soil, .15, .40))
    ffpi = np.where((rain >= 10) | (peak_rate >= 5), ffpi, 0)
    ffpi = np.where(ffpi >= 15, ffpi, 0)
    regions, count = label(ffpi >= 15, structure=np.ones((3, 3)))
    for n in range(1, count + 1):
        if np.sum(regions == n) < 3:
            ffpi[regions == n] = 0
    return lons, lats, ffpi, peak_hour


def plot(lons, lats, field, cycle, peak_hour, box, path, name):
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(box, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=.7)
    ax.add_feature(cfeature.BORDERS, linewidth=.6)
    mesh = ax.contourf(lons, lats, np.ma.masked_less(field, 15),
                       levels=[15, 25, 35, 50, 70, 85, 100],
                       cmap="turbo", extend="max", transform=ccrs.PlateCarree())
    fig.colorbar(mesh, ax=ax, label="Experimental GFS FFPI (0–100)", shrink=.8)
    ax.set_title(f"GFS Experimental Flash Flood Potential | {name}\n"
                 f"Run {cycle:%d %b %Y %HZ} | Window +6 to +72h | "
                 f"Domain peak rate +{peak_hour}h ({cycle + timedelta(hours=peak_hour):%d %b %HZ})\n"
                 "Forecast model guidance, not observed rainfall")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    cycle = choose_cycle()
    lon, lat, index, peak = calculate(cycle)
    plot(lon, lat, index, cycle, peak, [44.5, 51.5, 26, 31.5],
         OUT / "GFS_FFPI_KUWAIT_LATEST.png", "Kuwait")
    plot(lon, lat, index, cycle, peak, [25, 75, 5, 45],
         OUT / "GFS_FFPI_MIDDLE_EAST_LATEST.png", "Middle East")


if __name__ == "__main__":
    main()
