"""Independent GFS model guidance for Kuwait (never a radar observation)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


OUT = Path("output")
OUT.mkdir(exist_ok=True)
BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
BOX = (43, 56, 23, 34)  # west, east, south, north


def download_latest():
    now = datetime.now(timezone.utc)
    # NOMADS cycles may arrive several hours after initialization.
    cycles = [now.replace(hour=h, minute=0, second=0, microsecond=0)
              for h in (18, 12, 6, 0)]
    cycles += [c - timedelta(days=1) for c in cycles]
    for cycle in sorted((c for c in cycles if c <= now), reverse=True):
        params = {
            "file": "gfs.t%02dz.pgrb2.0p25.f006" % cycle.hour,
            "dir": "/gfs.%s/%02d/atmos" % (cycle.strftime("%Y%m%d"), cycle.hour),
            "var_REFC": "on", "var_PRATE": "on", "var_RH": "on",
            "lev_entire_atmosphere": "on", "lev_surface": "on",
            "lev_700_mb": "on", "subregion": "", "leftlon": BOX[0],
            "rightlon": BOX[1], "toplat": BOX[3], "bottomlat": BOX[2],
        }
        try:
            with urlopen(Request(BASE + "?" + urlencode(params),
                                 headers={"User-Agent": "ECMWF-FFPI-Kuwait/1.0"}),
                         timeout=45) as response:
                data = response.read()
            if not data.startswith(b"GRIB"):
                raise ValueError("NOMADS did not return GRIB data")
            path = OUT / "gfs_guidance_latest.grib2"
            path.write_bytes(data)
            return cycle, path
        except (OSError, ValueError) as exc:
            print(f"GFS cycle {cycle:%Y-%m-%d %HZ} unavailable: {exc}")
    raise RuntimeError("No GFS cycle available")


def field(path, short_name, level_type):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={
        "filter_by_keys": {"shortName": short_name,
                           "typeOfLevel": level_type}, "indexpath": ""})
    return ds[list(ds.data_vars)[0]]


def main():
    cycle, path = download_latest()
    refc = field(path, "refc", "entireAtmosphere")
    prate = field(path, "prate", "surface") * 3600  # kg m-2 s-1 -> mm/h
    rh = field(path, "r", "isobaricInhPa").sel(isobaricInhPa=700)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, data, levels, title, units in [
        (axes[0], refc, [5, 15, 25, 35, 45, 55, 65, 75],
         "Simulated composite reflectivity", "dBZ"),
        (axes[1], prate, [0.1, 0.5, 1, 2, 5, 10, 20, 40],
         "Forecast rain rate", "mm/h"),
        (axes[2], rh, np.arange(10, 101, 10),
         "700 hPa relative humidity", "%"),
    ]:
        ax.set_extent(BOX, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        image = ax.contourf(data.longitude, data.latitude, data,
                            levels=levels, cmap="turbo", extend="max",
                            transform=ccrs.PlateCarree())
        fig.colorbar(image, ax=ax, shrink=0.7, label=units)
        ax.set_title(title)
    valid = cycle + timedelta(hours=6)
    fig.suptitle(f"GFS 0.25° forecast guidance | Run {cycle:%d %b %Y %HZ} | "
                 f"Valid {valid:%d %b %Y %HZ} (+6h)\nModel simulation, not observed radar")
    fig.tight_layout()
    target = OUT / "GFS_GUIDANCE_KUWAIT_LATEST.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
