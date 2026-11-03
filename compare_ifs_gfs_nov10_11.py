"""Independent ECMWF IFS versus NOAA GFS rain forecast comparison for 10/11 Nov 2026.

Two fixed UTC-day windows (00Z–00Z); in Kuwait these are 03:00–03:00,
not midnight-to-midnight. Model accumulation is NOT a flood warning.
The two model products must share the SAME model initialization to compare.
No existing 72h ECMWF/GFS FFPI code or thresholds are touched.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

STARTS = (
    datetime(2026, 11, 10, tzinfo=timezone.utc),
    datetime(2026, 11, 11, tzinfo=timezone.utc),
)
END = datetime(2026, 11, 12, tzinfo=timezone.utc)
BOUND = (42.5, 54, 24.5, 33)  # west east south north; Kuwait, Qaysumah, Hafar
CITIES = (
    ("Al Qaysumah", 46.1273, 28.3112),
    ("Hafar Al Batin", 45.9666, 28.4361),
    ("Kuwait City", 47.98, 29.38),
)
OUTDIR = Path("output")
PREFIX = "ECMWF_GFS_RAIN_QAYSUMAH"
NOAA = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"


def allowed_steps(cycle):
    """For IFS 00/12Z: +0..144 every 3h, +150..360 every 6h."""
    if cycle.tzinfo is None or cycle.utcoffset() != timedelta(0):
        raise ValueError("Cycle must be timezone-aware UTC")
    if cycle.hour not in (0, 12) or cycle.minute or cycle.second:
        raise ValueError("Long-range IFS comparison requires 00/12 UTC")
    endpoints = (*STARTS, END)
    hours = tuple(int((when-cycle).total_seconds()/3600) for when in endpoints)
    for h in hours:
        if h < 0 or h > 360 or (h <= 144 and h % 3) or (h > 144 and h % 6):
            raise ValueError(f"IFS endpoint +{h}h unsupported")
    return hours


def cycles_from_now(now):
    """Newest possible 00/12Z candidates, not fabricated NOAA cycle times."""
    now = now.astimezone(timezone.utc)
    anchor = now.replace(hour=12 if now.hour >= 12 else 0,
                         minute=0, second=0, microsecond=0)
    for n in range(5):
        candidate = anchor - timedelta(hours=12*n)
        try:
            allowed_steps(candidate)
            yield candidate
        except ValueError:
            continue


def gfs_apcp_url(cycle, hour):
    params = {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}",
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
        "var_APCP": "on", "lev_surface": "on", "subregion": "",
        "leftlon": BOUND[0], "rightlon": BOUND[1],
        "bottomlat": BOUND[2], "toplat": BOUND[3],
    }
    return NOAA + "?" + urlencode(params)


def gfs_interval_hours(hour):
    """GFS APCP accumulation windows: <=120h 3h, later 6h."""
    return 3 if hour <= 120 else 6


def gfs_endpoint_steps(cycle):
    """Non-overlapping APCP intervals for the exact 10–12 Nov UTC period."""
    starts = [int((s - cycle).total_seconds()/3600) for s in STARTS]
    end = int((END - cycle).total_seconds()/3600)
    if starts[0] < 0 or end > 384:
        raise ValueError("GFS calendar window outside forecast")
    steps = []
    h = starts[0]
    while h < end:
        increment = 3 if h < 120 else 6
        h += increment
        if h > end:
            raise ValueError("GFS intervals do not evenly cover end")
        steps.append((h-increment, h))
    return steps


def read_gfs_interval(blob, start, end):
    """Decode exact non-overlapping surface APCP from filtered GRIB2.

    Return distinct latitude/longitude arrays and mm values. Reject mismatched
    stepRanges rather than silently double-counting overlapping accumulations.
    """
    import eccodes
    if not blob.startswith(b"GRIB"):
        raise ValueError(f"No NOAA GRIB2 for +{end}h")
    with TemporaryFile() as stream:
        stream.write(blob)
        stream.seek(0)
        while msg := eccodes.codes_grib_new_from_file(stream):
            try:
                if (eccodes.codes_get(msg, "shortName") != "tp" or
                    eccodes.codes_get(msg, "stepType") != "accum" or
                    eccodes.codes_get(msg, "typeOfLevel") != "surface"):
                    continue
                interval = (int(eccodes.codes_get(msg, "startStep")),
                            int(eccodes.codes_get(msg, "endStep")))
                if interval != (start, end):
                    continue
                nj, ni = int(eccodes.codes_get(msg, "Nj")), int(eccodes.codes_get(msg, "Ni"))
                raw = eccodes.codes_get_array(msg, "values").reshape((nj, ni))
                miss = float(eccodes.codes_get(msg, "missingValue"))
                data = np.where(np.isclose(raw, miss), np.nan, raw)
                lat = eccodes.codes_get_array(msg, "distinctLatitudes")
                lon = eccodes.codes_get_array(msg, "distinctLongitudes")
                return np.asarray(lon), np.asarray(lat), np.asarray(data)
            finally:
                eccodes.codes_release(msg)
    raise ValueError(f"GFS APCP exact {start}–{end}h accumulation missing")


def download_gfs(cycle, directory):
    """All APCP accumulation windows must come from one authentic GFS cycle."""
    steps = gfs_endpoint_steps(cycle)
    middle_step = int((STARTS[1]-cycle).total_seconds()/3600)
    segments = [None, None]
    lon0 = lat0 = None
    for n, (start, end) in enumerate(steps):
        url = gfs_apcp_url(cycle, end)
        with urlopen(Request(url, headers={"User-Agent": "rmethen-GFS-IFSRainComparison/1.0"}),
                     timeout=75) as response:
            blob = response.read()
        lon, lat, apcp = read_gfs_interval(blob, start, end)
        if lon0 is None:
            lon0, lat0 = lon, lat
            segments = [np.zeros_like(apcp, float), np.zeros_like(apcp, float)]
        elif apcp.shape != segments[0].shape or not (
            np.array_equal(lon, lon0) and np.array_equal(lat, lat0)):
            raise ValueError("GFS forecast grid changes across steps")
        if not np.isfinite(apcp).all() or (apcp < -1e-3).any():
            raise ValueError("GFS accumulation contains missing/negative values")
        idx = 0 if end <= middle_step else 1
        if start < middle_step < end:
            raise ValueError("GFS APCP interval straddles comparison day")
        segments[idx] += np.maximum(apcp, 0)
        print(f"GFS +{start}–+{end}h verified", flush=True)
    return lon0, lat0, segments


def read_ifs_tp(path, expected_hours):
    """Read cumulative ECMWF IFS tp (meters), then difference endpoints."""
    import xarray as xr
    with xr.open_dataset(
        path, engine="cfgrib",
        backend_kwargs={"filter_by_keys":{"shortName":"tp"}, "indexpath":""},
    ) as ds:
        if "tp" not in ds:
            raise ValueError("ECMWF IFS tp missing")
        da = ds["tp"]
        if "step" not in da.dims:
            raise ValueError("Need three IFS accumulated tp forecast steps")
        observed = np.asarray(da.step.values) / np.timedelta64(1,"h")
        if not np.array_equal(np.sort(observed), np.sort(np.asarray(expected_hours))):
            raise ValueError(f"IFS steps {observed} differ from expected {expected_hours}")
        lon, lat = np.asarray(da.longitude.values), np.asarray(da.latitude.values)
        if lon.ndim != 1 or lat.ndim != 1:
            raise ValueError("Expected 0.25-degree regular IFS grid")
        fields = [np.asarray(da.sel(step=np.timedelta64(h,"h")).values, float)
                  for h in expected_hours]
    differences = [(fields[i+1]-fields[i])*1000.0 for i in (0,1)]
    if any(not np.isfinite(p).all() or (p < -0.51).any() for p in differences):
        raise ValueError("IFS total precipitation contains invalid increments")
    return lon, lat, [np.maximum(a,0.0) for a in differences]


def download_ifs(cycle, directory):
    from ecmwf.opendata import Client
    h = allowed_steps(cycle)
    target = directory / "ifs_tp_calendar_nov10_11.grib2"
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    result = client.retrieve(
        date=cycle.strftime("%Y%m%d"), time=cycle.hour,
        type="fc", stream="oper", step=list(h), param=["tp"],
        target=str(target),
    )
    if result.datetime.replace(tzinfo=timezone.utc) != cycle:
        raise ValueError("IFS returned wrong forecast cycle")
    return read_ifs_tp(target, h)


def align_grid(lon, lat, data):
    """Normalize 1-D coordinates for consistent map/city sampling."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("Expected regular geographic grid")
    ix, iy = np.argsort(lon), np.argsort(lat)
    lon, lat = lon[ix], lat[iy]
    cropped_x = (lon >= BOUND[0]) & (lon <= BOUND[1])
    cropped_y = (lat >= BOUND[2]) & (lat <= BOUND[3])
    if cropped_x.sum() < 10 or cropped_y.sum() < 10:
        raise ValueError("Insufficient Kuwait/Hafar grid coverage")
    fields = []
    for f in data:
        if f.shape != (len(iy),len(ix)):
            raise ValueError("Grid and rainfall shapes differ")
        fields.append(np.asarray(f)[np.ix_(iy,ix)][np.ix_(cropped_y,cropped_x)])
    return lon[cropped_x], lat[cropped_y], fields


def nearest_cell(lon, lat, data, longitude, latitude):
    i = np.abs(lon-longitude).argmin()
    j = np.abs(lat-latitude).argmin()
    return float(data[j,i])


def render_panels(output, cycle, day_index, ecmwf, gfs):
    """Tall mobile-readable two-model maps with exactly the same 24/48h window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    start = STARTS[day_index] if day_index < 2 else STARTS[0]
    end = STARTS[day_index+1] if day_index == 0 else END
    title = (f"{start:%d %b} UTC 00Z – {end:%d %b} UTC 00Z"
             f" (Kuwait 03:00–03:00)")
    # Fixed bins, no auto-scaling between models or lead dates.
    levels = [0,1,3,5,10,20,30,50,75,100,150,250]
    cmap = ListedColormap(["#f4f7fa","#d4e8f2","#a9d5eb","#6caed6",
                            "#5eb982","#b9cf4a","#f2ca45","#f3a243",
                            "#e5663d","#cf3341","#922648"])
    norm = BoundaryNorm(levels, len(levels)-1, clip=True)
    crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9,13.5), facecolor="white")
    fig.suptitle(f"ECMWF IFS versus NOAA GFS | Kuwait & Hafar Al Batin\n"
                 f"{title} | Cycle {cycle:%Y-%m-%d %HZ}",
                 fontsize=14,weight="bold",y=.989)
    axes = [
        fig.add_axes([.08,.54,.75,.355],projection=crs),
        fig.add_axes([.08,.155,.75,.355],projection=crs),
    ]
    for ax, (name, (lon,lat,fields)) in zip(
        axes, [("ECMWF IFS",ecmwf),("NOAA GFS",gfs)]):
        values = fields[day_index] if day_index < 2 else fields[0]+fields[1]
        mesh = ax.pcolormesh(lon,lat,values,shading="nearest",
                             norm=norm,cmap=cmap,transform=crs,rasterized=True)
        ax.set_extent(BOUND,crs=crs)
        ax.coastlines("110m",color="#364452",linewidth=.9)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"),edgecolor="#69747e",
                       linewidth=.55)
        ax.set_xticks(np.arange(43,55,2),crs=crs)
        ax.set_yticks(np.arange(25,34,2),crs=crs)
        ax.tick_params(labelsize=9)
        for cname, x, y in CITIES:
            ax.plot(x,y,marker="o",color="black",ms=4,transform=crs)
            dy=-16 if cname=="Al Qaysumah" else 5
            dx=-67 if cname=="Hafar Al Batin" else 6
            ax.annotate(cname,(x,y),xytext=(dx,dy),textcoords="offset points",
                        fontsize=8,color="#10243a",
                        bbox={"facecolor":"white","alpha":.83,"edgecolor":"none","pad":1})
        parts = [f"{n}: {nearest_cell(lon,lat,values,x,y):.1f} mm"
                 for n,x,y in CITIES]
        ax.set_title(name+" | "+"  •  ".join(parts),
                     fontsize=9.5,weight="bold")
        cb = fig.colorbar(mesh,ax=ax,shrink=.78,pad=.028,
                          ticks=[1,5,10,20,30,50,75,100,150,250])
        cb.ax.tick_params(labelsize=8)
        cb.set_label("Total precipitation (mm)",fontsize=9)
    fig.text(.5,.072,"Same UTC calendar window and GFS/IFS 00Z or 12Z model cycle.",
             ha="center",fontsize=10,weight="bold")
    fig.text(.5,.047,"Model guidance only. DMI city forecast is ECMWF-derived;",
             ha="center",fontsize=9)
    fig.text(.5,.027,"daily precipitation is 03:00–03:00 Kuwait time. Not a flood alert.",
             ha="center",fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=155,facecolor="white",bbox_inches="tight")
    plt.close(fig)


def main(now=None):
    now = now or datetime.now(timezone.utc)
    OUTDIR.mkdir(exist_ok=True)
    status_file = OUTDIR / (PREFIX + "_STATUS.json")
    previous_cycle = None
    if status_file.exists():
        previous = json.loads(status_file.read_text())
        previous_cycle = datetime.fromisoformat(previous["cycle_utc"].replace("Z","+00:00"))
    if now >= END:
        print("Period ended; preserve archived Nov 10/11 maps", flush=True)
        return
    error = None
    for cycle in cycles_from_now(now):
        if previous_cycle is not None and cycle <= previous_cycle:
            print("No newer comparable 00/12Z cycle: keep previous published images")
            return
        with TemporaryDirectory() as tmp:
            try:
                tmp = Path(tmp)
                # NOAA and IFS must both succeed for the same cycle.
                # Test the late GFS forecast first before downloading all fields.
                gfs_latestor = gfs_endpoint_steps(cycle)[-1]
                with urlopen(
                    Request(gfs_apcp_url(cycle, gfs_latestor[1]),
                            headers={"User-Agent":"rmethen-GFS-IFSRainComparison/1.0"}),
                    timeout=75,
                ) as res:
                    last = res.read()
                read_gfs_interval(last, *gfs_latestor)
                lon_e, lat_e, maps_e = download_ifs(cycle,tmp)
                lon_g, lat_g, maps_g = download_gfs(cycle,tmp)
                ec = align_grid(lon_e,lat_e,maps_e)
                gf = align_grid(lon_g,lat_g,maps_g)
                print(f"Common ECMWF/GFS forecast cycle {cycle:%Y-%m-%d %HZ}",flush=True)
                tempout=tmp/"plots"
                for name,day in (("10NOV",0),("11NOV",1),("10_11NOV",2)):
                    render_panels(tempout/f"{PREFIX}_{name}.png",cycle,day,ec,gf)
                status = {
                    "cycle_utc":cycle.isoformat().replace("+00:00","Z"),
                    "generated_utc":now.isoformat().replace("+00:00","Z"),
                    "periods_utc":{"10NOV":"2026-11-10T00:00Z / 2026-11-11T00:00Z",
                                   "11NOV":"2026-11-11T00:00Z / 2026-11-12T00:00Z",
                                   "10_11NOV":"2026-11-10T00:00Z / 2026-11-12T00:00Z"},
                    "models":["ECMWF IFS 0.25deg","NOAA GFS 0.25deg"],
                    "city_mm":{},
                    "note":"UTC 00-00 equals Kuwait 03-03; forecast, not observations",
                }
                for n,x,y in CITIES:
                    status["city_mm"][n]={
                        model: [round(nearest_cell(p[0],p[1],v,x,y),1)
                                for v in (p[2][0],p[2][1],p[2][0]+p[2][1])]
                        for model,p in (("ECMWF",ec),("GFS",gf))
                    }
                (tempout/f"{PREFIX}_STATUS.json").write_text(
                    json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
                for p in tempout.iterdir():
                    p.replace(OUTDIR/p.name)
                print("Saved new ECMWF/GFS fixed-window maps and status",flush=True)
                return
            except (OSError,RuntimeError,ValueError,KeyError,ImportError) as exc:
                print(f"Incomplete common {cycle:%Y-%m-%d %HZ}: {exc}",flush=True)
                error=exc
    raise RuntimeError(f"No complete common ECMWF/GFS long-range cycle: {error}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now",help="Optional test UTC time YYYYMMDDHH")
    args=parser.parse_args()
    instant=(datetime.strptime(args.now,"%Y%m%d%H").replace(tzinfo=timezone.utc)
             if args.now else None)
    main(instant)
