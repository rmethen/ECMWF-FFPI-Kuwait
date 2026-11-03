"""Rolling ECMWF IFS / NOAA GFS precipitation comparison for the Gulf.

Ten consecutive 24-hour forecast windows beginning at the SAME 00/12Z model
cycle. No claim that these are Kuwait midnight-to-midnight calendar days:
00Z cycle -> 03:00–03:00 Kuwait, 12Z -> 15:00–15:00 Kuwait.
Experiment independent of the FFPI algorithm. No fake data on failure.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import numpy as np

from compare_ifs_gfs_nov10_11 import (
    BOUND, CITIES, align_grid, gfs_apcp_url, read_gfs_interval, nearest_cell,
)

PREFIX = "ECMWF_GFS_ROLLING_10DAY"
DAYS = 10
OUTDIR = Path("output")
KWT = timedelta(hours=3)
MAX_LEAD = DAYS * 24


def valid_cycle(cycle):
    if (cycle.tzinfo is None or cycle.utcoffset() != timedelta(0)
        or cycle.hour not in (0, 12) or cycle.minute or cycle.second
        or cycle.microsecond):
        raise ValueError("Rolling forecast requires a precise UTC 00/12Z cycle")
    return cycle


def candidate_cycles(now, limit=3):
    now = now.astimezone(timezone.utc)
    anchor = now.replace(hour=12 if now.hour >= 12 else 0,
                         minute=0, second=0, microsecond=0)
    return [anchor - timedelta(hours=12*i) for i in range(limit)]


def endpoints():
    return tuple(24*i for i in range(DAYS+1))


def gfs_intervals():
    """3h surface APCP intervals through +120h; 6h after +120h."""
    bounds = list(range(0, 121, 3)) + list(range(126, MAX_LEAD+1, 6))
    out = list(zip(bounds, bounds[1:]))
    if sum(end-start for start,end in out) != MAX_LEAD:
        raise AssertionError("GFS accumulation intervals are not complete")
    if any(a//24 != (b-1)//24 for a,b in out):
        raise AssertionError("An APCP interval crosses a 24h window")
    return out


def decode_ifs_cumulative(path):
    """IFS shortName=tp in METERS, accumulated from initialization."""
    import xarray as xr
    expected = set(endpoints()[1:])  # step 0 may not contain a tp message
    with xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"filter_by_keys":{"shortName":"tp"},
                                         "indexpath":""}) as ds:
        if "tp" not in ds:
            raise ValueError("IFS total precipitation (tp) missing")
        da = ds["tp"]
        if "step" not in da.dims:
            raise ValueError("IFS tp has no forecast step axis")
        steps = np.asarray(da.step.values) / np.timedelta64(1, "h")
        got = set(int(x) for x in steps)
        if got != expected:
            raise ValueError(f"IFS expected +24..+240 daily steps; got {sorted(got)}")
        lon = np.asarray(da.longitude.values)
        lat = np.asarray(da.latitude.values)
        if lon.ndim != 1 or lat.ndim != 1:
            raise ValueError("IFS tp must have regular geographical coordinates")
        fields = [np.asarray(da.sel(step=np.timedelta64(h,"h")).values, float)
                  for h in endpoints()[1:]]
    previous = np.zeros_like(fields[0])
    daily = []
    for arr in fields:
        increment = 1000*(arr-previous)
        if not np.isfinite(increment).all() or np.min(increment) < -0.51:
            raise ValueError("IFS precipitation increments invalid; not publishable")
        daily.append(np.maximum(increment,0))
        previous = arr
    return lon, lat, daily


def retrieve_ifs(cycle, target):
    from ecmwf.opendata import Client
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    received = client.retrieve(
        date=cycle.strftime("%Y%m%d"), time=cycle.hour, type="fc",
        stream="oper", step=list(endpoints()[1:]), param=["tp"],
        target=str(target),
    )
    actual = received.datetime
    if actual.tzinfo is None:
        actual = actual.replace(tzinfo=timezone.utc)
    if actual.astimezone(timezone.utc) != cycle:
        raise ValueError("ECMWF returned a different model cycle")
    return decode_ifs_cumulative(target)


def retrieve_gfs(cycle):
    """Download 60 NOAA GRIB intervals and sum each exact 24h day once."""
    intervals = gfs_intervals()
    lon0=lat0=None
    totals=None
    for k,(start,end) in enumerate(intervals):
        req=Request(
            gfs_apcp_url(cycle,end),
            headers={"User-Agent":"FFPI-rolling-IFS-GFS/1.0 (rainfall experiment)"})
        with urlopen(req,timeout=75) as res:
            raw=res.read()
        lon,lat,field=read_gfs_interval(raw,start,end)
        if totals is None:
            lon0,lat0=lon,lat
            totals=[np.zeros_like(field,dtype=float) for _ in range(DAYS)]
        if (field.shape != totals[0].shape or
            not np.array_equal(lon,lon0) or not np.array_equal(lat,lat0)):
            raise ValueError(f"GFS grid mismatch at +{end}")
        if not np.isfinite(field).all() or (field < -0.001).any():
            raise ValueError(f"Invalid NOAA rain at +{end}")
        totals[start//24]+=np.maximum(0,field)
        if (k+1)%10==0:
            print(f"Verified {k+1}/{len(intervals)} GFS APCP intervals",flush=True)
    return lon0,lat0,totals


def same_window_details(cycle):
    return [
        {
            "index":i+1,
            "start_utc":(cycle+timedelta(hours=24*i)).isoformat().replace("+00:00","Z"),
            "end_utc":(cycle+timedelta(hours=24*(i+1))).isoformat().replace("+00:00","Z"),
            "start_kuwait":(cycle+timedelta(hours=24*i)+KWT).strftime("%Y-%m-%d %H:%M"),
            "end_kuwait":(cycle+timedelta(hours=24*(i+1))+KWT).strftime("%Y-%m-%d %H:%M"),
            "image":f"{PREFIX}_D{i+1:02d}.png",
        } for i in range(DAYS)
    ]


def render_map(path, cycle, slot, ecmwf, gfs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm, ListedColormap
    levels=[0,1,3,5,10,20,30,50,75,100,150,250]
    colors=["#f4f7fa","#d4e8f2","#a9d5eb","#6caed6","#5eb982",
            "#b9cf4a","#f2ca45","#f3a243","#e5663d","#cf3341","#922648"]
    cmap=ListedColormap(colors)
    norm=BoundaryNorm(levels,len(colors),clip=True)
    start=cycle if slot is None else cycle+timedelta(days=slot)
    end=cycle+timedelta(days=DAYS if slot is None else slot+1)
    title="10-day total" if slot is None else f"Day {slot+1:02d} / 10"
    fig=plt.figure(figsize=(9,13.5),facecolor="white")
    fig.suptitle(
        f"ECMWF IFS vs NOAA GFS | {title}\n"
        f"{(start+KWT):%d %b %H:%M} to {(end+KWT):%d %b %H:%M} Kuwait"
        f" | Run {cycle:%Y-%m-%d %HZ}", fontsize=14,weight="bold",y=.987)
    proj=ccrs.PlateCarree()
    axes=[fig.add_axes([.08,.54,.76,.35],projection=proj),
          fig.add_axes([.08,.155,.76,.35],projection=proj)]
    for ax,(name,product) in zip(axes,[("ECMWF IFS",ecmwf),("NOAA GFS",gfs)]):
        lon,lat,days=product
        vals=(sum(days) if slot is None else days[slot])
        im=ax.pcolormesh(lon,lat,vals,norm=norm,cmap=cmap,
                         shading="nearest",transform=proj,rasterized=True)
        ax.set_extent(BOUND,crs=proj)
        ax.coastlines("110m",color="#344454",linewidth=.9)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"),
                       edgecolor="#6c7580",linewidth=.5)
        ax.set_xticks(np.arange(43,55,2),crs=proj)
        ax.set_yticks(np.arange(25,34,2),crs=proj)
        ax.tick_params(labelsize=9)
        for label,x,y in CITIES:
            ax.plot(x,y,"ko",ms=3.5,transform=proj)
            ax.annotate(label,(x,y),xytext=(-65 if label=="Hafar Al Batin" else 5,
                        -16 if label=="Al Qaysumah" else 6),
                        textcoords="offset points",fontsize=8,
                        bbox={"facecolor":"white","alpha":.85,"edgecolor":"none","pad":1})
        totals=[f"{s}: {nearest_cell(lon,lat,vals,x,y):.1f} mm" for s,x,y in CITIES]
        ax.set_title(name+" | "+"  ·  ".join(totals),fontsize=9.5,weight="bold")
        cb=fig.colorbar(im,ax=ax,pad=.025,shrink=.78,
                        ticks=[1,5,10,20,30,50,75,100,150,250])
        cb.set_label("Rain (mm)",fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.text(.50,.071,"Two independent model forecasts; one matching UTC cycle and valid window.",
             ha="center",fontsize=10,weight="bold")
    fig.text(.50,.043,"Days are 24h from forecast initialization (00Z or 12Z).",
             ha="center",fontsize=10)
    fig.text(.50,.020,"Forecast, not observations or a flood warning.",ha="center",fontsize=10)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,facecolor="white",dpi=122,bbox_inches="tight")
    plt.close(fig)


def build_status(cycle, generated, ecmwf, gfs):
    windows=same_window_details(cycle)
    cities={}
    for label,x,y in CITIES:
        city={}
        for name,(lon,lat,daily) in (("ECMWF",ecmwf),("GFS",gfs)):
            values=[round(nearest_cell(lon,lat,f,x,y),1) for f in daily]
            city[name]={"daily_mm":values,
                        "total_10d_mm":round(nearest_cell(lon,lat,sum(daily),x,y),1)}
        cities[label]=city
    return {
        "cycle_utc":cycle.isoformat().replace("+00:00","Z"),
        "generated_utc":generated.astimezone(timezone.utc).isoformat().replace("+00:00","Z"),
        "model_count":2,
        "models":["ECMWF IFS 0.25deg","NOAA GFS 0.25deg"],
        "type":"rolling 10 consecutive forecast 24h windows starting at cycle",
        "daily_windows":windows,
        "total_image":f"{PREFIX}_TOTAL.png",
        "cities":cities,
        "warning":"Kuwait local daily 03:00-03:00 (00Z run) or 15:00-15:00 (12Z run); no rain observations",
    }


def main(now=None, outdir=OUTDIR):
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    old_path=outdir/f"{PREFIX}_STATUS.json"
    last=None
    if old_path.exists():
        old=json.loads(old_path.read_text(encoding="utf-8"))
        last=datetime.fromisoformat(old["cycle_utc"].replace("Z","+00:00"))
    last_error=None
    for cycle in candidate_cycles(now):
        if last and cycle<=last:
            print(f"No newer comparable cycle since {last:%Y-%m-%d %HZ}",flush=True)
            return False
        with TemporaryDirectory() as td:
            try:
                folder=Path(td)
                # Verify GFS +240 exists before downloading 60 intervals.
                last_start,last_end=gfs_intervals()[-1]
                req=Request(gfs_apcp_url(cycle,last_end),
                            headers={"User-Agent":"FFPI-rolling-IFS-GFS/1.0"})
                with urlopen(req,timeout=75) as response:
                    final_blob=response.read()
                read_gfs_interval(final_blob,last_start,last_end)
                lon_e,lat_e,arr_e=retrieve_ifs(cycle,folder/"ifs_rolling_tp.grib2")
                lon_g,lat_g,arr_g=retrieve_gfs(cycle)
                ec=align_grid(lon_e,lat_e,arr_e)
                gf=align_grid(lon_g,lat_g,arr_g)
                out=folder/"maps"
                for i in range(DAYS):
                    render_map(out/f"{PREFIX}_D{i+1:02d}.png",cycle,i,ec,gf)
                render_map(out/f"{PREFIX}_TOTAL.png",cycle,None,ec,gf)
                status=build_status(cycle,now,ec,gf)
                (out/f"{PREFIX}_STATUS.json").write_text(
                    json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
                # Publish only after BOTH models and ALL 11 images succeeded.
                for file in sorted(out.iterdir()):
                    file.replace(outdir/file.name)
                print(f"Completed 10-day IFS/GFS cycle {cycle:%Y-%m-%d %HZ}",flush=True)
                return True
            except (OSError,ValueError,RuntimeError,KeyError,ImportError) as exc:
                print(f"Candidate {cycle:%Y-%m-%d %HZ} unavailable: {exc}",flush=True)
                last_error=exc
    raise RuntimeError(f"No complete same-cycle 10-day forecasts: {last_error}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now",help="UTC date YYYYMMDDHH for reproducible test")
    args=parser.parse_args()
    ts=(datetime.strptime(args.now,"%Y%m%d%H").replace(tzinfo=timezone.utc)
        if args.now else None)
    main(now=ts)
