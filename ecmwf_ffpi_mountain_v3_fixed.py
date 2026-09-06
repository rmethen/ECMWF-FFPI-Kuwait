from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.ndimage import label
from ecmwf.opendata import Client

# =========================================================
# ECMWF Experimental FFPI - Mountain V3 FIXED
# Wide Middle East map + North Oman diagnostics
# Plotting fix: pcolormesh instead of contourf to avoid
# Shapely/GEOS LinearRing errors on irregular masked polygons.
# =========================================================

OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)

GRIB_FILE = OUTDIR / "ecmwf_ffpi_mountain_v3_0_72h.grib2"
MAP_FILE = OUTDIR / "ECMWF_FFPI_MIDDLE_EAST_LATEST.png"
DIAG_FILE = OUTDIR / "ECMWF_FFPI_OMAN_DIAGNOSTIC.txt"

STEPS = list(range(3, 73, 3))

print("Downloading latest ECMWF IFS data...")

client = Client(source="ecmwf", model="ifs", resol="0p25")

result = client.retrieve(
    type="fc",
    stream="oper",
    step=STEPS,
    param=["tp", "tprate", "ro", "vsw"],
    target=str(GRIB_FILE),
)

run_time = result.datetime
print(f"ECMWF run: {run_time}")
print("Download completed.")

def open_grib(short_name, extra_filter=None):
    filter_keys = {"shortName": short_name}
    if extra_filter:
        filter_keys.update(extra_filter)
    return xr.open_dataset(
        GRIB_FILE,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": filter_keys, "indexpath": ""},
    )

print("Reading total precipitation...")
ds_tp = open_grib("tp")
tp = ds_tp["tp"] * 1000.0
tp72 = tp.max("step") if "step" in tp.dims else tp

print("Reading precipitation rate...")
ds_rate = open_grib("tprate")
rate_name = list(ds_rate.data_vars)[0]
rain_rate = ds_rate[rate_name] * 3600.0
max_rate = rain_rate.max("step") if "step" in rain_rate.dims else rain_rate

print("Reading runoff...")
ds_ro = open_grib("ro")
ro_name = list(ds_ro.data_vars)[0]
runoff = ds_ro[ro_name] * 1000.0
runoff72 = runoff.max("step") if "step" in runoff.dims else runoff

print("Reading soil moisture...")
try:
    ds_vsw = open_grib("vsw", {"level": 1})
except Exception:
    ds_vsw = open_grib("vsw")

vsw_name = list(ds_vsw.data_vars)[0]
soil = ds_vsw[vsw_name]

if "step" in soil.dims:
    soil = soil.max("step")
if "soilLayer" in soil.dims:
    soil = soil.isel(soilLayer=0)
if "depthBelowLandLayer" in soil.dims:
    soil = soil.isel(depthBelowLandLayer=0)

tp72, max_rate, runoff72, soil = xr.align(
    tp72, max_rate, runoff72, soil, join="inner"
)

def fixed_scale(x, low, high):
    return xr.where(
        x <= low,
        0.0,
        xr.where(x >= high, 1.0, (x - low) / (high - low))
    )

tp_score = fixed_scale(tp72, 10.0, 60.0)
rate_score = fixed_scale(max_rate, 3.0, 25.0)
runoff_score = fixed_scale(runoff72, 1.0, 20.0)
soil_score = fixed_scale(soil, 0.15, 0.40)

ffpi = (
    0.40 * rate_score
    + 0.30 * tp_score
    + 0.20 * runoff_score
    + 0.10 * soil_score
) * 100.0

rain_gate = (tp72 >= 10.0) | (max_rate >= 5.0)
ffpi = ffpi.where(rain_gate, 0.0)
ffpi = ffpi.where(ffpi >= 15.0, 0.0)

def remove_small_objects(data, min_pixels=3):
    arr = np.asarray(data.values).copy()
    mask = np.isfinite(arr) & (arr >= 15.0)
    structure = np.ones((3, 3), dtype=int)
    labeled, number = label(mask, structure=structure)
    for region in range(1, number + 1):
        region_mask = labeled == region
        if region_mask.sum() < min_pixels:
            arr[region_mask] = 0.0
    return xr.DataArray(
        arr, coords=data.coords, dims=data.dims, attrs=data.attrs
    )

ffpi_clean = remove_small_objects(ffpi, min_pixels=3)

# Preserve intense small/local cells
strong_local = (max_rate >= 8.0) | (runoff72 >= 5.0)
ffpi_clean = xr.where(
    strong_local & (ffpi >= 15.0),
    ffpi,
    ffpi_clean
)

# =========================================================
# North Oman diagnostic
# =========================================================
OMAN_LAT_MIN = 22.0
OMAN_LAT_MAX = 24.5
OMAN_LON_MIN = 56.0
OMAN_LON_MAX = 59.5

def subset_box(data, lat_min, lat_max, lon_min, lon_max):
    lat = data["latitude"]
    if lat[0] > lat[-1]:
        return data.sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        )
    return data.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )

oman_tp = subset_box(tp72, OMAN_LAT_MIN, OMAN_LAT_MAX, OMAN_LON_MIN, OMAN_LON_MAX)
oman_rate = subset_box(max_rate, OMAN_LAT_MIN, OMAN_LAT_MAX, OMAN_LON_MIN, OMAN_LON_MAX)
oman_ro = subset_box(runoff72, OMAN_LAT_MIN, OMAN_LAT_MAX, OMAN_LON_MIN, OMAN_LON_MAX)
oman_ffpi = subset_box(ffpi_clean, OMAN_LAT_MIN, OMAN_LAT_MAX, OMAN_LON_MIN, OMAN_LON_MAX)

def safe_max(x):
    a = np.asarray(x.values, dtype=float)
    return float(np.nanmax(a)) if a.size and np.isfinite(a).any() else float("nan")

diag = (
    "ECMWF Mountain V3 - North Oman Diagnostic\n"
    f"IFS Run: {run_time.strftime('%d %b %Y %H UTC')}\n"
    f"Box: {OMAN_LAT_MIN}-{OMAN_LAT_MAX}N, {OMAN_LON_MIN}-{OMAN_LON_MAX}E\n"
    f"Max 0-72h precipitation: {safe_max(oman_tp):.2f} mm\n"
    f"Max precipitation rate: {safe_max(oman_rate):.2f} mm/h\n"
    f"Max runoff: {safe_max(oman_ro):.2f} mm\n"
    f"Max cleaned FFPI: {safe_max(oman_ffpi):.2f}\n"
)

DIAG_FILE.write_text(diag, encoding="utf-8")
print(diag)

# =========================================================
# Wide Middle East map
# =========================================================
LAT_MIN = 5.0
LAT_MAX = 45.0
LON_MIN = 25.0
LON_MAX = 75.0

region = subset_box(ffpi_clean, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

print("Creating Middle East FFPI map...")

fig = plt.figure(figsize=(14, 11))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="0.94")
ax.add_feature(cfeature.OCEAN, facecolor="0.90")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.7)

levels = [15, 25, 35, 50, 70, 85, 100]
cmap = plt.get_cmap("turbo", len(levels) - 1)
norm = mcolors.BoundaryNorm(levels, cmap.N)

# Mask values below plotting threshold.
plot_data = region.where(region >= 15.0)

# Robust plotting: avoids GEOS LinearRing errors produced by contourf
mesh = ax.pcolormesh(
    region.longitude,
    region.latitude,
    plot_data,
    cmap=cmap,
    norm=norm,
    shading="auto",
    transform=ccrs.PlateCarree(),
)

cb = plt.colorbar(
    mesh,
    ax=ax,
    orientation="vertical",
    pad=0.025,
    shrink=0.85,
    boundaries=levels,
    ticks=levels,
    extend="max",
)
cb.set_label("Experimental FFPI (0-100)", fontsize=11)

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.4,
    alpha=0.5,
    linestyle="--",
)
gl.top_labels = False
gl.right_labels = False

run_text = run_time.strftime("%d %b %Y %H UTC")
plt.title(
    "Unbiased Experimental FFPI - ECMWF | Middle East\n"
    f"IFS Run: {run_text} | Forecast: +3 to +72 h\n"
    "Mountain V3 test | wide regional view | North Oman diagnostic",
    fontsize=14,
    weight="bold",
)

plt.figtext(0.99, 0.01, "© rmethen 2026", ha="right", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(MAP_FILE, dpi=180, bbox_inches="tight")
plt.close()

print("----------------------------------------")
print("ECMWF Mountain V3 completed successfully")
print(f"Saved map: {MAP_FILE}")
print(f"Saved diagnostic: {DIAG_FILE}")
print("----------------------------------------")
