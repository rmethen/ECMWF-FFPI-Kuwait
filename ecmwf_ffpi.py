from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.ndimage import label
from ecmwf.opendata import Client


# =========================================================
# ECMWF Experimental FFPI - Kuwait / Arabia
# Automatic IFS 0.25 degree run
# Forecast steps: 3h to 72h every 3 hours
# =========================================================

OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)

GRIB_FILE = OUTDIR / "ecmwf_ffpi_0_72h.grib2"
MAP_FILE = OUTDIR / "ECMWF_FFPI_KUWAIT_LATEST.png"
GULF_MAP_FILE = OUTDIR / "ECMWF_FFPI_GULF_LATEST.png"
STEPS = list(range(3, 73, 3))


# ---------------------------------------------------------
# 1. Download latest ECMWF IFS open data
# ---------------------------------------------------------

print("Downloading latest ECMWF IFS data...")

client = Client(
    source="ecmwf",
    model="ifs",
    resol="0p25",
)

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


# ---------------------------------------------------------
# 2. Helper to open GRIB parameter
# ---------------------------------------------------------

def open_grib(short_name, extra_filter=None):
    filter_keys = {"shortName": short_name}

    if extra_filter:
        filter_keys.update(extra_filter)

    ds = xr.open_dataset(
        GRIB_FILE,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": filter_keys,
            "indexpath": "",
        },
    )

    return ds


# ---------------------------------------------------------
# 3. Read precipitation
# ---------------------------------------------------------

print("Reading total precipitation...")

ds_tp = open_grib("tp")
tp = ds_tp["tp"]

# ECMWF tp is metres of water
tp_mm = tp * 1000.0

# maximum accumulated rainfall through 72 h
if "step" in tp_mm.dims:
    tp72 = tp_mm.max("step")
else:
    tp72 = tp_mm


# ---------------------------------------------------------
# 4. Read precipitation rate
# ---------------------------------------------------------

print("Reading precipitation rate...")

ds_rate = open_grib("tprate")

rate_name = list(ds_rate.data_vars)[0]
rain_rate = ds_rate[rate_name]

# kg m-2 s-1 is equivalent to mm/s
# convert to mm/hour
rain_rate_mmh = rain_rate * 3600.0

if "step" in rain_rate_mmh.dims:
    max_rate = rain_rate_mmh.max("step")
else:
    max_rate = rain_rate_mmh


# ---------------------------------------------------------
# 5. Read runoff
# ---------------------------------------------------------

print("Reading runoff...")

ds_ro = open_grib("ro")
ro_name = list(ds_ro.data_vars)[0]
runoff = ds_ro[ro_name]

# runoff is metres water equivalent
runoff_mm = runoff * 1000.0

if "step" in runoff_mm.dims:
    runoff72 = runoff_mm.max("step")
else:
    runoff72 = runoff_mm


# ---------------------------------------------------------
# 6. Read volumetric soil water
# Layer 1 is used as near-surface soil moisture proxy
# ---------------------------------------------------------

print("Reading soil moisture...")

try:
    ds_vsw = open_grib(
        "vsw",
        {"level": 1},
    )
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


# ---------------------------------------------------------
# 7. Align grids
# ---------------------------------------------------------

tp72, max_rate, runoff72, soil = xr.align(
    tp72,
    max_rate,
    runoff72,
    soil,
    join="inner",
)


# ---------------------------------------------------------
# 8. Fixed physical thresholds
#
# No map-relative normalization.
# Each component is scaled with fixed thresholds.
# ---------------------------------------------------------

def fixed_scale(x, low, high):
    return xr.where(
        x <= low,
        0.0,
        xr.where(
            x >= high,
            1.0,
            (x - low) / (high - low)
        )
    )


# Rain accumulation:
# weak below 10 mm
# maximum contribution at 60 mm
tp_score = fixed_scale(tp72, 10.0, 60.0)

# Rainfall intensity:
# weak below 3 mm/h
# maximum contribution at 25 mm/h
rate_score = fixed_scale(max_rate, 3.0, 25.0)

# Runoff:
# weak below 1 mm
# maximum contribution at 20 mm
runoff_score = fixed_scale(runoff72, 1.0, 20.0)

# Soil moisture:
# dry below 0.15
# strongly wet at 0.40
soil_score = fixed_scale(soil, 0.15, 0.40)


# ---------------------------------------------------------
# 9. Experimental FFPI
#
# Main weight is rainfall intensity.
# ---------------------------------------------------------

ffpi = (
    0.40 * rate_score
    + 0.30 * tp_score
    + 0.20 * runoff_score
    + 0.10 * soil_score
) * 100.0


# ---------------------------------------------------------
# 10. Dual rainfall gate
#
# Suppress false weak signals unless:
# TP >= 10 mm OR max rain rate >= 5 mm/h
# ---------------------------------------------------------

rain_gate = (tp72 >= 10.0) | (max_rate >= 5.0)

ffpi = ffpi.where(rain_gate, 0.0)


# ---------------------------------------------------------
# 11. Remove weak values
# ---------------------------------------------------------

ffpi = ffpi.where(ffpi >= 15.0, 0.0)


# ---------------------------------------------------------
# 12. Remove isolated single-pixel noise
# ---------------------------------------------------------

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
        arr,
        coords=data.coords,
        dims=data.dims,
        attrs=data.attrs,
    )


ffpi_clean = remove_small_objects(ffpi, min_pixels=3)


# ---------------------------------------------------------
# 13. Kuwait / Gulf regional subset
# ---------------------------------------------------------

LAT_MIN = 20.0
LAT_MAX = 34.0
LON_MIN = 38.0
LON_MAX = 60.0


lat_name = "latitude"
lon_name = "longitude"

lat = ffpi_clean[lat_name]

if lat[0] > lat[-1]:
    region = ffpi_clean.sel(
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX),
    )
else:
    region = ffpi_clean.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    )


# ---------------------------------------------------------
# 14. Plot
# ---------------------------------------------------------

print("Creating FFPI map...")

fig = plt.figure(figsize=(13, 10))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

ax.set_extent(
    [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    crs=ccrs.PlateCarree()
)

ax.add_feature(
    cfeature.LAND,
    facecolor="0.94"
)

ax.add_feature(
    cfeature.OCEAN,
    facecolor="0.90"
)

ax.add_feature(
    cfeature.COASTLINE,
    linewidth=0.8
)

ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.7
)

levels = [
    15,
    25,
    35,
    50,
    70,
    85,
    100,
]

plot = ax.contourf(
    region.longitude,
    region.latitude,
    region,
    levels=levels,
    cmap="turbo",
    extend="max",
    transform=ccrs.PlateCarree(),
)

cb = plt.colorbar(
    plot,
    ax=ax,
    orientation="vertical",
    pad=0.025,
    shrink=0.85,
)

cb.set_label(
    "Experimental FFPI (0–100)",
    fontsize=11
)

ax.gridlines(
    draw_labels=True,
    linewidth=0.4,
    alpha=0.5,
    linestyle="--",
)

run_text = run_time.strftime("%d %b %Y %H UTC")

plt.title(
    "Unbiased Experimental FFPI – ECMWF\n"
    f"IFS Run: {run_text} | Forecast: +3 to +72 h\n"
    "Fixed physical thresholds • rainfall gate • isolated-noise removal",
    fontsize=14,
    weight="bold",
)
plt.figtext(0.99, 0.01, "© rmethen 2026", ha="right", va="bottom", fontsize=9)
plt.tight_layout()

plt.savefig(
    MAP_FILE,
    dpi=180,
    bbox_inches="tight",
)

plt.close()
# Gulf regional map
GULF_LAT_MIN = 15.0
GULF_LAT_MAX = 33.0
GULF_LON_MIN = 34.0
GULF_LON_MAX = 60.0

if lat[0] > lat[-1]:
    gulf_region = ffpi_clean.sel(
        latitude=slice(GULF_LAT_MAX, GULF_LAT_MIN),
        longitude=slice(GULF_LON_MIN, GULF_LON_MAX),
    )
else:
    gulf_region = ffpi_clean.sel(
        latitude=slice(GULF_LAT_MIN, GULF_LAT_MAX),
        longitude=slice(GULF_LON_MIN, GULF_LON_MAX),
    )

fig_gulf = plt.figure(figsize=(13, 10))
ax_gulf = plt.axes(projection=ccrs.PlateCarree())

ax_gulf.set_extent(
    [GULF_LON_MIN, GULF_LON_MAX, GULF_LAT_MIN, GULF_LAT_MAX],
    crs=ccrs.PlateCarree()
)

ax_gulf.add_feature(cfeature.LAND, facecolor="0.94")
ax_gulf.add_feature(cfeature.OCEAN, facecolor="0.90")
ax_gulf.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax_gulf.add_feature(cfeature.BORDERS, linewidth=0.7)

plot_gulf = ax_gulf.contourf(
    gulf_region.longitude,
    gulf_region.latitude,
    gulf_region,
    levels=levels,
    cmap="turbo",
    extend="max",
    transform=ccrs.PlateCarree(),
)

cb_gulf = plt.colorbar(
    plot_gulf,
    ax=ax_gulf,
    orientation="vertical",
    pad=0.025,
    shrink=0.85,
)

cb_gulf.set_label("Experimental FFPI (0–100)", fontsize=11)

ax_gulf.gridlines(
    draw_labels=True,
    linewidth=0.4,
    alpha=0.5,
    linestyle="--",
)

plt.title(
    "Unbiased Experimental FFPI – ECMWF | Gulf Region\n"
    f"IFS Run: {run_text} | Forecast: +3 to +72 h\n"
    "Fixed physical thresholds • rainfall gate • isolated-noise removal",
    fontsize=14,
    weight="bold",
)

fig_gulf.text(
    0.99, 0.01, "© rmethen 2026",
    ha="right", va="bottom", fontsize=9
)

plt.tight_layout()
fig_gulf.savefig(GULF_MAP_FILE, dpi=180, bbox_inches="tight")
plt.close(fig_gulf)




print("----------------------------------------")
print("ECMWF FFPI completed successfully")
print(f"Saved map: {MAP_FILE}")

print("----------------------------------------")
