from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ecmwf.opendata import Client

# =========================================================
# ECMWF Experimental Hail Potential Index V1
# Middle East | IFS 0.25 degree | +3 to +72 h
#
# Components:
# 35% MUCAPE
# 25% Hail Growth Layer depth (-10C to -30C)
# 20% Deep-layer shear proxy (1000-500 hPa)
# 10% Freezing-level favourability
# 10% Relative humidity within hail-growth layer
#
# NOTE:
# This is an experimental hail-environment index, not a direct
# prediction of hailstone diameter.
# =========================================================

OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)

PL_GRIB = OUTDIR / "ecmwf_hail_v1_pressure.grib2"
SFC_GRIB = OUTDIR / "ecmwf_hail_v1_surface.grib2"

MAP_FILE = OUTDIR / "ECMWF_HAIL_INDEX_MIDDLE_EAST_LATEST.png"
DIAG_FILE = OUTDIR / "ECMWF_HAIL_INDEX_DIAGNOSTIC.txt"

STEPS = list(range(3, 73, 3))
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300]

LAT_MIN, LAT_MAX = 5.0, 45.0
LON_MIN, LON_MAX = 25.0, 75.0

client = Client(source="ecmwf", model="ifs", resol="0p25")

print("Downloading MUCAPE...")
sfc_result = client.retrieve(
    type="fc",
    stream="oper",
    step=STEPS,
    param=["mucape"],
    target=str(SFC_GRIB),
)

run_time = sfc_result.datetime

print("Downloading pressure-level fields...")
client.retrieve(
    type="fc",
    stream="oper",
    step=STEPS,
    param=["t", "u", "v", "r", "gh"],
    levelist=LEVELS,
    target=str(PL_GRIB),
)

print(f"ECMWF run: {run_time}")

def open_field(path, short_name):
    return xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": short_name},
            "indexpath": "",
        },
    )

print("Opening fields...")
ds_cape = open_field(SFC_GRIB, "mucape")
cape_name = list(ds_cape.data_vars)[0]
mucape = ds_cape[cape_name]

ds_t = open_field(PL_GRIB, "t")
ds_u = open_field(PL_GRIB, "u")
ds_v = open_field(PL_GRIB, "v")
ds_r = open_field(PL_GRIB, "r")
ds_gh = open_field(PL_GRIB, "gh")

t = ds_t[list(ds_t.data_vars)[0]]
u = ds_u[list(ds_u.data_vars)[0]]
v = ds_v[list(ds_v.data_vars)[0]]
rh = ds_r[list(ds_r.data_vars)[0]]
gh = ds_gh[list(ds_gh.data_vars)[0]]

# Detect pressure-level dimension
lev_dim = next(d for d in t.dims if "isobaric" in d.lower())

# Align all fields
mucape, t, u, v, rh, gh = xr.align(
    mucape, t, u, v, rh, gh, join="inner", exclude=[lev_dim]
)

# ---------------------------------------------------------
# Vertical interpolation helpers
# ---------------------------------------------------------

def _interp_height_at_temp(temp_prof, z_prof, target_k):
    temp_prof = np.asarray(temp_prof, dtype=float)
    z_prof = np.asarray(z_prof, dtype=float)

    ok = np.isfinite(temp_prof) & np.isfinite(z_prof)
    temp_prof = temp_prof[ok]
    z_prof = z_prof[ok]

    if temp_prof.size < 2:
        return np.nan

    # Sort by height
    order = np.argsort(z_prof)
    temp_prof = temp_prof[order]
    z_prof = z_prof[order]

    for i in range(len(temp_prof) - 1):
        t1, t2 = temp_prof[i], temp_prof[i + 1]
        z1, z2 = z_prof[i], z_prof[i + 1]

        if (t1 - target_k) * (t2 - target_k) <= 0 and t1 != t2:
            frac = (target_k - t1) / (t2 - t1)
            return z1 + frac * (z2 - z1)

    return np.nan

def _layer_rh(temp_prof, rh_prof):
    temp_prof = np.asarray(temp_prof, dtype=float)
    rh_prof = np.asarray(rh_prof, dtype=float)

    mask = (
        np.isfinite(temp_prof)
        & np.isfinite(rh_prof)
        & (temp_prof <= 263.15)   # -10 C
        & (temp_prof >= 243.15)   # -30 C
    )

    if mask.sum() == 0:
        return np.nan

    return float(np.nanmean(rh_prof[mask]))

def interp_height_at_temp(temp, height, target_k):
    return xr.apply_ufunc(
        _interp_height_at_temp,
        temp,
        height,
        input_core_dims=[[lev_dim], [lev_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
        kwargs={"target_k": target_k},
    )

def layer_rh(temp, relhum):
    return xr.apply_ufunc(
        _layer_rh,
        temp,
        relhum,
        input_core_dims=[[lev_dim], [lev_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )

print("Calculating thermodynamic layers...")
z0 = interp_height_at_temp(t, gh, 273.15)
z10 = interp_height_at_temp(t, gh, 263.15)
z30 = interp_height_at_temp(t, gh, 243.15)

hgl_depth = (z30 - z10).clip(min=0.0)
growth_rh = layer_rh(t, rh)

print("Calculating deep-layer shear proxy...")
u1000 = u.sel({lev_dim: 1000})
v1000 = v.sel({lev_dim: 1000})
u500 = u.sel({lev_dim: 500})
v500 = v.sel({lev_dim: 500})

shear = np.hypot(u500 - u1000, v500 - v1000)

# ---------------------------------------------------------
# Fixed scores, 0-1
# ---------------------------------------------------------

def scale(x, low, high):
    return xr.where(
        x <= low,
        0.0,
        xr.where(x >= high, 1.0, (x - low) / (high - low))
    )

cape_score = scale(mucape, 250.0, 2500.0)
hgl_score = scale(hgl_depth, 1000.0, 3500.0)
shear_score = scale(shear, 7.5, 25.0)
rh_score = scale(growth_rh, 45.0, 80.0)

# Lower freezing level favours survival of hail to the ground.
freeze_score = xr.where(
    z0 <= 2500.0, 1.0,
    xr.where(
        z0 >= 5000.0, 0.0,
        (5000.0 - z0) / 2500.0
    )
)

hpi = (
    0.35 * cape_score
    + 0.25 * hgl_score
    + 0.20 * shear_score
    + 0.10 * freeze_score
    + 0.10 * rh_score
) * 100.0

# Require at least modest instability and a valid hail-growth layer
hail_gate = (
    (mucape >= 250.0)
    & np.isfinite(hgl_depth)
    & (hgl_depth >= 750.0)
)

hpi = hpi.where(hail_gate, 0.0)

# Maximum hail environment during +3 to +72 h
if "step" in hpi.dims:
    hpi72 = hpi.max("step")
else:
    hpi72 = hpi

# Diagnostics: maxima over the whole Middle East domain
def subset_box(data):
    lat = data["latitude"]
    if lat[0] > lat[-1]:
        return data.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX),
        )
    return data.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    )

region = subset_box(hpi72)

def safe_max(x):
    a = np.asarray(x.values, dtype=float)
    return float(np.nanmax(a)) if a.size and np.isfinite(a).any() else float("nan")

diag = (
    "ECMWF Experimental Hail Potential Index V1\n"
    f"IFS Run: {run_time.strftime('%d %b %Y %H UTC')}\n"
    "Forecast: +3 to +72 h\n"
    f"Max HPI in Middle East domain: {safe_max(region):.1f}\n"
    "Weights: MUCAPE 35%, HGL depth 25%, shear proxy 20%, "
    "freezing level 10%, HGL RH 10%\n"
    "Deep-layer shear uses 1000-500 hPa as a ~0-6 km proxy.\n"
    "Experimental product: not a direct hail-size forecast.\n"
)
DIAG_FILE.write_text(diag, encoding="utf-8")
print(diag)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

print("Creating Middle East hail map...")

levels = [20, 35, 50, 65, 80, 90, 100]
cmap = plt.get_cmap("turbo", len(levels) - 1)
norm = mcolors.BoundaryNorm(levels, cmap.N)

plot_data = region.where(region >= 20.0)

cities = {
    "Kuwait": (47.98, 29.38),
    "Riyadh": (46.68, 24.71),
    "Jeddah": (39.19, 21.49),
    "Makkah": (39.86, 21.39),
    "Taif": (40.42, 21.27),
    "Muscat": (58.41, 23.59),
}

def save_plain():
    fig, ax = plt.subplots(figsize=(14, 10))

    mesh = ax.pcolormesh(
        region.longitude,
        region.latitude,
        plot_data,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.arange(25, 76, 5))
    ax.set_yticks(np.arange(5, 46, 5))
    ax.grid(True, linewidth=0.4, alpha=0.4, linestyle="--")

    for name, (lon, lat) in cities.items():
        ax.plot(lon, lat, marker="o", markersize=4)
        ax.text(lon + 0.3, lat + 0.2, name, fontsize=8)

    cb = plt.colorbar(
        mesh,
        ax=ax,
        pad=0.02,
        shrink=0.90,
        boundaries=levels,
        ticks=levels,
        extend="max",
    )
    cb.set_label("Experimental Hail Potential Index (0-100)")

    ax.set_title(
        "ECMWF Experimental Hail Potential Index V1 | Middle East\n"
        f"IFS Run: {run_time.strftime('%d %b %Y %H UTC')} | +3 to +72 h\n"
        "MUCAPE + hail growth layer + deep-layer shear + freezing level + RH",
        fontsize=14,
        weight="bold",
    )

    plt.figtext(
        0.99, 0.01, "© rmethen 2026",
        ha="right", va="bottom", fontsize=9
    )
    plt.tight_layout()
    plt.savefig(MAP_FILE, dpi=180, bbox_inches="tight")
    plt.close()

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(14, 11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.LAND, facecolor="0.94")
    ax.add_feature(cfeature.OCEAN, facecolor="0.90")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)

    mesh = ax.pcolormesh(
        region.longitude,
        region.latitude,
        plot_data,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    ax.set_xticks(np.arange(25, 76, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(5, 46, 5), crs=ccrs.PlateCarree())

    for name, (lon, lat) in cities.items():
        ax.plot(
            lon, lat,
            marker="o",
            markersize=4,
            transform=ccrs.PlateCarree(),
        )
        ax.text(
            lon + 0.3, lat + 0.2, name,
            fontsize=8,
            transform=ccrs.PlateCarree(),
        )

    cb = plt.colorbar(
        mesh,
        ax=ax,
        pad=0.025,
        shrink=0.85,
        boundaries=levels,
        ticks=levels,
        extend="max",
    )
    cb.set_label("Experimental Hail Potential Index (0-100)")

    plt.title(
        "ECMWF Experimental Hail Potential Index V1 | Middle East\n"
        f"IFS Run: {run_time.strftime('%d %b %Y %H UTC')} | +3 to +72 h\n"
        "MUCAPE + hail growth layer + deep-layer shear + freezing level + RH",
        fontsize=14,
        weight="bold",
    )

    plt.figtext(
        0.99, 0.01, "© rmethen 2026",
        ha="right", va="bottom", fontsize=9
    )

    plt.tight_layout()

    try:
        plt.savefig(MAP_FILE, dpi=180, bbox_inches="tight")
        plt.close()
    except Exception:
        plt.close()
        save_plain()

except Exception:
    save_plain()

print("----------------------------------------")
print("ECMWF Hail Potential Index V1 completed")
print(f"Saved map: {MAP_FILE}")
print(f"Saved diagnostic: {DIAG_FILE}")
print("----------------------------------------")
