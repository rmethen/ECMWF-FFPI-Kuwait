"""Non-operational diagnostic maps for the unchanged 72-hour ECMWF FFPI.

The first three panels are *not* subjected to the FFPI rainfall gate,
minimum score or 3-pixel filter. The fourth explains those filters.
All forecast fields use the exact data fetched for the operational FFPI.
"""
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import xarray as xr


DIAGNOSTIC_TARGETS = {
    "KUWAIT": ([44.5, 51.5, 26.0, 31.5],
               "ECMWF_FFPI_DIAGNOSTIC_KUWAIT_LATEST.png"),
    "MIDDLE_EAST": ([25.0, 75.0, 5.0, 45.0],
                    "ECMWF_FFPI_DIAGNOSTIC_MIDDLE_EAST_LATEST.png"),
}


def classify_filter_reasons(raw_ffpi, rain_gate, final_ffpi):
    """Return exact, mutually exclusive codes describing the published mask.

    0: no raw FFPI signal, 1: raw score <15 despite passing rainfall gate,
    2: fails rainfall gate (rain <10 mm AND rate <5 mm/h),
    3: passes gate and score >=15 but removed as isolated 1-2 grid cells,
    4: appears in final/published FFPI.
    The logic is diagnostic only; it cannot modify the operational index.
    """
    raw = np.asarray(raw_ffpi, dtype=float)
    gate = np.asarray(rain_gate, dtype=bool)
    final = np.asarray(final_ffpi, dtype=float)
    if raw.shape != gate.shape or raw.shape != final.shape:
        raise ValueError("Raw score, rain gate and final FFPI must align")

    valid = np.isfinite(raw) & np.isfinite(final)
    result = np.zeros(raw.shape, dtype=np.uint8)
    result[valid & (raw > 0) & gate & (raw < 15)] = 1
    result[valid & (raw > 0) & ~gate] = 2
    result[valid & (raw >= 15) & gate & (final < 15)] = 3
    result[valid & (final >= 15)] = 4
    return result


def subset(field, bounds):
    west, east, south, north = bounds
    latitude = field["latitude"].values
    lat_slice = slice(north, south) if latitude[0] > latitude[-1] else slice(south, north)
    return field.sel(latitude=lat_slice, longitude=slice(west, east))


def _base_map(ax, bounds):
    ax.set_extent(bounds, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.96")
    ax.add_feature(cfeature.OCEAN, facecolor="0.95")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.65)
    ax.add_feature(cfeature.BORDERS, linewidth=0.45)
    gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, linewidth=0.3,
                      alpha=0.4, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}


def render_diagnostic_map(path, name, bounds, run_time,
                          rain_mm, peak_rate_mmh, raw_ffpi, gate, final_ffpi):
    """Write a four-panel quality-control map, without editing FFPI outputs."""
    fields = [rain_mm, peak_rate_mmh, raw_ffpi, gate, final_ffpi]
    rain, rate, raw, passed, final = [subset(f, bounds) for f in fields]
    classification = classify_filter_reasons(raw.values, passed.values, final.values)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig.suptitle(
        f"ECMWF FFPI Diagnostics | {name}\n"
        f"IFS Run: {run_time:%d %b %Y %H UTC} | +3 to +72h | "
        "Forecast guidance (not observations)",
        fontsize=16, weight="bold", y=0.99,
    )

    panels = [
        ("A. Total precipitation, 72 h (mm)", rain.values, "YlGnBu", 100.0, "mm"),
        ("B. Maximum instantaneous rain rate (mm/h)", rate.values,
         "turbo", 25.0, "mm/h"),
        ("C. Raw FFPI before ALL filters (0-100)", raw.values,
         "magma", 100.0, "score"),
    ]
    for ax, (heading, values, cmap, maximum, label) in zip(axes.flat[:3], panels):
        _base_map(ax, bounds)
        # Zero has no meaningful color; weak >0 signals remain visible.
        colored = np.ma.masked_where(~np.isfinite(values) | (values <= 0), values)
        mesh = ax.pcolormesh(rain.longitude.values, rain.latitude.values,
                             colored, shading="auto", cmap=cmap, vmin=0,
                             vmax=maximum, transform=ccrs.PlateCarree())
        cb = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.018, extend="max")
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(heading, fontsize=11, weight="bold")

    status_ax = axes.flat[3]
    _base_map(status_ax, bounds)
    colors = ["#efc64c", "#e36a26", "#9e57c5", "#218c57"]
    labels = ["Score below 15", "Rainfall gate failed",
              "Isolated 1-2 cells removed", "Retained in FFPI"]
    status_cmap = ListedColormap(colors)
    status_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], status_cmap.N)
    status_ax.pcolormesh(
        rain.longitude.values, rain.latitude.values,
        np.ma.masked_where(classification == 0, classification),
        cmap=status_cmap, norm=status_norm, shading="auto",
        transform=ccrs.PlateCarree(),
    )
    status_ax.set_title("D. Why FFPI colors remain or disappear", fontsize=11, weight="bold")
    status_ax.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, labels)],
        fontsize=8, loc="lower left", framealpha=0.9,
    )

    fig.text(
        0.5, 0.027,
        "Rain gate: accumulation >=10 mm OR peak rate >=5 mm/h. "
        "Published FFPI additionally requires score >=15 and >=3 connected cells.\n"
        "Panels A-C use the SAME ECMWF 72h forecast, without FFPI masking. "
        "A-C are diagnostic, NOT flood warnings.",
        ha="center", va="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94), h_pad=2.3, w_pad=1.2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    counts = np.bincount(classification.ravel(), minlength=5)
    print(f"Diagnostic {name}: class counts (none, score, gate, noise, shown) = "
          f"{counts[:5].tolist()}; saved {path}", flush=True)


def create_diagnostic_maps(outdir, run_time, rainfall_mm, max_rate_mmh,
                           unfiltered_ffpi, rain_gate, published_ffpi):
    """Render the same data for Kuwait and the wider Middle East."""
    for name, (bounds, filename) in DIAGNOSTIC_TARGETS.items():
        render_diagnostic_map(
            Path(outdir) / filename, name.replace("_", " "), bounds, run_time,
            rainfall_mm, max_rate_mmh, unfiltered_ffpi, rain_gate,
            published_ffpi,
        )
