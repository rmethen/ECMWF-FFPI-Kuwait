"""Kuwait multi-hazard diagnostic functions (experimental, not calibrated probabilities).

Pure NumPy calculations so these can be tested without network or GRIB access.
Input arrays must share a grid; missing values propagate as NaN.
"""
import numpy as np


def _scale(x, low, high):
    x = np.asarray(x, dtype=float)
    return np.clip((x - low) / (high - low), 0, 1)


def _weighted(parts):
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a, _ in parts])
    weights = [w for _, w in parts]
    out = sum(a * w for a, w in zip(arrays, weights)) / sum(weights)
    valid = np.logical_and.reduce([np.isfinite(a) for a in arrays])
    return np.where(valid, np.clip(out * 100, 0, 100), np.nan)


def flood_guidance(rate_mm_h, accum_6h_mm, runoff_mm, soil_wetness_fraction):
    """Flood-environment guidance (0-100), not a flood probability.

    rate: mm/h; accum_6h: mm/6h; runoff: mm for matching 6h;
    soil wetness: fraction 0..1 (NOT volumetric soil water in m3/m3
    unless converted to a normalized wetness fraction).
    """
    return _weighted([
        (_scale(rate_mm_h, 2, 25), 0.35),
        (_scale(accum_6h_mm, 5, 75), 0.35),
        (_scale(runoff_mm, 0, 25), 0.20),
        (_scale(soil_wetness_fraction, 0.3, 0.95), 0.10),
    ])


def lightning_environment(mucape_j_kg, omega700_pa_s, rh700_percent,
                          lapse_700_500_k_km):
    """Lightning-favourable environment, NOT observed lightning or probability.

    Negative pressure vertical velocity omega denotes ascent.
    """
    return _weighted([
        (_scale(mucape_j_kg, 50, 1800), 0.35),
        (_scale(-np.asarray(omega700_pa_s), 0.03, 0.8), 0.30),
        (_scale(rh700_percent, 40, 90), 0.15),
        (_scale(lapse_700_500_k_km, 5, 8.5), 0.20),
    ])


def hail_environment(mucape_j_kg, wbz_m, lapse_700_500_k_km,
                     shear_0_6km_ms, omega700_pa_s):
    """Hail-favourable environment, NOT hail size or observed hail.

    WBZ in meters above ground level. Penalize very high WBZ, not
    treat very low WBZ as sufficient for hail by itself.
    """
    wbz = np.asarray(wbz_m, dtype=float)
    wbz_score = np.clip((4200 - wbz) / 2200, 0, 1)
    return _weighted([
        (_scale(mucape_j_kg, 100, 1800), 0.20),
        (wbz_score, 0.25),
        (_scale(lapse_700_500_k_km, 5.5, 8.5), 0.25),
        (_scale(shear_0_6km_ms, 8, 28), 0.20),
        (_scale(-np.asarray(omega700_pa_s), 0.03, 0.8), 0.10),
    ])


def diagnostics(field, threshold=15):
    """Explain blank-looking maps when all valid values fall below color levels."""
    a = np.asarray(field, dtype=float)
    valid = a[np.isfinite(a)]
    return {
        "finite_cells": int(valid.size),
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "fraction_above_plot_floor": float(np.mean(valid >= threshold)) if valid.size else None,
    }


def combined_guidance(flood, lightning, hail):
    """Maximum of three diagnostics; only valid where ALL inputs are finite."""
    f, l, h = np.broadcast_arrays(np.asarray(flood, dtype=float),
                                  np.asarray(lightning, dtype=float),
                                  np.asarray(hail, dtype=float))
    return np.where(np.isfinite(f) & np.isfinite(l) & np.isfinite(h),
                    np.maximum(np.maximum(f, l), h), np.nan)
