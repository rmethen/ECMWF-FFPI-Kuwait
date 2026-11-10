# Kuwait multi-hazard experimental integration

This branch prepares independent Kuwait-focused layers for flash floods, lightning environment and hail environment, without changing the operational FFPI map.

## Verified existing inputs
- ECMWF FFPI: `ecmwf_ffpi.py` retrieves IFS `tp,tprate,ro,vsw` for +3 to +72 hours.
- GFS FFPI: `gfs_ffpi.py` retrieves APCP, PRATE, WATR and SOILW.
- Existing hail V1: `ecmwf_hail_index_v1.py` retrieves MUCAPE and pressure-level temperature, winds, RH and geopotential height.

## Required implementation checks
1. Diagnose the blank Kuwait FFPI map: log finite-cell count, minimum, maximum, and fraction of FFPI values >=15; do not mistake values below the first plotted contour for missing data.
2. Produce Kuwait-specific separate maps for flash floods, lightning-favourable environment and hail-favourable environment, plus a clearly labelled combined *guidance* overlay.
3. Compute lightning proxy only from available, verified instability/moisture/lift inputs. Never label it observed lightning or claim a calibrated lightning probability.
4. Reuse hail V1 as a baseline, but assess wet-bulb-zero altitude, 700–500 hPa lapse rate, deep-layer shear and forcing before implementing a V2-like Kuwait diagnostic. Never equate V1 with the other repository's V2.
5. For each map, annotate IFS/GFS initialization, valid forecast time, input sources, and missing-data state. Prefer +3/+6/+12/+24 h snapshots where available; do not fabricate +1 h data.
6. Add offline synthetic-data unit tests for all derived layers, mask handling, zero-risk cases, and plotting, then test real GRIB ingestion before publishing.

No real-time lightning or hail observations are currently integrated. These are experimental forecast-environment indices, not warnings.
