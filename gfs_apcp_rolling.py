"""Decode NOAA GFS APCP fields with variable accumulation windows.

A forecast at +6h may encode 0–6h accumulation rather than 3–6h.
Convert every source window to since-cycle cumulative before differencing.
"""
import numpy as np
from tempfile import TemporaryFile


def decode_gfs_apcp(blob, expected_end):
    """Read authentic GRIB APCP ending at forecast hour expected_end."""
    import eccodes
    if not blob.startswith(b"GRIB"):
        raise ValueError("NOAA response is not a GRIB")
    found=[]
    with TemporaryFile() as handle:
        handle.write(blob)
        handle.seek(0)
        while msg:=eccodes.codes_grib_new_from_file(handle):
            try:
                if (eccodes.codes_get(msg,"shortName")!="tp" or
                    eccodes.codes_get(msg,"stepType")!="accum" or
                    eccodes.codes_get(msg,"typeOfLevel")!="surface"):
                    continue
                a=int(eccodes.codes_get(msg,"startStep"))
                b=int(eccodes.codes_get(msg,"endStep"))
                if b!=expected_end or a<0 or a>=b:
                    continue
                ny,nx=int(eccodes.codes_get(msg,"Nj")),int(eccodes.codes_get(msg,"Ni"))
                raw=np.asarray(eccodes.codes_get_array(msg,"values")).reshape(ny,nx)
                missing=float(eccodes.codes_get(msg,"missingValue"))
                rain=np.where(np.isclose(raw,missing),np.nan,raw)
                lons=np.asarray(eccodes.codes_get_array(msg,"distinctLongitudes"))
                lats=np.asarray(eccodes.codes_get_array(msg,"distinctLatitudes"))
                found.append((a,lons,lats,rain))
            finally:
                eccodes.codes_release(msg)
    if not found:
        raise ValueError(f"No GFS APCP ending at +{expected_end}")
    # The shortest valid window, if multiple APCP messages are supplied.
    return max(found,key=lambda record:record[0])


def advance_gfs_cumulative(cumulative, previous_end, source_start, end, source_rain):
    """Extract exactly previous_end..end from accumulated source_start..end.

    +6h APCP with 0–6h step range minus existing 0–3h cumulative yields 3–6h.
    Accept source_start only if its verified cumulative total is available.
    """
    if source_start not in cumulative or previous_end not in cumulative:
        raise ValueError(
            f"Cannot derive +{previous_end}–+{end} from +{source_start}–+{end}")
    if cumulative[source_start].shape!=source_rain.shape:
        raise ValueError("GFS grid shape changed")
    total=cumulative[source_start]+source_rain
    segment=total-cumulative[previous_end]
    if not np.isfinite(segment).all() or (segment < -0.5).any():
        raise ValueError(f"Invalid negative/missing increment at +{end}")
    cumulative[end]=total
    return np.maximum(segment,0)
