"""Optional JMA comparison maps; failures must never block the ECMWF product."""

import logging
import os
from pathlib import Path
import struct
import tempfile
import time
from urllib.request import Request, urlopen
import zlib

LOGGER = logging.getLogger(__name__)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def valid_png(path):
    """Check the complete PNG container, chunk CRCs and basic dimensions."""
    try:
        with open(path, "rb") as image:
            if image.read(8) != PNG_SIGNATURE:
                return False
            seen_header = False
            seen_data = False
            while True:
                length_bytes = image.read(4)
                if len(length_bytes) != 4:
                    return False
                length = struct.unpack(">I", length_bytes)[0]
                kind = image.read(4)
                if len(kind) != 4 or length > 32 * 1024 * 1024:
                    return False
                payload = image.read(length)
                crc_bytes = image.read(4)
                if len(payload) != length or len(crc_bytes) != 4:
                    return False
                if zlib.crc32(kind + payload) != struct.unpack(">I", crc_bytes)[0]:
                    return False
                if not seen_header:
                    if kind != b"IHDR" or length != 13:
                        return False
                    width, height = struct.unpack(">II", payload[:8])
                    if not width or not height:
                        return False
                    seen_header = True
                if kind == b"IDAT":
                    seen_data = True
                if kind == b"IEND":
                    return length == 0 and seen_data and image.read(1) == b""
    except (OSError, ValueError, struct.error):
        return False


def download_jma_map(url, destination, attempts=3, timeout=15, delay=1):
    """Atomically replace a map only after validating a complete PNG."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".png", delete=False) as tmp:
                temporary = Path(tmp.name)
                with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as response:
                    while chunk := response.read(65536):
                        tmp.write(chunk)
            if not valid_png(temporary):
                raise ValueError("invalid or incomplete PNG")
            os.replace(temporary, destination)
            LOGGER.info("JMA updated: %s", destination)
            return True
        except (OSError, ValueError) as exc:
            LOGGER.warning("JMA attempt %d/%d failed for %s: %s", attempt + 1, attempts, destination, exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if attempt + 1 < attempts:
            time.sleep(delay * 2 ** attempt)

    if valid_png(destination):
        LOGGER.warning("JMA unavailable; retaining previous valid map: %s (may be stale)", destination)
    else:
        destination.unlink(missing_ok=True)
        LOGGER.warning("JMA unavailable; skipping map: %s", destination)
    return False


def download_jma_maps(maps, **kwargs):
    for url, destination in maps:
        download_jma_map(url, destination, **kwargs)
