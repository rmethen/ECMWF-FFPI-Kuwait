import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from jma_maps import download_jma_map, download_jma_maps, valid_png


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


PNG = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
    + chunk(b"IEND", b"")
)


class JmaResilienceTests(unittest.TestCase):
    def test_connection_refused_then_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "map.png"
            with patch("jma_maps.urlopen", side_effect=[ConnectionRefusedError(111, "refused"), io.BytesIO(PNG)]) as fetch, patch("jma_maps.time.sleep") as sleep:
                self.assertTrue(download_jma_map("https://example.test/map.png", destination))
            self.assertEqual(fetch.call_count, 2)
            sleep.assert_called_once_with(1)
            self.assertTrue(valid_png(destination))

    def test_persistent_jma_failure_does_not_stop_ffpi_and_preserves_valid_map(self):
        # The script reaches this optional section only after printing ECMWF success.
        source = Path(__file__).resolve().parent.joinpath("ecmwf_ffpi.py").read_text()
        self.assertLess(source.index('print("ECMWF FFPI completed successfully")'), source.index("download_jma_maps(["))
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "previous.png"
            missing = Path(directory) / "missing.png"
            previous.write_bytes(PNG)
            with patch("jma_maps.urlopen", side_effect=ConnectionRefusedError(111, "refused")) as fetch, patch("jma_maps.time.sleep"):
                download_jma_maps([("https://example.test/a", previous), ("https://example.test/b", missing)])
            self.assertEqual(fetch.call_count, 6)
            self.assertEqual(previous.read_bytes(), PNG)
            self.assertFalse(missing.exists())

    def test_invalid_download_cannot_replace_previous_valid_png(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "map.png"
            destination.write_bytes(PNG)
            with patch("jma_maps.urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(b"<html>error</html>")), patch("jma_maps.time.sleep"):
                self.assertFalse(download_jma_map("https://example.test/map.png", destination))
            self.assertEqual(destination.read_bytes(), PNG)


if __name__ == "__main__":
    unittest.main()
