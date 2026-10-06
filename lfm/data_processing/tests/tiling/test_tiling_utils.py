from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lfm.data_processing.tiling import resolve_notebook_source_index


class NotebookSourceIndexResolutionTestCase(unittest.TestCase):
    def test_default_sources_use_existing_protected_shared_geopackage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache_dir = root / "cache"
            for source_name in ("wac", "nac", "static"):
                with self.subTest(source_name=source_name):
                    data_dir = root / source_name
                    data_dir.mkdir()
                    shared_index = data_dir / "output_index.gpkg"
                    shared_index.touch()

                    resolution = resolve_notebook_source_index(
                        source_name=source_name,
                        data_dir=data_dir,
                        cache_dir=cache_dir,
                        default_data_dir=data_dir,
                    )

                    self.assertEqual(resolution.index_path, shared_index)
                    self.assertTrue(resolution.uses_shared_default)
                    self.assertFalse(resolution.rebuild_invalid_index)

    def test_custom_source_uses_replaceable_per_clone_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            custom_data_dir = root / "custom-wac"
            resolution = resolve_notebook_source_index(
                source_name="wac",
                data_dir=custom_data_dir,
                cache_dir=root / "cache",
                default_data_dir=root / "canonical-wac",
            )

            self.assertEqual(
                resolution.index_path,
                root / "cache" / "wac_index.gpkg",
            )
            self.assertFalse(resolution.uses_shared_default)
            self.assertTrue(resolution.rebuild_invalid_index)

    def test_default_source_requires_shared_geopackage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "nac"
            data_dir.mkdir()

            with self.assertRaisesRegex(
                FileNotFoundError,
                "default NAC directory requires its shared raster index",
            ):
                resolve_notebook_source_index(
                    source_name="nac",
                    data_dir=data_dir,
                    cache_dir=Path(temporary_directory) / "cache",
                    default_data_dir=data_dir,
                )

    def test_unknown_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "source_name must be one of"):
                resolve_notebook_source_index(
                    source_name="altimetry",  # type: ignore[arg-type]
                    data_dir=Path(temporary_directory) / "data",
                    cache_dir=Path(temporary_directory) / "cache",
                )


if __name__ == "__main__":
    unittest.main()
