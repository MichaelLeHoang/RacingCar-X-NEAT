import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from storage import Storage
from test_core import valid_track


class StorageCompatibilityTests(unittest.TestCase):
    def test_v1_track_loads_for_repair_but_is_excluded_from_valid_list(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(directory)
            broken = {"schema_version": 1, "name": "Old Broken", "tiles": [
                {"x": 1, "y": 1, "kind": "straight", "rotation": 0}
            ]}
            (storage.tracks_dir / "old.rctrack").write_text(json.dumps(broken), encoding="utf-8")
            self.assertEqual(1, len(storage.custom_tracks()))
            self.assertEqual(0, len(storage.custom_tracks(valid_only=True)))

    def test_v2_track_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(directory); track = valid_track("Round Trip")
            storage.save_track(track)
            loaded = storage.custom_tracks(valid_only=True)[0]
            self.assertEqual(track.signature(), loaded.signature())
            self.assertEqual(2, loaded.schema_version)

    def test_invalid_import_is_rejected_without_deleting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(directory)
            path = storage.imports_dir / "bad.rctrack"
            path.write_text(json.dumps({"schema_version": 1, "name": "Bad", "tiles": []}), encoding="utf-8")
            report = storage.import_inbox()
            self.assertEqual(0, report.imported)
            self.assertEqual(1, len(report.issues))
            self.assertTrue(path.exists())
            self.assertEqual([], storage.custom_tracks())

    def test_track_export_is_non_executable_json(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(directory); track = valid_track("Exported")
            path = storage.export_track(track)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("Exported", data["name"])
            self.assertEqual(2, data["schema_version"])


if __name__ == "__main__":
    unittest.main()
