import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from interactive_app import ASSET_DIR, InteractiveApp, load_car_sprite
from test_core import valid_track


class EditorInteractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); os.environ["RACING_DATA_DIR"] = self.directory.name
        self.app = InteractiveApp()

    def tearDown(self):
        pygame.quit(); self.directory.cleanup()

    def test_car_sprites_share_size_and_transparent_corners(self):
        for filename in ("WhiteCar.png", "RedCar.png", "green-car.png", "purple-car.png", "grey-car.png"):
            sprite = load_car_sprite(ASSET_DIR / filename)
            self.assertEqual((32, 64), sprite.get_size())
            self.assertEqual(0, sprite.get_at((0, 0)).a)

    def test_multiple_training_profiles_keep_independent_configuration(self):
        self.app.model_name = "Speedster"; self.app.training_skin = "red"; self.app.add_training_profile()
        self.app.model_name = "Corner Pro"; self.app.training_skin = "green"; self.app.switch_training_profile(0)
        self.assertEqual("Speedster", self.app.model_name); self.assertEqual("red", self.app.training_skin)
        self.app.switch_training_profile(1)
        self.assertEqual("Corner Pro", self.app.model_name); self.assertEqual("green", self.app.training_skin)

    def test_placement_overwrite_requires_confirmation(self):
        self.app._place_editor_tile((3, 4), "straight", 0)
        original = self.app.editor_tiles[(3, 4)]
        self.app._place_editor_tile((3, 4), "corner", 90)
        self.assertIsNotNone(self.app.pending_replace)
        self.assertIs(original, self.app.editor_tiles[(3, 4)])

    def test_undo_and_redo_cover_placement(self):
        self.app._place_editor_tile((3, 4), "straight", 0)
        self.assertIn((3, 4), self.app.editor_tiles)
        self.app._undo(); self.assertNotIn((3, 4), self.app.editor_tiles)
        self.app._redo(); self.assertIn((3, 4), self.app.editor_tiles)

    def test_invalid_track_cannot_save_and_valid_track_appears_immediately(self):
        self.app._place_editor_tile((1, 1), "straight", 0)
        self.app.save_editor_track()
        self.assertEqual([], self.app.storage.custom_tracks())
        track = valid_track("Saved Circuit")
        self.app.editor_tiles = {tile.cell: tile for tile in track.tiles}
        self.app.editor_name = track.name
        self.app.save_editor_track()
        self.assertEqual("Saved Circuit", self.app.custom[0].name)

    def test_testing_valid_track_enters_preparation_without_saving(self):
        track = valid_track("Unsaved Test")
        self.app.editor_tiles = {tile.cell: tile for tile in track.tiles}; self.app.editor_name = track.name
        before = len(self.app.storage.custom_tracks())
        self.app.prepare_race(self.app.editor_track())
        self.assertEqual("race", self.app.scene)
        self.assertEqual("PREPARING", self.app.race_session.state.value)
        self.assertEqual(before, len(self.app.storage.custom_tracks()))


if __name__ == "__main__":
    unittest.main()
