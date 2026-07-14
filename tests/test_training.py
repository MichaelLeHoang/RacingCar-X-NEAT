import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from interactive_app import InteractiveApp


class ContinuousTrainingTests(unittest.TestCase):
    def test_stop_discards_partial_generation_and_keeps_champion(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["RACING_DATA_DIR"] = directory
            app = InteractiveApp()
            track = app.campaign[0]
            track.timeout = .05
            app.training_tracks = [track]
            app.training_speed = 0
            app.run_training_generation()
            champion = app.champion
            generation = app.training_generation
            app.training_active = True
            app.training_stop_requested = True
            app.run_training_generation()
            self.assertFalse(app.training_active)
            self.assertIsNotNone(app.champion)
            self.assertEqual(generation, app.training_generation)
            self.assertTrue(app.show_training_save_modal)
            pygame.quit()

    def test_validated_generation_opens_save_modal(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["RACING_DATA_DIR"] = directory
            app = InteractiveApp()
            track = app.campaign[0]
            track.timeout = .05
            app.training_tracks = [track]
            app.training_active = True
            app.training_speed = 0
            app.validate_champion = lambda: {track.track_id: True}
            app.run_training_generation()
            self.assertFalse(app.training_active)
            self.assertIsNotNone(app.champion)
            self.assertTrue(app.show_training_save_modal)
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
