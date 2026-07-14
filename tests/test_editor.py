import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from interactive_app import ASSET_DIR, InteractiveApp, load_car_sprite


class EditorInteractionTests(unittest.TestCase):
    def test_car_sprites_share_size_and_transparent_corners(self):
        pygame.display.set_mode((1, 1))
        for filename in ("WhiteCar.png", "RedCar.png", "green-car.png", "purple-car.png", "grey-car.png"):
            sprite = load_car_sprite(ASSET_DIR / filename)
            self.assertEqual((32, 64), sprite.get_size())
            self.assertEqual(0, sprite.get_at((0, 0)).a)

    def test_multiple_training_profiles_keep_independent_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["RACING_DATA_DIR"] = directory
            app = InteractiveApp()
            app.model_name = "Speedster"; app.training_skin = "red"
            app.add_training_profile()
            app.model_name = "Corner Pro"; app.training_skin = "green"
            app.switch_training_profile(0)
            self.assertEqual("Speedster", app.model_name)
            self.assertEqual("red", app.training_skin)
            app.switch_training_profile(1)
            self.assertEqual("Corner Pro", app.model_name)
            self.assertEqual("green", app.training_skin)
            pygame.quit()

    def test_drag_piece_and_rotate_with_r(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["RACING_DATA_DIR"] = directory
            app = InteractiveApp()
            app.logical_mouse = lambda: (1000, 210)
            app.event_editor(pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=(1000, 210)
            ))
            self.assertEqual("straight", app.editor_drag_kind)
            app.event_editor(pygame.event.Event(
                pygame.KEYDOWN, key=pygame.K_r, unicode="r", mod=0
            ))
            app.logical_mouse = lambda: (42 + 3 * 64 + 20, 105 + 4 * 64 + 20)
            app.event_editor(pygame.event.Event(
                pygame.MOUSEBUTTONUP, button=1, pos=(0, 0)
            ))
            self.assertEqual("straight", app.editor_tiles[(3, 4)].kind)
            self.assertEqual(90, app.editor_tiles[(3, 4)].rotation)
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
