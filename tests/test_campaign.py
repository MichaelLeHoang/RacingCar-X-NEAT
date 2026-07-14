import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import neat
import pygame

from interactive_app import BASE_DIR, InteractiveApp
from racing_core import ModelRecord, load_neat_config, serialize_genome


def make_model(name):
    config = load_neat_config(BASE_DIR)
    genome = next(iter(neat.Population(config).population.values()))
    return ModelRecord(name, "white", serialize_genome(genome), 1, 1.0)


class CampaignInteractionTests(unittest.TestCase):
    def test_inventory_pages_rename_and_delete_use_same_models(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["RACING_DATA_DIR"] = directory
            app = InteractiveApp()
            for index in range(7):
                app.storage.save_model(make_model(f"Model {index + 1}"))
            app.refresh_models()

            self.assertEqual(7, len(app.models))
            self.assertEqual(2, app.inventory_pages())
            self.assertEqual(6, len(app.model_cards()))
            app.inventory_page = 1
            self.assertEqual(1, len(app.model_cards()))

            model = app.model_cards()[0][1]
            app.selected_model_id = model.model_id
            app.begin_rename(model)
            app.rename_text = "Renamed Racer"
            app.confirm_rename()
            self.assertEqual("Renamed Racer", app.selected_model().name)

            app.delete_model = app.selected_model()
            app.confirm_delete_model()
            self.assertEqual(6, len(app.models))
            self.assertEqual(6, len(app.storage.models()))
            pygame.quit()

    def test_finish_mask_completion_unlocks_and_starts_next_level(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["RACING_DATA_DIR"] = directory
            app = InteractiveApp()
            model = make_model("Campaign Racer")
            app.storage.save_model(model)
            app.refresh_models(model.model_id)
            model = app.selected_model()
            app.start_race(app.campaign[0], model, 1)

            car = app.race["car"]
            car.finish_armed = True
            car.x, car.y = 90, 250
            app.update_race()

            self.assertEqual("COMPLETE", app.race_result)
            self.assertEqual(2, app.progress["unlocked"])
            self.assertTrue(app.progress["completed"]["1"])
            app.start_next_level()
            self.assertEqual("race", app.scene)
            self.assertEqual(2, app.race["level"])
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
