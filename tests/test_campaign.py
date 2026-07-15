import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import neat
import pygame

from campaign import CampaignProgress
from interactive_app import BASE_DIR, InteractiveApp
from race_session import RaceState
from racing_core import ModelRecord, load_neat_config, serialize_genome
from training_session import TrainingMode


def make_model(name="Campaign Racer", controller="five-sensor-v1"):
    config = load_neat_config(BASE_DIR)
    genome = next(iter(neat.Population(config).population.values()))
    model = ModelRecord(name, "white", serialize_genome(genome), 1, 1.0)
    model.controller_version = controller
    return model


class CampaignInteractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        os.environ["RACING_DATA_DIR"] = self.directory.name
        self.app = InteractiveApp()

    def tearDown(self):
        pygame.quit()
        self.directory.cleanup()

    def test_locked_level_cannot_open_and_selection_does_not_start_timer(self):
        self.assertFalse(self.app.prepare_race(self.app.campaign[1], 2))
        self.assertIsNone(self.app.race_session)
        self.assertTrue(self.app.prepare_race(self.app.campaign[0], 1))
        self.assertEqual(RaceState.PREPARING, self.app.race_session.state)
        self.assertIsNone(self.app.race_session.started_at)
        self.assertEqual(0.0, self.app.race_session.elapsed_time)

    def test_race_begins_only_after_valid_drop_and_countdown(self):
        model = make_model(); self.app.storage.save_model(model); self.app.refresh_models(model.model_id)
        self.app.prepare_race(self.app.campaign[0], 1)
        self.assertIsNone(self.app.race_session.car)
        self.assertTrue(self.app.drop_model(self.app.selected_model()))
        self.assertEqual(RaceState.COUNTDOWN, self.app.race_session.state)
        self.assertEqual(0.0, self.app.race_session.elapsed_time)
        for _ in range(48): self.app.update_race()
        self.assertEqual(RaceState.RUNNING, self.app.race_session.state)
        self.assertIsNotNone(self.app.race_session.started_at)
        self.app.update_race()
        self.assertGreater(self.app.race_session.elapsed_time, 0)

    def test_incompatible_drop_is_rejected_without_creating_car(self):
        model = make_model(controller="future-v2")
        self.app.prepare_race(self.app.campaign[0], 1)
        self.assertFalse(self.app.drop_model(model))
        self.assertEqual(RaceState.PREPARING, self.app.race_session.state)
        self.assertIsNone(self.app.race_session.car)

    def test_inventory_pagination_rename_delete_and_count_share_storage(self):
        for index in range(7): self.app.storage.save_model(make_model(f"Model {index + 1}"))
        self.app.refresh_models()
        self.assertEqual(7, len(self.app.models)); self.assertEqual(2, self.app.inventory_pages())
        self.assertEqual(6, len(self.app.model_cards()))
        self.app.inventory_page = 1
        model = self.app.model_cards()[0][1]
        self.app.begin_rename(model); self.app.rename_text = "Renamed"; self.app.confirm_rename()
        self.assertEqual("Renamed", self.app.selected_model().name)
        self.app.delete_model = self.app.selected_model(); self.app.confirm_delete_model()
        self.assertEqual(6, len(self.app.models)); self.assertEqual(6, len(self.app.storage.models()))

    def test_play_and_train_inventory_page_over_the_same_model_records(self):
        for index in range(7): self.app.storage.save_model(make_model(f"Shared {index + 1}"))
        self.app.refresh_models()
        play_ids = [model.model_id for _, model in self.app.model_cards()]
        train_ids = [model.model_id for _, model in self.app.training_inventory_cards()]
        self.assertEqual(play_ids, train_ids)
        self.assertEqual(7, len(self.app.models))
        cards = [rect for rect, _ in self.app.training_inventory_cards()]
        layout = self.app.training_inventory_layout()
        self.assertLess(max(rect.bottom for rect in cards), layout["actions_y"])

    def test_small_training_inventory_uses_compact_non_overlapping_layout(self):
        for index in range(3): self.app.storage.save_model(make_model(f"Compact {index + 1}"))
        self.app.refresh_models()
        cards = [rect for rect, _ in self.app.training_inventory_cards()]
        layout = self.app.training_inventory_layout()
        self.assertEqual(545, layout["modal_height"])
        self.assertLess(max(rect.bottom for rect in cards), layout["actions_y"])

    def test_train_this_car_uses_failed_campaign_level_and_saved_genome(self):
        model = make_model(); self.app.storage.save_model(model); self.app.refresh_models(model.model_id)
        self.app.progress_record.record_completion(1, 10.0)
        failed_track = self.app.campaign[1]
        self.app.start_race(failed_track, self.app.selected_model(), 2)
        self.app.race_session._finish(RaceState.CRASHED)
        self.app.logical_mouse = lambda: (620, 470)
        self.app.event_race(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1))
        self.assertEqual("train", self.app.scene)
        self.assertEqual(model.model_id, self.app.training_seed.model_id)
        self.assertEqual(TrainingMode.CUSTOM, self.app.training_mode)
        self.assertEqual([failed_track.track_id],
                         [track.track_id for track in self.app._configured_training_tracks()])
        self.assertEqual(failed_track.track_id, self.app.training_target_track.track_id)

    def test_completion_freezes_time_saves_best_and_unlocks_next(self):
        model = make_model(); self.app.storage.save_model(model); self.app.refresh_models(model.model_id)
        self.app.start_race(self.app.campaign[0], self.app.selected_model(), 1)
        session = self.app.race_session
        session.state = RaceState.RUNNING; session.elapsed_time = 12.5
        session._finish(RaceState.COMPLETE)
        self.app.update_race()
        frozen = session.displayed_time
        for _ in range(10): self.app.update_race()
        self.assertEqual(frozen, session.displayed_time)
        self.assertEqual(2, self.app.progress_record.unlocked)
        self.assertEqual(12.5, self.app.progress_record.best_times["1"])
        self.app.start_next_level()
        self.assertEqual(RaceState.PREPARING, self.app.race_session.state)
        self.assertEqual(2, self.app.race_session.level)

    def test_campaign_progress_only_unlocks_sequentially(self):
        progress = CampaignProgress()
        with self.assertRaises(ValueError): progress.record_completion(2, 4.0)
        self.assertTrue(progress.record_completion(1, 8.0))
        self.assertFalse(progress.record_completion(1, 9.0))
        self.assertEqual(8.0, progress.best_times["1"])


if __name__ == "__main__":
    unittest.main()
