import copy
import os
import tempfile
import unittest
import warnings

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import neat
import pygame

from campaign import campaign_tracks
from interactive_app import BASE_DIR, CAR_SPECS, InteractiveApp
from racing_core import ModelRecord, load_neat_config, serialize_genome
from training_session import TrainingMode, TrainingProfile, TrainingSession, TrainingState


class TrainingSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init(); pygame.display.set_mode((1, 1))
        cls.sprites = {name: pygame.Surface((32, 64), pygame.SRCALPHA) for name in CAR_SPECS}
        for sprite in cls.sprites.values(): pygame.draw.rect(sprite, (255, 255, 255), (6, 2, 20, 60))

    def population(self):
        return neat.Population(load_neat_config(BASE_DIR))

    def test_multi_track_aggregation_prefers_generalization(self):
        tracks = [copy.deepcopy(campaign_tracks()[0]), copy.deepcopy(campaign_tracks()[0])]
        session = TrainingSession(self.population(), TrainingProfile(TrainingMode.CUSTOM, "white", tracks=tracks),
                                  self.sprites, CAR_SPECS)
        keys = list(session.population.population)
        session.track_scores = {key: [0.0, 0.0] for key in keys}
        session.track_scores[keys[0]] = [100.0, 0.0]
        session.track_scores[keys[1]] = [45.0, 45.0]
        session._finish_generation()
        self.assertEqual(keys[1], session.candidate_champion.key)

    def test_completing_every_track_dominates_partial_suite_speed(self):
        tracks = [campaign_tracks()[1], campaign_tracks()[2]]
        session = TrainingSession(self.population(), TrainingProfile(TrainingMode.CUSTOM, "white", tracks=tracks),
                                  self.sprites, CAR_SPECS)
        keys = list(session.population.population)
        session.generation_tracks = tracks
        session.track_scores = {key: [0.0, 0.0] for key in keys}
        session.track_completions = {key: [False, False] for key in keys}
        session.track_scores[keys[0]] = [50_000.0, 50_000.0]
        session.track_completions[keys[0]] = [True, False]
        session.track_scores[keys[1]] = [10_000.0, 10_000.0]
        session.track_completions[keys[1]] = [True, True]
        session._finish_generation()
        self.assertEqual(keys[1], session.candidate_champion.key)

    def test_faster_all_track_champion_wins_after_completion_is_equal(self):
        tracks = [campaign_tracks()[1], campaign_tracks()[2]]
        session = TrainingSession(self.population(), TrainingProfile(TrainingMode.CUSTOM, "white", tracks=tracks),
                                  self.sprites, CAR_SPECS)
        keys = list(session.population.population)
        session.generation_tracks = tracks
        session.track_scores = {key: [0.0, 0.0] for key in keys}
        session.track_completions = {key: [False, False] for key in keys}
        session.track_scores[keys[0]] = [12_000.0, 12_000.0]
        session.track_scores[keys[1]] = [18_000.0, 18_000.0]
        session.track_completions[keys[0]] = [True, True]
        session.track_completions[keys[1]] = [True, True]
        session._finish_generation()
        self.assertEqual(keys[1], session.candidate_champion.key)

    def test_one_track_success_does_not_validate_multi_track_scope(self):
        tracks = [campaign_tracks()[0], campaign_tracks()[1]]
        session = TrainingSession(self.population(), TrainingProfile(TrainingMode.CUSTOM, "white", tracks=tracks),
                                  self.sprites, CAR_SPECS)
        session.validation_results = [{"track_id": tracks[0].track_id, "passed": True}]
        self.assertFalse(session.champion_validated)

    def test_random_curriculum_uses_separate_held_out_seeds(self):
        profile = TrainingProfile(TrainingMode.RANDOM_CURRICULUM, "white", base_seed=77,
                                  difficulty_range=(3, 6))
        session = TrainingSession(self.population(), profile, self.sprites, CAR_SPECS)
        training = session._curriculum_tracks()
        held_out = session._held_out_tracks()
        training_seeds = {track.generation["seed"] for track in training}
        held_seeds = {track.generation["seed"] for track in held_out}
        self.assertEqual(3, len(held_seeds))
        self.assertTrue(training_seeds.isdisjoint(held_seeds))
        self.assertEqual(held_seeds, set(session.validation_scope["held_out_seeds"]))
        self.assertEqual(77, session.validation_scope["base_seed"])

    def test_stop_discards_partial_generation_and_preserves_completed_champion(self):
        population = self.population()
        previous = copy.deepcopy(next(iter(population.population.values())))
        previous.fitness = 12.0
        population.best_genome = previous
        track = copy.deepcopy(campaign_tracks()[0]); track.timeout = .05
        session = TrainingSession(population, TrainingProfile(TrainingMode.ORIGINAL, "white", tracks=[track]),
                                  self.sprites, CAR_SPECS)
        session.start(); session.advance(1); session.stop()
        self.assertEqual(TrainingState.STOPPED, session.state)
        self.assertEqual(previous.key, session.completed_champion.key)
        self.assertEqual(12.0, session.completed_champion.fitness)

    def test_advance_is_bounded_and_pause_does_not_step(self):
        track = copy.deepcopy(campaign_tracks()[0]); track.timeout = .05
        session = TrainingSession(self.population(), TrainingProfile(TrainingMode.ORIGINAL, "white", tracks=[track]),
                                  self.sprites, CAR_SPECS)
        session.start(); before = [entry.car.elapsed for entry in session.active]
        session.advance(1); after = [entry.car.elapsed for entry in session.active]
        self.assertTrue(all(value <= 1 / 60 for value in after))
        session.pause(); paused = [entry.car.elapsed for entry in session.active]
        session.advance(20)
        self.assertEqual(paused, [entry.car.elapsed for entry in session.active])


class TrainingStorageStatusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); os.environ["RACING_DATA_DIR"] = self.directory.name
        self.app = InteractiveApp()

    def tearDown(self):
        pygame.quit(); self.directory.cleanup()

    def test_save_current_best_is_draft(self):
        champion = next(iter(neat.Population(load_neat_config(BASE_DIR)).population.values()))
        champion.fitness = 8.0
        self.app.champion = champion; self.app.training_generation = 1
        self.app.save_champion(force_draft=True)
        self.assertEqual("draft", self.app.storage.models()[0].status)

    def test_validated_save_is_blocked_without_complete_suite(self):
        champion = next(iter(neat.Population(load_neat_config(BASE_DIR)).population.values()))
        champion.fitness = 8.0
        self.app.champion = champion
        self.app.save_champion(force_validated=True)
        self.assertEqual([], self.app.storage.models())

    def test_start_switches_the_train_scene_to_live_simulation_rendering(self):
        self.app.start_training()
        self.assertEqual(TrainingState.RUNNING, self.app.training_session.state)
        called = []
        self.app.draw_training_live = lambda: called.append(True)
        self.app.draw_train()
        self.assertEqual([True], called)

    def test_continue_training_restores_genome_generation_and_track_scope(self):
        genome = next(iter(neat.Population(load_neat_config(BASE_DIR)).population.values()))
        track = self.app.campaign[2]
        model = ModelRecord("Continuing Racer", "purple", serialize_genome(genome), 7, 12.5,
                            trained_tracks=[track.track_id],
                            validation_scope={"mode": "custom", "track_ids": [track.track_id],
                                              "difficulty_range": [2, 5], "base_seed": 88})
        self.app.storage.save_model(model); self.app.refresh_models(model.model_id)
        self.assertTrue(self.app.continue_training_model(self.app.selected_model()))
        self.assertEqual("Continuing Racer", self.app.model_name)
        self.assertEqual("purple", self.app.training_skin)
        self.assertEqual(TrainingMode.CUSTOM, self.app.training_mode)
        self.assertEqual([track.track_id],
                         [item.track_id for item in self.app._configured_training_tracks()])
        self.app.init_population()
        self.assertEqual(7, self.app.population.generation)
        self.assertEqual(12.5, self.app.population.best_genome.fitness)
        self.app.start_training(); self.app.training_session.stop()
        self.assertEqual(12.5, self.app.training_session.completed_champion.fitness)

    def test_random_stop_opens_save_choices_and_draft_saves_champion(self):
        population = neat.Population(load_neat_config(BASE_DIR))
        champion = copy.deepcopy(next(iter(population.population.values())))
        champion.fitness = 42.5
        population.best_genome = copy.deepcopy(champion)
        profile = TrainingProfile(TrainingMode.RANDOM_CURRICULUM, "white", base_seed=19)
        session = TrainingSession(population, profile, self.app.skins, CAR_SPECS)
        session.completed_champion = champion
        session.state = TrainingState.RUNNING
        self.app.training_session = session
        self.app.logical_mouse = lambda: (1080, 400)
        self.app.event_train(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1))
        self.assertEqual(TrainingState.STOPPED, session.state)
        self.assertTrue(self.app.show_training_save_modal)
        self.assertEqual(42.5, self.app.champion.fitness)

        self.app.logical_mouse = lambda: (390, 480)
        self.app.event_train(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1))
        self.assertFalse(self.app.show_training_save_modal)
        self.assertEqual("draft", self.app.storage.models()[0].status)

    def test_stop_before_first_completed_generation_explains_save_is_unavailable(self):
        population = neat.Population(load_neat_config(BASE_DIR))
        profile = TrainingProfile(TrainingMode.RANDOM_CURRICULUM, "white", base_seed=23)
        session = TrainingSession(population, profile, self.app.skins, CAR_SPECS)
        session.completed_champion = None
        session.state = TrainingState.RUNNING
        self.app.training_session = session
        self.app.stop_training()
        self.assertTrue(self.app.show_training_save_modal)
        self.assertIsNone(self.app.champion)
        self.app.background()
        self.app.draw_train()

    def test_save_best_and_validated_replace_same_main_track_inventory_car(self):
        population = neat.Population(load_neat_config(BASE_DIR))
        champion = copy.deepcopy(next(iter(population.population.values())))
        champion.fitness = 123.0
        track = self.app.campaign[0]
        profile = TrainingProfile(TrainingMode.ORIGINAL, "white", tracks=[track])
        session = TrainingSession(population, profile, self.app.skins, CAR_SPECS)
        session.completed_champion = champion
        self.app.training_session = session
        self.app.training_mode = TrainingMode.ORIGINAL

        self.app.training_generation = 4
        first = self.app.save_champion(force_draft=True)
        self.app.training_generation = 5
        second = self.app.save_champion(force_draft=True)
        self.assertEqual(first.model_id, second.model_id)
        self.assertEqual(1, len(self.app.storage.models()))
        self.assertEqual(5, self.app.storage.models()[0].generation)

        session.validation_results = [{"track_id": track.track_id, "passed": True,
                                       "elapsed": 8.0, "reason": "complete"}]
        self.app.training_generation = 6
        validated = self.app.save_champion(force_validated=True)
        models = self.app.storage.models()
        self.assertEqual(1, len(models))
        self.assertEqual("validated", models[0].status)
        self.assertEqual(validated.model_id, models[0].model_id)
        self.assertEqual(self.app.training_lineage_id, models[0].lineage_id)

    def test_continued_model_reserves_saved_innovations_before_mutation(self):
        config = load_neat_config(BASE_DIR)
        source_population = neat.Population(config)
        genome = copy.deepcopy(next(iter(source_population.population.values())))
        genome.mutate_add_node(config.genome_config)
        genome.mutate_add_node(config.genome_config)
        genome.fitness = 10.0
        model = ModelRecord("Innovation Safe", "white", serialize_genome(genome), 3, 10.0)
        self.app.storage.save_model(model); self.app.refresh_models(model.model_id)
        self.app.continue_training_model(self.app.selected_model())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.app.init_population()
        collisions = [warning for warning in caught
                      if "Innovation number collision" in str(warning.message)]
        self.assertEqual([], collisions)
        saved_max = max(connection.innovation for connection in genome.connections.values())
        tracker = self.app.population.reproduction.innovation_tracker
        self.assertGreaterEqual(tracker.get_current_innovation_number(), saved_max)


if __name__ == "__main__":
    unittest.main()
