import json
import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import neat
import pygame

from racing_core import (
    Car, ModelRecord, Storage, Tile, TrackDefinition, campaign_tracks,
    create_track_runtime, deserialize_genome, load_neat_config, serialize_genome,
    piece_atlas, validate_track,
)


class TrackTests(unittest.TestCase):
    def test_campaign_has_ten_valid_tracks(self):
        tracks = campaign_tracks()
        self.assertEqual(10, len(tracks))
        for track in tracks:
            errors, path = validate_track(track)
            self.assertEqual([], errors)
            if track.runtime_type == "component":
                self.assertEqual(len(track.tiles), len(path))
            runtime = create_track_runtime(track)
            self.assertEqual(3, len(runtime.checkpoints))

    def test_first_lap_uses_legacy_geometry(self):
        track = campaign_tracks()[0]
        runtime = create_track_runtime(track)
        self.assertEqual("legacy_bitmap", track.runtime_type)
        self.assertEqual((900, 750), runtime.size)
        self.assertEqual((100, 170), runtime.spawn_position)
        self.assertEqual((24, 25), runtime.origin)
        image = pygame.transform.smoothscale(
            pygame.image.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "imgs", "WhiteCar.png")),
            (32, 64),
        )
        car = Car(image, runtime)
        finished, _, _ = car.update_progress(runtime)
        self.assertFalse(finished)
        self.assertFalse(car.finish_armed)
        car.x, car.y = 90, 80
        finished, _, _ = car.update_progress(runtime)
        self.assertFalse(finished)
        self.assertTrue(car.finish_armed)
        car.x, car.y = 90, 250
        self.assertTrue(car.finish_overlap(runtime))
        finished, _, _ = car.update_progress(runtime)
        self.assertTrue(finished)
        self.assertEqual(0, car.next_checkpoint)

    def test_component_track_cannot_finish_before_leaving_start(self):
        runtime = create_track_runtime(campaign_tracks()[1])
        image = pygame.Surface((32, 64), pygame.SRCALPHA)
        image.fill((255, 255, 255, 255))
        car = Car(image, runtime)
        finished, _, _ = car.update_progress(runtime)
        self.assertFalse(finished)
        self.assertFalse(car.finish_armed)

    def test_piece_rotation_changes_ports(self):
        tile = Tile(0, 0, "corner", 0)
        self.assertEqual(("N", "E"), tile.ports)
        tile.rotation = 90
        self.assertEqual(("E", "S"), tile.ports)
        tile.rotation = 180
        self.assertEqual(("S", "W"), tile.ports)
        tile.rotation = 270
        self.assertEqual(("W", "N"), tile.ports)

    def test_source_piece_rotations_are_cached(self):
        atlas = piece_atlas()
        straight = atlas.surface("straight", 0)
        self.assertIs(straight, atlas.surface("straight", 0))
        self.assertEqual((64, 64), atlas.surface("corner", 90).get_size())

    def test_invalid_track_is_rejected(self):
        errors, path = validate_track(TrackDefinition("Broken", [Tile(1, 1, "straight")]))
        self.assertTrue(errors)
        self.assertEqual([], path)


class PersistenceTests(unittest.TestCase):
    def test_genome_json_round_trip_preserves_outputs(self):
        config = load_neat_config(os.path.dirname(os.path.dirname(__file__)))
        population = neat.Population(config)
        genome = next(iter(population.population.values()))
        before = neat.nn.FeedForwardNetwork.create(genome, config).activate((.1, .2, .3, .4, .5))
        encoded = json.loads(json.dumps(serialize_genome(genome)))
        restored = deserialize_genome(encoded)
        after = neat.nn.FeedForwardNetwork.create(restored, config).activate((.1, .2, .3, .4, .5))
        self.assertEqual(before, after)

    def test_atomic_model_track_and_progress_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(directory)
            config = load_neat_config(os.path.dirname(os.path.dirname(__file__)))
            genome = next(iter(neat.Population(config).population.values()))
            model = ModelRecord("Test", "white", serialize_genome(genome), 1, 2.5)
            storage.save_model(model)
            self.assertEqual("Test", storage.models()[0].name)
            track = campaign_tracks()[0]
            track.source = "custom"; track.track_id = "custom-test"
            storage.save_track(track)
            self.assertEqual("custom-test", storage.custom_tracks()[0].track_id)
            progress = {"unlocked": 4, "completed": {}, "best_times": {}}
            storage.save_progress(progress)
            self.assertEqual(4, storage.progress()["unlocked"])


if __name__ == "__main__":
    pygame.init()
    unittest.main()
