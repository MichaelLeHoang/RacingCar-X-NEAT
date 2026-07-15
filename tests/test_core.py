import json
import os
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import neat
import pygame

from campaign import campaign_tracks
from racing_core import (
    ModelRecord, Storage, Tile, TrackDefinition, create_track_runtime,
    deserialize_genome, load_neat_config, piece_atlas, serialize_genome,
    track_from_path, validate_track,
)
from track_geometry import OPPOSITE, PORTS, VECTORS


BASE = os.path.dirname(os.path.dirname(__file__))


def rectangle_path(left=1, top=1, right=8, bottom=6):
    return ([(left + 2, top)] + [(x, top) for x in range(left + 3, right + 1)]
            + [(right, y) for y in range(top + 1, bottom + 1)]
            + [(x, bottom) for x in range(right - 1, left - 1, -1)]
            + [(left, y) for y in range(bottom - 1, top - 1, -1)]
            + [(left + 1, top)])


def valid_track(name="Valid"):
    return track_from_path(name, rectangle_path(), 2, source="custom")


class TrackGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((1, 1))

    def test_campaign_is_stable_unique_and_strictly_harder(self):
        tracks = campaign_tracks()
        self.assertEqual(10, len(tracks))
        signatures, scores = set(), []
        for track in tracks:
            result = validate_track(track)
            self.assertTrue(result.valid, result.messages)
            runtime = create_track_runtime(track)
            self.assertGreater(len(runtime.gates), 0)
            signatures.add(track.signature() if track.tiles else ("legacy",))
            scores.append(result.metrics.difficulty_score)
        self.assertEqual(10, len(signatures))
        self.assertEqual(scores, sorted(scores))
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])))
        self.assertEqual(max(validate_track(track).metrics.tile_count for track in tracks[1:]),
                         validate_track(tracks[-1]).metrics.tile_count)

    def test_legacy_bitmap_geometry_is_preserved(self):
        runtime = create_track_runtime(campaign_tracks()[0])
        self.assertEqual((900, 750), runtime.size)
        self.assertEqual((100, 170), runtime.spawn_position)
        self.assertEqual((24, 25), runtime.origin)
        self.assertGreater(len(runtime.gates), 12)

    def test_all_rotations_preserve_dimensions_and_semantics(self):
        atlas = piece_atlas()
        for kind, rotations in PORTS.items():
            for rotation, ports in rotations.items():
                self.assertEqual(2, len(ports))
                for layer in ("visual", "road", "curb", "finish"):
                    self.assertEqual((64, 64), atlas.layer(kind, rotation, layer).get_size())

    def test_every_valid_port_pair_has_continuous_visual_and_road_seam(self):
        atlas = piece_atlas()
        for kind_a, rotations_a in PORTS.items():
            for rotation_a, ports_a in rotations_a.items():
                for direction in ports_a:
                    for kind_b, rotations_b in PORTS.items():
                        for rotation_b, ports_b in rotations_b.items():
                            if OPPOSITE[direction] not in ports_b:
                                continue
                            road_a = pygame.mask.from_surface(atlas.layer(kind_a, rotation_a, "road"))
                            road_b = pygame.mask.from_surface(atlas.layer(kind_b, rotation_b, "road"))
                            visual_a = atlas.layer(kind_a, rotation_a, "visual")
                            visual_b = atlas.layer(kind_b, rotation_b, "visual")
                            if direction == "E":
                                values_a = [road_a.get_at((63, y)) for y in range(64)]
                                values_b = [road_b.get_at((0, y)) for y in range(64)]
                                colors_a = [visual_a.get_at((63, y)) for y in range(64)]
                                colors_b = [visual_b.get_at((0, y)) for y in range(64)]
                                alpha = [(visual_a.get_at((63, y)).a, visual_b.get_at((0, y)).a) for y in range(64)]
                            elif direction == "W":
                                values_a = [road_a.get_at((0, y)) for y in range(64)]
                                values_b = [road_b.get_at((63, y)) for y in range(64)]
                                colors_a = [visual_a.get_at((0, y)) for y in range(64)]
                                colors_b = [visual_b.get_at((63, y)) for y in range(64)]
                                alpha = [(visual_a.get_at((0, y)).a, visual_b.get_at((63, y)).a) for y in range(64)]
                            elif direction == "N":
                                values_a = [road_a.get_at((x, 0)) for x in range(64)]
                                values_b = [road_b.get_at((x, 63)) for x in range(64)]
                                colors_a = [visual_a.get_at((x, 0)) for x in range(64)]
                                colors_b = [visual_b.get_at((x, 63)) for x in range(64)]
                                alpha = [(visual_a.get_at((x, 0)).a, visual_b.get_at((x, 63)).a) for x in range(64)]
                            else:
                                values_a = [road_a.get_at((x, 63)) for x in range(64)]
                                values_b = [road_b.get_at((x, 0)) for x in range(64)]
                                colors_a = [visual_a.get_at((x, 63)) for x in range(64)]
                                colors_b = [visual_b.get_at((x, 0)) for x in range(64)]
                                alpha = [(visual_a.get_at((x, 63)).a, visual_b.get_at((x, 0)).a) for x in range(64)]
                            self.assertEqual(values_a, values_b)
                            self.assertEqual(colors_a, colors_b)
                            self.assertEqual(44, sum(values_a))
                            self.assertTrue(all(a and b for (a, b), road in zip(alpha, values_a) if road))

    def test_unconnected_parallel_straights_do_not_merge(self):
        atlas = piece_atlas()
        mask = pygame.mask.from_surface(atlas.layer("straight", 0, "road"))
        self.assertFalse(any(mask.get_at((0, y)) or mask.get_at((63, y)) for y in range(64)))

    def test_start_orientation_and_drop_area_are_mask_contained(self):
        track = valid_track()
        runtime = create_track_runtime(track)
        self.assertEqual(runtime.drop_zone.mask.count(),
                         runtime.road_mask.overlap_area(runtime.drop_zone.mask, runtime.drop_zone.offset))
        start = next(tile for tile in track.tiles if tile.kind == "start_finish")
        expected = {0: 0, 90: 270, 180: 180, 270: 90}[start.rotation]
        self.assertEqual(expected, runtime.spawn_angle)


class ValidationTests(unittest.TestCase):
    def codes(self, track):
        return {issue.code for issue in validate_track(track).errors}

    def test_missing_and_multiple_start_lines(self):
        track = valid_track()
        start = next(tile for tile in track.tiles if tile.kind == "start_finish")
        start.kind = "straight"
        self.assertIn("MISSING_START", self.codes(track))
        track = valid_track()
        track.tiles[4].kind = "start_finish"
        self.assertIn("MULTIPLE_STARTS", self.codes(track))

    def test_open_endpoint_and_mismatched_rotation(self):
        track = valid_track(); removed = track.tiles.pop(5)
        self.assertIn("OPEN_CONNECTION", self.codes(track))
        track = valid_track(); track.tiles[5].rotation = (track.tiles[5].rotation + 90) % 360
        self.assertIn("MISMATCHED_ROTATION", self.codes(track))

    def test_duplicate_and_out_of_bounds_cells(self):
        track = valid_track(); track.tiles.append(Tile(track.tiles[0].x, track.tiles[0].y, "straight"))
        self.assertIn("DUPLICATE_CELL", self.codes(track))
        track = valid_track(); track.tiles[3].x = 99
        self.assertIn("OUT_OF_BOUNDS", self.codes(track))

    def test_disconnected_island_subloop_and_branch_are_diagnosed(self):
        first = valid_track()
        second = track_from_path("Second", rectangle_path(4, 2, 11, 7), 2)
        for tile in second.tiles:
            tile.x += 14
            if tile.kind == "start_finish": tile.kind = "straight"
        combined = TrackDefinition("Combined", first.tiles + second.tiles, grid_size=(30, 12))
        codes = self.codes(combined)
        self.assertIn("DISCONNECTED_ISLAND", codes)
        self.assertIn("SUB_LOOP", codes)
        branched = valid_track(); branched.tiles.append(Tile(4, 2, "branch", 0))
        self.assertIn("BRANCH", self.codes(branched))

    def test_insufficient_start_clearance_has_affected_cells(self):
        path = rectangle_path()
        # Move the start marker to the last straight before a corner.
        from track_geometry import infer_tile
        tiles = [infer_tile(cell, path[i - 1], path[(i + 1) % len(path)], i == 6)
                 for i, cell in enumerate(path)]
        result = validate_track(TrackDefinition("Tight start", tiles))
        issue = next(issue for issue in result.errors if issue.code == "INSUFFICIENT_START_CLEARANCE")
        self.assertGreaterEqual(len(issue.cells), 2)

    def test_structured_result_keeps_live_metrics_when_invalid(self):
        result = validate_track(TrackDefinition("Broken", [Tile(1, 1, "corner")]))
        self.assertFalse(result.valid)
        self.assertEqual(1, result.metrics.tile_count)
        self.assertEqual(1, result.metrics.corner_count)
        self.assertTrue(result.errors[0].code)

    def test_parallel_mask_separated_lanes_are_valid_with_a_warning(self):
        path = ([(4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (8, 2)]
                + [(x, 2) for x in range(7, 0, -1)]
                + [(1, 1), (2, 1), (3, 1)])
        track = track_from_path("Parallel lanes", path, 4)
        result = validate_track(track)
        self.assertTrue(result.valid)
        self.assertTrue(any(issue.code == "PARALLEL_ADJACENCY" and issue.severity == "warning"
                            for issue in result.errors))
        create_track_runtime(track)


class PersistenceTests(unittest.TestCase):
    def test_controller_contract_remains_five_inputs_four_outputs(self):
        config = load_neat_config(BASE)
        self.assertEqual(5, config.genome_config.num_inputs)
        self.assertEqual(4, config.genome_config.num_outputs)
        self.assertTrue(config.no_fitness_termination)

    def test_genome_json_round_trip_preserves_outputs(self):
        config = load_neat_config(BASE)
        genome = next(iter(neat.Population(config).population.values()))
        before = neat.nn.FeedForwardNetwork.create(genome, config).activate((.1, .2, .3, .4, .5))
        restored = deserialize_genome(json.loads(json.dumps(serialize_genome(genome))))
        after = neat.nn.FeedForwardNetwork.create(restored, config).activate((.1, .2, .3, .4, .5))
        self.assertEqual(before, after)

    def test_v1_model_migrates_and_v2_round_trips(self):
        config = load_neat_config(BASE)
        genome = next(iter(neat.Population(config).population.values()))
        v1 = {"schema_version": 1, "name": "Old", "skin": "white",
              "genome": serialize_genome(genome), "generation": 2, "fitness": 3.5,
              "controller_version": "five-sensor-v1"}
        model = ModelRecord.from_dict(v1)
        self.assertEqual(2, model.schema_version)
        self.assertEqual(model.model_id, model.lineage_id)
        self.assertEqual("Old", ModelRecord.from_dict(json.loads(json.dumps(model.to_dict()))).name)

    def test_atomic_model_track_and_progress_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(directory)
            config = load_neat_config(BASE)
            genome = next(iter(neat.Population(config).population.values()))
            model = ModelRecord("Test", "white", serialize_genome(genome), 1, 2.5)
            storage.save_model(model)
            storage.save_track(valid_track())
            storage.save_progress({"unlocked": 4, "completed": {}, "best_times": {}})
            self.assertEqual("Test", storage.models()[0].name)
            self.assertEqual(1, len(storage.custom_tracks(valid_only=True)))
            self.assertEqual(4, storage.progress()["unlocked"])


if __name__ == "__main__":
    unittest.main()
