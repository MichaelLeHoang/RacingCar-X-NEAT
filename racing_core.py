"""Compatibility-facing core primitives for Racing Car X NEAT.

New subsystems live in focused modules, while this module keeps the public
imports used by existing saved tests and integrations stable.
"""

from __future__ import annotations

import math
from itertools import count
from pathlib import Path

import neat
import pygame
from neat.genes import DefaultConnectionGene, DefaultNodeGene

from campaign import CampaignProgress, campaign_tracks
from storage import CONTROLLER_VERSION, DEFAULT_CAR_STATS, ModelRecord, Storage
from track_assets import (
    LegacyBitmapRuntime, TrackAssetAtlas as TrackPieceAtlas, TrackRuntime,
    create_track_runtime, piece_atlas,
)
from track_generator import TrackGenerationConstraints, generate_track
from track_geometry import (
    CURB_OUTER_WIDTH as CURB_WIDTH, E, GRID_SIZE, N, OPPOSITE, PORTS, ROAD_WIDTH,
    S, TILE_SIZE, VECTORS, W, Direction, Tile, TrackDefinition, TrackMetrics,
    TrackValidationResult, ValidationIssue, calculate_metrics, infer_tile,
    track_from_path, validate_track,
)


LOGICAL_SIZE = (1280, 800)
ASSET_DIR = Path(__file__).resolve().parent / "imgs"
SENSOR_RANGE = 150
SENSOR_ANGLES = (90, -90, 0, 45, -45)
STATIONARY_SECONDS = 5.0
STATIONARY_DISTANCE = 3.0


def serialize_genome(genome) -> dict:
    return {
        "key": int(genome.key),
        "fitness": genome.fitness,
        "nodes": [{
            "key": key,
            "bias": node.bias,
            "response": node.response,
            "activation": node.activation,
            "aggregation": node.aggregation,
            "time_constant": getattr(node, "time_constant", 1.0),
        } for key, node in genome.nodes.items()],
        "connections": [{
            "in": key[0],
            "out": key[1],
            "innovation": connection.innovation,
            "weight": connection.weight,
            "enabled": connection.enabled,
        } for key, connection in genome.connections.items()],
    }


def deserialize_genome(data: dict, key=None):
    genome = neat.DefaultGenome(int(data["key"] if key is None else key))
    genome.fitness = data.get("fitness")
    for item in data["nodes"]:
        node = DefaultNodeGene(int(item["key"]))
        node.bias = float(item["bias"])
        node.response = float(item["response"])
        node.activation = item["activation"]
        node.aggregation = item["aggregation"]
        node.time_constant = float(item.get("time_constant", 1.0))
        genome.nodes[node.key] = node
    for item in data["connections"]:
        pair = int(item["in"]), int(item["out"])
        innovation = int(item.get("innovation", 0))
        connection = DefaultConnectionGene(pair, innovation=innovation)
        connection.weight = float(item["weight"])
        connection.enabled = bool(item["enabled"])
        genome.connections[pair] = connection
    return genome


def reconcile_genome_innovations(genome, population):
    """Rebase a restored genome onto a fresh population's innovation namespace.

    Saved genomes legitimately carry historical innovation numbers.  A new
    neat-python population starts its tracker from the initial connections, so
    without advancing/reconciling that tracker it can later reuse a saved
    number for a different connection and warn during crossover.
    """
    tracker = population.reproduction.innovation_tracker
    pair_to_innovation = {}
    innovation_to_pair = {}
    maximum = tracker.get_current_innovation_number()
    for current in population.population.values():
        for pair, connection in current.connections.items():
            pair_to_innovation.setdefault(pair, connection.innovation)
            innovation_to_pair.setdefault(connection.innovation, pair)
            maximum = max(maximum, int(connection.innovation))
    for connection in genome.connections.values():
        maximum = max(maximum, int(connection.innovation or 0))
    tracker.global_counter = maximum

    for pair, connection in genome.connections.items():
        if pair in pair_to_innovation:
            connection.innovation = pair_to_innovation[pair]
        elif (not connection.innovation
              or (connection.innovation in innovation_to_pair
                  and innovation_to_pair[connection.innovation] != pair)):
            tracker.global_counter += 1
            connection.innovation = tracker.global_counter
        pair_to_innovation[pair] = connection.innovation
        innovation_to_pair[connection.innovation] = pair
    restored_innovations = [int(connection.innovation or 0)
                            for connection in genome.connections.values()]
    if restored_innovations:
        tracker.global_counter = max(tracker.global_counter, *restored_innovations)

    node_keys = [key for key in genome.nodes if key >= 0]
    for current in population.population.values():
        node_keys.extend(key for key in current.nodes if key >= 0)
    population.config.genome_config.node_indexer = count(max(node_keys, default=-1) + 1)
    population.config.genome_config.innovation_tracker = tracker
    return genome


class Car:
    def __init__(self, image: pygame.Surface, runtime, stats=None):
        stats = {**DEFAULT_CAR_STATS, **(stats or {})}
        self.image = image
        self.max_vel = float(stats["max_speed"])
        self.rotation_vel = float(stats["turn_speed"])
        self.acceleration = float(stats["acceleration"])
        self.angle = runtime.spawn_angle
        self.vel = 0.0
        if getattr(runtime, "spawn_position", None) is not None:
            self.x, self.y = runtime.spawn_position
        else:
            self.x = runtime.spawn_center[0] - image.get_width() / 2
            self.y = runtime.spawn_center[1] - image.get_height() / 2
        self.sensor_values = (0.0,) * 5
        self.next_checkpoint = 0
        self.stationary_elapsed = 0.0
        self.stationary_anchor = self.x, self.y
        self.distance = 0.0
        self.elapsed = 0.0
        self.finish_armed = False
        self.left_start = False
        self.last_progress, self.progress_segment = runtime.progress_at(self.center)
        self.furthest_progress = self.last_progress
        self.backward_progress = 0.0
        self.angle_without_progress = 0.0
        self.oscillations = 0
        self._last_progress_delta = 0.0

    @property
    def center(self):
        return self.x + self.image.get_width() / 2, self.y + self.image.get_height() / 2

    def step(self, action: int):
        old_angle = self.angle
        if action == 0:
            self.angle += self.rotation_vel
        elif action == 1:
            self.angle -= self.rotation_vel
        if action in (0, 1, 2):
            self.vel = min(self.max_vel, self.vel + self.acceleration)
        else:
            self.vel = max(0.0, self.vel - self.acceleration / 2)
        radians = math.radians(self.angle)
        self.x -= math.sin(radians) * self.vel
        self.y -= math.cos(radians) * self.vel
        self.distance += self.vel
        self.angle_without_progress += abs(self.angle - old_angle)

    def rotated(self):
        image = pygame.transform.rotate(self.image, self.angle)
        rect = image.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)
        return image, rect

    def _mask_overlap(self, target_mask: pygame.mask.Mask) -> bool:
        image, rect = self.rotated()
        return target_mask.overlap(pygame.mask.from_surface(image), (int(rect.x), int(rect.y))) is not None

    def crashed(self, runtime) -> bool:
        return self._mask_overlap(runtime.collision_mask)

    def finish_overlap(self, runtime) -> bool:
        return self._mask_overlap(runtime.finish_mask)

    def sensors(self, runtime):
        values = []
        for relative in SENSOR_ANGLES:
            radians = math.radians(self.angle + relative)
            for distance in range(0, SENSOR_RANGE + 1, 2):
                x = int(self.center[0] - math.sin(radians) * distance)
                y = int(self.center[1] - math.cos(radians) * distance)
                if (not (0 <= x < runtime.size[0] and 0 <= y < runtime.size[1])
                        or runtime.collision_mask.get_at((x, y))):
                    values.append(distance / SENSOR_RANGE)
                    break
            else:
                values.append(1.0)
        self.sensor_values = tuple(values)
        return self.sensor_values

    def _gate_overlap(self, gate) -> bool:
        return self._mask_overlap(gate.mask)

    def update_progress(self, runtime, dt=1 / 60):
        self.elapsed += dt
        if math.dist(self.center, runtime.start_center) >= ROAD_WIDTH * 1.5:
            self.left_start = True
        if self.next_checkpoint < len(runtime.gates):
            gate = runtime.gates[self.next_checkpoint]
            if self._gate_overlap(gate):
                self.next_checkpoint += 1
                self.angle_without_progress = 0.0
        progress, segment = runtime.progress_at(self.center)
        delta = progress - self.last_progress
        lap_length = max(1.0, runtime.metrics.estimated_lap_distance)
        if delta < -lap_length * .5:
            delta += lap_length
        elif delta > lap_length * .5:
            delta -= lap_length
        if delta < 0:
            self.backward_progress += -delta
        if self._last_progress_delta * delta < 0 and abs(delta) > .25:
            self.oscillations += 1
        self._last_progress_delta = delta
        self.last_progress = progress
        self.progress_segment = segment
        self.furthest_progress = max(self.furthest_progress, progress)
        # The legacy bitmap has a real finish-line mask but its hand-authored
        # progress gates are intentionally approximate.  Requiring every small
        # gate made an otherwise successful Original Track lap run forever
        # after visibly crossing the line.  Semantic tracks retain strict gate
        # ordering; the legacy lap arms once it has left the start area.
        legacy_finish = runtime.definition.runtime_type == "legacy_bitmap"
        self.finish_armed = self.left_start and (
            legacy_finish or self.next_checkpoint == len(runtime.gates)
        )
        finished = self.finish_armed and self.finish_overlap(runtime)
        if math.dist((self.x, self.y), self.stationary_anchor) >= STATIONARY_DISTANCE:
            self.stationary_anchor = self.x, self.y
            self.stationary_elapsed = 0.0
        else:
            self.stationary_elapsed += dt
        stalled = self.stationary_elapsed >= STATIONARY_SECONDS
        timeout = self.elapsed >= runtime.definition.timeout
        return finished, stalled, timeout


def load_neat_config(base_dir):
    return neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        str(Path(base_dir) / "config_feedforward.txt"),
    )


__all__ = [
    "ASSET_DIR", "CONTROLLER_VERSION", "CURB_WIDTH", "Car", "CampaignProgress",
    "DEFAULT_CAR_STATS", "Direction", "E", "GRID_SIZE", "LOGICAL_SIZE",
    "LegacyBitmapRuntime", "ModelRecord", "N", "OPPOSITE", "PORTS", "ROAD_WIDTH",
    "S", "SENSOR_ANGLES", "SENSOR_RANGE", "Storage", "TILE_SIZE", "Tile",
    "TrackDefinition", "TrackGenerationConstraints", "TrackMetrics", "TrackPieceAtlas",
    "TrackRuntime", "TrackValidationResult", "ValidationIssue", "VECTORS", "W",
    "calculate_metrics", "campaign_tracks", "create_track_runtime", "deserialize_genome",
    "generate_track", "infer_tile", "load_neat_config", "piece_atlas",
    "reconcile_genome_innovations", "serialize_genome", "track_from_path", "validate_track",
]
