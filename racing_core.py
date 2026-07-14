from __future__ import annotations

import copy
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import neat
import pygame
from neat.genes import DefaultConnectionGene, DefaultNodeGene


LOGICAL_SIZE = (1280, 800)
ASSET_DIR = Path(__file__).resolve().parent / "imgs"
GRID_SIZE = (14, 10)
TILE_SIZE = 64
ROAD_WIDTH = 38
CURB_WIDTH = 54
SENSOR_RANGE = 150
SENSOR_ANGLES = (90, -90, 0, 45, -45)
STATIONARY_SECONDS = 5.0
STATIONARY_DISTANCE = 3.0
DEFAULT_CAR_STATS = {"max_speed": 4.0, "acceleration": .2, "turn_speed": 4.0}

N, E, S, W = "N", "E", "S", "W"
VECTORS = {N: (0, -1), E: (1, 0), S: (0, 1), W: (-1, 0)}
OPPOSITE = {N: S, E: W, S: N, W: E}

PORTS = {
    "straight": {
        0: (N, S), 90: (E, W), 180: (N, S), 270: (E, W),
    },
    "start_finish": {
        0: (N, S), 90: (E, W), 180: (N, S), 270: (E, W),
    },
    "corner": {
        0: (N, E), 90: (E, S), 180: (S, W), 270: (W, N),
    },
}


@dataclass
class Tile:
    x: int
    y: int
    kind: str
    rotation: int = 0

    @property
    def ports(self):
        return PORTS[self.kind][self.rotation % 360]


@dataclass
class TrackDefinition:
    name: str
    tiles: list[Tile]
    timeout: float = 30.0
    difficulty: int = 1
    source: str = "custom"
    track_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 1
    runtime_type: str = "component"

    def to_dict(self):
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported track schema")
        tiles = [Tile(**tile) for tile in data["tiles"]]
        return cls(
            name=str(data["name"]), tiles=tiles,
            timeout=float(data.get("timeout", 30)),
            difficulty=int(data.get("difficulty", 1)),
            source=str(data.get("source", "custom")),
            track_id=str(data.get("track_id", uuid.uuid4())),
            runtime_type=str(data.get("runtime_type", "component")),
        )


def validate_track(track: TrackDefinition):
    if track.runtime_type == "legacy_bitmap":
        return [], []
    errors = []
    if len(track.tiles) < 8:
        errors.append("Use at least 8 track pieces")
    starts = [tile for tile in track.tiles if tile.kind == "start_finish"]
    if len(starts) != 1:
        errors.append("Include exactly one start/finish piece")
    cells = {(tile.x, tile.y): tile for tile in track.tiles}
    if len(cells) != len(track.tiles):
        errors.append("Two pieces cannot occupy the same cell")
    if any(tile.kind not in PORTS for tile in track.tiles):
        errors.append("Unknown track piece")
    if errors:
        return errors, []

    adjacency = {}
    for cell, tile in cells.items():
        neighbors = []
        for port in tile.ports:
            dx, dy = VECTORS[port]
            other = cells.get((cell[0] + dx, cell[1] + dy))
            if other is None or OPPOSITE[port] not in other.ports:
                errors.append(f"Unmatched connection at {cell}")
            else:
                neighbors.append((other.x, other.y))
        adjacency[cell] = neighbors
    if errors:
        return sorted(set(errors)), []
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        errors.append("Track must be one non-branching loop")
        return errors, []

    start = (starts[0].x, starts[0].y)
    path = [start]
    previous = None
    current = start
    while True:
        choices = [cell for cell in adjacency[current] if cell != previous]
        if not choices:
            errors.append("Track is not a closed loop")
            break
        nxt = choices[0]
        if nxt == start:
            break
        if nxt in path:
            errors.append("Track contains a smaller disconnected loop")
            break
        path.append(nxt)
        previous, current = current, nxt
        if len(path) > len(cells):
            errors.append("Track traversal failed")
            break
    if len(path) != len(cells):
        errors.append("All pieces must belong to the same loop")
    return errors, path if not errors else []


def infer_tile(cell, previous, following, start=False):
    ports = []
    for other in (previous, following):
        dx, dy = other[0] - cell[0], other[1] - cell[1]
        ports.append({(0, -1): N, (1, 0): E, (0, 1): S, (-1, 0): W}[(dx, dy)])
    target = set(ports)
    kind = "start_finish" if start else "straight"
    for rotation, candidate in PORTS[kind].items():
        if set(candidate) == target:
            return Tile(*cell, kind, rotation)
    for rotation, candidate in PORTS["corner"].items():
        if set(candidate) == target:
            return Tile(*cell, "corner", rotation)
    raise ValueError("Path contains a non-orthogonal connection")


def track_from_path(name, path, difficulty, timeout):
    tiles = []
    for index, cell in enumerate(path):
        tiles.append(infer_tile(
            cell, path[index - 1], path[(index + 1) % len(path)], index == 0
        ))
    return TrackDefinition(
        name=name, tiles=tiles, difficulty=difficulty, timeout=timeout,
        source="campaign", track_id=f"level-{difficulty}",
    )


def rectangular_path(left, top, right, bottom):
    path = [(left, y) for y in range(bottom - 1, top - 1, -1)]
    path += [(x, top) for x in range(left + 1, right + 1)]
    path += [(right, y) for y in range(top + 1, bottom + 1)]
    path += [(x, bottom) for x in range(right - 1, left - 1, -1)]
    return path


def campaign_tracks():
    specs = [
        ("First Lap", 2, 1, 7, 7, 34), ("Long Turn", 1, 1, 8, 7, 33),
        ("Wide Circuit", 1, 1, 10, 7, 35), ("Fast Box", 2, 1, 11, 7, 32),
        ("Outer Ring", 1, 0, 12, 8, 38), ("Precision", 3, 1, 10, 8, 29),
        ("Endurance", 1, 0, 12, 9, 36), ("Redline", 2, 0, 12, 8, 29),
        ("Expert Loop", 1, 1, 11, 9, 28), ("Final Circuit", 1, 0, 12, 9, 26),
    ]
    tracks = [track_from_path(name, rectangular_path(l, t, r, b), i, timeout)
              for i, (name, l, t, r, b, timeout) in enumerate(specs, 1)]
    tracks[0] = TrackDefinition(
        name="First Lap", tiles=[], timeout=45, difficulty=1,
        source="campaign", track_id="level-1", runtime_type="legacy_bitmap",
    )
    return tracks


class TrackPieceAtlas:
    """Reusable editor pieces extracted from the repository's original track."""
    SOURCE_RECTS = {
        "straight": pygame.Rect(0, 165, 125, 125),
        "corner": pygame.Rect(710, 0, 190, 190),
    }

    def __init__(self):
        self.source = pygame.image.load(ASSET_DIR / "track.png")
        self.finish = pygame.image.load(ASSET_DIR / "finish.png")
        self.cache = {}

    def surface(self, kind, rotation):
        key = (kind, rotation % 360)
        if key in self.cache:
            return self.cache[key]
        source_kind = "straight" if kind == "start_finish" else kind
        base = self.source.subsurface(self.SOURCE_RECTS[source_kind]).copy()
        base = pygame.transform.smoothscale(base, (TILE_SIZE, TILE_SIZE))
        # The sampled outer top-right bend connects west to south. Rotate it
        # into the atlas's canonical north-to-east orientation.
        if source_kind == "corner":
            base = pygame.transform.rotate(base, 180)
        if kind == "start_finish":
            line = pygame.transform.smoothscale(self.finish, (TILE_SIZE, 10))
            base.blit(line, (0, TILE_SIZE // 2 - 5))
        rotated = pygame.transform.rotate(base, rotation % 360)
        if rotated.get_size() != (TILE_SIZE, TILE_SIZE):
            rotated = pygame.transform.smoothscale(rotated, (TILE_SIZE, TILE_SIZE))
        self.cache[key] = rotated
        return rotated


PIECE_ATLAS = None


def piece_atlas():
    global PIECE_ATLAS
    if PIECE_ATLAS is None:
        PIECE_ATLAS = TrackPieceAtlas()
    return PIECE_ATLAS


class TrackRuntime:
    def __init__(self, definition: TrackDefinition):
        self.definition = definition
        errors, self.path = validate_track(definition)
        if errors:
            raise ValueError("; ".join(errors))
        self.size = (GRID_SIZE[0] * TILE_SIZE, GRID_SIZE[1] * TILE_SIZE)
        self.origin = (28, 82)
        self.surface = pygame.Surface(self.size, pygame.SRCALPHA)
        self._render()
        self.border_mask = self.road_mask.copy()
        self.border_mask.invert()
        start = self.path[0]
        nxt = self.path[1]
        self.spawn_center = self.center(start)
        self.spawn_angle = self.angle_for(start, nxt)
        fractions = (0.25, 0.5, 0.75)
        self.checkpoints = [self.center(self.path[int(len(self.path) * f) % len(self.path)])
                            for f in fractions]
        self.start_center = self.center(start)
        self.finish_mask = self._finish_mask(start, nxt)

    def _finish_mask(self, start, following):
        surface = pygame.Surface(self.size, pygame.SRCALPHA)
        cx, cy = self.center(start)
        dx, dy = following[0] - start[0], following[1] - start[1]
        if dx:
            pygame.draw.line(surface, (255, 255, 255, 255),
                             (cx, cy - ROAD_WIDTH // 2), (cx, cy + ROAD_WIDTH // 2), 8)
        else:
            pygame.draw.line(surface, (255, 255, 255, 255),
                             (cx - ROAD_WIDTH // 2, cy), (cx + ROAD_WIDTH // 2, cy), 8)
        return pygame.mask.from_surface(surface)

    def center(self, cell):
        return (cell[0] * TILE_SIZE + TILE_SIZE // 2,
                cell[1] * TILE_SIZE + TILE_SIZE // 2)

    @staticmethod
    def angle_for(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        return {(0, -1): 0, (-1, 0): 90, (0, 1): 180, (1, 0): 270}[(dx, dy)]

    def _render(self):
        points = [self.center(cell) for cell in self.path]
        closed = points + [points[0]]
        road_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        atlas = piece_atlas()
        for tile in self.definition.tiles:
            self.surface.blit(atlas.surface(tile.kind, tile.rotation),
                              (tile.x * TILE_SIZE, tile.y * TILE_SIZE))
        pygame.draw.lines(road_surface, (255, 255, 255, 255), False, closed, ROAD_WIDTH)
        for point in points:
            pygame.draw.circle(road_surface, (255, 255, 255, 255), point, ROAD_WIDTH // 2)
        self.road_mask = pygame.mask.from_surface(road_surface)


class LegacyBitmapRuntime:
    def __init__(self, definition):
        self.definition = definition
        self.size = (900, 750)
        self.origin = (24, 25)
        self.surface = pygame.Surface(self.size, pygame.SRCALPHA)
        track = pygame.transform.scale_by(pygame.image.load(ASSET_DIR / "track.png"), .85)
        border = pygame.transform.scale_by(pygame.image.load(ASSET_DIR / "track-border.png"), .85)
        finish = pygame.image.load(ASSET_DIR / "finish.png")
        self.surface.blit(track, (70, 0)); self.surface.blit(finish, (80, 270)); self.surface.blit(border, (70, 0))
        border_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        border_surface.blit(border, (70, 0))
        self.border_mask = pygame.mask.from_surface(border_surface)
        finish_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        finish_surface.blit(finish, (80, 270))
        self.finish_mask = pygame.mask.from_surface(finish_surface)
        self.spawn_position = (100, 170); self.spawn_angle = 0
        self.spawn_center = (117.5, 205)
        self.start_center = (130, 280)
        self.checkpoints = [(285, 300), (705, 300), (495, 635)]


def create_track_runtime(definition):
    if definition.runtime_type == "legacy_bitmap":
        return LegacyBitmapRuntime(definition)
    return TrackRuntime(definition)


@dataclass
class ModelRecord:
    name: str
    skin: str
    genome: dict
    generation: int
    fitness: float
    status: str = "draft"
    trained_tracks: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    attempts: int = 0
    wins: int = 0
    best_times: dict = field(default_factory=dict)
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 1
    controller_version: str = "five-sensor-v1"
    car_stats: dict = field(default_factory=lambda: dict(DEFAULT_CAR_STATS))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != 1 or data.get("controller_version") != "five-sensor-v1":
            raise ValueError("Incompatible model")
        return cls(**data)


def serialize_genome(genome):
    return {
        "key": int(genome.key), "fitness": genome.fitness,
        "nodes": [{
            "key": key, "bias": node.bias, "response": node.response,
            "activation": node.activation, "aggregation": node.aggregation,
            "time_constant": getattr(node, "time_constant", 1.0),
        } for key, node in genome.nodes.items()],
        "connections": [{
            "in": key[0], "out": key[1], "innovation": connection.innovation,
            "weight": connection.weight, "enabled": connection.enabled,
        } for key, connection in genome.connections.items()],
    }


def deserialize_genome(data, key=None):
    genome = neat.DefaultGenome(int(data["key"] if key is None else key))
    genome.fitness = data.get("fitness")
    for item in data["nodes"]:
        node = DefaultNodeGene(int(item["key"]))
        node.bias = float(item["bias"]); node.response = float(item["response"])
        node.activation = item["activation"]; node.aggregation = item["aggregation"]
        node.time_constant = float(item.get("time_constant", 1.0))
        genome.nodes[node.key] = node
    for item in data["connections"]:
        pair = (int(item["in"]), int(item["out"]))
        connection = DefaultConnectionGene(pair, innovation=int(item["innovation"]))
        connection.weight = float(item["weight"]); connection.enabled = bool(item["enabled"])
        genome.connections[pair] = connection
    return genome


class Storage:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get(
            "RACING_DATA_DIR", Path.home() / ".racing_car_x_neat"
        ))
        self.models_dir = self.root / "models"; self.tracks_dir = self.root / "tracks"
        self.exports_dir = self.root / "exports"; self.imports_dir = self.root / "imports"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.tracks_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.imports_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.root / "progress.json"

    @staticmethod
    def _write(path, data):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)

    def models(self):
        records = []
        for path in self.models_dir.glob("*.rcmodel"):
            try: records.append(ModelRecord.from_dict(json.loads(path.read_text())))
            except (ValueError, KeyError, json.JSONDecodeError): pass
        records.sort(key=lambda model: (model.name.casefold(), model.model_id))
        return records

    def save_model(self, model):
        self._write(self.models_dir / f"{model.model_id}.rcmodel", model.to_dict())

    def delete_model(self, model_id):
        path = self.models_dir / f"{model_id}.rcmodel"
        if path.exists(): path.unlink()

    def export_model(self, model):
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model.name)
        path = self.exports_dir / f"{safe_name or model.model_id}.rcmodel"
        self._write(path, model.to_dict())
        return path

    def import_inbox(self):
        imported = 0
        for path in self.imports_dir.iterdir():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if path.suffix == ".rcmodel":
                    model = ModelRecord.from_dict(data); model.model_id = str(uuid.uuid4())
                    self.save_model(model); imported += 1
                elif path.suffix == ".rctrack":
                    track = TrackDefinition.from_dict(data); track.track_id = str(uuid.uuid4())
                    errors, _ = validate_track(track)
                    if errors: raise ValueError(errors[0])
                    self.save_track(track); imported += 1
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
        return imported

    def custom_tracks(self):
        tracks = []
        for path in self.tracks_dir.glob("*.rctrack"):
            try: tracks.append(TrackDefinition.from_dict(json.loads(path.read_text())))
            except (ValueError, KeyError, json.JSONDecodeError): pass
        return tracks

    def save_track(self, track):
        self._write(self.tracks_dir / f"{track.track_id}.rctrack", track.to_dict())

    def progress(self):
        default = {"unlocked": 1, "completed": {}, "best_times": {}}
        if not self.progress_path.exists(): return default
        try: return {**default, **json.loads(self.progress_path.read_text())}
        except json.JSONDecodeError: return default

    def save_progress(self, progress):
        self._write(self.progress_path, progress)


class Car:
    def __init__(self, image, runtime: TrackRuntime, stats=None):
        stats = {**DEFAULT_CAR_STATS, **(stats or {})}
        self.image = image; self.max_vel = float(stats["max_speed"])
        self.rotation_vel = float(stats["turn_speed"]); self.acceleration = float(stats["acceleration"])
        self.angle = runtime.spawn_angle; self.vel = 0.0
        if hasattr(runtime, "spawn_position"):
            self.x, self.y = runtime.spawn_position
        else:
            self.x = runtime.spawn_center[0] - image.get_width() / 2
            self.y = runtime.spawn_center[1] - image.get_height() / 2
        self.sensor_values = (0.0,) * 5; self.next_checkpoint = 0
        self.started = time.perf_counter(); self.stationary_elapsed = 0.0
        self.stationary_anchor = (self.x, self.y); self.distance = 0.0; self.elapsed = 0.0
        # A lap is armed only after the car has left the start/finish area.
        # This prevents component tracks, which spawn on the line, from
        # completing on their first frame while still making the pixel mask
        # authoritative when the car returns.
        self.finish_armed = False

    @property
    def center(self): return (self.x + self.image.get_width()/2, self.y + self.image.get_height()/2)

    def step(self, action):
        if action == 0: self.angle += self.rotation_vel
        elif action == 1: self.angle -= self.rotation_vel
        if action in (0, 1, 2): self.vel = min(self.max_vel, self.vel + self.acceleration)
        else: self.vel = max(0, self.vel - self.acceleration / 2)
        radians = math.radians(self.angle)
        self.x -= math.sin(radians) * self.vel; self.y -= math.cos(radians) * self.vel
        self.distance += self.vel

    def rotated(self):
        image = pygame.transform.rotate(self.image, self.angle)
        rect = image.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)
        return image, rect

    def crashed(self, runtime):
        image, rect = self.rotated(); mask = pygame.mask.from_surface(image)
        return runtime.border_mask.overlap(mask, (int(rect.x), int(rect.y))) is not None

    def finish_overlap(self, runtime):
        image, rect = self.rotated(); mask = pygame.mask.from_surface(image)
        return runtime.finish_mask.overlap(mask, (int(rect.x), int(rect.y))) is not None

    def sensors(self, runtime):
        values = []
        for relative in SENSOR_ANGLES:
            radians = math.radians(self.angle + relative)
            for distance in range(0, SENSOR_RANGE + 1, 2):
                x = int(self.center[0] - math.sin(radians) * distance)
                y = int(self.center[1] - math.cos(radians) * distance)
                if not (0 <= x < runtime.size[0] and 0 <= y < runtime.size[1]) or runtime.border_mask.get_at((x, y)):
                    values.append(distance / SENSOR_RANGE); break
            else: values.append(1.0)
        self.sensor_values = tuple(values); return self.sensor_values

    def update_progress(self, runtime, dt=1/60):
        self.elapsed += dt
        if self.next_checkpoint < len(runtime.checkpoints):
            target = runtime.checkpoints[self.next_checkpoint]
            if math.dist(self.center, target) < ROAD_WIDTH:
                self.next_checkpoint += 1
        finish_overlap = self.finish_overlap(runtime)
        if (not self.finish_armed and not finish_overlap and
                math.dist(self.center, runtime.start_center) >= ROAD_WIDTH * 3):
            self.finish_armed = True
        finished = self.finish_armed and finish_overlap
        if math.dist((self.x, self.y), self.stationary_anchor) >= STATIONARY_DISTANCE:
            self.stationary_anchor = (self.x, self.y); self.stationary_elapsed = 0.0
        else:
            self.stationary_elapsed += dt
        stalled = self.stationary_elapsed >= STATIONARY_SECONDS
        timeout = self.elapsed >= runtime.definition.timeout
        return finished, stalled, timeout


def load_neat_config(base_dir):
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       str(Path(base_dir) / "config_feedforward.txt"))
