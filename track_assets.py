"""Canonical tile artwork, masks, and track runtime composition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pygame

from track_geometry import (
    CURB_OUTER_WIDTH, FORWARD_PORT, GRID_SIZE, ROAD_WIDTH, TILE_SIZE, VECTORS,
    TrackDefinition, Tile, validate_track,
)


BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "imgs"
TILE_ASSET_DIR = ASSET_DIR / "tiles"

ASPHALT = (111, 112, 116, 255)
CURB_RED = (210, 28, 48, 255)
CURB_WHITE = (244, 244, 239, 255)
OUTLINE = (9, 10, 11, 255)
MASK_COLOR = (255, 255, 255, 255)


def _geometry(kind: str, x: float, y: float) -> tuple[float, float, float]:
    """Return distance from centerline, arc length, and total length."""
    if kind in ("straight", "start_finish"):
        return abs(x - TILE_SIZE / 2), y, float(TILE_SIZE)
    if kind == "corner":
        cx, cy = TILE_SIZE, 0
        radius = TILE_SIZE / 2
        angle = math.atan2(y - cy, x - cx)
        distance = abs(math.hypot(x - cx, y - cy) - radius)
        length = abs(math.pi - angle) * radius
        return distance, length, math.pi * radius / 2
    raise ValueError(f"Unknown tile kind: {kind}")


def _curb_color(length: float, total: float) -> tuple[int, int, int, int]:
    # The phase is symmetric, so both connector endpoints are white even when
    # a tile is rotated 180 degrees.
    endpoint_distance = min(length, max(0.0, total - length))
    return CURB_WHITE if int(endpoint_distance // 8) % 2 == 0 else CURB_RED


def build_tile_layers(kind: str) -> dict[str, pygame.Surface]:
    """Build canonical rotation-zero visual and semantic mask layers."""
    if kind not in ("straight", "corner", "start_finish"):
        raise ValueError(kind)
    visual = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    road = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    curb = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    finish = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    road_half = ROAD_WIDTH / 2
    curb_half = CURB_OUTER_WIDTH / 2
    outline_half = curb_half + 2
    for y in range(TILE_SIZE):
        for x in range(TILE_SIZE):
            distance, length, total = _geometry(kind, x + .5, y + .5)
            if distance <= outline_half:
                visual.set_at((x, y), OUTLINE)
            if distance <= curb_half:
                visual.set_at((x, y), _curb_color(length, total))
                if distance > road_half:
                    curb.set_at((x, y), MASK_COLOR)
            if distance <= road_half:
                visual.set_at((x, y), ASPHALT)
                road.set_at((x, y), MASK_COLOR)
    if kind == "start_finish":
        top = TILE_SIZE // 2 - 4
        for y in range(top, top + 8):
            for x in range(TILE_SIZE):
                if road.get_at((x, y)).a:
                    color = (248, 248, 248, 255) if ((x // 4 + y // 4) % 2) else OUTLINE
                    visual.set_at((x, y), color)
                    finish.set_at((x, y), MASK_COLOR)
    return {"visual": visual, "road": road, "curb": curb, "finish": finish}


def write_canonical_assets(directory: Path = TILE_ASSET_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for kind in ("straight", "corner", "start_finish"):
        for layer, surface in build_tile_layers(kind).items():
            path = directory / f"{kind}-{layer}.png"
            pygame.image.save(surface, path)
            written.append(path)
    return written


class TrackAssetAtlas:
    def __init__(self, directory: Path = TILE_ASSET_DIR):
        self.directory = directory
        self._base: dict[tuple[str, str], pygame.Surface] = {}
        self._cache: dict[tuple[str, int, str], pygame.Surface] = {}

    def _load(self, kind: str, layer: str) -> pygame.Surface:
        key = kind, layer
        if key not in self._base:
            path = self.directory / f"{kind}-{layer}.png"
            if path.exists():
                self._base[key] = pygame.image.load(path)
            else:
                self._base[key] = build_tile_layers(kind)[layer]
        return self._base[key]

    def layer(self, kind: str, rotation: int, layer: str = "visual") -> pygame.Surface:
        rotation %= 360
        key = kind, rotation, layer
        if key not in self._cache:
            surface = self._load(kind, layer)
            rotated = pygame.transform.rotate(surface, -rotation)
            if rotated.get_size() != (TILE_SIZE, TILE_SIZE):
                rotated = pygame.transform.smoothscale(rotated, (TILE_SIZE, TILE_SIZE))
            self._cache[key] = rotated
        return self._cache[key]

    def surface(self, kind: str, rotation: int) -> pygame.Surface:
        return self.layer(kind, rotation, "visual")


PIECE_ATLAS: TrackAssetAtlas | None = None


def piece_atlas() -> TrackAssetAtlas:
    global PIECE_ATLAS
    if PIECE_ATLAS is None:
        PIECE_ATLAS = TrackAssetAtlas()
    return PIECE_ATLAS


@dataclass(frozen=True)
class ProgressGate:
    index: int
    cell: tuple[int, int]
    center: tuple[float, float]
    mask: pygame.mask.Mask


@dataclass(frozen=True)
class DropZone:
    center: tuple[float, float]
    angle: int
    size: tuple[int, int]
    mask: pygame.mask.Mask
    offset: tuple[int, int]


def _gate_mask(size: tuple[int, int], center: tuple[float, float], radius: int = 18) -> pygame.mask.Mask:
    surface = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.circle(surface, MASK_COLOR, (round(center[0]), round(center[1])), radius)
    return pygame.mask.from_surface(surface)


class TrackRuntime:
    def __init__(self, definition: TrackDefinition):
        result = validate_track(definition)
        if not result.valid:
            raise ValueError("; ".join(result.messages))
        self.definition = definition
        self.path = result.path
        self.metrics = result.metrics
        self.size = (definition.grid_size[0] * TILE_SIZE, definition.grid_size[1] * TILE_SIZE)
        self.origin = (28, 82)
        self.surface = pygame.Surface(self.size, pygame.SRCALPHA)
        road_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        curb_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        finish_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        atlas = piece_atlas()
        for tile in definition.tiles:
            position = tile.x * TILE_SIZE, tile.y * TILE_SIZE
            self.surface.blit(atlas.layer(tile.kind, tile.rotation, "visual"), position)
            road_surface.blit(atlas.layer(tile.kind, tile.rotation, "road"), position)
            curb_surface.blit(atlas.layer(tile.kind, tile.rotation, "curb"), position)
            if tile.kind == "start_finish":
                finish_surface.blit(atlas.layer(tile.kind, tile.rotation, "finish"), position)
        self.road_mask = pygame.mask.from_surface(road_surface)
        self.curb_mask = pygame.mask.from_surface(curb_surface)
        self.collision_mask = self.road_mask.copy()
        self.collision_mask.invert()
        self.border_mask = self.collision_mask
        self.finish_mask = pygame.mask.from_surface(finish_surface)
        self.start_center = self.center(self.path[0])
        start_tile = next(tile for tile in definition.tiles if tile.kind == "start_finish")
        forward = start_tile.forward_port
        dx, dy = VECTORS[forward]
        self.spawn_angle = {"N": 0, "W": 90, "S": 180, "E": 270}[forward]
        self.spawn_center = (self.start_center[0] - dx * 48, self.start_center[1] - dy * 48)
        self.spawn_position = None
        self.drop_zone = self._drop_zone(self.spawn_center, self.spawn_angle)
        if self.road_mask.overlap_area(self.drop_zone.mask, self.drop_zone.offset) != self.drop_zone.mask.count():
            raise ValueError("Spawn/drop area does not fit inside the road mask")
        self.gates = [ProgressGate(index, cell, self.center(cell),
                                   _gate_mask(self.size, self.center(cell)))
                      for index, cell in enumerate(self.path[1:], 1)]
        self.checkpoints = [gate.center for gate in self.gates]

    def center(self, cell: tuple[int, int]) -> tuple[float, float]:
        return cell[0] * TILE_SIZE + TILE_SIZE / 2, cell[1] * TILE_SIZE + TILE_SIZE / 2

    def _drop_zone(self, center: tuple[float, float], angle: int) -> DropZone:
        base = pygame.Surface((40, 64), pygame.SRCALPHA)
        base.fill(MASK_COLOR)
        rotated = pygame.transform.rotate(base, angle)
        rect = rotated.get_rect(center=center)
        return DropZone(center, angle, (40, 64), pygame.mask.from_surface(rotated), rect.topleft)

    def progress_at(self, point: tuple[float, float]) -> tuple[float, int]:
        """Project a point onto the ordered tile-center polyline."""
        points = [self.center(cell) for cell in self.path] + [self.center(self.path[0])]
        best_distance = float("inf")
        best_progress = 0.0
        best_segment = 0
        cumulative = 0.0
        for index, (a, b) in enumerate(zip(points, points[1:])):
            vx, vy = b[0] - a[0], b[1] - a[1]
            length_sq = vx * vx + vy * vy
            t = max(0.0, min(1.0, ((point[0] - a[0]) * vx + (point[1] - a[1]) * vy) / length_sq))
            projected = a[0] + vx * t, a[1] + vy * t
            distance = math.dist(point, projected)
            length = math.sqrt(length_sq)
            if distance < best_distance:
                best_distance = distance
                best_progress = cumulative + length * t
                best_segment = index
            cumulative += length
        return best_progress, best_segment


class LegacyBitmapRuntime:
    """Exact legacy visual/collision placement plus ordered progress metadata."""

    LEGACY_CENTERS = [
        (117, 205), (130, 90), (255, 70), (255, 360), (345, 400),
        (380, 155), (710, 155), (750, 300), (475, 350), (700, 400),
        (710, 610), (650, 700), (570, 530), (500, 510), (450, 680),
        (350, 700), (120, 500), (95, 350),
    ]

    def __init__(self, definition: TrackDefinition):
        self.definition = definition
        self.metrics = validate_track(definition).metrics
        self.size = (900, 750)
        self.origin = (24, 25)
        self.surface = pygame.Surface(self.size, pygame.SRCALPHA)
        track = pygame.transform.scale_by(pygame.image.load(ASSET_DIR / "track.png"), .85)
        border = pygame.transform.scale_by(pygame.image.load(ASSET_DIR / "track-border.png"), .85)
        finish = pygame.image.load(ASSET_DIR / "finish.png")
        self.surface.blit(track, (70, 0))
        self.surface.blit(finish, (80, 270))
        self.surface.blit(border, (70, 0))
        border_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        border_surface.blit(border, (70, 0))
        self.border_mask = pygame.mask.from_surface(border_surface)
        self.collision_mask = self.border_mask
        road_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        road_surface.blit(track, (70, 0))
        self.road_mask = pygame.mask.from_surface(road_surface)
        self.curb_mask = self.border_mask
        finish_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        finish_surface.blit(finish, (80, 270))
        self.finish_mask = pygame.mask.from_surface(finish_surface)
        self.spawn_position = (100, 170)
        self.spawn_angle = 0
        self.spawn_center = (116, 202)
        self.start_center = (130, 280)
        self.path = list(range(len(self.LEGACY_CENTERS)))
        self.gates = [ProgressGate(index, (index, 0), center, _gate_mask(self.size, center, 24))
                      for index, center in enumerate(self.LEGACY_CENTERS[1:], 1)]
        self.checkpoints = [gate.center for gate in self.gates]
        self.drop_zone = self._drop_zone()

    def _drop_zone(self) -> DropZone:
        base = pygame.Surface((40, 64), pygame.SRCALPHA)
        base.fill(MASK_COLOR)
        rect = base.get_rect(center=self.spawn_center)
        return DropZone(self.spawn_center, self.spawn_angle, (40, 64),
                        pygame.mask.from_surface(base), rect.topleft)

    def progress_at(self, point: tuple[float, float]) -> tuple[float, int]:
        best = min(range(len(self.LEGACY_CENTERS)),
                   key=lambda index: math.dist(point, self.LEGACY_CENTERS[index]))
        return best * TILE_SIZE, best


def create_track_runtime(definition: TrackDefinition):
    if definition.runtime_type == "legacy_bitmap":
        return LegacyBitmapRuntime(definition)
    return TrackRuntime(definition)
