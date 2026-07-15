"""Semantic track definitions, validation, metrics, and shared geometry.

The module deliberately contains no scene code.  Rendering, collision, the
editor, campaign play, and training all consume the same definitions here.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable


TILE_SIZE = 64
GRID_SIZE = (14, 10)
ROAD_WIDTH = 44
CURB_OUTER_WIDTH = 56


class Direction(str, Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"


N, E, S, W = (direction.value for direction in Direction)
VECTORS = {N: (0, -1), E: (1, 0), S: (0, 1), W: (-1, 0)}
OPPOSITE = {N: S, E: W, S: N, W: E}
FORWARD_PORT = {0: N, 90: E, 180: S, 270: W}

PORTS = {
    "straight": {0: (N, S), 90: (E, W), 180: (N, S), 270: (E, W)},
    "start_finish": {0: (N, S), 90: (E, W), 180: (N, S), 270: (E, W)},
    "corner": {0: (N, E), 90: (E, S), 180: (S, W), 270: (W, N)},
}


@dataclass
class Tile:
    x: int
    y: int
    kind: str
    rotation: int = 0
    preview: dict = field(default_factory=dict)

    @property
    def cell(self) -> tuple[int, int]:
        return self.x, self.y

    @property
    def ports(self) -> tuple[str, ...]:
        return PORTS.get(self.kind, {}).get(self.rotation % 360, ())

    @property
    def forward_port(self) -> str | None:
        if self.kind != "start_finish":
            return None
        return FORWARD_PORT.get(self.rotation % 360)


@dataclass
class TrackMetrics:
    tile_count: int = 0
    corner_count: int = 0
    turn_density: float = 0.0
    longest_straight: int = 0
    shortest_straight_between_turns: int = 0
    alternating_turns: int = 0
    consecutive_technical_sections: int = 0
    estimated_lap_distance: float = 0.0
    recovery_distance: float = 0.0
    difficulty_score: float = 1.0
    recommended_timeout: float = 45.0
    open_connection_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackDefinition:
    name: str
    tiles: list[Tile]
    timeout: float = 30.0
    difficulty: int = 1
    source: str = "custom"
    track_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 2
    runtime_type: str = "component"
    grid_size: tuple[int, int] = GRID_SIZE
    metrics: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["grid_size"] = list(self.grid_size)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrackDefinition":
        version = int(data.get("schema_version", 1))
        if version not in (1, 2):
            raise ValueError("Unsupported track schema")
        tiles = [Tile(**tile) for tile in data.get("tiles", [])]
        return cls(
            name=str(data["name"]),
            tiles=tiles,
            timeout=float(data.get("timeout", 30.0)),
            difficulty=int(data.get("difficulty", 1)),
            source=str(data.get("source", "custom")),
            track_id=str(data.get("track_id") or uuid.uuid4()),
            schema_version=2,
            runtime_type=str(data.get("runtime_type", "component")),
            grid_size=tuple(data.get("grid_size", GRID_SIZE)),
            metrics=dict(data.get("metrics", {})),
            generation=dict(data.get("generation", {})),
        )

    def signature(self) -> tuple:
        return tuple(sorted((tile.x, tile.y, tile.kind, tile.rotation % 360)
                            for tile in self.tiles))


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    cells: tuple[tuple[int, int], ...] = ()
    severity: str = "error"


@dataclass
class TrackValidationResult:
    valid: bool
    errors: list[ValidationIssue]
    path: list[tuple[int, int]]
    metrics: TrackMetrics

    @property
    def messages(self) -> list[str]:
        return [issue.message for issue in self.errors]

    def issues_for(self, cell: tuple[int, int]) -> list[ValidationIssue]:
        return [issue for issue in self.errors if cell in issue.cells]

    def __iter__(self):
        """Compatibility with the former ``errors, path`` return contract."""
        yield [issue.message for issue in self.errors if issue.severity == "error"]
        yield self.path


def _issue(code: str, message: str, cells: Iterable[tuple[int, int]] = ()) -> ValidationIssue:
    return ValidationIssue(code, message, tuple(cells))


def _warning(code: str, message: str, cells: Iterable[tuple[int, int]] = ()) -> ValidationIssue:
    return ValidationIssue(code, message, tuple(cells), "warning")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def calculate_metrics(track: TrackDefinition, path: list[tuple[int, int]],
                      open_connections: int = 0) -> TrackMetrics:
    if track.runtime_type == "legacy_bitmap":
        values = {
            "tile_count": 32, "corner_count": 8, "turn_density": .25,
            "longest_straight": 8, "shortest_straight_between_turns": 3,
            "alternating_turns": 2, "consecutive_technical_sections": 1,
            "estimated_lap_distance": 2060.0, "recovery_distance": 5.0,
            "difficulty_score": 1.0, "recommended_timeout": 45.0,
        }
        values.update(track.metrics)
        return TrackMetrics(**{key: values[key] for key in TrackMetrics.__dataclass_fields__
                               if key in values}, open_connection_count=open_connections)

    cells = {tile.cell: tile for tile in track.tiles}
    if not path:
        corner_count = sum(tile.kind == "corner" for tile in track.tiles)
        tile_count = len(track.tiles)
        estimated = ((tile_count - corner_count) * TILE_SIZE
                     + corner_count * math.pi * TILE_SIZE / 4)
        return TrackMetrics(
            tile_count=tile_count,
            corner_count=corner_count,
            turn_density=round(corner_count / tile_count, 4) if tile_count else 0.0,
            estimated_lap_distance=round(estimated, 2),
            open_connection_count=open_connections,
        )
    corners = [index for index, cell in enumerate(path)
               if cells.get(cell) and cells[cell].kind == "corner"]
    tile_count = len(path) or len(track.tiles)
    corner_count = len(corners)
    straight_runs: list[int] = []
    turn_signs: list[int] = []
    if path and corners:
        for position, corner_index in enumerate(corners):
            next_corner = corners[(position + 1) % len(corners)]
            gap = (next_corner - corner_index - 1) % len(path)
            straight_runs.append(gap)
            previous = path[corner_index - 1]
            current = path[corner_index]
            following = path[(corner_index + 1) % len(path)]
            ax, ay = current[0] - previous[0], current[1] - previous[1]
            bx, by = following[0] - current[0], following[1] - current[1]
            turn_signs.append(1 if ax * by - ay * bx > 0 else -1)
    alternating = sum(1 for index, sign in enumerate(turn_signs)
                      if len(turn_signs) > 1 and sign != turn_signs[index - 1])
    technical = sum(1 for run in straight_runs if run <= 2)
    longest = max(straight_runs, default=0)
    shortest = min(straight_runs, default=0)
    recovery = sum(straight_runs) / len(straight_runs) if straight_runs else 0.0
    lap_distance = sum(math.pi * TILE_SIZE / 4 if cells[cell].kind == "corner" else TILE_SIZE
                       for cell in path) if path else 0.0
    density = corner_count / tile_count if tile_count else 0.0
    pressure = (
        .25 * _clamp((density - .15) / .35)
        + .15 * _clamp((corner_count - 4) / 12)
        + .15 * (1 - _clamp((shortest - 1) / 5))
        + .15 * (alternating / max(1, corner_count))
        + .15 * (technical / max(1, corner_count))
        + .10 * (1 - _clamp((recovery - 1) / 5))
        + .05 * _clamp((lap_distance - 768) / 2048)
    )
    score = round(1 + 9 * _clamp(pressure), 3)
    timeout = round(_clamp(
        lap_distance / 120.0 * (2.5 - .08 * score) + corner_count * .25,
        20.0, 60.0,
    ), 1)
    return TrackMetrics(
        tile_count=tile_count,
        corner_count=corner_count,
        turn_density=round(density, 4),
        longest_straight=longest,
        shortest_straight_between_turns=shortest,
        alternating_turns=alternating,
        consecutive_technical_sections=technical,
        estimated_lap_distance=round(lap_distance, 2),
        recovery_distance=round(recovery, 2),
        difficulty_score=score,
        recommended_timeout=timeout,
        open_connection_count=open_connections,
    )


def validate_track(track: TrackDefinition) -> TrackValidationResult:
    if track.runtime_type == "legacy_bitmap":
        metrics = calculate_metrics(track, [])
        if track.source == "campaign" and track.track_id == "level-1":
            return TrackValidationResult(True, [], [], metrics)
        issue = _issue("LEGACY_RUNTIME_RESERVED",
                       "Only the built-in Level 1 may use legacy bitmap geometry.")
        return TrackValidationResult(False, [issue], [], metrics)

    issues: list[ValidationIssue] = []
    width, height = track.grid_size
    if len(track.tiles) < 12:
        issues.append(_issue("TOO_FEW_TILES", "Add track pieces until the track has at least 12.",
                             (tile.cell for tile in track.tiles)))
    if sum(tile.kind == "corner" for tile in track.tiles) < 4:
        issues.append(_issue("TOO_FEW_CORNERS", "Add at least four corner pieces."))

    starts = [tile for tile in track.tiles if tile.kind == "start_finish"]
    if not starts:
        issues.append(_issue("MISSING_START", "Add one Start / Finish piece."))
    elif len(starts) > 1:
        issues.append(_issue("MULTIPLE_STARTS", "Keep exactly one Start / Finish piece.",
                             (tile.cell for tile in starts)))

    grouped: dict[tuple[int, int], list[Tile]] = {}
    for tile in track.tiles:
        grouped.setdefault(tile.cell, []).append(tile)
        if not (0 <= tile.x < width and 0 <= tile.y < height):
            issues.append(_issue("OUT_OF_BOUNDS", "This piece is outside the track grid.", [tile.cell]))
        if tile.kind not in PORTS:
            if tile.kind == "branch":
                issues.append(_issue("BRANCH", "The track cannot contain a branch.", [tile.cell]))
            issues.append(_issue("UNKNOWN_TILE", "This track contains an unknown piece.", [tile.cell]))
        elif tile.rotation % 360 not in (0, 90, 180, 270) or len(tile.ports) != 2:
            issues.append(_issue("INVALID_PORTS", "Every piece must have exactly two valid ports.", [tile.cell]))
    for cell, duplicates in grouped.items():
        if len(duplicates) > 1:
            issues.append(_issue("DUPLICATE_CELL", "Two pieces occupy this grid cell.", [cell]))

    cells = {cell: values[0] for cell, values in grouped.items()}
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in cells}
    open_count = 0
    for cell, tile in cells.items():
        for port in tile.ports:
            dx, dy = VECTORS[port]
            neighbor_cell = cell[0] + dx, cell[1] + dy
            neighbor = cells.get(neighbor_cell)
            if neighbor is None:
                open_count += 1
                issues.append(_issue("OPEN_CONNECTION", "This piece has an open connection.", [cell]))
            elif OPPOSITE[port] not in neighbor.ports:
                open_count += 1
                issues.append(_issue("MISMATCHED_ROTATION",
                                     "This piece is rotated away from its neighbor.",
                                     [cell, neighbor_cell]))
            else:
                adjacency[cell].append(neighbor_cell)

    # Adjacent tiles that do not connect are visually ambiguous and are not
    # permitted even if both tiles otherwise belong to the loop.
    seen_pairs = set()
    for cell, tile in cells.items():
        for port, (dx, dy) in VECTORS.items():
            neighbor_cell = cell[0] + dx, cell[1] + dy
            neighbor = cells.get(neighbor_cell)
            pair = tuple(sorted((cell, neighbor_cell)))
            if neighbor and pair not in seen_pairs:
                seen_pairs.add(pair)
                if not (port in tile.ports and OPPOSITE[port] in neighbor.ports):
                    issues.append(_warning(
                        "PARALLEL_ADJACENCY",
                        "Parallel lanes touch as grid neighbors but remain separated by their masks.",
                        pair,
                    ))

    for cell, neighbors in adjacency.items():
        degree = len(set(neighbors))
        if degree > 2:
            issues.append(_issue("BRANCH", "The track cannot contain a branch.", [cell]))
        elif degree < 2 and len(cells[cell].ports) == 2:
            # OPEN_CONNECTION already gives the local reason; this code makes
            # degree failures directly addressable by editor tooling.
            issues.append(_issue("INVALID_DEGREE", "This piece is not part of a closed loop.", [cell]))

    path: list[tuple[int, int]] = []
    if len(starts) == 1 and starts[0].cell in cells and starts[0].forward_port:
        start = starts[0]
        dx, dy = VECTORS[start.forward_port]
        following = start.x + dx, start.y + dy
        if following in adjacency.get(start.cell, []):
            previous, current = start.cell, following
            path = [start.cell]
            while current != start.cell and current not in path and len(path) <= len(cells):
                path.append(current)
                choices = [candidate for candidate in adjacency.get(current, [])
                           if candidate != previous]
                if len(choices) != 1:
                    break
                previous, current = current, choices[0]

    visited = set()
    components: list[set[tuple[int, int]]] = []
    for cell in cells:
        if cell in visited:
            continue
        component = set()
        stack = [cell]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency.get(current, []))
        visited.update(component)
        components.append(component)
    if len(components) > 1:
        smallest = min(components, key=len)
        issues.append(_issue("DISCONNECTED_ISLAND", "The track contains a disconnected section.", smallest))
        if any(len(component) >= 4 and all(len(set(adjacency[c])) == 2 for c in component)
               for component in components):
            issues.append(_issue("SUB_LOOP", "The track contains a smaller internal loop.", smallest))
    if cells and len(path) != len(cells):
        issues.append(_issue("NOT_SINGLE_LOOP", "All pieces must form one ordered closed loop.", cells))

    if len(starts) == 1 and len(path) == len(cells) and len(path) >= 3:
        start = starts[0]
        approach = cells[path[-1]]
        aligned = set(start.ports)
        if approach.kind != "straight" or set(approach.ports) != aligned:
            issues.append(_issue(
                "INSUFFICIENT_START_CLEARANCE",
                "The start line needs a straight approach for the car drop area.",
                [start.cell, approach.cell],
            ))

    # Deduplicate exact issue instances without losing stable display order.
    unique: list[ValidationIssue] = []
    keys = set()
    for issue in issues:
        key = issue.code, issue.message, tuple(sorted(issue.cells))
        if key not in keys:
            keys.add(key)
            unique.append(issue)
    blocking = [issue for issue in unique if issue.severity == "error"]
    valid = not blocking
    complete_path = path if valid and len(path) == len(cells) else []
    metrics = calculate_metrics(track, complete_path, open_count)
    return TrackValidationResult(valid, unique, complete_path, metrics)


def infer_tile(cell: tuple[int, int], previous: tuple[int, int],
               following: tuple[int, int], start: bool = False) -> Tile:
    directions = []
    for other in (previous, following):
        delta = other[0] - cell[0], other[1] - cell[1]
        directions.append({value: key for key, value in VECTORS.items()}[delta])
    target = set(directions)
    if start:
        forward = directions[1]
        rotation = {value: key for key, value in FORWARD_PORT.items()}[forward]
        if set(PORTS["start_finish"][rotation]) != target:
            raise ValueError("Start/finish must lie on a straight path")
        return Tile(*cell, "start_finish", rotation)
    for kind in ("straight", "corner"):
        for rotation, ports in PORTS[kind].items():
            if set(ports) == target:
                return Tile(*cell, kind, rotation)
    raise ValueError("Path contains a non-orthogonal or branching connection")


def track_from_path(name: str, path: list[tuple[int, int]], difficulty: int,
                    timeout: float | None = None, *, source: str = "campaign",
                    track_id: str | None = None, grid_size: tuple[int, int] = GRID_SIZE,
                    generation: dict | None = None) -> TrackDefinition:
    tiles = [infer_tile(cell, path[index - 1], path[(index + 1) % len(path)], index == 0)
             for index, cell in enumerate(path)]
    track = TrackDefinition(name, tiles, timeout or 30.0, difficulty, source,
                            track_id or str(uuid.uuid4()), grid_size=grid_size,
                            generation=generation or {})
    result = validate_track(track)
    if not result.valid:
        raise ValueError("; ".join(result.messages))
    track.metrics = result.metrics.to_dict()
    track.timeout = float(timeout if timeout is not None else result.metrics.recommended_timeout)
    return track
