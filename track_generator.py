"""Deterministic constrained simple-cycle track generation."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from track_geometry import GRID_SIZE, TrackDefinition, track_from_path, validate_track


@dataclass(frozen=True)
class TrackGenerationConstraints:
    min_tiles: int = 12
    max_tiles: int = 46
    min_corners: int = 4
    max_corners: int = 24
    minimum_start_clearance: int = 1
    min_difficulty_score: float | None = None
    max_difficulty_score: float | None = None
    attempt_limit: int = 200
    node_limit_per_attempt: int = 50_000


@dataclass(frozen=True)
class GenerationStats:
    attempts: int
    visited_nodes: int
    used_fallback: bool = False


def _neighbors(cell: tuple[int, int]):
    x, y = cell
    return [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]


def _within(cell: tuple[int, int], size: tuple[int, int]) -> bool:
    return 0 <= cell[0] < size[0] and 0 <= cell[1] < size[1]


def _valid_extension(candidate: tuple[int, int], current: tuple[int, int],
                     used: set[tuple[int, int]], size: tuple[int, int],
                     reserved: set[tuple[int, int]]) -> bool:
    if not _within(candidate, size) or candidate in used or candidate in reserved:
        return False
    # An induced cycle has no orthogonally adjacent non-consecutive cells.
    return all(neighbor == current or neighbor not in used for neighbor in _neighbors(candidate))


def _fallback_path(size: tuple[int, int], difficulty: int, seed: int = 0) -> list[tuple[int, int]]:
    width, height = size
    inset = 1
    rng = random.Random((seed << 4) ^ difficulty)
    left = inset + rng.randint(0, max(0, min(2, width - 9)))
    top = inset + rng.randint(0, max(0, min(1, height - 7)))
    right = min(width - 2, left + 6 + min(4, difficulty // 3) + rng.randint(0, 2))
    bottom = min(height - 2, top + 4 + min(2, difficulty // 5) + rng.randint(0, 1))
    if right - left < 4 or bottom - top < 3:
        raise ValueError("Grid is too small for a valid fallback track")
    # Start on the top straight with at least one straight on each side.
    path = [(left + 2, top)]
    path += [(x, top) for x in range(left + 3, right + 1)]
    path += [(right, y) for y in range(top + 1, bottom + 1)]
    path += [(x, bottom) for x in range(right - 1, left - 1, -1)]
    path += [(left, y) for y in range(bottom - 1, top - 1, -1)]
    path += [(left + 1, top)]
    return path


def _difficulty_bounds(difficulty: int, constraints: TrackGenerationConstraints) -> tuple[float, float]:
    default_high = {1: 3.0, 2: 3.6, 3: 4.6}.get(difficulty, min(10.0, difficulty + 1.15))
    return (
        constraints.min_difficulty_score if constraints.min_difficulty_score is not None
        else max(1.0, difficulty - 1.15),
        constraints.max_difficulty_score if constraints.max_difficulty_score is not None
        else default_high,
    )


def generate_track(seed: int, difficulty: int, grid_size: tuple[int, int] = GRID_SIZE,
                   constraints: TrackGenerationConstraints | None = None) -> TrackDefinition:
    constraints = constraints or TrackGenerationConstraints()
    if not 1 <= difficulty <= 10:
        raise ValueError("difficulty must be between 1 and 10")
    if grid_size[0] < 7 or grid_size[1] < 6:
        raise ValueError("grid_size must be at least 7×6")
    minimum = max(12, constraints.min_tiles)
    maximum = min(constraints.max_tiles, grid_size[0] * grid_size[1])
    low_score, high_score = _difficulty_bounds(difficulty, constraints)
    total_nodes = 0

    for attempt in range(1, constraints.attempt_limit + 1):
        rng = random.Random((int(seed) << 16) ^ (difficulty << 8) ^ attempt)
        horizontal = rng.choice((True, False))
        if horizontal:
            direction = (1, 0)
            sx = rng.randint(3, grid_size[0] - 4)
            sy = rng.randint(1, grid_size[1] - 2)
        else:
            direction = (0, 1)
            sx = rng.randint(1, grid_size[0] - 2)
            sy = rng.randint(3, grid_size[1] - 4)
        start = sx, sy
        departure = sx + direction[0], sy + direction[1]
        post_departure = sx + 2 * direction[0], sy + 2 * direction[1]
        approach = sx - direction[0], sy - direction[1]
        target = sx - 2 * direction[0], sy - 2 * direction[1]
        if not all(_within(cell, grid_size)
                   for cell in (start, departure, post_departure, approach, target)):
            continue

        desired = min(maximum, max(minimum, 12 + difficulty * 3 + rng.randint(-2, 3)))
        path = [start, departure, post_departure]
        used = set(path)
        reserved = {approach, target}
        found_path = None
        for _ in range(min(constraints.node_limit_per_attempt, maximum * 4)):
            total_nodes += 1
            current = path[-1]
            if (target in _neighbors(current) and len(path) + 2 >= minimum
                    and not any(n in used and n != current for n in _neighbors(target))):
                candidate_path = path + [target, approach]
                try:
                    candidate_track = track_from_path(
                        f"Generated {seed}", candidate_path, difficulty,
                        source="generated", track_id=f"generated-{seed}-{difficulty}",
                        grid_size=grid_size,
                    )
                    candidate_metrics = validate_track(candidate_track).metrics
                    if (constraints.min_corners <= candidate_metrics.corner_count <= constraints.max_corners
                            and low_score <= candidate_metrics.difficulty_score <= high_score):
                        found_path = candidate_path
                        break
                except ValueError:
                    pass
                # The layout closed but missed its requested band; retry with
                # a fresh walk rather than exploring an exponential tree.
                break
            if len(path) >= maximum - 2:
                break
            choices = [cell for cell in _neighbors(current)
                       if _valid_extension(cell, current, used, grid_size, reserved)]
            # Do not pass alongside the approach tile; that would prevent the
            # final induced-cycle closure.  Cells beside target remain valid
            # because one of them must become its predecessor.
            choices = [cell for cell in choices if approach not in _neighbors(cell)]
            if not choices:
                break
            rng.shuffle(choices)
            def desirability(cell):
                distance = abs(cell[0] - target[0]) + abs(cell[1] - target[1])
                onward = sum(_valid_extension(n, cell, used | {cell}, grid_size, reserved)
                             for n in _neighbors(cell))
                if len(path) < desired:
                    return distance + onward * 1.5 + rng.random()
                return -distance + onward * .4 + rng.random()
            choices.sort(key=desirability, reverse=True)
            pool = choices[:min(2, len(choices))]
            selected = rng.choice(pool)
            path.append(selected)
            used.add(selected)
        if found_path:
            return track_from_path(
                f"Generated {seed}", found_path, difficulty, source="generated",
                track_id=f"generated-{seed}-{difficulty}", grid_size=grid_size,
                generation={"seed": seed, "difficulty": difficulty, "attempt": attempt,
                            "visited_nodes": total_nodes, "fallback": False},
            )

    fallback = track_from_path(
        f"Generated {seed}", _fallback_path(grid_size, difficulty, seed), difficulty,
        source="generated", track_id=f"generated-{seed}-{difficulty}", grid_size=grid_size,
        generation={"seed": seed, "difficulty": difficulty,
                    "attempt": constraints.attempt_limit, "visited_nodes": total_nodes,
                    "fallback": True},
    )
    return fallback


def generate_with_stats(seed: int, difficulty: int, grid_size: tuple[int, int] = GRID_SIZE,
                        constraints: TrackGenerationConstraints | None = None
                        ) -> tuple[TrackDefinition, GenerationStats]:
    track = generate_track(seed, difficulty, grid_size, constraints)
    generation = track.generation
    return track, GenerationStats(
        attempts=int(generation.get("attempt", 0)),
        visited_nodes=int(generation.get("visited_nodes", 0)),
        used_fallback=bool(generation.get("fallback", False)),
    )
