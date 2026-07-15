"""Stable campaign catalog and progress rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from track_geometry import TrackDefinition


BASE_DIR = Path(__file__).resolve().parent
CAMPAIGN_PATH = BASE_DIR / "data" / "campaign_tracks.json"


LEVEL_NAMES = (
    "First Lap", "Open Circuit", "Offset Run", "Compact Hairpin", "Switchback",
    "Technical Sprint", "Endurance Mix", "Alternating Apex", "Precision Ring",
    "Final Circuit",
)


def legacy_track() -> TrackDefinition:
    return TrackDefinition(
        name=LEVEL_NAMES[0], tiles=[], timeout=45.0, difficulty=1,
        source="campaign", track_id="level-1", runtime_type="legacy_bitmap",
        metrics={"difficulty_score": 1.0, "recommended_timeout": 45.0},
    )


def campaign_tracks(path: Path = CAMPAIGN_PATH) -> list[TrackDefinition]:
    tracks = [legacy_track()]
    if not path.exists():
        # Development fallback; normal builds commit the generated definitions.
        from track_generator import TrackGenerationConstraints, generate_track
        for level, seed in enumerate((1200, 3781, 3863, 4138, 4316, 4409, 4644, 4928, 7150), 2):
            constraints = TrackGenerationConstraints(max_tiles=60) if level == 10 else None
            track = generate_track(seed, level, constraints=constraints)
            track.name = LEVEL_NAMES[level - 1]
            track.track_id = f"level-{level}"
            track.source = "campaign"
            tracks.append(track)
        return tracks
    content = json.loads(path.read_text(encoding="utf-8"))
    component_tracks = [TrackDefinition.from_dict(item) for item in content["tracks"]]
    if len(component_tracks) != 9:
        raise ValueError("Campaign catalog must contain levels 2 through 10")
    return tracks + component_tracks


@dataclass
class CampaignProgress:
    unlocked: int = 1
    completed: dict[str, bool] = field(default_factory=dict)
    best_times: dict[str, float] = field(default_factory=dict)
    schema_version: int = 2

    @classmethod
    def from_dict(cls, data: dict) -> "CampaignProgress":
        return cls(
            unlocked=max(1, min(10, int(data.get("unlocked", 1)))),
            completed={str(key): bool(value) for key, value in data.get("completed", {}).items()},
            best_times={str(key): float(value) for key, value in data.get("best_times", {}).items()},
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "unlocked": self.unlocked,
            "completed": self.completed,
            "best_times": self.best_times,
        }

    def is_unlocked(self, level: int) -> bool:
        return 1 <= level <= self.unlocked

    def record_completion(self, level: int, elapsed: float) -> bool:
        if not self.is_unlocked(level):
            raise ValueError("A locked level cannot be completed")
        key = str(level)
        previous = self.best_times.get(key)
        personal_best = previous is None or elapsed < previous
        if personal_best:
            self.best_times[key] = elapsed
        self.completed[key] = True
        if level == self.unlocked and level < 10:
            self.unlocked = level + 1
        return personal_best
