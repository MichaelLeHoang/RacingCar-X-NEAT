"""Generate and commit the selected deterministic campaign definitions."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaign import LEVEL_NAMES
from track_generator import TrackGenerationConstraints, generate_track


SEEDS = (1200, 3781, 3863, 4138, 4316, 4409, 4644, 4928, 7150)


if __name__ == "__main__":
    tracks = []
    previous_score = 1.0
    for level, seed in enumerate(SEEDS, 2):
        constraints = TrackGenerationConstraints(max_tiles=60) if level == 10 else None
        track = generate_track(seed, level, constraints=constraints)
        track.name = LEVEL_NAMES[level - 1]
        track.track_id = f"level-{level}"
        track.source = "campaign"
        score = float(track.metrics["difficulty_score"])
        if score <= previous_score:
            raise RuntimeError(f"Level {level} is not harder than its predecessor")
        previous_score = score
        tracks.append(track.to_dict())
    destination = ROOT / "data" / "campaign_tracks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"schema_version": 1, "tracks": tracks}, indent=2),
                           encoding="utf-8")
    print(destination.relative_to(ROOT))
