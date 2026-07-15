"""Rebuild the checked-in canonical track component artwork and masks."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pygame

from track_assets import write_canonical_assets


if __name__ == "__main__":
    pygame.init()
    for output in write_canonical_assets():
        print(output.relative_to(ROOT))
    pygame.quit()

