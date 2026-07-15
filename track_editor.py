"""UI-independent Track Builder state, history, and placement policy."""

from __future__ import annotations

import copy
import uuid

from track_geometry import OPPOSITE, PORTS, VECTORS, Tile, TrackDefinition, validate_track


class TrackEditorState:
    def __init__(self):
        self.tiles: dict[tuple[int, int], Tile] = {}
        self.tool = "select"
        self.kind = "straight"
        self.rotation = 0
        self.selected = None
        self.history: list[dict] = []
        self.redo_stack: list[dict] = []
        self.name = "Custom Track"
        self.track_id = str(uuid.uuid4())
        self.pending_replace: tuple[tuple[int, int], Tile] | None = None

    def snapshot(self):
        return {cell: copy.deepcopy(tile) for cell, tile in self.tiles.items()}

    def push_history(self):
        self.history.append(self.snapshot())
        self.redo_stack.clear()

    def definition(self):
        return TrackDefinition(self.name.strip() or "Custom Track", list(self.tiles.values()),
                               source="custom", track_id=self.track_id)

    @property
    def validation(self):
        return validate_track(self.definition())

    def suggested_rotation(self, kind: str, cell: tuple[int, int]) -> int:
        candidates = []
        for rotation in (0, 90, 180, 270):
            ports = PORTS[kind][rotation]
            matches = mismatches = 0
            for direction, (dx, dy) in VECTORS.items():
                neighbor = self.tiles.get((cell[0] + dx, cell[1] + dy))
                if neighbor is None:
                    continue
                connects = direction in ports
                reciprocal = OPPOSITE[direction] in neighbor.ports
                if connects and reciprocal:
                    matches += 1
                elif connects != reciprocal:
                    mismatches += 1
            if matches == 1 and mismatches == 0:
                candidates.append(rotation)
        return candidates[0] if len(candidates) == 1 else self.rotation

    def place(self, cell: tuple[int, int], kind: str, rotation: int | None = None) -> bool:
        rotation = self.suggested_rotation(kind, cell) if rotation is None else rotation
        tile = Tile(*cell, kind, rotation)
        if cell in self.tiles:
            self.pending_replace = cell, tile
            return False
        self.push_history()
        self.tiles[cell] = tile
        self.selected = cell
        return True

    def confirm_replace(self):
        if self.pending_replace:
            cell, tile = self.pending_replace
            self.push_history()
            self.tiles[cell] = tile
            self.selected = cell
            self.pending_replace = None

    def undo(self):
        if self.history:
            self.redo_stack.append(self.snapshot())
            self.tiles = self.history.pop()
            self.selected = None

    def redo(self):
        if self.redo_stack:
            self.history.append(self.snapshot())
            self.tiles = self.redo_stack.pop()
            self.selected = None

    def delete_selected(self):
        if self.selected in self.tiles:
            self.push_history()
            del self.tiles[self.selected]
            self.selected = None

    def rotate_selected(self):
        if self.selected in self.tiles:
            self.push_history()
            self.tiles[self.selected].rotation = (self.tiles[self.selected].rotation + 90) % 360
        else:
            self.rotation = (self.rotation + 90) % 360

    def clear(self):
        if self.tiles:
            self.push_history()
            self.tiles = {}
            self.selected = None

    def load(self, track: TrackDefinition):
        self.tiles = {tile.cell: copy.deepcopy(tile) for tile in track.tiles}
        self.name = track.name
        self.track_id = track.track_id
        self.history = []
        self.redo_stack = []
        self.selected = None
        self.pending_replace = None

    def new(self):
        self.tiles = {}
        self.name = "Custom Track"
        self.track_id = str(uuid.uuid4())
        self.history = []
        self.redo_stack = []
        self.selected = None
        self.pending_replace = None

