from __future__ import annotations

import copy
import math
import os
import time
import uuid
from collections import deque
from pathlib import Path

import neat
import pygame

from campaign import CampaignProgress, campaign_tracks
from race_session import RaceSession, RaceState, TERMINAL_STATES
from racing_core import (
    DEFAULT_CAR_STATS, GRID_SIZE, LOGICAL_SIZE, SENSOR_ANGLES, SENSOR_RANGE, TILE_SIZE,
    ModelRecord, Storage, create_track_runtime,
    deserialize_genome, load_neat_config, piece_atlas, reconcile_genome_innovations,
    serialize_genome, validate_track,
)
from storage import CONTROLLER_VERSION
from track_geometry import OPPOSITE, VECTORS
from track_editor import TrackEditorState
from training_session import TrainingMode, TrainingProfile, TrainingSession, TrainingState
from ui.components import Button, COLORS, PillButton, draw_modal, font


BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "imgs"

CAR_SPECS = {
    "white": {"max_speed": 4.0, "acceleration": .20, "turn_speed": 4.0},
    "red": {"max_speed": 4.4, "acceleration": .22, "turn_speed": 3.7},
    "green": {"max_speed": 3.8, "acceleration": .24, "turn_speed": 4.5},
    "purple": {"max_speed": 4.1, "acceleration": .18, "turn_speed": 4.2},
    "grey": {"max_speed": 4.0, "acceleration": .20, "turn_speed": 4.0},
}


def load_car_sprite(path):
    """Remove connected opaque-black canvas pixels and normalize to 32×64."""
    source = pygame.image.load(path).convert_alpha()
    width, height = source.get_size()
    queue, seen = deque(), set()
    border = ([(x, 0) for x in range(width)] + [(x, height - 1) for x in range(width)]
              + [(0, y) for y in range(height)] + [(width - 1, y) for y in range(height)])
    for point in border:
        color = source.get_at(point)
        if color.a and max(color.r, color.g, color.b) <= 8:
            queue.append(point)
            seen.add(point)
    while queue:
        x, y = queue.popleft()
        source.set_at((x, y), (0, 0, 0, 0))
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            nx, ny = neighbor
            if 0 <= nx < width and 0 <= ny < height and neighbor not in seen:
                color = source.get_at(neighbor)
                if color.a and max(color.r, color.g, color.b) <= 8:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return pygame.transform.smoothscale(source, (32, 64))


class TrainingStopped(Exception):
    """Compatibility marker retained for external callers."""


class InteractiveApp:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode(LOGICAL_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Racing Car X NEAT")
        self.canvas = pygame.Surface(LOGICAL_SIZE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.scene = "home"
        self.storage = Storage()
        self.campaign = campaign_tracks()
        self.custom = self.storage.custom_tracks()
        self.models = self.storage.models()
        self.progress_record = self.storage.progress_record()
        self.progress = self.progress_record.to_dict()
        self.message = ""
        self.grass = pygame.transform.scale(pygame.image.load(ASSET_DIR / "grass.jpg"), LOGICAL_SIZE)
        self.skins = {
            "white": load_car_sprite(ASSET_DIR / "WhiteCar.png"),
            "red": load_car_sprite(ASSET_DIR / "RedCar.png"),
            "green": load_car_sprite(ASSET_DIR / "green-car.png"),
            "purple": load_car_sprite(ASSET_DIR / "purple-car.png"),
            "grey": load_car_sprite(ASSET_DIR / "grey-car.png"),
        }
        self.selected_model_id = self.models[0].model_id if self.models else None
        self.inventory_page = 0
        self.drag_model = None
        self.drag_origin = None
        self.drag_return = None
        self.rename_model = None
        self.rename_text = ""
        self.delete_model = None
        self.race_session: RaceSession | None = None
        self.race = None
        self.race_result = None
        self.race_persisted = False
        self._attempt_saved = False
        self.training_mode = TrainingMode.ORIGINAL
        self.training_tracks = []
        self.training_target_track = None
        self.training_lineage_id = str(uuid.uuid4())
        self.training_skin = "white"
        self.training_seed = None
        self.training_base_seed = 1
        self.training_difficulty = (1, 5)
        self.training_speed = 1
        self.training_session: TrainingSession | None = None
        self.population = None
        self.champion = None
        self.champion_validation = {}
        self.training_generation = 0
        self.training_active = False
        self.training_paused = False
        self.training_stop_requested = False
        self.training_finisher = None
        self.show_training_save_modal = False
        self.show_training_inventory = False
        self.model_name = "New Racer"
        self.typing_name = False
        self.typing_seed = False
        self.training_profiles = []
        self.active_training_profile = 0
        self.add_training_profile(initial=True)
        self.editor_state = TrackEditorState()
        self.editor_drag_kind = None
        self.editor_typing = False
        self.editor_error = ""
        self.show_track_catalog = False
        self.track_thumbnail_cache = {}

    # ---------- application shell ----------
    def logical_mouse(self):
        wx, wy = self.window.get_size()
        mx, my = pygame.mouse.get_pos()
        scale = min(wx / LOGICAL_SIZE[0], wy / LOGICAL_SIZE[1])
        ox = (wx - LOGICAL_SIZE[0] * scale) / 2
        oy = (wy - LOGICAL_SIZE[1] * scale) / 2
        return (mx - ox) / scale, (my - oy) / scale

    def background(self):
        self.canvas.blit(self.grass, (0, 0))
        shade = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        shade.fill((8, 14, 18, 178))
        self.canvas.blit(shade, (0, 0))

    def title(self, text, subtitle=""):
        self.canvas.blit(font(46, True).render(text, True, COLORS["white"]), (42, 22))
        if subtitle:
            self.canvas.blit(font(19).render(subtitle, True, COLORS["muted"]), (44, 74))

    def present(self):
        size = self.window.get_size()
        scale = min(size[0] / 1280, size[1] / 800)
        fitted = int(1280 * scale), int(800 * scale)
        frame = pygame.transform.smoothscale(self.canvas, fitted)
        self.window.fill((5, 8, 10))
        self.window.blit(frame, ((size[0] - fitted[0]) // 2, (size[1] - fitted[1]) // 2))
        pygame.display.flip()

    def buttons(self):
        if self.scene == "home":
            return [Button((475, 320 + index * 92, 330, 62), label, action)
                    for index, (label, action) in enumerate((
                        ("Play", "levels"), ("Train", "train"), ("Track Builder", "editor")
                    ))]
        target = "levels" if self.scene == "race" else "home"
        return [Button((1090, 24, 145, 42), "Back", target)]

    def handle_common(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.rename_model or self.delete_model or self.pending_replace:
                self.rename_model = self.delete_model = self.pending_replace = None
            elif self.show_training_save_modal:
                self.show_training_save_modal = False
            elif self.show_training_inventory:
                self.show_training_inventory = False
            elif self.scene == "home":
                self.running = False
            elif self.scene == "race":
                if self.race_session and self.race_session.state == RaceState.RUNNING:
                    self.race_session.pause()
                else:
                    self.scene = "levels"
            else:
                self.scene = "home"
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for button in self.buttons():
                if button.enabled and button.rect.collidepoint(self.logical_mouse()):
                    self.scene = button.action
                    self.message = ""
                    return True
        return False

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if not self.handle_common(event):
                    getattr(self, f"event_{self.scene}")(event)
            getattr(self, f"update_{self.scene}")()
            self.background()
            getattr(self, f"draw_{self.scene}")()
            mouse = self.logical_mouse()
            for button in self.buttons():
                button.draw(self.canvas, mouse)
            if self.message:
                self.canvas.blit(font(17, True).render(self.message, True, COLORS["cyan"]), (44, 770))
            self.present()
            self.clock.tick(60)
        pygame.quit()

    # ---------- home and levels ----------
    def event_home(self, event):
        pass

    def update_home(self):
        pass

    def draw_home(self):
        logo = font(64, True).render("RACING CAR", True, COLORS["white"])
        accent = font(64, True).render("X NEAT", True, COLORS["red"])
        self.canvas.blit(logo, logo.get_rect(center=(640, 145)))
        self.canvas.blit(accent, accent.get_rect(center=(640, 215)))
        self.canvas.blit(font(21).render("Evolve. Race. Build.", True, COLORS["muted"]), (535, 265))

    def level_cards(self):
        return [(pygame.Rect(55 + (index % 5) * 245, 135 + (index // 5) * 270, 215, 220),
                 track, index + 1) for index, track in enumerate(self.campaign)]

    def event_levels(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse = self.logical_mouse()
            for rect, track, level in self.level_cards():
                if rect.collidepoint(mouse) and self.progress_record.is_unlocked(level):
                    self.prepare_race(track, level)
                    return

    def update_levels(self):
        pass

    def track_thumbnail(self, track, size=(187, 112)):
        key = track.track_id, size
        if key not in self.track_thumbnail_cache:
            runtime = create_track_runtime(track)
            preview = pygame.Surface(size, pygame.SRCALPHA)
            preview.fill((28, 55, 32, 255))
            scale = min(size[0] / runtime.size[0], size[1] / runtime.size[1])
            fitted = max(1, int(runtime.size[0] * scale)), max(1, int(runtime.size[1] * scale))
            image = pygame.transform.smoothscale(runtime.surface, fitted)
            preview.blit(image, ((size[0] - fitted[0]) // 2, (size[1] - fitted[1]) // 2))
            self.track_thumbnail_cache[key] = preview
        return self.track_thumbnail_cache[key]

    def draw_levels(self):
        self.title("Level Selection", "Choose an available level, then place a trained car at its start")
        mouse = self.logical_mouse()
        for rect, track, level in self.level_cards():
            unlocked = self.progress_record.is_unlocked(level)
            completed = self.progress_record.completed.get(str(level), False)
            color = COLORS["card"] if unlocked else (29, 34, 38)
            pygame.draw.rect(self.canvas, color, rect, border_radius=22)
            preview = self.track_thumbnail(track)
            if not unlocked:
                preview = preview.copy()
                overlay = pygame.Surface(preview.get_size(), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                preview.blit(overlay, (0, 0))
            self.canvas.blit(preview, (rect.x + 14, rect.y + 43))
            self.canvas.blit(font(25, True).render(f"{level}", True,
                            COLORS["cyan"] if unlocked else COLORS["muted"]), (rect.x + 15, rect.y + 10))
            self.canvas.blit(font(16, True).render(track.name[:20], True,
                            COLORS["white"] if unlocked else COLORS["muted"]), (rect.x + 48, rect.y + 15))
            if not unlocked:
                # Primitive lock icon keeps the application asset-independent.
                pygame.draw.rect(self.canvas, COLORS["muted"], (rect.centerx - 13, rect.bottom - 45, 26, 22), 2, 4)
                pygame.draw.arc(self.canvas, COLORS["muted"], (rect.centerx - 10, rect.bottom - 60, 20, 25), math.pi, 0, 2)
            elif completed:
                best = self.progress_record.best_times.get(str(level), 0.0)
                self.canvas.blit(font(18, True).render("✓", True, COLORS["green"]), (rect.x + 15, rect.bottom - 39))
                self.canvas.blit(font(15).render(f"Best {best:.2f}s", True, COLORS["white"]),
                                 (rect.x + 48, rect.bottom - 36))
            if unlocked and rect.collidepoint(mouse):
                pygame.draw.rect(self.canvas, COLORS["cyan"], rect, 3, border_radius=22)

    # ---------- shared inventory ----------
    def refresh_models(self, selected_id=None):
        selected_id = selected_id or self.selected_model_id
        self.models = self.storage.models()
        ids = [model.model_id for model in self.models]
        self.selected_model_id = selected_id if selected_id in ids else (ids[0] if ids else None)
        self.inventory_page = min(self.inventory_page, self.inventory_pages() - 1)

    def selected_model(self):
        return next((model for model in self.models if model.model_id == self.selected_model_id), None)

    def inventory_pages(self):
        return max(1, math.ceil(len(self.models) / 6))

    def model_cards(self):
        start = self.inventory_page * 6
        return [(pygame.Rect(950, 162 + index * 78, 280, 68), model)
                for index, model in enumerate(self.models[start:start + 6])]

    def training_inventory_cards(self):
        """The Train scene pages over the exact same records as Play."""
        start = self.inventory_page * 6
        return [(pygame.Rect(185 + (index % 3) * 305,
                                  235 + (index // 3) * 145, 285, 130), model)
                for index, model in enumerate(self.models[start:start + 6])]

    def training_inventory_layout(self):
        """Compact one-row pages while reserving room for a full six-card page."""
        start = self.inventory_page * 6
        visible = len(self.models[start:start + 6])
        if visible <= 3:
            return {"modal_height": 545, "actions_y": 420,
                    "scope_y": 470, "footer_y": 510}
        return {"modal_height": 620, "actions_y": 535,
                "scope_y": 590, "footer_y": 625}

    def begin_rename(self, model):
        if model:
            self.rename_model = model
            self.rename_text = model.name

    def confirm_rename(self):
        if self.rename_model:
            selected = self.rename_model.model_id
            self.storage.rename_model(selected, self.rename_text)
            self.rename_model = None
            self.refresh_models(selected)
            self.message = "Model renamed"

    def confirm_delete_model(self):
        if self.delete_model:
            name = self.delete_model.name
            self.storage.delete_model(self.delete_model.model_id)
            self.delete_model = None
            self.refresh_models()
            self.message = f"Deleted {name}"

    def _event_model_modal(self, event):
        mouse = self.logical_mouse()
        if self.rename_model and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.confirm_rename()
            elif event.key == pygame.K_BACKSPACE:
                self.rename_text = self.rename_text[:-1]
            elif event.unicode.isprintable() and len(self.rename_text) < 24:
                self.rename_text += event.unicode
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if pygame.Rect(430, 430, 190, 48).collidepoint(mouse):
                self.confirm_rename() if self.rename_model else self.confirm_delete_model()
            elif pygame.Rect(660, 430, 190, 48).collidepoint(mouse):
                self.rename_model = self.delete_model = None
            return True
        return bool(self.rename_model or self.delete_model)

    def _draw_model_modal(self):
        mouse = self.logical_mouse()
        if self.rename_model:
            draw_modal(self.canvas, (385, 275, 510, 235), "Rename model", ("Up to 24 characters",))
            pygame.draw.rect(self.canvas, COLORS["card"], (420, 355, 440, 48), border_radius=16)
            self.canvas.blit(font(20).render(self.rename_text, True, COLORS["white"]), (437, 367))
            Button((430, 430, 190, 48), "Save Name", enabled=bool(self.rename_text.strip())).draw(self.canvas, mouse)
            Button((660, 430, 190, 48), "Cancel").draw(self.canvas, mouse)
        elif self.delete_model:
            draw_modal(self.canvas, (385, 275, 510, 235), "Delete model?",
                       (f"{self.delete_model.name} will be removed permanently.", "This cannot be undone."))
            Button((430, 430, 190, 48), "Delete Model").draw(self.canvas, mouse)
            Button((660, 430, 190, 48), "Cancel").draw(self.canvas, mouse)

    # ---------- race preparation and race ----------
    def prepare_race(self, track, level=None):
        if level and not self.progress_record.is_unlocked(level):
            self.message = "That level is locked"
            return False
        runtime = create_track_runtime(track)
        self.race_session = RaceSession(runtime, level)
        self.race = {"runtime": runtime, "model": None, "net": None, "car": None, "level": level}
        self.race_result = None
        self.race_persisted = False
        self._attempt_saved = False
        self.drag_model = None
        self.scene = "race"
        self.refresh_models()
        return True

    def _controller_for(self, model):
        genome = deserialize_genome(model.genome)
        return neat.nn.FeedForwardNetwork.create(genome, load_neat_config(BASE_DIR))

    def drop_model(self, model):
        try:
            controller = self._controller_for(model)
            accepted = self.race_session.accept_drop(model, self.skins[model.skin], controller)
        except (ValueError, KeyError, TypeError):
            accepted = False
        if accepted:
            self.race.update({"model": model, "net": controller, "car": self.race_session.car})
            self.selected_model_id = model.model_id
        return accepted

    def start_race(self, track, model, level=None):
        """Compatibility helper: prepare and accept a supplied model."""
        self.prepare_race(track, level)
        self.drop_model(model)

    def start_next_level(self):
        level = self.race_session.level if self.race_session else None
        if level and level < 10:
            self.prepare_race(self.campaign[level], level + 1)
        else:
            self.scene = "levels"

    def retry_race(self):
        if self.race_session:
            self.prepare_race(self.race_session.runtime.definition, self.race_session.level)

    def _drop_rect(self):
        zone = self.race_session.runtime.drop_zone
        ox, oy = self.race_session.runtime.origin
        return pygame.Rect(ox + zone.offset[0], oy + zone.offset[1],
                           zone.mask.get_size()[0], zone.mask.get_size()[1])

    def event_race(self, event):
        if self.rename_model or self.delete_model:
            self._event_model_modal(event)
            return
        mouse = self.logical_mouse()
        session = self.race_session
        if session.terminal and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if session.state == RaceState.COMPLETE:
                if pygame.Rect(310, 450, 200, 48).collidepoint(mouse): self.start_next_level()
                elif pygame.Rect(540, 450, 200, 48).collidepoint(mouse): self.retry_race()
                elif pygame.Rect(770, 450, 200, 48).collidepoint(mouse): self.scene = "levels"
            else:
                if pygame.Rect(310, 450, 200, 48).collidepoint(mouse): self.retry_race()
                elif pygame.Rect(540, 450, 200, 48).collidepoint(mouse):
                    if self.continue_training_model(session.model, session.runtime.definition):
                        self.scene = "train"
                elif pygame.Rect(770, 450, 200, 48).collidepoint(mouse): self.scene = "levels"
            return
        if session.state == RaceState.PAUSED and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if pygame.Rect(535, 405, 210, 50).collidepoint(mouse): session.resume()
            return
        if session.state != RaceState.PREPARING:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                session.pause() if session.state == RaceState.RUNNING else session.resume()
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, model in self.model_cards():
                if rect.collidepoint(mouse):
                    self.selected_model_id = model.model_id
                    self.drag_model = model
                    self.drag_origin = rect.center
                    return
            if pygame.Rect(955, 646, 80, 36).collidepoint(mouse) and self.inventory_page > 0:
                self.inventory_page -= 1
            elif pygame.Rect(1045, 646, 80, 36).collidepoint(mouse) and self.inventory_page + 1 < self.inventory_pages():
                self.inventory_page += 1
            elif pygame.Rect(1140, 646, 90, 36).collidepoint(mouse): self.begin_rename(self.selected_model())
            elif pygame.Rect(1138, 690, 92, 36).collidepoint(mouse): self.delete_model = self.selected_model()
            elif pygame.Rect(1045, 690, 84, 36).collidepoint(mouse) and self.selected_model():
                path = self.storage.export_model(self.selected_model()); self.message = f"Exported {path.name}"
            elif pygame.Rect(955, 690, 80, 36).collidepoint(mouse):
                report = self.storage.import_inbox(); self.refresh_models(); self.custom = self.storage.custom_tracks()
                self.message = f"Imported {report.imported} item(s)"
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drag_model:
            if self._drop_rect().collidepoint(mouse) and self.drop_model(self.drag_model):
                self.message = "Car accepted"
            else:
                self.drag_return = {"model": self.drag_model, "start": mouse,
                                    "end": self.drag_origin or mouse, "time": time.perf_counter()}
            self.drag_model = None

    def update_race(self):
        if self.drag_return and time.perf_counter() - self.drag_return["time"] > .22:
            self.drag_return = None
        session = self.race_session
        if not session:
            return
        previous = session.state
        session.update(1 / 60)
        if previous == RaceState.COUNTDOWN and session.state == RaceState.RUNNING and not self._attempt_saved:
            self.storage.save_model(session.model)
            self._attempt_saved = True
        self.race_result = session.state.value if session.terminal else None
        if session.terminal and not self.race_persisted:
            self._persist_race()

    def _persist_race(self):
        session = self.race_session
        model = session.model
        if session.state == RaceState.COMPLETE:
            elapsed = session.finished_elapsed_time
            previous = model.best_times.get(session.runtime.definition.track_id)
            session.new_personal_best = previous is None or elapsed < previous
            if session.new_personal_best:
                model.best_times[session.runtime.definition.track_id] = elapsed
            model.wins += 1
            if session.level:
                self.progress_record.record_completion(session.level, elapsed)
                self.progress = self.progress_record.to_dict()
                self.storage.save_progress(self.progress_record)
        self.storage.save_model(model)
        self.refresh_models(model.model_id)
        self.race_persisted = True

    def draw_runtime(self, runtime, cars):
        ox, oy = runtime.origin
        self.canvas.blit(runtime.surface, (ox, oy))
        for car in cars:
            cx, cy = car.center
            for angle, value in zip(SENSOR_ANGLES, car.sensor_values):
                radians = math.radians(car.angle + angle)
                distance = value * SENSOR_RANGE
                end = ox + cx - math.sin(radians) * distance, oy + cy - math.cos(radians) * distance
                pygame.draw.line(self.canvas, COLORS["cyan"], (ox + cx, oy + cy), end, 1)
            image, rect = car.rotated()
            self.canvas.blit(image, (ox + rect.x, oy + rect.y))

    def _draw_drop_zone(self):
        rect = self._drop_rect()
        highlight = self.drag_model and rect.collidepoint(self.logical_mouse())
        color = COLORS["green"] if highlight else COLORS["cyan"]
        phase = int(time.perf_counter() * 10) % 12
        for x in range(rect.left - phase, rect.right, 12):
            pygame.draw.line(self.canvas, color, (max(rect.left, x), rect.top), (min(rect.right, x + 6), rect.top), 2)
            pygame.draw.line(self.canvas, color, (max(rect.left, x), rect.bottom), (min(rect.right, x + 6), rect.bottom), 2)
        for y in range(rect.top - phase, rect.bottom, 12):
            pygame.draw.line(self.canvas, color, (rect.left, max(rect.top, y)), (rect.left, min(rect.bottom, y + 6)), 2)
            pygame.draw.line(self.canvas, color, (rect.right, max(rect.top, y)), (rect.right, min(rect.bottom, y + 6)), 2)
        label = font(13, True).render("DROP CAR TO START", True, color)
        self.canvas.blit(label, label.get_rect(midbottom=(rect.centerx, rect.top - 5)))

    def draw_model_ghost(self, model, position):
        image = self.skins[model.skin]
        self.canvas.blit(image, image.get_rect(center=(position[0], position[1] - 10)))
        label = font(14, True).render(model.name, True, COLORS["white"])
        self.canvas.blit(label, label.get_rect(midtop=(position[0], position[1] + 25)))

    def draw_race(self):
        session = self.race_session
        if not session:
            return
        if session.runtime.definition.runtime_type != "legacy_bitmap":
            self.title(session.runtime.definition.name,
                       "Prepare your car" if session.state == RaceState.PREPARING else "Ordered lap in progress")
        self.draw_runtime(session.runtime, [session.car] if session.car else [])
        if session.state == RaceState.PREPARING:
            self._draw_drop_zone()
            pygame.draw.rect(self.canvas, COLORS["panel"], (935, 95, 310, 650), border_radius=24)
            count = len(self.models)
            self.canvas.blit(font(21, True).render(f"MODEL DRAWER · {count}", True, COLORS["white"]), (955, 115))
            if not self.models:
                self.canvas.blit(font(17).render("No trained cars yet.", True, COLORS["muted"]), (955, 185))
                self.canvas.blit(font(15).render("Use Train, then save a Draft", True, COLORS["muted"]), (955, 215))
                self.canvas.blit(font(15).render("or Validated Champion.", True, COLORS["muted"]), (955, 238))
            mouse = self.logical_mouse()
            for rect, model in self.model_cards():
                pygame.draw.rect(self.canvas, COLORS["card"], rect, border_radius=15)
                icon = pygame.transform.smoothscale(self.skins[model.skin], (24, 48))
                self.canvas.blit(icon, icon.get_rect(midleft=(rect.x + 16, rect.centery)))
                self.canvas.blit(font(16, True).render(model.name[:20], True, COLORS["white"]), (rect.x + 48, rect.y + 7))
                status_color = COLORS["green"] if model.status == "validated" else COLORS["muted"]
                self.canvas.blit(font(13).render(f"{model.status.title()} · Gen {model.generation}", True, status_color), (rect.x + 48, rect.y + 29))
                self.canvas.blit(font(12).render(f"Fitness {model.fitness:.1f} · Tracks {model.trained_track_count}", True, COLORS["muted"]), (rect.x + 48, rect.y + 48))
                if model.model_id == self.selected_model_id:
                    pygame.draw.rect(self.canvas, COLORS["cyan"], rect, 2, border_radius=15)
            Button((955, 646, 80, 36), "Prev", enabled=self.inventory_page > 0).draw(self.canvas, mouse)
            Button((1045, 646, 80, 36), "Next", enabled=self.inventory_page + 1 < self.inventory_pages()).draw(self.canvas, mouse)
            Button((1140, 646, 90, 36), "Rename", enabled=self.selected_model() is not None).draw(self.canvas, mouse)
            Button((955, 690, 80, 36), "Import").draw(self.canvas, mouse)
            Button((1045, 690, 84, 36), "Export", enabled=self.selected_model() is not None).draw(self.canvas, mouse)
            Button((1138, 690, 92, 36), "Delete", enabled=self.selected_model() is not None).draw(self.canvas, mouse)
            if self.drag_model:
                self.draw_model_ghost(self.drag_model, mouse)
            elif self.drag_return:
                elapsed = min(1.0, (time.perf_counter() - self.drag_return["time"]) / .22)
                eased = 1 - (1 - elapsed) ** 3
                start, end = self.drag_return["start"], self.drag_return["end"]
                point = start[0] + (end[0] - start[0]) * eased, start[1] + (end[1] - start[1]) * eased
                self.draw_model_ghost(self.drag_return["model"], point)
        else:
            pygame.draw.rect(self.canvas, COLORS["ink"], (950, 105, 280, 150), border_radius=20)
            self.canvas.blit(font(24, True).render(f"{session.displayed_time:.2f}s", True, COLORS["white"]), (972, 125))
            self.canvas.blit(font(17).render(session.state.value, True, COLORS["cyan"]), (972, 165))
            if session.car:
                self.canvas.blit(font(15).render(f"Gates {session.car.next_checkpoint}/{len(session.runtime.gates)}", True, COLORS["muted"]), (972, 198))
                self.canvas.blit(font(15).render(session.model.name, True, COLORS["muted"]), (972, 222))
        if session.state == RaceState.COUNTDOWN:
            label = "READY" if session.countdown_remaining > .3 else "GO"
            self.canvas.blit(font(64, True).render(label, True, COLORS["cyan"]),
                             font(64, True).render(label, True, COLORS["cyan"]).get_rect(center=(640, 390)))
        elif session.state == RaceState.PAUSED:
            draw_modal(self.canvas, (430, 290, 420, 210), "Paused", ("Race time is frozen.",))
            Button((535, 405, 210, 50), "Resume").draw(self.canvas, self.logical_mouse())
        elif session.terminal:
            title = "Lap Complete" if session.state == RaceState.COMPLETE else session.state.value.title()
            lines = [f"Frozen time: {session.displayed_time:.2f}s"]
            if session.state == RaceState.COMPLETE and session.new_personal_best:
                lines.append("New personal best!")
            draw_modal(self.canvas, (260, 245, 760, 300), title, lines)
            labels = ("Next Level", "Retry", "Level Select") if session.state == RaceState.COMPLETE else ("Retry", "Train This Car", "Level Select")
            for index, label in enumerate(labels):
                Button((310 + index * 230, 450, 200, 48), label).draw(self.canvas, self.logical_mouse())
        self._draw_model_modal()

    # ---------- training ----------
    def capture_training_profile(self):
        return {"name": self.model_name, "skin": self.training_skin, "seed": self.training_seed,
                "tracks": list(self.training_tracks), "mode": self.training_mode,
                "target_track": self.training_target_track,
                "lineage_id": self.training_lineage_id,
                "base_seed": self.training_base_seed, "difficulty": self.training_difficulty,
                "population": self.population, "session": self.training_session,
                "generation": self.training_generation, "champion": self.champion,
                "validation": dict(self.champion_validation)}

    def apply_training_profile(self, state):
        self.model_name, self.training_skin, self.training_seed = state["name"], state["skin"], state["seed"]
        self.training_tracks, self.training_mode = list(state["tracks"]), state.get("mode", TrainingMode.ORIGINAL)
        self.training_target_track = state.get("target_track")
        self.training_lineage_id = state.get("lineage_id", str(uuid.uuid4()))
        self.training_base_seed = state.get("base_seed", 1)
        self.training_difficulty = tuple(state.get("difficulty", (1, 5)))
        self.population, self.training_session = state["population"], state.get("session")
        self.training_generation, self.champion = state["generation"], state["champion"]
        self.champion_validation = dict(state["validation"])

    def save_active_training_profile(self):
        if self.training_profiles:
            self.training_profiles[self.active_training_profile] = self.capture_training_profile()

    def add_training_profile(self, initial=False):
        if not initial:
            self.save_active_training_profile()
            self.model_name = f"Model {len(self.training_profiles) + 1}"
            self.training_skin = "white"; self.training_seed = None; self.training_tracks = []
            self.training_target_track = None
            self.training_lineage_id = str(uuid.uuid4())
            self.training_mode = TrainingMode.ORIGINAL; self.population = None; self.training_session = None
            self.training_base_seed = 1; self.training_difficulty = (1, 5)
            self.training_generation = 0; self.champion = None; self.champion_validation = {}
        self.training_profiles.append(self.capture_training_profile())
        self.active_training_profile = len(self.training_profiles) - 1

    def switch_training_profile(self, index):
        if index != self.active_training_profile and 0 <= index < len(self.training_profiles):
            self.save_active_training_profile(); self.active_training_profile = index
            self.apply_training_profile(self.training_profiles[index])

    def init_population(self):
        config = load_neat_config(BASE_DIR)
        self.population = neat.Population(config)
        if self.training_seed:
            seed = deserialize_genome(self.training_seed.genome)
            seed = reconcile_genome_innovations(seed, self.population)
            seed.fitness = float(self.training_seed.fitness)
            # The saved genome is itself a completed champion.  Keeping it as
            # Population.best_genome means Stop can never discard the model a
            # user explicitly chose to continue from.
            self.population.best_genome = copy.deepcopy(seed)
            keys = list(self.population.population)
            for index, key in enumerate(keys[:max(1, int(len(keys) * .4))]):
                clone = copy.deepcopy(seed); clone.key = key; clone.fitness = None
                if index: clone.mutate(config.genome_config)
                self.population.population[key] = clone
            self.population.generation = max(0, int(self.training_seed.generation))
            self.population.species.speciate(config, self.population.population, self.population.generation)

    def training_track_options(self):
        """Saved custom tracks plus an unsaved campaign retry target, de-duplicated."""
        tracks = []
        if self.training_target_track is not None:
            tracks.append(self.training_target_track)
        tracks.extend(self.storage.custom_tracks(valid_only=True))
        seen = set()
        return [track for track in tracks
                if not (track.track_id in seen or seen.add(track.track_id))]

    def _track_by_id(self, track_id):
        return next((track for track in [*self.campaign, *self.storage.custom_tracks(valid_only=True)]
                     if track.track_id == track_id), None)

    def continue_training_model(self, model, failed_track=None):
        """Seed a new NEAT population from a saved genome and restore its suite."""
        if model is None or model.controller_version != CONTROLLER_VERSION:
            self.message = "That model is not compatible with the five-sensor controller"
            return False
        try:
            champion = deserialize_genome(model.genome)
        except (ValueError, TypeError, KeyError):
            self.message = "That saved genome could not be loaded"
            return False

        self.training_seed = model
        self.training_lineage_id = model.lineage_id or model.model_id
        self.selected_model_id = model.model_id
        self.model_name = model.name
        self.training_skin = model.skin
        self.training_generation = model.generation
        champion.fitness = model.fitness
        self.champion = champion
        self.champion_validation = dict(model.validation)
        self.population = None
        self.training_session = None
        self.training_target_track = failed_track

        if failed_track is not None:
            self.training_mode = (TrainingMode.ORIGINAL
                                  if failed_track.runtime_type == "legacy_bitmap"
                                  else TrainingMode.CUSTOM)
            self.training_tracks = [failed_track]
            self.message = f"Continuing {model.name} on failed track: {failed_track.name}"
        else:
            scope = dict(model.validation_scope or {})
            try:
                self.training_mode = TrainingMode(scope.get("mode", TrainingMode.ORIGINAL.value))
            except ValueError:
                self.training_mode = TrainingMode.ORIGINAL
            self.training_base_seed = int(scope.get("base_seed", 1))
            difficulty = scope.get("difficulty_range", (1, 5))
            self.training_difficulty = tuple(difficulty) if len(difficulty) == 2 else (1, 5)
            track_ids = scope.get("track_ids") or model.trained_tracks
            self.training_tracks = [track for track_id in track_ids
                                    if (track := self._track_by_id(track_id)) is not None]
            if self.training_mode == TrainingMode.ORIGINAL:
                self.training_tracks = [self.campaign[0]]
            elif self.training_mode == TrainingMode.CUSTOM and not self.training_tracks:
                self.message = "Model loaded; choose at least one available track to continue"
                self.save_active_training_profile()
                return True
            self.message = f"Loaded {model.name} for continued training"
        self.save_active_training_profile()
        return True

    def _event_training_inventory(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        mouse = self.logical_mouse()
        layout = self.training_inventory_layout()
        actions_y, footer_y = layout["actions_y"], layout["footer_y"]
        for rect, model in self.training_inventory_cards():
            if rect.collidepoint(mouse):
                self.selected_model_id = model.model_id
                return
        if pygame.Rect(185, actions_y, 100, 38).collidepoint(mouse) and self.inventory_page > 0:
            self.inventory_page -= 1
        elif (pygame.Rect(295, actions_y, 100, 38).collidepoint(mouse)
              and self.inventory_page + 1 < self.inventory_pages()):
            self.inventory_page += 1
        elif pygame.Rect(420, actions_y, 120, 38).collidepoint(mouse):
            self.begin_rename(self.selected_model())
        elif pygame.Rect(550, actions_y, 120, 38).collidepoint(mouse):
            self.delete_model = self.selected_model()
        elif pygame.Rect(680, actions_y, 120, 38).collidepoint(mouse):
            report = self.storage.import_inbox()
            self.refresh_models()
            self.custom = self.storage.custom_tracks()
            self.message = f"Imported {report.imported} item(s)"
        elif (pygame.Rect(810, actions_y, 120, 38).collidepoint(mouse)
              and self.selected_model()):
            path = self.storage.export_model(self.selected_model())
            self.message = f"Exported {path.name}"
        elif pygame.Rect(390, footer_y, 235, 48).collidepoint(mouse):
            if self.continue_training_model(self.selected_model()):
                self.show_training_inventory = False
        elif pygame.Rect(655, footer_y, 235, 48).collidepoint(mouse):
            self.show_training_inventory = False

    def _configured_training_tracks(self):
        if self.training_mode == TrainingMode.ORIGINAL:
            return [self.campaign[0]]
        if self.training_mode == TrainingMode.CUSTOM:
            return [track for track in self.training_tracks if validate_track(track).valid]
        return []

    def start_training(self):
        if self.population is None:
            self.init_population()
        tracks = self._configured_training_tracks()
        if self.training_mode == TrainingMode.CUSTOM and not tracks:
            self.message = "Select at least one valid custom track"
            return
        profile = TrainingProfile(self.training_mode, self.training_skin, self.model_name,
                                  tracks, self.training_base_seed, self.training_difficulty)
        self.training_session = TrainingSession(self.population, profile, self.skins, CAR_SPECS)
        self.training_session.start()
        self.training_active = True
        self.training_paused = False
        self.show_training_save_modal = False
        self.message = "Training started"

    def stop_training(self):
        """Stop safely, snapshot the completed champion, and offer persistence."""
        session = self.training_session
        if session is None:
            return
        session.stop()
        self.training_active = False
        self.training_paused = False
        self.training_generation = session.generation
        self.champion = session.completed_champion
        self.champion_validation = {
            item["track_id"]: item["passed"] for item in session.validation_results
        }
        self.show_training_save_modal = True
        if self.champion is None:
            self.message = "Training stopped before a generation completed"
        else:
            self.message = "Training stopped; choose how to save the completed champion"
        self.save_active_training_profile()

    def _event_training_save_modal(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        mouse = self.logical_mouse()
        if pygame.Rect(285, 455, 210, 48).collidepoint(mouse):
            if self.save_champion(force_draft=True) is not None:
                self.show_training_save_modal = False
        elif pygame.Rect(535, 455, 210, 48).collidepoint(mouse):
            if self.save_champion(force_validated=True) is not None:
                self.show_training_save_modal = False
        elif pygame.Rect(785, 455, 210, 48).collidepoint(mouse):
            self.show_training_save_modal = False

    def event_train(self, event):
        mouse = self.logical_mouse()
        if self.rename_model or self.delete_model:
            self._event_model_modal(event); return
        if self.show_training_save_modal:
            self._event_training_save_modal(event); return
        if self.show_training_inventory:
            self._event_training_inventory(event); return
        live_states = (TrainingState.RUNNING, TrainingState.VALIDATING, TrainingState.PAUSED)
        if self.training_session and self.training_session.state in live_states:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if self.training_session.state == TrainingState.PAUSED: self.training_session.resume()
                else: self.training_session.pause()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(970, 315, 240, 48).collidepoint(mouse):
                    if self.training_session.state == TrainingState.PAUSED: self.training_session.resume()
                    else: self.training_session.pause()
                elif pygame.Rect(970, 378, 240, 48).collidepoint(mouse):
                    self.stop_training()
                for index, speed in enumerate((1, 2, 4, 0)):
                    if pygame.Rect(955 + index * 70, 500, 60, 38).collidepoint(mouse): self.training_speed = speed
            return
        if event.type == pygame.KEYDOWN and self.typing_seed:
            if event.key == pygame.K_RETURN:
                self.typing_seed = False
            elif event.key == pygame.K_BACKSPACE:
                self.training_base_seed = int(str(self.training_base_seed)[:-1] or "0")
            elif event.unicode.isdigit() and len(str(self.training_base_seed)) < 10:
                self.training_base_seed = int(("" if self.training_base_seed == 0 else str(self.training_base_seed)) + event.unicode)
            return
        if event.type == pygame.KEYDOWN and self.typing_name:
            if event.key == pygame.K_RETURN: self.typing_name = False
            elif event.key == pygame.K_BACKSPACE: self.model_name = self.model_name[:-1]
            elif event.unicode.isprintable() and len(self.model_name) < 24: self.model_name += event.unicode
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if pygame.Rect(40, 125, 285, 44).collidepoint(mouse): self.typing_name = True
            for index, mode in enumerate(TrainingMode):
                if pygame.Rect(365 + index * 220, 105, 200, 42).collidepoint(mouse):
                    self.training_mode = mode
            if self.training_mode == TrainingMode.RANDOM_CURRICULUM:
                if pygame.Rect(390, 185, 260, 44).collidepoint(mouse): self.typing_seed = True
                low, high = self.training_difficulty
                if pygame.Rect(390, 255, 44, 36).collidepoint(mouse): low = max(1, low - 1)
                if pygame.Rect(440, 255, 44, 36).collidepoint(mouse): low = min(high, low + 1)
                if pygame.Rect(560, 255, 44, 36).collidepoint(mouse): high = max(low, high - 1)
                if pygame.Rect(610, 255, 44, 36).collidepoint(mouse): high = min(10, high + 1)
                self.training_difficulty = (low, high)
            for index, skin in enumerate(self.skins):
                if pygame.Rect(45 + index * 62, 205, 50, 76).collidepoint(mouse): self.training_skin = skin
            if self.training_mode == TrainingMode.CUSTOM:
                for index, track in enumerate(self.training_track_options()[:8]):
                    if pygame.Rect(365 + (index % 2) * 285, 180 + (index // 2) * 120, 265, 100).collidepoint(mouse):
                        selected = next((item for item in self.training_tracks
                                         if item.track_id == track.track_id), None)
                        if selected: self.training_tracks.remove(selected)
                        else: self.training_tracks.append(track)
            if pygame.Rect(45, 320, 280, 48).collidepoint(mouse):
                if not self.training_session or self.training_session.state in (TrainingState.STOPPED, TrainingState.IDLE): self.start_training()
                elif self.training_session.state == TrainingState.PAUSED: self.training_session.resume()
                else: self.training_session.pause()
            if pygame.Rect(45, 378, 280, 48).collidepoint(mouse) and self.training_session:
                self.stop_training()
            if pygame.Rect(45, 436, 280, 44).collidepoint(mouse): self.save_champion(force_draft=True)
            if pygame.Rect(45, 494, 280, 44).collidepoint(mouse): self.save_champion(force_validated=True)
            for index, speed in enumerate((1, 2, 4, 0)):
                if pygame.Rect(45 + index * 72, 575, 62, 38).collidepoint(mouse): self.training_speed = speed
            if pygame.Rect(45, 700, 44, 44).collidepoint(mouse) and len(self.training_profiles) < 6: self.add_training_profile()
            for index in range(len(self.training_profiles)):
                if pygame.Rect(100 + index * 145, 695, 132, 48).collidepoint(mouse): self.switch_training_profile(index)
            if pygame.Rect(365, 615, 270, 48).collidepoint(mouse):
                self.refresh_models(); self.show_training_inventory = True
            if pygame.Rect(650, 615, 240, 48).collidepoint(mouse):
                self.continue_training_model(self.selected_model())

    def update_train(self):
        session = self.training_session
        if not session:
            return
        if self.training_speed == 0:
            deadline = time.perf_counter() + .012
            steps = 0
            while steps < 240 and time.perf_counter() < deadline:
                session.advance(1)
                steps += 1
        else:
            session.advance(self.training_speed)
        self.training_generation = session.generation
        self.champion = session.completed_champion
        self.champion_validation = {item["track_id"]: item["passed"] for item in session.validation_results}
        self.training_active = session.state in (TrainingState.RUNNING, TrainingState.VALIDATING)
        self.training_paused = session.state == TrainingState.PAUSED
        if session.state == TrainingState.VALIDATED:
            self.message = "Validated champion ready to save"
        self.save_active_training_profile()

    def run_training_generation(self):
        """Compatibility helper used by headless callers; UI uses bounded updates."""
        if self.training_stop_requested:
            if self.training_session: self.training_session.stop()
            self.training_active = False; self.training_stop_requested = False
            self.show_training_save_modal = self.champion is not None
            return
        if not self.training_session:
            # Older callers populate training_tracks directly.
            if self.training_tracks and self.training_mode == TrainingMode.ORIGINAL:
                self.training_mode = (TrainingMode.ORIGINAL if self.training_tracks[0].runtime_type == "legacy_bitmap"
                                      else TrainingMode.CUSTOM)
            self.start_training()
        target = self.training_session.completed_generations + 1
        guard = 0
        while self.training_session.completed_generations < target and guard < 100_000:
            self.training_session.advance(1); guard += 1
        self.update_train()

    def validate_champion(self):
        return dict(self.champion_validation)

    def save_champion(self, force_draft=False, force_validated=False):
        session = self.training_session
        champion = session.completed_champion if session else self.champion
        if champion is None:
            self.message = "Complete at least one generation first"; return None
        validated = bool(session and session.champion_validated)
        if force_validated and not validated:
            self.message = "The complete validation suite has not passed"; return None
        status = "validated" if validated and not force_draft else "draft"
        scope = session.validation_scope if session else {}
        results = list(session.validation_results) if session else []
        model = ModelRecord(
            self.model_name.strip() or "Unnamed Racer", self.training_skin,
            serialize_genome(champion), self.training_generation,
            float(champion.fitness or 0.0), status=status,
            trained_tracks=[track.track_id for track in self._configured_training_tracks()],
            validation={item["track_id"]: item["passed"] for item in results},
            car_stats=dict(CAR_SPECS[self.training_skin]),
            validation_scope=scope, validation_results=results,
            lineage_id=self.training_lineage_id,
        )
        model = self.storage.save_training_snapshot(model)
        self.refresh_models(model.model_id)
        self.message = f"Saved {model.name} as {status.title()}"
        return model

    def draw_train(self):
        if self.training_session and self.training_session.state in (
                TrainingState.RUNNING, TrainingState.VALIDATING, TrainingState.PAUSED):
            self.draw_training_live()
            return
        self.title("Train", "NEAT training stays responsive and validates across the selected mode suite")
        mouse = self.logical_mouse()
        self.canvas.blit(font(15, True).render("MODEL NAME", True, COLORS["muted"]), (45, 102))
        pygame.draw.rect(self.canvas, COLORS["card"], (40, 125, 285, 44), border_radius=16)
        self.canvas.blit(font(18).render(self.model_name, True, COLORS["white"]), (54, 137))
        self.canvas.blit(font(15, True).render("CAR SKIN", True, COLORS["muted"]), (45, 180))
        for index, (skin, image) in enumerate(self.skins.items()):
            rect = pygame.Rect(45 + index * 62, 205, 50, 76)
            pygame.draw.rect(self.canvas, COLORS["cyan"] if skin == self.training_skin else COLORS["card"], rect, 2, border_radius=12)
            self.canvas.blit(image, image.get_rect(center=rect.center))
        for index, mode in enumerate(TrainingMode):
            label = {TrainingMode.ORIGINAL: "Original Track", TrainingMode.CUSTOM: "Custom Tracks",
                     TrainingMode.RANDOM_CURRICULUM: "Random Curriculum"}[mode]
            Button((365 + index * 220, 105, 200, 42), label, selected=self.training_mode == mode).draw(self.canvas, mouse)
        if self.training_mode == TrainingMode.ORIGINAL:
            preview = self.track_thumbnail(self.campaign[0], (520, 300)); self.canvas.blit(preview, (450, 185))
        elif self.training_mode == TrainingMode.CUSTOM:
            valid = self.training_track_options()
            if not valid:
                self.canvas.blit(font(18).render("No valid custom tracks. Build and save one first.", True, COLORS["muted"]), (380, 190))
            for index, track in enumerate(valid[:8]):
                rect = pygame.Rect(365 + (index % 2) * 285, 180 + (index // 2) * 120, 265, 100)
                pygame.draw.rect(self.canvas, COLORS["card"], rect, border_radius=15)
                self.canvas.blit(self.track_thumbnail(track, (115, 70)), (rect.x + 10, rect.y + 12))
                retry = self.training_target_track and track.track_id == self.training_target_track.track_id
                label = f"Retry · {track.name}" if retry else track.name
                self.canvas.blit(font(15, True).render(label[:20], True, COLORS["white"]), (rect.x + 135, rect.y + 18))
                self.canvas.blit(font(13).render(f"Difficulty {track.metrics.get('difficulty_score', 1):.1f}", True, COLORS["muted"]), (rect.x + 135, rect.y + 45))
                if any(item.track_id == track.track_id for item in self.training_tracks):
                    pygame.draw.rect(self.canvas, COLORS["cyan"], rect, 3, border_radius=15)
        else:
            self.canvas.blit(font(15, True).render("BASE SEED", True, COLORS["muted"]), (390, 162))
            pygame.draw.rect(self.canvas, COLORS["card"], (390, 185, 260, 44), border_radius=15)
            self.canvas.blit(font(20).render(str(self.training_base_seed), True, COLORS["white"]), (405, 196))
            self.canvas.blit(font(15, True).render("DIFFICULTY RANGE", True, COLORS["muted"]), (390, 235))
            Button((390, 255, 44, 36), "−").draw(self.canvas, mouse)
            Button((440, 255, 44, 36), "+").draw(self.canvas, mouse)
            self.canvas.blit(font(20, True).render(str(self.training_difficulty[0]), True, COLORS["white"]), (505, 262))
            Button((560, 255, 44, 36), "−").draw(self.canvas, mouse)
            Button((610, 255, 44, 36), "+").draw(self.canvas, mouse)
            self.canvas.blit(font(20, True).render(str(self.training_difficulty[1]), True, COLORS["white"]), (675, 262))
            self.canvas.blit(font(17).render("Three rotating training tracks per generation", True, COLORS["muted"]), (390, 315))
            self.canvas.blit(font(17).render("Three separate held-out seeds for validation", True, COLORS["muted"]), (390, 347))
        state = self.training_session.state.value if self.training_session else "IDLE"
        action = "Resume" if state == "PAUSED" else "Pause" if state in ("RUNNING", "VALIDATING") else "Start"
        Button((45, 320, 280, 48), action).draw(self.canvas, mouse)
        Button((45, 378, 280, 48), "Stop", enabled=self.training_session is not None).draw(self.canvas, mouse)
        Button((45, 436, 280, 44), "Save Current Best", enabled=self.champion is not None).draw(self.canvas, mouse)
        validated = bool(self.training_session and self.training_session.champion_validated)
        Button((45, 494, 280, 44), "Save Validated Champion", enabled=validated).draw(self.canvas, mouse)
        self.canvas.blit(font(14, True).render("SIMULATION SPEED", True, COLORS["muted"]), (45, 550))
        for index, speed in enumerate((1, 2, 4, 0)):
            Button((45 + index * 72, 575, 62, 38), "Max" if speed == 0 else f"{speed}×",
                   selected=self.training_speed == speed).draw(self.canvas, mouse)
        fitness = float(self.champion.fitness or 0) if self.champion else 0.0
        self.canvas.blit(font(16, True).render(f"State {state} · Generation {self.training_generation}", True, COLORS["white"]), (45, 625))
        self.canvas.blit(font(15).render(f"Completed champion fitness {fitness:.2f}", True, COLORS["muted"]), (45, 648))
        self.canvas.blit(font(13, True).render("TRAINING PROFILES", True, COLORS["muted"]), (45, 677))
        Button((45, 700, 44, 44), "+", enabled=len(self.training_profiles) < 6).draw(self.canvas, mouse)
        for index, profile in enumerate(self.training_profiles):
            Button((100 + index * 145, 695, 132, 48), profile["name"][:12], selected=index == self.active_training_profile).draw(self.canvas, mouse)
        selected_model = self.selected_model()
        Button((365, 615, 270, 48), f"Model Inventory · {len(self.models)}").draw(self.canvas, mouse)
        Button((650, 615, 240, 48), "Continue Selected",
               enabled=selected_model is not None).draw(self.canvas, mouse)
        selected_name = selected_model.name if selected_model else "No saved models"
        self.canvas.blit(font(14).render(f"Selected: {selected_name[:32]}", True, COLORS["muted"]), (370, 672))
        if self.training_seed:
            self.canvas.blit(font(14, True).render(
                f"Seeded from generation {self.training_seed.generation}", True, COLORS["cyan"]), (650, 672))
        if self.show_training_inventory:
            self.draw_training_inventory()
        if self.show_training_save_modal:
            self.draw_training_save_modal()
        self._draw_model_modal()

    def draw_training_save_modal(self):
        session = self.training_session
        champion = session.completed_champion if session else self.champion
        validated = bool(session and session.champion_validated)
        if champion is None:
            lines = (
                "No generation finished before Stop was pressed.",
                "Partial genome fitness is discarded so it cannot be mislabeled as a champion.",
            )
        else:
            fitness = float(champion.fitness or 0.0)
            lines = (
                f"Latest completed champion: Generation {self.training_generation} · Fitness {fitness:.2f}",
                "Save it as a Draft, or as Validated when the complete suite has passed.",
            )
        draw_modal(self.canvas, (230, 205, 820, 345), "Training Stopped", lines)
        mouse = self.logical_mouse()
        Button((285, 455, 210, 48), "Save Draft", enabled=champion is not None).draw(self.canvas, mouse)
        Button((535, 455, 210, 48), "Save Validated",
               enabled=champion is not None and validated).draw(self.canvas, mouse)
        Button((785, 455, 210, 48), "Don't Save").draw(self.canvas, mouse)

    def draw_training_inventory(self):
        mouse = self.logical_mouse()
        layout = self.training_inventory_layout()
        actions_y = layout["actions_y"]
        draw_modal(self.canvas, (140, 95, 1000, layout["modal_height"]), "Saved Model Inventory",
                   (f"{len(self.models)} model(s) · shared with the Play drawer",
                    "Select a genome to retrain it with its saved validation scope."))
        if not self.models:
            self.canvas.blit(font(20).render("No saved models yet. Save a current best or champion first.",
                                             True, COLORS["muted"]), (315, 300))
        for rect, model in self.training_inventory_cards():
            pygame.draw.rect(self.canvas, COLORS["card"], rect, border_radius=18)
            icon = pygame.transform.smoothscale(self.skins[model.skin], (28, 56))
            self.canvas.blit(icon, icon.get_rect(midleft=(rect.x + 18, rect.centery)))
            self.canvas.blit(font(17, True).render(model.name[:22], True, COLORS["white"]),
                             (rect.x + 55, rect.y + 14))
            status_color = COLORS["green"] if model.status == "validated" else COLORS["muted"]
            self.canvas.blit(font(14).render(
                f"{model.status.title()} · Generation {model.generation}", True, status_color),
                (rect.x + 55, rect.y + 43))
            self.canvas.blit(font(13).render(
                f"Fitness {model.fitness:.1f} · Tracks {model.trained_track_count}",
                True, COLORS["muted"]), (rect.x + 55, rect.y + 69))
            best = min(model.best_times.values()) if model.best_times else None
            result = f"Best campaign {best:.2f}s" if best is not None else "No campaign finish yet"
            self.canvas.blit(font(13).render(result, True, COLORS["muted"]),
                             (rect.x + 55, rect.y + 94))
            if model.model_id == self.selected_model_id:
                pygame.draw.rect(self.canvas, COLORS["cyan"], rect, 3, border_radius=18)
        Button((185, actions_y, 100, 38), "Prev", enabled=self.inventory_page > 0).draw(self.canvas, mouse)
        Button((295, actions_y, 100, 38), "Next",
               enabled=self.inventory_page + 1 < self.inventory_pages()).draw(self.canvas, mouse)
        selected = self.selected_model()
        Button((420, actions_y, 120, 38), "Rename", enabled=selected is not None).draw(self.canvas, mouse)
        Button((550, actions_y, 120, 38), "Delete", enabled=selected is not None).draw(self.canvas, mouse)
        Button((680, actions_y, 120, 38), "Import").draw(self.canvas, mouse)
        Button((810, actions_y, 120, 38), "Export", enabled=selected is not None).draw(self.canvas, mouse)
        if selected:
            scope = selected.validation_scope.get("mode", "original").replace("_", " ").title()
            self.canvas.blit(font(15).render(
                f"Continue {selected.name[:24]} · prior scope: {scope}", True, COLORS["white"]),
                (390, layout["scope_y"]))
        Button((390, layout["footer_y"], 235, 48), "Continue Training", enabled=selected is not None).draw(self.canvas, mouse)
        Button((655, layout["footer_y"], 235, 48), "Close").draw(self.canvas, mouse)

    def draw_training_live(self):
        session = self.training_session
        entries = session.active if session.state == TrainingState.RUNNING or (
            session.state == TrainingState.PAUSED and session.active) else (
            [session._validation_entry] if session._validation_entry else []
        )
        cars = [entry.car for entry in entries]
        self.draw_runtime(session.runtime, cars)
        state_label = {
            TrainingState.RUNNING: "TRAINING",
            TrainingState.VALIDATING: "VALIDATING CHAMPION",
            TrainingState.PAUSED: "TRAINING PAUSED",
        }[session.state]
        banner = pygame.Surface((880, 58), pygame.SRCALPHA); banner.fill((10, 15, 18, 210))
        self.canvas.blit(banner, (35, 18))
        self.canvas.blit(font(27, True).render(state_label, True, COLORS["white"]), (55, 31))
        track_name = session.runtime.definition.name
        self.canvas.blit(font(17).render(f"Generation {session.generation} · {track_name} · {len(cars)} active", True, COLORS["muted"]), (330, 37))
        pygame.draw.rect(self.canvas, COLORS["panel"], (940, 85, 300, 620), border_radius=24)
        self.canvas.blit(font(24, True).render("Training Session", True, COLORS["white"]), (965, 110))
        if session.state == TrainingState.VALIDATING or (
                session.state == TrainingState.PAUSED
                and getattr(session, "_paused_from", None) == TrainingState.VALIDATING):
            suite_position = f"Validation track  {session._validation_index + 1}/{len(session._validation_tracks)}"
        else:
            suite_position = f"Suite track  {session.track_index + 1}/{len(session.generation_tracks)}"
        details = (
            f"State  {session.state.value}",
            f"Generation  {session.generation}",
            f"Track  {track_name[:20]}",
            f"Cars active  {len(cars)}",
            suite_position,
            f"Completed generations  {session.completed_generations}",
        )
        for index, line in enumerate(details):
            self.canvas.blit(font(16).render(line, True, COLORS["muted"]), (965, 155 + index * 25))
        action = "Resume" if session.state == TrainingState.PAUSED else "Pause"
        Button((970, 315, 240, 48), action).draw(self.canvas, self.logical_mouse())
        Button((970, 378, 240, 48), "Stop Training").draw(self.canvas, self.logical_mouse())
        self.canvas.blit(font(14, True).render("SIMULATION SPEED", True, COLORS["muted"]), (965, 468))
        for index, speed in enumerate((1, 2, 4, 0)):
            Button((955 + index * 70, 500, 60, 38), "Max" if speed == 0 else f"{speed}×",
                   selected=self.training_speed == speed).draw(self.canvas, self.logical_mouse())
        self.canvas.blit(font(14).render("Space also pauses or resumes.", True, COLORS["muted"]), (965, 565))
        self.canvas.blit(font(14).render("Stop keeps the latest completed", True, COLORS["muted"]), (965, 590))
        self.canvas.blit(font(14).render("champion and discards partial fitness.", True, COLORS["muted"]), (965, 612))

    # ---------- track builder ----------
    editor_tiles = property(lambda self: self.editor_state.tiles,
                            lambda self, value: setattr(self.editor_state, "tiles", value))
    editor_tool = property(lambda self: self.editor_state.tool,
                           lambda self, value: setattr(self.editor_state, "tool", value))
    editor_kind = property(lambda self: self.editor_state.kind,
                           lambda self, value: setattr(self.editor_state, "kind", value))
    editor_rotation = property(lambda self: self.editor_state.rotation,
                               lambda self, value: setattr(self.editor_state, "rotation", value))
    editor_selected = property(lambda self: self.editor_state.selected,
                               lambda self, value: setattr(self.editor_state, "selected", value))
    editor_history = property(lambda self: self.editor_state.history,
                              lambda self, value: setattr(self.editor_state, "history", value))
    editor_redo = property(lambda self: self.editor_state.redo_stack,
                           lambda self, value: setattr(self.editor_state, "redo_stack", value))
    editor_name = property(lambda self: self.editor_state.name,
                           lambda self, value: setattr(self.editor_state, "name", value))
    editor_track_id = property(lambda self: self.editor_state.track_id,
                               lambda self, value: setattr(self.editor_state, "track_id", value))
    pending_replace = property(lambda self: self.editor_state.pending_replace,
                               lambda self, value: setattr(self.editor_state, "pending_replace", value))

    @property
    def editor_origin(self):
        return 42, 105

    def snapshot(self):
        return self.editor_state.snapshot()

    def _push_history(self):
        self.editor_state.push_history()

    def editor_track(self):
        return self.editor_state.definition()

    def _auto_rotation(self, kind, cell):
        return self.editor_state.suggested_rotation(kind, cell)

    def _place_editor_tile(self, cell, kind, rotation=None):
        return self.editor_state.place(cell, kind, rotation)

    def _undo(self):
        self.editor_state.undo()

    def _redo(self):
        self.editor_state.redo()

    def event_editor(self, event):
        mouse = self.logical_mouse(); ox, oy = self.editor_origin
        if self.show_track_catalog:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                tracks = self.storage.custom_tracks()
                for index, track in enumerate(tracks[:8]):
                    if pygame.Rect(390, 215 + index * 48, 500, 40).collidepoint(mouse):
                        self.editor_state.load(track)
                        self.show_track_catalog = False; return
                if pygame.Rect(430, 650, 190, 44).collidepoint(mouse):
                    self.editor_state.new()
                    self.show_track_catalog = False
                elif pygame.Rect(660, 650, 190, 44).collidepoint(mouse): self.show_track_catalog = False
            return
        if self.pending_replace:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(430, 430, 190, 48).collidepoint(mouse):
                    self.editor_state.confirm_replace()
                elif pygame.Rect(660, 430, 190, 48).collidepoint(mouse): self.pending_replace = None
            return
        if event.type == pygame.KEYDOWN:
            if self.editor_typing:
                if event.key == pygame.K_RETURN: self.editor_typing = False
                elif event.key == pygame.K_BACKSPACE: self.editor_name = self.editor_name[:-1]
                elif event.unicode.isprintable() and len(self.editor_name) < 28: self.editor_name += event.unicode
                return
            ctrl = event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META)
            if event.key == pygame.K_z and ctrl:
                self._redo() if event.mod & pygame.KMOD_SHIFT else self._undo(); return
            if event.key == pygame.K_r:
                self.editor_state.rotate_selected()
            elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE) and self.editor_selected in self.editor_tiles:
                self.editor_state.delete_selected()
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.editor_drag_kind:
            cell = int((mouse[0] - ox) // TILE_SIZE), int((mouse[1] - oy) // TILE_SIZE)
            if 0 <= cell[0] < GRID_SIZE[0] and 0 <= cell[1] < GRID_SIZE[1]: self._place_editor_tile(cell, self.editor_drag_kind)
            self.editor_drag_kind = None; return
        if event.type != pygame.MOUSEBUTTONDOWN:
            return
        if pygame.Rect(960, 98, 275, 38).collidepoint(mouse): self.editor_typing = True; return
        tools = {"select": pygame.Rect(960, 150, 82, 36), "straight": pygame.Rect(1050, 150, 82, 36),
                 "corner": pygame.Rect(1140, 150, 82, 36), "start_finish": pygame.Rect(960, 194, 262, 36)}
        for tool, rect in tools.items():
            if rect.collidepoint(mouse):
                self.editor_tool = tool
                if tool != "select": self.editor_kind = tool; self.editor_drag_kind = tool
                return
        actions = {
            "rotate": pygame.Rect(960, 242, 125, 36), "delete": pygame.Rect(1095, 242, 125, 36),
            "undo": pygame.Rect(960, 286, 82, 36), "redo": pygame.Rect(1050, 286, 82, 36),
            "clear": pygame.Rect(1140, 286, 82, 36), "test": pygame.Rect(960, 330, 125, 38),
            "save": pygame.Rect(1095, 330, 125, 38), "import": pygame.Rect(960, 376, 125, 34),
            "export": pygame.Rect(1095, 376, 125, 34), "catalog": pygame.Rect(960, 416, 260, 34),
        }
        action = next((name for name, rect in actions.items() if rect.collidepoint(mouse)), None)
        result = validate_track(self.editor_track())
        if action == "rotate": pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, unicode="r", mod=0))
        elif action == "delete" and self.editor_selected in self.editor_tiles:
            self.editor_state.delete_selected()
        elif action == "undo": self._undo()
        elif action == "redo": self._redo()
        elif action == "clear" and self.editor_tiles: self.editor_state.clear()
        elif action == "save": self.save_editor_track()
        elif action == "test":
            if result.valid: self.prepare_race(self.editor_track())
            else: self.editor_error = result.messages[0]
        elif action == "import":
            report = self.storage.import_inbox(); self.custom = self.storage.custom_tracks(); self.message = f"Imported {report.imported_tracks} track(s)"
        elif action == "export" and result.valid:
            path = self.storage.export_track(self.editor_track()); self.message = f"Exported {path.name}"
        elif action == "catalog": self.show_track_catalog = True
        if action: return
        cell = int((mouse[0] - ox) // TILE_SIZE), int((mouse[1] - oy) // TILE_SIZE)
        if 0 <= cell[0] < GRID_SIZE[0] and 0 <= cell[1] < GRID_SIZE[1]:
            if event.button == 3 and cell in self.editor_tiles:
                self._push_history(); del self.editor_tiles[cell]; self.editor_selected = None
            elif event.button == 1:
                if self.editor_tool == "select": self.editor_selected = cell if cell in self.editor_tiles else None
                else: self._place_editor_tile(cell, self.editor_kind)

    def save_editor_track(self):
        track = self.editor_track(); result = validate_track(track)
        if not result.valid:
            self.editor_error = result.messages[0]; return
        track.timeout = result.metrics.recommended_timeout
        self.storage.save_track(track); self.custom = self.storage.custom_tracks(); self.editor_error = ""
        self.message = f"Saved {track.name}"

    def update_editor(self):
        pass

    def draw_piece(self, tile, origin):
        self.canvas.blit(piece_atlas().surface(tile.kind, tile.rotation),
                         (origin[0] + tile.x * TILE_SIZE, origin[1] + tile.y * TILE_SIZE))

    def draw_editor(self):
        self.title("Track Builder", "Drag, rotate, inspect every connection, then Test or Save")
        ox, oy = self.editor_origin; mouse = self.logical_mouse()
        area = pygame.Rect(ox, oy, GRID_SIZE[0] * TILE_SIZE, GRID_SIZE[1] * TILE_SIZE)
        pygame.draw.rect(self.canvas, (43, 90, 48), area)
        for x in range(GRID_SIZE[0] + 1): pygame.draw.line(self.canvas, (70, 115, 72), (ox + x * TILE_SIZE, oy), (ox + x * TILE_SIZE, area.bottom))
        for y in range(GRID_SIZE[1] + 1): pygame.draw.line(self.canvas, (70, 115, 72), (ox, oy + y * TILE_SIZE), (area.right, oy + y * TILE_SIZE))
        result = validate_track(self.editor_track())
        invalid_cells = {cell for issue in result.errors if issue.severity == "error" for cell in issue.cells}
        for tile in self.editor_tiles.values(): self.draw_piece(tile, (ox, oy))
        for cell, tile in self.editor_tiles.items():
            for port in tile.ports:
                dx, dy = VECTORS[port]; neighbor = self.editor_tiles.get((cell[0] + dx, cell[1] + dy))
                center = ox + cell[0] * TILE_SIZE + TILE_SIZE // 2, oy + cell[1] * TILE_SIZE + TILE_SIZE // 2
                endpoint = center[0] + dx * (TILE_SIZE // 2 - 5), center[1] + dy * (TILE_SIZE // 2 - 5)
                matching = neighbor and OPPOSITE[port] in neighbor.ports
                pygame.draw.circle(self.canvas, COLORS["green"] if matching else COLORS["red"], endpoint, 4)
        for cell in invalid_cells:
            if 0 <= cell[0] < GRID_SIZE[0] and 0 <= cell[1] < GRID_SIZE[1]:
                overlay = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA); overlay.fill((215, 40, 62, 65))
                self.canvas.blit(overlay, (ox + cell[0] * TILE_SIZE, oy + cell[1] * TILE_SIZE))
                pygame.draw.rect(self.canvas, COLORS["red"], (ox + cell[0] * TILE_SIZE, oy + cell[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE), 2)
        if self.editor_drag_kind:
            cell = int((mouse[0] - ox) // TILE_SIZE), int((mouse[1] - oy) // TILE_SIZE)
            if 0 <= cell[0] < GRID_SIZE[0] and 0 <= cell[1] < GRID_SIZE[1]:
                rotation = self._auto_rotation(self.editor_drag_kind, cell)
                ghost = piece_atlas().surface(self.editor_drag_kind, rotation).copy(); ghost.set_alpha(145)
                self.canvas.blit(ghost, (ox + cell[0] * TILE_SIZE, oy + cell[1] * TILE_SIZE))
        if self.editor_selected:
            pygame.draw.rect(self.canvas, COLORS["cyan"], (ox + self.editor_selected[0] * TILE_SIZE, oy + self.editor_selected[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)
        pygame.draw.rect(self.canvas, COLORS["panel"], (948, 88, 302, 660), border_radius=22)
        pygame.draw.rect(self.canvas, COLORS["ink"], (960, 98, 275, 38), border_radius=14)
        self.canvas.blit(font(17).render(self.editor_name, True, COLORS["white"]), (974, 108))
        for label, rect, selected in (
            ("Select", (960, 150, 82, 36), self.editor_tool == "select"),
            ("Straight", (1050, 150, 82, 36), self.editor_tool == "straight"),
            ("Corner", (1140, 150, 82, 36), self.editor_tool == "corner"),
            ("Start / Finish", (960, 194, 262, 36), self.editor_tool == "start_finish"),
        ): Button(rect, label, selected=selected).draw(self.canvas, mouse)
        Button((960, 242, 125, 36), "Rotate").draw(self.canvas, mouse); Button((1095, 242, 125, 36), "Delete").draw(self.canvas, mouse)
        Button((960, 286, 82, 36), "Undo", enabled=bool(self.editor_history)).draw(self.canvas, mouse)
        Button((1050, 286, 82, 36), "Redo", enabled=bool(self.editor_redo)).draw(self.canvas, mouse)
        Button((1140, 286, 82, 36), "Clear", enabled=bool(self.editor_tiles)).draw(self.canvas, mouse)
        Button((960, 330, 125, 38), "Test", enabled=result.valid and bool(self.models)).draw(self.canvas, mouse)
        Button((1095, 330, 125, 38), "Save", enabled=result.valid).draw(self.canvas, mouse)
        Button((960, 376, 125, 34), "Import").draw(self.canvas, mouse); Button((1095, 376, 125, 34), "Export", enabled=result.valid).draw(self.canvas, mouse)
        Button((960, 416, 260, 34), f"Saved Tracks · {len(self.custom)}").draw(self.canvas, mouse)
        metrics = result.metrics
        info = (f"Pieces {len(self.editor_tiles)} · Corners {metrics.corner_count}",
                f"Open ports {metrics.open_connection_count}",
                f"Length {metrics.estimated_lap_distance:.0f}px",
                f"Difficulty {metrics.difficulty_score:.1f}",
                "VALID · READY TO SAVE" if result.valid else "INVALID · KEEP EDITING")
        for index, line in enumerate(info):
            color = COLORS["green"] if result.valid and index == 4 else COLORS["red"] if index == 4 else COLORS["white"]
            self.canvas.blit(font(14, index == 4).render(line, True, color), (965, 465 + index * 23))
        self.canvas.blit(font(14, True).render("VALIDATION", True, COLORS["muted"]), (965, 590))
        for index, issue in enumerate(result.errors[:5]):
            text = issue.message if len(issue.message) <= 38 else issue.message[:35] + "…"
            issue_color = COLORS["red"] if issue.severity == "error" else (240, 190, 70)
            self.canvas.blit(font(13).render(f"• {text}", True, issue_color), (965, 615 + index * 22))
        if self.pending_replace:
            draw_modal(self.canvas, (385, 275, 510, 235), "Replace this piece?", ("The occupied cell will be overwritten.",))
            Button((430, 430, 190, 48), "Replace").draw(self.canvas, mouse); Button((660, 430, 190, 48), "Cancel").draw(self.canvas, mouse)
        elif self.show_track_catalog:
            tracks = self.storage.custom_tracks()
            draw_modal(self.canvas, (350, 145, 580, 585), "Saved Tracks",
                       ("Select a track to edit or repair.",))
            for index, track in enumerate(tracks[:8]):
                rect = pygame.Rect(390, 215 + index * 48, 500, 40)
                valid = validate_track(track).valid
                pygame.draw.rect(self.canvas, COLORS["card"], rect, border_radius=12)
                self.canvas.blit(font(16, True).render(track.name[:30], True, COLORS["white"]), (rect.x + 14, rect.y + 10))
                status = "Valid" if valid else "Needs repair"
                self.canvas.blit(font(14).render(status, True, COLORS["green"] if valid else COLORS["red"]), (rect.right - 105, rect.y + 11))
            Button((430, 650, 190, 44), "New Track").draw(self.canvas, mouse)
            Button((660, 650, 190, 44), "Close").draw(self.canvas, mouse)


def main():
    InteractiveApp().run()


if __name__ == "__main__":
    main()
