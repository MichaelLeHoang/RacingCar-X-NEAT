from __future__ import annotations

import copy
import math
import os
import time
from collections import deque
from pathlib import Path

import neat
import pygame

from racing_core import (
    DEFAULT_CAR_STATS, GRID_SIZE, LOGICAL_SIZE, ROAD_WIDTH, SENSOR_ANGLES, SENSOR_RANGE, TILE_SIZE,
    Car, ModelRecord, Storage, Tile, TrackDefinition,
    campaign_tracks, create_track_runtime, deserialize_genome, load_neat_config,
    piece_atlas, serialize_genome, validate_track,
)


BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "imgs"
COLORS = {
    "ink": (19, 24, 29), "panel": (26, 33, 39), "card": (42, 51, 58),
    "white": (245, 247, 248), "muted": (170, 183, 190), "cyan": (33, 210, 235),
    "red": (215, 40, 62), "green": (60, 205, 120), "road": (105, 106, 110),
}

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
    width, height = source.get_size(); queue = deque(); seen = set()
    border = ([(x, 0) for x in range(width)] + [(x, height-1) for x in range(width)] +
              [(0, y) for y in range(height)] + [(width-1, y) for y in range(height)])
    for point in border:
        color = source.get_at(point)
        if color.a and max(color.r, color.g, color.b) <= 8:
            queue.append(point); seen.add(point)
    while queue:
        x, y = queue.popleft(); source.set_at((x, y), (0, 0, 0, 0))
        for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0 <= nx < width and 0 <= ny < height and (nx,ny) not in seen:
                color = source.get_at((nx,ny))
                if color.a and max(color.r,color.g,color.b) <= 8:
                    seen.add((nx,ny));queue.append((nx,ny))
    return pygame.transform.smoothscale(source, (32, 64))


class TrainingStopped(Exception):
    pass


def font(size, bold=False):
    return pygame.font.SysFont("arial", size, bold=bold)


class PillButton:
    def __init__(self, rect, label, action, enabled=True):
        self.rect = pygame.Rect(rect); self.label = label; self.action = action; self.enabled = enabled

    def draw(self, surface, mouse):
        hovered = self.rect.collidepoint(mouse) and self.enabled
        color = COLORS["cyan"] if hovered else (58, 71, 80)
        if not self.enabled: color = (55, 60, 64)
        pygame.draw.rect(surface, color, self.rect, border_radius=self.rect.height // 2)
        text_color = COLORS["ink"] if hovered else (COLORS["white"] if self.enabled else (110, 118, 122))
        label = font(20, True).render(self.label, True, text_color)
        surface.blit(label, label.get_rect(center=self.rect.center))


class InteractiveApp:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode(LOGICAL_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Racing Car X NEAT")
        self.canvas = pygame.Surface(LOGICAL_SIZE)
        self.clock = pygame.time.Clock(); self.running = True; self.scene = "home"
        self.storage = Storage(); self.campaign = campaign_tracks()
        self.custom = self.storage.custom_tracks(); self.models = self.storage.models()
        self.progress = self.storage.progress(); self.message = ""
        self.grass = pygame.transform.scale(pygame.image.load(ASSET_DIR / "grass.jpg"), LOGICAL_SIZE)
        self.skins = {
            "white": load_car_sprite(ASSET_DIR / "WhiteCar.png"),
            "red": load_car_sprite(ASSET_DIR / "RedCar.png"),
            "green": load_car_sprite(ASSET_DIR / "green-car.png"),
            "purple": load_car_sprite(ASSET_DIR / "purple-car.png"),
            "grey": load_car_sprite(ASSET_DIR / "grey-car.png"),
        }
        self.drag_model = None; self.hover_model = None; self.drag_return = None
        self.inventory_page = 0
        self.selected_model_id = self.models[0].model_id if self.models else None
        self.rename_model = None; self.rename_text = ""; self.delete_model = None
        self.race = None; self.race_result = None
        self.training_tracks = []; self.training_skin = "white"; self.training_seed = None
        self.population = None; self.champion = None; self.training_generation = 0
        self.champion_validation = {}
        self.training_active = False; self.training_stop_requested = False
        self.training_paused = False; self.training_speed = 1
        self.training_finisher = None
        self.show_training_save_modal = False
        self.training_track_index = 0; self.model_name = "New Racer"; self.typing_name = False
        self.editor_tiles = {}; self.editor_kind = "straight"; self.editor_rotation = 0
        self.editor_drag_kind = None
        self.editor_selected = None; self.editor_history = []; self.editor_name = "Custom Track"
        self.editor_typing = False; self.editor_error = ""
        self.track_thumbnail_cache = {}
        self.training_profiles = []; self.active_training_profile = 0
        self.add_training_profile(initial=True)

    def logical_mouse(self):
        wx, wy = self.window.get_size(); mx, my = pygame.mouse.get_pos()
        scale = min(wx / LOGICAL_SIZE[0], wy / LOGICAL_SIZE[1])
        ox = (wx - LOGICAL_SIZE[0] * scale) / 2; oy = (wy - LOGICAL_SIZE[1] * scale) / 2
        return ((mx - ox) / scale, (my - oy) / scale)

    def present(self):
        size = self.window.get_size(); scale = min(size[0]/1280, size[1]/800)
        scaled_size = (int(1280*scale), int(800*scale))
        frame = pygame.transform.smoothscale(self.canvas, scaled_size)
        self.window.fill((5, 8, 10)); self.window.blit(frame, ((size[0]-scaled_size[0])//2, (size[1]-scaled_size[1])//2))
        pygame.display.flip()

    def title(self, text, subtitle=""):
        self.canvas.blit(font(46, True).render(text, True, COLORS["white"]), (42, 24))
        if subtitle: self.canvas.blit(font(20).render(subtitle, True, COLORS["muted"]), (44, 78))

    def background(self):
        self.canvas.blit(self.grass, (0, 0)); shade = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        shade.fill((8, 14, 18, 175)); self.canvas.blit(shade, (0, 0))

    def buttons(self):
        if self.scene == "home":
            return [PillButton((475, 320+i*92, 330, 62), label, action)
                    for i, (label, action) in enumerate((("Get Started", "campaign"), ("Train", "train"), ("Custom Tracks", "editor")))]
        return [PillButton((1080, 24, 150, 44), "Back", "home")]

    def handle_common(self, event):
        if event.type == pygame.QUIT: self.running = False; return True
        if self.scene == "campaign" and (self.rename_model or self.delete_model):
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.rename_model = None; self.delete_model = None
                return True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                return True
        if self.scene == "train" and self.show_training_save_modal:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.show_training_save_modal = False
                return True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.scene == "home": self.running = False
            else: self.scene = "home"
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse = self.logical_mouse()
            for button in self.buttons():
                if button.enabled and button.rect.collidepoint(mouse):
                    self.scene = button.action; self.message = ""; return True
        return False

    def run(self):
        while self.running:
            events = pygame.event.get()
            for event in events:
                if self.handle_common(event): continue
                getattr(self, f"event_{self.scene}")(event)
            getattr(self, f"update_{self.scene}")()
            self.background(); getattr(self, f"draw_{self.scene}")()
            mouse = self.logical_mouse()
            for button in self.buttons(): button.draw(self.canvas, mouse)
            if self.message:
                notice = font(18, True).render(self.message, True, COLORS["cyan"])
                self.canvas.blit(notice, (44, 760))
            self.present(); self.clock.tick(60)
        pygame.quit()

    # Home
    def event_home(self, event): pass
    def update_home(self): pass
    def draw_home(self):
        logo = font(64, True).render("RACING CAR", True, COLORS["white"])
        x = font(64, True).render("X NEAT", True, COLORS["red"])
        self.canvas.blit(logo, logo.get_rect(center=(640, 145))); self.canvas.blit(x, x.get_rect(center=(640, 215)))
        self.canvas.blit(font(21).render("Evolve. Race. Build.", True, COLORS["muted"]), (535, 265))

    # Campaign and inventory
    def refresh_models(self, selected_id=None):
        selected_id = selected_id or self.selected_model_id
        self.models = self.storage.models()
        model_ids = [model.model_id for model in self.models]
        self.selected_model_id = selected_id if selected_id in model_ids else (model_ids[0] if model_ids else None)
        if self.selected_model_id:
            self.inventory_page = model_ids.index(self.selected_model_id) // 6
        else:
            self.inventory_page = 0

    def selected_model(self):
        return next((model for model in self.models if model.model_id == self.selected_model_id), None)

    def model_cards(self):
        start = self.inventory_page * 6
        return [(pygame.Rect(40 + i*180, 620, 160, 120), model)
                for i, model in enumerate(self.models[start:start + 6])]

    def inventory_pages(self):
        return max(1, math.ceil(len(self.models) / 6))

    def begin_rename(self, model):
        if model:
            self.rename_model = model
            self.rename_text = model.name

    def confirm_rename(self):
        if not self.rename_model:
            return
        self.rename_model.name = self.rename_text.strip() or self.rename_model.name
        selected_id = self.rename_model.model_id
        self.storage.save_model(self.rename_model)
        self.rename_model = None
        self.refresh_models(selected_id)
        self.message = "Model renamed"

    def confirm_delete_model(self):
        if not self.delete_model:
            return
        name = self.delete_model.name
        self.storage.delete_model(self.delete_model.model_id)
        self.delete_model = None
        self.refresh_models()
        self.message = f"Deleted {name}"

    def level_cards(self):
        cards = []
        for i, track in enumerate(self.campaign):
            row, col = divmod(i, 5); cards.append((pygame.Rect(70+col*235, 150+row*190, 190, 140), track, i+1))
        return cards

    def event_campaign(self, event):
        mouse = self.logical_mouse()
        if self.delete_model:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.confirm_delete_model()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(430, 430, 190, 48).collidepoint(mouse):
                    self.confirm_delete_model()
                elif pygame.Rect(660, 430, 190, 48).collidepoint(mouse):
                    self.delete_model = None
            return
        if event.type == pygame.KEYDOWN and self.rename_model:
            if event.key == pygame.K_RETURN:
                self.confirm_rename()
            elif event.key == pygame.K_ESCAPE: self.rename_model = None
            elif event.key == pygame.K_BACKSPACE: self.rename_text = self.rename_text[:-1]
            elif event.unicode.isprintable() and len(self.rename_text) < 24: self.rename_text += event.unicode
            return
        if self.rename_model:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(430, 430, 190, 48).collidepoint(mouse):
                    self.confirm_rename()
                elif pygame.Rect(660, 430, 190, 48).collidepoint(mouse):
                    self.rename_model = None
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            selected = self.selected_model()
            if pygame.Rect(1115, 565, 125, 38).collidepoint(mouse):
                self.begin_rename(selected); return
            if pygame.Rect(1115, 610, 125, 38).collidepoint(mouse):
                self.delete_model = selected; return
            if pygame.Rect(1115, 655, 125, 38).collidepoint(mouse) and selected:
                path=self.storage.export_model(selected);self.message=f"Exported to {path}";return
            if pygame.Rect(1115, 700, 125, 38).collidepoint(mouse):
                count=self.storage.import_inbox();self.refresh_models();self.custom=self.storage.custom_tracks();self.message=f"Imported {count} item(s)";return
            if pygame.Rect(850, 570, 90, 36).collidepoint(mouse) and self.inventory_page > 0:
                self.inventory_page -= 1
                self.selected_model_id = self.models[self.inventory_page * 6].model_id
                return
            if pygame.Rect(950, 570, 90, 36).collidepoint(mouse) and self.inventory_page + 1 < self.inventory_pages():
                self.inventory_page += 1
                self.selected_model_id = self.models[self.inventory_page * 6].model_id
                return
            for rect, model in self.model_cards():
                if rect.collidepoint(mouse):
                    self.selected_model_id = model.model_id
                    self.drag_model = model
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            for rect, model in self.model_cards():
                if rect.collidepoint(mouse):
                    self.selected_model_id = model.model_id
                    self.begin_rename(model)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drag_model:
            dropped=False
            for rect, track, level in self.level_cards():
                if rect.collidepoint(mouse) and level <= self.progress["unlocked"]:
                    self.start_race(track, self.drag_model, level);dropped=True;break
            if not dropped:
                source=next((rect.center for rect,model in self.model_cards() if model==self.drag_model),mouse)
                self.drag_return={"model":self.drag_model,"start":mouse,"end":source,"time":time.perf_counter()}
            self.drag_model = None

    def update_campaign(self):
        if self.drag_return and time.perf_counter()-self.drag_return["time"]>.22:self.drag_return=None

    def draw_model_ghost(self,model,position,scale=1.0):
        image=self.skins[model.skin]
        image=pygame.transform.smoothscale(image,(max(1,int(image.get_width()*scale)),max(1,int(image.get_height()*scale))))
        shadow=pygame.Surface((130,100),pygame.SRCALPHA);pygame.draw.ellipse(shadow,(0,0,0,110),(25,68,80,20))
        self.canvas.blit(shadow,shadow.get_rect(center=position));self.canvas.blit(image,image.get_rect(center=(position[0],position[1]-12)))
        label=font(15,True).render(model.name,True,COLORS["white"]);box=label.get_rect(midtop=(position[0],position[1]+38)).inflate(18,10)
        pygame.draw.rect(self.canvas,(15,20,24),box,border_radius=box.height//2);self.canvas.blit(label,label.get_rect(center=box.center))

    def draw_campaign(self):
        self.title("Campaign", "Drag a saved car onto an unlocked level")
        for rect, track, level in self.level_cards():
            unlocked = level <= self.progress["unlocked"]
            pygame.draw.rect(self.canvas, COLORS["card"] if unlocked else (35, 40, 44), rect, border_radius=22)
            color = COLORS["cyan"] if unlocked else (90, 98, 102)
            self.canvas.blit(font(30, True).render(str(level), True, color), (rect.x+18, rect.y+14))
            self.canvas.blit(font(18, True).render(track.name, True, COLORS["white"] if unlocked else COLORS["muted"]), (rect.x+18, rect.y+57))
            status = "Unlocked" if unlocked else "Locked"
            self.canvas.blit(font(15).render(status, True, color), (rect.x+18, rect.y+99))
            if self.drag_model and rect.collidepoint(self.logical_mouse()):
                pygame.draw.rect(self.canvas,COLORS["cyan"] if unlocked else COLORS["red"],rect,4,border_radius=22)
        count_label = f"MODEL INVENTORY · {len(self.models)} MODEL{'S' if len(self.models) != 1 else ''}"
        self.canvas.blit(font(22, True).render(count_label, True, COLORS["white"]), (42, 575))
        if not self.models: self.canvas.blit(font(18).render("No models yet — train and save one first.", True, COLORS["muted"]), (42, 630))
        mouse = self.logical_mouse(); self.hover_model = None
        for rect, model in self.model_cards():
            pygame.draw.rect(self.canvas, COLORS["card"], rect, border_radius=18)
            self.canvas.blit(self.skins[model.skin], (rect.x+12, rect.y+25))
            self.canvas.blit(font(17, True).render(model.name[:13], True, COLORS["white"]), (rect.x+58, rect.y+20))
            self.canvas.blit(font(14).render(model.status.title(), True, COLORS["green"] if model.status=="validated" else COLORS["muted"]), (rect.x+58, rect.y+48))
            if model.model_id == self.selected_model_id:
                pygame.draw.rect(self.canvas, COLORS["cyan"], rect, 3, border_radius=18)
            if model==self.drag_model:
                faded=pygame.Surface(rect.size,pygame.SRCALPHA);faded.fill((10,15,18,155));self.canvas.blit(faded,rect)
            if rect.collidepoint(mouse): self.hover_model = model; pygame.draw.rect(self.canvas, COLORS["white"], rect, 2, border_radius=18)
        pages = self.inventory_pages()
        PillButton((850,570,90,36),"Prev","",self.inventory_page > 0).draw(self.canvas,mouse)
        PillButton((950,570,90,36),"Next","",self.inventory_page + 1 < pages).draw(self.canvas,mouse)
        self.canvas.blit(font(13).render(f"{self.inventory_page + 1}/{pages}",True,COLORS["muted"]),(1048,581))
        selected = self.selected_model()
        PillButton((1115,565,125,38),"Rename","",selected is not None).draw(self.canvas,mouse)
        PillButton((1115,610,125,38),"Delete","",selected is not None).draw(self.canvas,mouse)
        PillButton((1115,655,125,38),"Export","",selected is not None).draw(self.canvas,mouse)
        PillButton((1115,700,125,38),"Import","",True).draw(self.canvas,mouse)
        if self.hover_model:
            stats={**DEFAULT_CAR_STATS,**self.hover_model.car_stats}
            passed=sum(bool(value) for value in self.hover_model.validation.values())
            lines=(f"Gen {self.hover_model.generation} · Fitness {self.hover_model.fitness:.1f} · Wins {self.hover_model.wins}/{self.hover_model.attempts}",
                   f"Trained tracks {len(self.hover_model.trained_tracks)} · Passed {passed} · {self.hover_model.status.title()}",
                   f"Speed {stats['max_speed']:.1f} · Acceleration {stats['acceleration']:.2f} · Turning {stats['turn_speed']:.1f}")
            rendered=[font(15).render(text,True,COLORS["white"]) for text in lines]
            width=max(item.get_width() for item in rendered)+28;box=pygame.Rect(1230-width,515,width,81)
            pygame.draw.rect(self.canvas,COLORS["ink"],box,border_radius=12)
            for i,item in enumerate(rendered):self.canvas.blit(item,(box.x+14,box.y+9+i*23))
        if self.rename_model:
            shade=pygame.Surface(LOGICAL_SIZE,pygame.SRCALPHA);shade.fill((0,0,0,145));self.canvas.blit(shade,(0,0))
            box=pygame.Rect(385,275,510,235);pygame.draw.rect(self.canvas,COLORS["ink"],box,border_radius=28)
            self.canvas.blit(font(28,True).render("Rename model",True,COLORS["white"]),(box.x+34,box.y+28))
            pygame.draw.rect(self.canvas,COLORS["card"],(box.x+34,box.y+75,442,48),border_radius=18)
            self.canvas.blit(font(20).render(self.rename_text,True,COLORS["white"]),(box.x+49,box.y+88))
            self.canvas.blit(font(14).render("Up to 24 characters",True,COLORS["muted"]),(box.x+36,box.y+132))
            PillButton((430,430,190,48),"Save Name","",bool(self.rename_text.strip())).draw(self.canvas,mouse)
            PillButton((660,430,190,48),"Cancel","",True).draw(self.canvas,mouse)
        elif self.delete_model:
            shade=pygame.Surface(LOGICAL_SIZE,pygame.SRCALPHA);shade.fill((0,0,0,145));self.canvas.blit(shade,(0,0))
            box=pygame.Rect(385,275,510,235);pygame.draw.rect(self.canvas,COLORS["ink"],box,border_radius=28)
            self.canvas.blit(font(28,True).render("Delete model?",True,COLORS["white"]),(box.x+34,box.y+30))
            self.canvas.blit(font(17).render(f"{self.delete_model.name} will be removed permanently.",True,COLORS["muted"]),(box.x+34,box.y+86))
            self.canvas.blit(font(16).render("This cannot be undone.",True,COLORS["red"]),(box.x+34,box.y+119))
            PillButton((430,430,190,48),"Delete Model","",True).draw(self.canvas,mouse)
            PillButton((660,430,190,48),"Cancel","",True).draw(self.canvas,mouse)
        if self.drag_model:
            over_valid=any(rect.collidepoint(mouse) and level<=self.progress["unlocked"] for rect,_,level in self.level_cards())
            self.draw_model_ghost(self.drag_model,mouse,1.15 if over_valid else 1.0)
        elif self.drag_return:
            elapsed=min(1.0,(time.perf_counter()-self.drag_return["time"])/.22);ease=1-(1-elapsed)**3
            start=self.drag_return["start"];end=self.drag_return["end"]
            point=(start[0]+(end[0]-start[0])*ease,start[1]+(end[1]-start[1])*ease)
            self.draw_model_ghost(self.drag_return["model"],point)

    # Race
    def start_race(self, track, model, level=None):
        runtime = create_track_runtime(track); genome = deserialize_genome(model.genome)
        net = neat.nn.FeedForwardNetwork.create(genome, load_neat_config(BASE_DIR))
        car = Car(self.skins[model.skin], runtime, model.car_stats)
        self.race = {"runtime": runtime, "model": model, "net": net, "car": car, "level": level, "started": time.perf_counter()}
        model.attempts += 1; self.storage.save_model(model); self.race_result = None; self.scene = "race"

    def start_next_level(self):
        if not self.race or not self.race["level"] or self.race["level"] >= len(self.campaign):
            self.scene = "home"
            return
        next_level = self.race["level"] + 1
        self.start_race(self.campaign[next_level - 1], self.race["model"], next_level)

    def event_race(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.race_result:
            mouse = self.logical_mouse()
            if self.race_result == "COMPLETE":
                if pygame.Rect(405, 430, 210, 50).collidepoint(mouse):
                    self.start_next_level()
                elif pygame.Rect(650, 430, 210, 50).collidepoint(mouse):
                    self.scene = "home"
            else:
                if pygame.Rect(325, 430, 190, 50).collidepoint(mouse):
                    self.start_race(self.race["runtime"].definition, self.race["model"], self.race["level"])
                elif pygame.Rect(545, 430, 190, 50).collidepoint(mouse):
                    self.training_seed = self.race["model"]
                    self.training_tracks = [self.race["runtime"].definition]
                    self.scene = "train"
                elif pygame.Rect(765, 430, 190, 50).collidepoint(mouse):
                    self.scene = "home"
            return
        if event.type == pygame.KEYDOWN and self.race_result:
            if event.key == pygame.K_r: self.start_race(self.race["runtime"].definition, self.race["model"], self.race["level"])
            elif event.key == pygame.K_n and self.race_result == "COMPLETE": self.start_next_level()
            elif event.key == pygame.K_m: self.scene = "home"
            elif event.key == pygame.K_t:
                self.training_seed = self.race["model"]; self.training_tracks = [self.race["runtime"].definition]; self.scene = "train"

    def update_race(self):
        if not self.race or self.race_result: return
        r = self.race; car = r["car"]; runtime = r["runtime"]
        output = r["net"].activate(car.sensors(runtime)); car.step(max(range(4), key=output.__getitem__))
        now = time.perf_counter(); finished, stalled, timeout = car.update_progress(runtime)
        if finished:
            self.race_result = "COMPLETE"; elapsed = now-r["started"]; model = r["model"]
            model.wins += 1; previous = model.best_times.get(runtime.definition.track_id)
            model.best_times[runtime.definition.track_id] = min(previous, elapsed) if previous else elapsed
            if r["level"]:
                self.progress["completed"][str(r["level"])] = True
                self.progress["best_times"][str(r["level"])] = min(
                    self.progress["best_times"].get(str(r["level"]), elapsed), elapsed
                )
                self.progress["unlocked"] = max(self.progress["unlocked"], min(10, r["level"]+1)); self.storage.save_progress(self.progress)
            self.storage.save_model(model)
        elif car.crashed(runtime): self.race_result = "CRASHED"
        elif stalled: self.race_result = "STALLED"
        elif timeout: self.race_result = "TIME OUT"

    def draw_runtime(self, runtime, cars):
        ox, oy = runtime.origin; self.canvas.blit(runtime.surface, (ox, oy))
        for car in cars:
            cx, cy = car.center
            for angle, value in zip(SENSOR_ANGLES, car.sensor_values):
                rad = math.radians(car.angle+angle); distance=value*SENSOR_RANGE
                end=(ox+cx-math.sin(rad)*distance, oy+cy-math.cos(rad)*distance)
                pygame.draw.line(self.canvas, COLORS["cyan"], (ox+cx,oy+cy), end, 1)
            image, rect=car.rotated(); self.canvas.blit(image,(ox+rect.x,oy+rect.y))

    def draw_race(self):
        if self.race and self.race["runtime"].definition.runtime_type!="legacy_bitmap":
            self.title(self.race["runtime"].definition.name)
        if not self.race: return
        self.draw_runtime(self.race["runtime"], [self.race["car"]])
        elapsed=time.perf_counter()-self.race["started"]
        hud=pygame.Rect(940,110,300,180); pygame.draw.rect(self.canvas,(10,15,18),hud,border_radius=22)
        if self.race["runtime"].definition.runtime_type=="legacy_bitmap":
            self.canvas.blit(font(30,True).render(self.race["runtime"].definition.name,True,COLORS["white"]),(960,55))
        armed = "Yes" if self.race["car"].finish_armed else "No"
        lines=(f"Time  {elapsed:.1f}s",f"Finish armed  {armed}",f"Model  {self.race['model'].name}")
        for i,line in enumerate(lines): self.canvas.blit(font(21,True).render(line,True,COLORS["white"]),(hud.x+22,hud.y+24+i*42))
        if self.race_result:
            shade=pygame.Surface(LOGICAL_SIZE,pygame.SRCALPHA);shade.fill((0,0,0,135));self.canvas.blit(shade,(0,0))
            if self.race_result == "COMPLETE":
                box=pygame.Rect(350,250,580,280);pygame.draw.rect(self.canvas,COLORS["ink"],box,border_radius=28)
                self.canvas.blit(font(38,True).render("Congratulations!",True,COLORS["green"]),(box.x+38,box.y+30))
                level=self.race["level"]
                detail=(f"Level {level} complete · {self.race['runtime'].definition.name}"
                        if level else f"Completed {self.race['runtime'].definition.name}")
                self.canvas.blit(font(20).render(detail,True,COLORS["white"]),(box.x+40,box.y+91))
                self.canvas.blit(font(17).render(f"Race time {elapsed:.1f}s · the next level is now unlocked.",True,COLORS["muted"]),(box.x+40,box.y+128))
                next_label="Next Level" if level and level < len(self.campaign) else "Campaign Complete"
                PillButton((405,430,210,50),next_label,"",True).draw(self.canvas,self.logical_mouse())
                PillButton((650,430,210,50),"Main Menu","",True).draw(self.canvas,self.logical_mouse())
            else:
                box=pygame.Rect(285,250,710,280);pygame.draw.rect(self.canvas,COLORS["ink"],box,border_radius=28)
                self.canvas.blit(font(38,True).render(self.race_result,True,COLORS["red"]),(box.x+38,box.y+34))
                self.canvas.blit(font(18).render("Try again, train this model, or return to the main menu.",True,COLORS["muted"]),(box.x+40,box.y+100))
                PillButton((325,430,190,50),"Retry","",True).draw(self.canvas,self.logical_mouse())
                PillButton((545,430,190,50),"Train Model","",True).draw(self.canvas,self.logical_mouse())
                PillButton((765,430,190,50),"Main Menu","",True).draw(self.canvas,self.logical_mouse())

    # Training
    def all_train_tracks(self): return self.campaign[:self.progress["unlocked"]] + self.custom

    def capture_training_profile(self):
        return {
            "name":self.model_name,"skin":self.training_skin,"seed":self.training_seed,
            "tracks":list(self.training_tracks),"population":self.population,
            "champion":self.champion,"generation":self.training_generation,
            "validation":dict(self.champion_validation),"track_index":self.training_track_index,
            "finisher":self.training_finisher,
        }

    def apply_training_profile(self,state):
        self.model_name=state["name"];self.training_skin=state["skin"]
        self.training_seed=state["seed"];self.training_tracks=list(state["tracks"])
        self.population=state["population"];self.champion=state["champion"]
        self.training_generation=state["generation"];self.champion_validation=dict(state["validation"])
        self.training_track_index=state["track_index"]
        self.training_finisher=state.get("finisher")

    def save_active_training_profile(self):
        if self.training_profiles:
            self.training_profiles[self.active_training_profile]=self.capture_training_profile()

    def add_training_profile(self,initial=False):
        if not initial:self.save_active_training_profile()
        if initial:
            self.training_profiles.append(self.capture_training_profile());return
        number=len(self.training_profiles)+1
        self.model_name=f"Model {number}";self.training_skin="white";self.training_seed=None
        self.training_tracks=[];self.population=None;self.champion=None;self.training_generation=0
        self.champion_validation={};self.training_track_index=0;self.training_finisher=None
        self.training_profiles.append(self.capture_training_profile());self.active_training_profile=len(self.training_profiles)-1

    def switch_training_profile(self,index):
        if index==self.active_training_profile or not 0<=index<len(self.training_profiles):return
        self.save_active_training_profile();self.active_training_profile=index
        self.apply_training_profile(self.training_profiles[index])

    def training_track_cards(self):
        return [(pygame.Rect(385+(i%3)*275,125+(i//3)*150,255,132),track)
                for i,track in enumerate(self.all_train_tracks()[:9])]

    def track_thumbnail(self,track,size=(231,92)):
        key=(track.track_id,size)
        if key not in self.track_thumbnail_cache:
            runtime=create_track_runtime(track);preview=pygame.Surface(size)
            preview.fill((28,55,32));source_size=runtime.surface.get_size()
            scale=min(size[0]/source_size[0],size[1]/source_size[1])
            fitted_size=(max(1,int(source_size[0]*scale)),max(1,int(source_size[1]*scale)))
            fitted=pygame.transform.smoothscale(runtime.surface,fitted_size)
            preview.blit(fitted,((size[0]-fitted_size[0])//2,(size[1]-fitted_size[1])//2));self.track_thumbnail_cache[key]=preview
        return self.track_thumbnail_cache[key]

    def event_train(self, event):
        mouse=self.logical_mouse()
        if self.show_training_save_modal:
            if event.type==pygame.KEYDOWN and event.key==pygame.K_RETURN:
                self.save_champion();self.show_training_save_modal=False
            if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                if pygame.Rect(430,430,190,48).collidepoint(mouse):
                    self.save_champion();self.show_training_save_modal=False
                elif pygame.Rect(660,430,190,48).collidepoint(mouse):self.show_training_save_modal=False
            return
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            if pygame.Rect(40,130,300,44).collidepoint(mouse): self.typing_name=True
            for rect,track in self.training_track_cards():
                if rect.collidepoint(mouse):
                    if track in self.training_tracks: self.training_tracks.remove(track)
                    else: self.training_tracks.append(track)
            for i,skin in enumerate(self.skins):
                if pygame.Rect(45+i*68,210,54,80).collidepoint(mouse): self.training_skin=skin
            if pygame.Rect(45,718,44,44).collidepoint(mouse) and len(self.training_profiles)<6:
                self.add_training_profile();return
            for i,_ in enumerate(self.training_profiles):
                if pygame.Rect(100+i*150,710,136,54).collidepoint(mouse):self.switch_training_profile(i);return
            if pygame.Rect(45,330,270,50).collidepoint(mouse):
                if self.training_active:
                    self.training_stop_requested=True
                elif self.training_tracks:
                    self.training_active=True;self.training_stop_requested=False
                else:self.message="Select at least one track"
            if pygame.Rect(45,400,270,50).collidepoint(mouse): self.save_champion()
            for i,speed in enumerate((1,2,4,0)):
                if pygame.Rect(45+i*72,485,62,40).collidepoint(mouse):self.training_speed=speed
        if event.type==pygame.KEYDOWN and self.typing_name:
            if event.key==pygame.K_RETURN: self.typing_name=False
            elif event.key==pygame.K_BACKSPACE: self.model_name=self.model_name[:-1]
            elif event.unicode.isprintable() and len(self.model_name)<24: self.model_name+=event.unicode

    def update_train(self):
        if self.training_active:
            self.run_training_generation()

    def init_population(self):
        config=load_neat_config(BASE_DIR); self.population=neat.Population(config)
        if self.training_seed:
            seed=deserialize_genome(self.training_seed.genome)
            keys=list(self.population.population)
            for i,key in enumerate(keys[:max(1,int(len(keys)*.4))]):
                clone=copy.deepcopy(seed); clone.key=key; clone.fitness=None
                if i: clone.mutate(config.genome_config)
                self.population.population[key]=clone
            self.population.species.speciate(
                config, self.population.population, self.population.generation
            )

    def run_training_generation(self):
        if not self.training_tracks: self.message="Select at least one track"; return
        if self.population is None: self.init_population()
        track=self.training_tracks[self.training_track_index%len(self.training_tracks)]; self.training_track_index+=1
        runtime=create_track_runtime(track); config=self.population.config; skin=self.skins[self.training_skin]
        stats=CAR_SPECS[self.training_skin];self.training_finisher=None
        def evaluate(genomes, _):
            cars=[]; nets=[]; active=[]
            for _,genome in genomes:
                genome.fitness=0.0; active.append(genome); cars.append(Car(skin,runtime,stats)); nets.append(neat.nn.FeedForwardNetwork.create(genome,config))
            clock=pygame.time.Clock(); frame=0
            while cars and frame<int(track.timeout*60):
                for event in pygame.event.get():
                    if event.type==pygame.QUIT:self.running=False;self.training_stop_requested=True
                    if event.type==pygame.KEYDOWN and event.key==pygame.K_SPACE:self.training_paused=not self.training_paused
                    if event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE:self.training_stop_requested=True
                    if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                        mouse=self.logical_mouse()
                        if pygame.Rect(1000,700,220,52).collidepoint(mouse):self.training_stop_requested=True
                        for i,speed in enumerate((1,2,4,0)):
                            if pygame.Rect(930+i*70,635,60,38).collidepoint(mouse):self.training_speed=speed
                if self.training_stop_requested:
                    if active:
                        partial=max(active,key=lambda genome:genome.fitness)
                        if self.champion is None or (partial.fitness or 0)>(self.champion.fitness or 0):self.champion=copy.deepcopy(partial)
                    raise TrainingStopped
                if self.training_paused:
                    self.draw_training_runtime(runtime,cars,track,frame,True);clock.tick(30);continue
                frame+=1
                for i,car in enumerate(cars):
                    output=nets[i].activate(car.sensors(runtime)); car.step(max(range(4),key=output.__getitem__)); active[i].fitness+=car.vel*.01-.001
                for i in range(len(cars)-1,-1,-1):
                    finished,stalled,timeout=cars[i].update_progress(runtime)
                    if finished:
                        active[i].fitness+=1000;self.training_finisher=copy.deepcopy(active[i])
                        cars.clear();nets.clear();active.clear();break
                    elif cars[i].crashed(runtime) or stalled or timeout: active[i].fitness-=2; del cars[i],nets[i],active[i]
                render_every=10 if self.training_speed==0 else 1
                if frame%render_every==0:self.draw_training_runtime(runtime,cars,track,frame,False)
                if self.training_speed:clock.tick(60*self.training_speed)
        try:
            candidate=self.population.run(evaluate,1)
        except TrainingStopped:
            self.training_active=False;self.training_stop_requested=False
            self.training_track_index=max(0,self.training_track_index-1)
            self.message="Training stopped · last completed champion preserved"
            self.show_training_save_modal=self.champion is not None
            self.save_active_training_profile()
            return
        self.champion=candidate;self.training_generation+=1
        if self.training_finisher:
            self.champion=self.training_finisher
            self.champion_validation={selected.track_id:selected.track_id==track.track_id for selected in self.training_tracks}
            self.training_active=False
            self.message=f"Training complete · a car finished {track.name} · save best model"
            self.show_training_save_modal=True
            self.save_active_training_profile();return
        self.champion_validation=self.validate_champion()
        self.message=f"Generation {self.training_generation} complete · fitness {self.champion.fitness:.2f}"
        if self.champion_validation and all(self.champion_validation.values()):
            self.training_active=False
            self.message=f"Validated after generation {self.training_generation} · ready to save"
            self.show_training_save_modal=True
        self.save_active_training_profile()

    def draw_training_runtime(self,runtime,cars,track,frame,paused):
        self.background();state="Training paused" if paused else "Training"
        speed="Max" if self.training_speed==0 else f"{self.training_speed}×"
        if track.runtime_type!="legacy_bitmap":self.title(state,f"Generation {self.training_generation} · {track.name} · {len(cars)} alive · {frame/60:.1f}s · {speed}")
        self.draw_runtime(runtime,cars)
        if track.runtime_type=="legacy_bitmap":
            self.canvas.blit(font(27,True).render(state,True,COLORS["white"]),(945,70))
            details=(f"Generation {self.training_generation}",track.name,f"{len(cars)} alive",f"{frame/60:.1f}s · {speed}")
            for i,text in enumerate(details):self.canvas.blit(font(18).render(text,True,COLORS["muted"]),(945,112+i*30))
        for i,value in enumerate((1,2,4,0)):
            label="Max" if value==0 else f"{value}×"
            PillButton((930+i*70,635,60,38),label,"",True).draw(self.canvas,self.logical_mouse())
            if value==self.training_speed:pygame.draw.rect(self.canvas,COLORS["cyan"],(930+i*70,635,60,38),2,border_radius=19)
        PillButton((1000,700,220,52),"Stop Training","",True).draw(self.canvas,self.logical_mouse())
        self.present()

    def validate_champion(self):
        if not self.champion: return {}
        config=load_neat_config(BASE_DIR); results={}
        for track in self.training_tracks:
            runtime=create_track_runtime(track); car=Car(self.skins[self.training_skin],runtime,CAR_SPECS[self.training_skin])
            net=neat.nn.FeedForwardNetwork.create(self.champion,config)
            passed=False
            for frame in range(int(track.timeout*60)):
                output=net.activate(car.sensors(runtime)); car.step(max(range(4),key=output.__getitem__))
                finished, stalled, timeout = car.update_progress(runtime)
                if finished: passed=True; break
                if car.crashed(runtime) or stalled or timeout: break
            results[track.track_id]=passed
        return results

    def save_champion(self):
        if not self.champion: self.message="Train at least one generation first"; return
        validation=self.champion_validation or self.validate_champion(); status="validated" if validation and all(validation.values()) else "draft"
        model=ModelRecord(name=self.model_name or "Unnamed Racer",skin=self.training_skin,genome=serialize_genome(self.champion),generation=self.training_generation,fitness=float(self.champion.fitness or 0),status=status,trained_tracks=[t.track_id for t in self.training_tracks],validation=validation,car_stats=dict(CAR_SPECS[self.training_skin]))
        self.storage.save_model(model); self.refresh_models(model.model_id); self.message=f"Saved {model.name} as {status}"
        self.save_active_training_profile()

    def draw_train(self):
        self.title("Train", "Configure models, choose track snapshots, and evolve until a car finishes")
        self.canvas.blit(font(17,True).render("MODEL NAME",True,COLORS["muted"]),(45,105)); pygame.draw.rect(self.canvas,COLORS["card"],(40,130,300,44),border_radius=18)
        self.canvas.blit(font(19).render(self.model_name or "Type a name",True,COLORS["white"]),(55,141))
        self.canvas.blit(font(17,True).render("CAR SKIN",True,COLORS["muted"]),(45,185))
        for i,(skin,image) in enumerate(self.skins.items()):
            rect=pygame.Rect(45+i*68,210,54,80); pygame.draw.rect(self.canvas,COLORS["cyan"] if skin==self.training_skin else COLORS["card"],rect,2,border_radius=14); self.canvas.blit(image,image.get_rect(center=rect.center))
        train_label="Stop Training" if self.training_active else "Start Training"
        PillButton((45,330,270,50),train_label,"",bool(self.training_tracks)).draw(self.canvas,self.logical_mouse())
        PillButton((45,400,270,50),"Save Current Best","",self.champion is not None).draw(self.canvas,self.logical_mouse())
        self.canvas.blit(font(17,True).render("SPEED",True,COLORS["muted"]),(45,460))
        for i,value in enumerate((1,2,4,0)):
            label="Max" if value==0 else f"{value}×";rect=(45+i*72,485,62,40)
            PillButton(rect,label,"",True).draw(self.canvas,self.logical_mouse())
            if value==self.training_speed:pygame.draw.rect(self.canvas,COLORS["cyan"],rect,2,border_radius=20)
        stats=CAR_SPECS[self.training_skin]
        self.canvas.blit(font(16,True).render("CAR PERFORMANCE",True,COLORS["muted"]),(45,545))
        self.canvas.blit(font(15).render(f"Speed {stats['max_speed']:.1f} · Accel {stats['acceleration']:.2f}",True,COLORS["white"]),(45,570))
        self.canvas.blit(font(15).render(f"Turning {stats['turn_speed']:.1f}",True,COLORS["white"]),(45,592))
        passed=sum(bool(value) for value in self.champion_validation.values())
        fitness=float(self.champion.fitness or 0) if self.champion else 0.0
        status="Running" if self.training_active else ("Complete" if self.training_finisher else ("Best ready" if self.champion else "Not started"))
        self.canvas.blit(font(16,True).render("TRAINING STATS",True,COLORS["muted"]),(45,618))
        self.canvas.blit(font(15).render(f"Generation {self.training_generation} · Best fitness {fitness:.1f}",True,COLORS["white"]),(45,642))
        self.canvas.blit(font(15).render(f"Tracks {len(self.training_tracks)} · Passed {passed} · {status}",True,COLORS["white"]),(45,664))
        if self.training_seed:self.canvas.blit(font(13).render(f"Seed: {self.training_seed.name}",True,COLORS["cyan"]),(45,686))
        self.canvas.blit(font(17,True).render("TRACKS",True,COLORS["muted"]),(385,100))
        for rect,track in self.training_track_cards():
            selected=track in self.training_tracks
            pygame.draw.rect(self.canvas,COLORS["card"],rect,border_radius=18)
            preview=self.track_thumbnail(track);self.canvas.blit(preview,(rect.x+12,rect.y+10))
            overlay=pygame.Surface((rect.width-24,26),pygame.SRCALPHA);overlay.fill((10,15,18,190));self.canvas.blit(overlay,(rect.x+12,rect.bottom-36))
            self.canvas.blit(font(15,True).render(track.name,True,COLORS["white"]),(rect.x+22,rect.bottom-32))
            if selected:pygame.draw.rect(self.canvas,COLORS["cyan"],rect,3,border_radius=18)
        self.canvas.blit(font(15,True).render("TRAINING MODELS",True,COLORS["muted"]),(45,692))
        PillButton((45,718,44,44),"+","",len(self.training_profiles)<6).draw(self.canvas,self.logical_mouse())
        self.training_profiles[self.active_training_profile]=self.capture_training_profile()
        for i,state in enumerate(self.training_profiles):
            rect=pygame.Rect(100+i*150,710,136,54);selected=i==self.active_training_profile
            pygame.draw.rect(self.canvas,COLORS["card"],rect,border_radius=16)
            self.canvas.blit(self.skins[state["skin"]],(rect.x+8,rect.y-4))
            self.canvas.blit(font(14,True).render(state["name"][:11],True,COLORS["white"]),(rect.x+46,rect.y+10))
            self.canvas.blit(font(12).render(f"Gen {state['generation']}",True,COLORS["muted"]),(rect.x+46,rect.y+32))
            if selected:pygame.draw.rect(self.canvas,COLORS["cyan"],rect,2,border_radius=16)
        if self.show_training_save_modal:
            shade=pygame.Surface(LOGICAL_SIZE,pygame.SRCALPHA);shade.fill((0,0,0,145));self.canvas.blit(shade,(0,0))
            box=pygame.Rect(385,275,510,235);pygame.draw.rect(self.canvas,COLORS["ink"],box,border_radius=28)
            heading="Training complete" if self.training_finisher else "Training stopped"
            self.canvas.blit(font(30,True).render(heading,True,COLORS["white"]),(box.x+34,box.y+30))
            self.canvas.blit(font(17).render(f"Best fitness: {fitness:.1f} · Generation: {self.training_generation}",True,COLORS["muted"]),(box.x+34,box.y+82))
            self.canvas.blit(font(16).render("Save the best model to inventory?",True,COLORS["white"]),(box.x+34,box.y+116))
            PillButton((430,430,190,48),"Save Best Model","",self.champion is not None).draw(self.canvas,self.logical_mouse())
            PillButton((660,430,190,48),"Not Now","",True).draw(self.canvas,self.logical_mouse())

    # Editor
    @property
    def editor_origin(self): return (42,105)
    def snapshot(self): return {cell:copy.deepcopy(tile) for cell,tile in self.editor_tiles.items()}
    def editor_track(self):
        timeout=max(20,min(180,len(self.editor_tiles)*1.5))
        return TrackDefinition(self.editor_name or "Custom Track",list(self.editor_tiles.values()),timeout=timeout)

    def event_editor(self,event):
        mouse=self.logical_mouse(); ox,oy=self.editor_origin
        if event.type==pygame.MOUSEBUTTONUP and event.button==1 and self.editor_drag_kind:
            gx=int((mouse[0]-ox)//TILE_SIZE);gy=int((mouse[1]-oy)//TILE_SIZE)
            if 0<=gx<GRID_SIZE[0] and 0<=gy<GRID_SIZE[1]:
                self.editor_history.append(self.snapshot())
                self.editor_tiles[(gx,gy)]=Tile(gx,gy,self.editor_drag_kind,self.editor_rotation)
                self.editor_selected=(gx,gy)
            self.editor_drag_kind=None
            return
        if event.type==pygame.KEYDOWN:
            if self.editor_typing:
                if event.key==pygame.K_RETURN:self.editor_typing=False
                elif event.key==pygame.K_BACKSPACE:self.editor_name=self.editor_name[:-1]
                elif event.unicode.isprintable() and len(self.editor_name)<28:self.editor_name+=event.unicode
                return
            if event.key==pygame.K_r:
                self.editor_history.append(self.snapshot())
                if self.editor_selected in self.editor_tiles:self.editor_tiles[self.editor_selected].rotation=(self.editor_tiles[self.editor_selected].rotation+90)%360
                else:self.editor_rotation=(self.editor_rotation+90)%360
            elif event.key in (pygame.K_DELETE,pygame.K_BACKSPACE) and self.editor_selected in self.editor_tiles:
                self.editor_history.append(self.snapshot());del self.editor_tiles[self.editor_selected];self.editor_selected=None
            elif event.key==pygame.K_z and (event.mod&(pygame.KMOD_CTRL|pygame.KMOD_META)) and self.editor_history:self.editor_tiles=self.editor_history.pop()
        if event.type==pygame.MOUSEBUTTONDOWN:
            if pygame.Rect(965,105,270,42).collidepoint(mouse):self.editor_typing=True;return
            palette={"straight":pygame.Rect(965,190,125,48),"corner":pygame.Rect(1110,190,125,48),"start_finish":pygame.Rect(965,255,270,48)}
            for kind,rect in palette.items():
                if rect.collidepoint(mouse):self.editor_kind=kind;self.editor_drag_kind=kind;self.editor_selected=None;return
            gx=int((mouse[0]-ox)//TILE_SIZE);gy=int((mouse[1]-oy)//TILE_SIZE)
            if 0<=gx<GRID_SIZE[0] and 0<=gy<GRID_SIZE[1]:
                cell=(gx,gy)
                if event.button==1:
                    self.editor_history.append(self.snapshot())
                    if cell in self.editor_tiles:self.editor_selected=cell
                    else:self.editor_tiles[cell]=Tile(gx,gy,self.editor_kind,self.editor_rotation);self.editor_selected=cell
                elif event.button==3 and cell in self.editor_tiles:self.editor_history.append(self.snapshot());del self.editor_tiles[cell]
            if pygame.Rect(965,330,125,44).collidepoint(mouse):pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_r,unicode="r",mod=0))
            if pygame.Rect(1110,330,125,44).collidepoint(mouse) and self.editor_history:self.editor_tiles=self.editor_history.pop()
            if pygame.Rect(965,395,270,44).collidepoint(mouse):self.save_editor_track()
            if pygame.Rect(965,455,125,44).collidepoint(mouse) and self.editor_selected in self.editor_tiles:
                self.editor_history.append(self.snapshot());del self.editor_tiles[self.editor_selected];self.editor_selected=None
            if pygame.Rect(1110,455,125,44).collidepoint(mouse):
                self.editor_history.append(self.snapshot());self.editor_tiles={};self.editor_selected=None
            if pygame.Rect(965,515,270,44).collidepoint(mouse):
                track=self.editor_track();errors,_=validate_track(track)
                if errors:self.editor_error=errors[0]
                elif not self.models:self.message="Save or train a model before testing"
                else:self.start_race(track,self.models[0])

    def save_editor_track(self):
        track=self.editor_track();errors,_=validate_track(track)
        if errors:self.editor_error=errors[0];return
        self.storage.save_track(track);self.custom=self.storage.custom_tracks();self.editor_error="";self.message=f"Saved {track.name}"

    def update_editor(self):pass

    def draw_piece(self,tile,origin):
        x=origin[0]+tile.x*TILE_SIZE;y=origin[1]+tile.y*TILE_SIZE
        self.canvas.blit(piece_atlas().surface(tile.kind,tile.rotation),(x,y))

    def draw_editor(self):
        self.title("Custom Track", "Place pieces · R rotates · right-click deletes · Ctrl/Cmd+Z undoes")
        ox,oy=self.editor_origin;area=pygame.Rect(ox,oy,GRID_SIZE[0]*TILE_SIZE,GRID_SIZE[1]*TILE_SIZE)
        pygame.draw.rect(self.canvas,(43,90,48),area)
        for x in range(GRID_SIZE[0]+1):pygame.draw.line(self.canvas,(70,115,72),(ox+x*TILE_SIZE,oy),(ox+x*TILE_SIZE,area.bottom))
        for y in range(GRID_SIZE[1]+1):pygame.draw.line(self.canvas,(70,115,72),(ox,oy+y*TILE_SIZE),(area.right,oy+y*TILE_SIZE))
        for tile in self.editor_tiles.values():self.draw_piece(tile,(ox,oy))
        if self.editor_drag_kind:
            mx,my=self.logical_mouse();gx=int((mx-ox)//TILE_SIZE);gy=int((my-oy)//TILE_SIZE)
            if 0<=gx<GRID_SIZE[0] and 0<=gy<GRID_SIZE[1]:
                self.draw_piece(Tile(gx,gy,self.editor_drag_kind,self.editor_rotation),(ox,oy))
                pygame.draw.rect(self.canvas,COLORS["cyan"],(ox+gx*TILE_SIZE,oy+gy*TILE_SIZE,TILE_SIZE,TILE_SIZE),2)
        if self.editor_selected:
            rect=pygame.Rect(ox+self.editor_selected[0]*TILE_SIZE,oy+self.editor_selected[1]*TILE_SIZE,TILE_SIZE,TILE_SIZE);pygame.draw.rect(self.canvas,COLORS["cyan"],rect,3)
        pygame.draw.rect(self.canvas,COLORS["card"],(950,90,300,620),border_radius=24)
        pygame.draw.rect(self.canvas,COLORS["ink"],(965,105,270,42),border_radius=16);self.canvas.blit(font(18).render(self.editor_name,True,COLORS["white"]),(980,115))
        for rect,label,selected in ((pygame.Rect(965,190,125,48),"Straight",self.editor_kind=="straight"),(pygame.Rect(1110,190,125,48),"Corner",self.editor_kind=="corner"),(pygame.Rect(965,255,270,48),"Start / Finish",self.editor_kind=="start_finish")):
            pygame.draw.rect(self.canvas,COLORS["cyan"] if selected else COLORS["ink"],rect,2 if selected else 0,border_radius=rect.height//2);self.canvas.blit(font(17,True).render(label,True,COLORS["white"]),font(17,True).render(label,True,COLORS["white"]).get_rect(center=rect.center))
        PillButton((965,330,125,44),"Rotate · R","").draw(self.canvas,self.logical_mouse());PillButton((1110,330,125,44),"Undo","").draw(self.canvas,self.logical_mouse())
        errors,_=validate_track(self.editor_track());PillButton((965,395,270,44),"Save Track","",not errors).draw(self.canvas,self.logical_mouse())
        PillButton((965,455,125,44),"Delete","",self.editor_selected in self.editor_tiles).draw(self.canvas,self.logical_mouse());PillButton((1110,455,125,44),"Clear","",bool(self.editor_tiles)).draw(self.canvas,self.logical_mouse())
        PillButton((965,515,270,44),"Test with first model","",not errors and bool(self.models)).draw(self.canvas,self.logical_mouse())
        status="Ready to save" if not errors else (self.editor_error or errors[0]);color=COLORS["green"] if not errors else COLORS["red"]
        words=[];line=""
        for word in status.split():
            if len(line)+len(word)>28:words.append(line);line=word
            else:line=(line+" "+word).strip()
        words.append(line)
        for i,text in enumerate(words):self.canvas.blit(font(16).render(text,True,color),(970,585+i*22))


def main():
    InteractiveApp().run()


if __name__ == "__main__":
    main()
