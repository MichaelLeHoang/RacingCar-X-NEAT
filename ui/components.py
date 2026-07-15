from __future__ import annotations

import pygame


COLORS = {
    "ink": (19, 24, 29), "panel": (26, 33, 39), "card": (42, 51, 58),
    "white": (245, 247, 248), "muted": (170, 183, 190), "cyan": (33, 210, 235),
    "red": (215, 40, 62), "green": (60, 205, 120), "road": (111, 112, 116),
}


def font(size: int, bold: bool = False):
    return pygame.font.SysFont("arial", size, bold=bold)


class Button:
    def __init__(self, rect, label: str, action=None, enabled: bool = True,
                 selected: bool = False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.action = action
        self.enabled = enabled
        self.selected = selected

    def draw(self, surface: pygame.Surface, mouse):
        hovered = self.enabled and self.rect.collidepoint(mouse)
        if self.selected:
            color = COLORS["cyan"]
        elif hovered:
            color = COLORS["cyan"]
        else:
            color = (58, 71, 80) if self.enabled else (55, 60, 64)
        pygame.draw.rect(surface, color, self.rect, border_radius=self.rect.height // 2)
        text_color = COLORS["ink"] if hovered or self.selected else (
            COLORS["white"] if self.enabled else (110, 118, 122)
        )
        label = font(min(20, max(13, self.rect.height // 2)), True).render(
            self.label, True, text_color
        )
        surface.blit(label, label.get_rect(center=self.rect.center))


PillButton = Button


def draw_modal(surface: pygame.Surface, rect, title: str, lines=()):
    shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 150))
    surface.blit(shade, (0, 0))
    box = pygame.Rect(rect)
    pygame.draw.rect(surface, COLORS["ink"], box, border_radius=28)
    surface.blit(font(30, True).render(title, True, COLORS["white"]), (box.x + 34, box.y + 26))
    for index, line in enumerate(lines):
        surface.blit(font(17).render(str(line), True, COLORS["muted"]),
                     (box.x + 34, box.y + 78 + index * 27))
    return box

