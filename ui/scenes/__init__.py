from enum import Enum


class SceneId(str, Enum):
    HOME = "home"
    LEVELS = "levels"
    RACE = "race"
    TRAIN = "train"
    EDITOR = "editor"

