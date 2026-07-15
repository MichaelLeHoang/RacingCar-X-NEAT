"""Versioned, non-executable model/track/campaign persistence."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from campaign import CampaignProgress
from track_geometry import TrackDefinition, validate_track


CONTROLLER_VERSION = "five-sensor-v1"
DEFAULT_CAR_STATS = {"max_speed": 4.0, "acceleration": .2, "turn_speed": 4.0}


@dataclass
class ModelRecord:
    name: str
    skin: str
    genome: dict
    generation: int
    fitness: float
    status: str = "draft"
    trained_tracks: list[str] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    attempts: int = 0
    wins: int = 0
    best_times: dict[str, float] = field(default_factory=dict)
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 2
    controller_version: str = CONTROLLER_VERSION
    car_stats: dict = field(default_factory=lambda: dict(DEFAULT_CAR_STATS))
    validation_scope: dict = field(default_factory=dict)
    validation_results: list[dict] = field(default_factory=list)
    lineage_id: str = ""

    def __post_init__(self):
        if not self.lineage_id:
            self.lineage_id = self.model_id

    @property
    def trained_track_count(self) -> int:
        return len(set(self.trained_tracks))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["schema_version"] = 2
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ModelRecord":
        version = int(data.get("schema_version", 1))
        if version not in (1, 2):
            raise ValueError("Unsupported model schema")
        controller = str(data.get("controller_version", CONTROLLER_VERSION))
        if controller != CONTROLLER_VERSION:
            raise ValueError("Incompatible controller version")
        known = cls.__dataclass_fields__
        migrated = {key: value for key, value in data.items() if key in known}
        migrated["schema_version"] = 2
        migrated["controller_version"] = controller
        migrated.setdefault("validation_scope", {})
        migrated.setdefault("validation_results", [
            {"track_id": str(track_id), "passed": bool(passed)}
            for track_id, passed in data.get("validation", {}).items()
        ])
        migrated.setdefault("validation", dict(data.get("validation", {})))
        migrated.setdefault("car_stats", dict(DEFAULT_CAR_STATS))
        model = cls(**migrated)
        model.name = model.name.strip() or "Unnamed Racer"
        model.best_times = {str(key): float(value) for key, value in model.best_times.items()}
        if model.status not in ("draft", "validated"):
            model.status = "draft"
        return model


@dataclass(frozen=True)
class ImportIssue:
    filename: str
    code: str
    message: str


@dataclass
class ImportReport:
    imported_models: int = 0
    imported_tracks: int = 0
    issues: list[ImportIssue] = field(default_factory=list)

    @property
    def imported(self) -> int:
        return self.imported_models + self.imported_tracks

    def __int__(self):
        return self.imported

    def __str__(self):
        return str(self.imported)

    def __eq__(self, other):
        if isinstance(other, int):
            return self.imported == other
        return super().__eq__(other)


class Storage:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get(
            "RACING_DATA_DIR", Path.home() / ".racing_car_x_neat"
        ))
        self.models_dir = self.root / "models"
        self.tracks_dir = self.root / "tracks"
        self.exports_dir = self.root / "exports"
        self.imports_dir = self.root / "imports"
        for directory in (self.models_dir, self.tracks_dir, self.exports_dir, self.imports_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.root / "progress.json"

    @staticmethod
    def _write(path: Path, data: dict):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)

    def models(self) -> list[ModelRecord]:
        records = []
        for path in self.models_dir.glob("*.rcmodel"):
            try:
                records.append(ModelRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        records.sort(key=lambda model: (model.name.casefold(), model.model_id))
        return records

    def save_model(self, model: ModelRecord):
        model.schema_version = 2
        self._write(self.models_dir / f"{model.model_id}.rcmodel", model.to_dict())

    @staticmethod
    def _training_family(model: ModelRecord) -> tuple:
        scope = json.dumps(model.validation_scope or {}, sort_keys=True, separators=(",", ":"))
        return (model.name.strip().casefold(), model.skin,
                tuple(sorted(set(model.trained_tracks))), scope)

    def save_training_snapshot(self, model: ModelRecord) -> ModelRecord:
        """Keep one inventory car per training lineage for Drafts and Champions."""
        records = self.models()
        family = self._training_family(model)
        members = [record for record in records if (
            record.lineage_id == model.lineage_id
            or (record.status == "draft" and record.lineage_id == record.model_id
                and self._training_family(record) == family)
        )]
        if members:
            exact = [record for record in members if record.lineage_id == model.lineage_id]
            target = max(exact or members, key=lambda item: (item.generation, item.fitness))
            model.model_id = target.model_id
            model.attempts = max([model.attempts, *(item.attempts for item in members)])
            model.wins = max([model.wins, *(item.wins for item in members)])
            for item in members:
                for track_id, elapsed in item.best_times.items():
                    previous = model.best_times.get(track_id)
                    model.best_times[track_id] = elapsed if previous is None else min(previous, elapsed)
                if item.model_id != model.model_id:
                    self.delete_model(item.model_id)
        self.save_model(model)
        return model

    def save_validated_champion(self, model: ModelRecord) -> ModelRecord:
        """Backward-compatible name for callers from the first schema-v2 release."""
        return self.save_training_snapshot(model)

    def rename_model(self, model_id: str, name: str) -> ModelRecord:
        model = next((item for item in self.models() if item.model_id == model_id), None)
        if model is None:
            raise KeyError(model_id)
        model.name = name.strip() or model.name
        self.save_model(model)
        return model

    def delete_model(self, model_id: str):
        path = self.models_dir / f"{model_id}.rcmodel"
        if path.exists():
            path.unlink()

    @staticmethod
    def _safe_name(name: str, fallback: str) -> str:
        safe = "".join(character if character.isalnum() or character in "-_" else "_"
                       for character in name)
        return safe or fallback

    def export_model(self, model: ModelRecord) -> Path:
        path = self.exports_dir / f"{self._safe_name(model.name, model.model_id)}.rcmodel"
        self._write(path, model.to_dict())
        return path

    def custom_tracks(self, valid_only: bool = False) -> list[TrackDefinition]:
        tracks = []
        for path in self.tracks_dir.glob("*.rctrack"):
            try:
                track = TrackDefinition.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            if not valid_only or validate_track(track).valid:
                tracks.append(track)
        tracks.sort(key=lambda track: (track.name.casefold(), track.track_id))
        return tracks

    def save_track(self, track: TrackDefinition):
        result = validate_track(track)
        if not result.valid:
            raise ValueError("; ".join(result.messages))
        track.schema_version = 2
        track.metrics = result.metrics.to_dict()
        self._write(self.tracks_dir / f"{track.track_id}.rctrack", track.to_dict())

    def delete_track(self, track_id: str):
        path = self.tracks_dir / f"{track_id}.rctrack"
        if path.exists():
            path.unlink()

    def export_track(self, track: TrackDefinition) -> Path:
        result = validate_track(track)
        if not result.valid:
            raise ValueError("Cannot export an invalid track")
        path = self.exports_dir / f"{self._safe_name(track.name, track.track_id)}.rctrack"
        self._write(path, track.to_dict())
        return path

    def import_inbox(self) -> ImportReport:
        report = ImportReport()
        for path in sorted(self.imports_dir.iterdir()):
            if not path.is_file() or path.suffix not in (".rcmodel", ".rctrack"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if path.suffix == ".rcmodel":
                    model = ModelRecord.from_dict(data)
                    model.model_id = str(uuid.uuid4())
                    self.save_model(model)
                    report.imported_models += 1
                else:
                    track = TrackDefinition.from_dict(data)
                    result = validate_track(track)
                    if not result.valid:
                        raise ValueError("; ".join(result.messages))
                    track.track_id = str(uuid.uuid4())
                    track.source = "custom"
                    self.save_track(track)
                    report.imported_tracks += 1
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                report.issues.append(ImportIssue(path.name, "IMPORT_REJECTED", str(error)))
        return report

    def progress_record(self) -> CampaignProgress:
        if not self.progress_path.exists():
            return CampaignProgress()
        try:
            return CampaignProgress.from_dict(json.loads(self.progress_path.read_text(encoding="utf-8")))
        except (ValueError, TypeError, json.JSONDecodeError):
            return CampaignProgress()

    def progress(self) -> dict:
        """Compatibility accessor used by earlier callers and saved tests."""
        return self.progress_record().to_dict()

    def save_progress(self, progress: CampaignProgress | dict):
        record = progress if isinstance(progress, CampaignProgress) else CampaignProgress.from_dict(progress)
        self._write(self.progress_path, record.to_dict())
