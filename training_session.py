"""Responsive, deterministic NEAT training and validation sessions."""

from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass, field
from enum import Enum

import neat

from racing_core import Car, TILE_SIZE, create_track_runtime
from track_generator import TrackGenerationConstraints, generate_track


class TrainingMode(str, Enum):
    ORIGINAL = "original"
    CUSTOM = "custom"
    RANDOM_CURRICULUM = "random_curriculum"


class TrainingState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    VALIDATING = "VALIDATING"
    STOPPED = "STOPPED"
    VALIDATED = "VALIDATED"


@dataclass(frozen=True)
class FitnessPolicy:
    progress_per_tile: float = 10.0
    checkpoint_reward: float = 5.0
    speed_reward: float = .01
    completion_reward: float = 10_000.0
    time_efficiency_reward: float = 10_000.0
    suite_track_completion_reward: float = 100_000.0
    suite_all_tracks_reward: float = 100_000.0
    collision_penalty: float = 25.0
    backward_per_tile: float = 15.0
    spin_penalty: float = 2.0
    stall_penalty: float = 25.0
    timeout_penalty: float = 10.0
    oscillation_penalty: float = 1.0


@dataclass
class TrainingProfile:
    mode: TrainingMode
    skin: str
    name: str = "New Racer"
    tracks: list = field(default_factory=list)
    base_seed: int = 1
    difficulty_range: tuple[int, int] = (1, 5)
    random_tracks_per_generation: int = 3

    def validation_scope(self, held_out_seeds: list[int] | None = None) -> dict:
        return {
            "mode": self.mode.value,
            "track_ids": [track.track_id for track in self.tracks],
            "base_seed": self.base_seed,
            "difficulty_range": list(self.difficulty_range),
            "held_out_seeds": list(held_out_seeds or []),
        }


@dataclass
class EvaluationEntry:
    genome: object
    network: object
    car: Car
    score: float = 0.0
    last_progress: float = 0.0
    last_checkpoint: int = 0
    last_backward: float = 0.0
    last_oscillations: int = 0
    completed: bool = False


class TrainingSession:
    """Advance a NEAT generation a bounded number of simulation steps."""

    def __init__(self, population: neat.Population, profile: TrainingProfile,
                 sprites: dict[str, object], car_stats: dict[str, dict],
                 policy: FitnessPolicy | None = None):
        self.population = population
        self.profile = profile
        self.sprites = sprites
        self.car_stats = car_stats
        self.policy = policy or FitnessPolicy()
        self.state = TrainingState.IDLE
        self.generation = population.generation
        self.generation_tracks: list = []
        self.track_index = 0
        self.runtime = None
        self.active: list[EvaluationEntry] = []
        self.track_scores: dict[int, list[float]] = {}
        self.track_completions: dict[int, list[bool]] = {}
        self.completed_champion = copy.deepcopy(population.best_genome)
        self.candidate_champion = None
        self.validation_results: list[dict] = []
        self.held_out_seeds = [profile.base_seed + 1_000_001 + index for index in range(3)]
        self._validation_tracks: list = []
        self._validation_index = 0
        self._validation_entry: EvaluationEntry | None = None
        self.completed_generations = 0

    @property
    def validation_scope(self) -> dict:
        return self.profile.validation_scope(
            self.held_out_seeds if self.profile.mode == TrainingMode.RANDOM_CURRICULUM else []
        )

    @property
    def champion_validated(self) -> bool:
        expected = 3 if self.profile.mode == TrainingMode.RANDOM_CURRICULUM else len(self.profile.tracks)
        return (expected > 0 and len(self.validation_results) == expected
                and all(item["passed"] for item in self.validation_results))

    def _curriculum_tracks(self) -> list:
        low, high = self.profile.difficulty_range
        tracks = []
        span = high - low + 1
        for index in range(self.profile.random_tracks_per_generation):
            sequence = self.completed_generations * self.profile.random_tracks_per_generation + index
            seed = self.profile.base_seed + sequence * 10_003
            difficulty = low + sequence % span
            tracks.append(generate_track(seed, difficulty))
        return tracks

    def _training_tracks(self) -> list:
        if self.profile.mode == TrainingMode.RANDOM_CURRICULUM:
            return self._curriculum_tracks()
        return list(self.profile.tracks)

    def _held_out_tracks(self) -> list:
        if self.profile.mode != TrainingMode.RANDOM_CURRICULUM:
            return list(self.profile.tracks)
        low, high = self.profile.difficulty_range
        span = high - low + 1
        return [generate_track(seed, low + index % span)
                for index, seed in enumerate(self.held_out_seeds)]

    def start(self):
        if self.state in (TrainingState.PAUSED, TrainingState.RUNNING, TrainingState.VALIDATING):
            return
        if not self._training_tracks():
            raise ValueError("Select at least one valid training track")
        self.state = TrainingState.RUNNING
        self._begin_generation()

    def pause(self):
        if self.state in (TrainingState.RUNNING, TrainingState.VALIDATING):
            self._paused_from = self.state
            self.state = TrainingState.PAUSED

    def resume(self):
        if self.state == TrainingState.PAUSED:
            self.state = getattr(self, "_paused_from", TrainingState.RUNNING)

    def stop(self):
        if self.state in (TrainingState.RUNNING, TrainingState.PAUSED, TrainingState.VALIDATING):
            self.active.clear()
            self._validation_entry = None
            self.state = TrainingState.STOPPED

    def _begin_generation(self):
        self.generation_tracks = self._training_tracks()
        self.track_index = 0
        self.track_scores = {key: [] for key in self.population.population}
        self.track_completions = {key: [] for key in self.population.population}
        self.population.reporters.start_generation(self.population.generation)
        for genome in self.population.population.values():
            genome.fitness = 0.0
        self._begin_track()

    def _begin_track(self):
        track = self.generation_tracks[self.track_index]
        self.runtime = create_track_runtime(track)
        self.active = []
        config = self.population.config
        for genome in self.population.population.values():
            car = Car(self.sprites[self.profile.skin], self.runtime,
                      self.car_stats[self.profile.skin])
            self.active.append(EvaluationEntry(
                genome=genome,
                network=neat.nn.FeedForwardNetwork.create(genome, config),
                car=car,
                last_progress=car.last_progress,
            ))

    def _step_entry(self, entry: EvaluationEntry) -> bool:
        car = entry.car
        runtime = self.runtime
        output = entry.network.activate(car.sensors(runtime))
        car.step(max(range(4), key=output.__getitem__))
        previous_progress = car.last_progress
        previous_checkpoint = car.next_checkpoint
        previous_backward = car.backward_progress
        previous_oscillations = car.oscillations
        finished, stalled, timeout = car.update_progress(runtime)
        delta = car.last_progress - previous_progress
        lap = max(TILE_SIZE, runtime.metrics.estimated_lap_distance)
        if delta < -lap * .5:
            delta += lap
        elif delta > lap * .5:
            delta -= lap
        if delta > 0:
            entry.score += delta / TILE_SIZE * self.policy.progress_per_tile
        entry.score -= max(0.0, car.backward_progress - previous_backward) / TILE_SIZE * self.policy.backward_per_tile
        if car.next_checkpoint > previous_checkpoint:
            entry.score += (car.next_checkpoint - previous_checkpoint) * self.policy.checkpoint_reward
        entry.score += self.policy.speed_reward * (car.vel / max(.01, car.max_vel))
        if car.angle_without_progress >= 360:
            entry.score -= self.policy.spin_penalty
            car.angle_without_progress %= 360
        if car.oscillations > previous_oscillations:
            entry.score -= self.policy.oscillation_penalty
        crashed = car.crashed(runtime)
        if finished:
            entry.score += self.policy.completion_reward
            time_efficiency = max(0.0, 1.0 - car.elapsed / max(.01, runtime.definition.timeout))
            entry.score += time_efficiency * self.policy.time_efficiency_reward
            entry.completed = True
            return True
        if crashed:
            entry.score -= self.policy.collision_penalty
            return True
        if stalled:
            entry.score -= self.policy.stall_penalty
            return True
        if timeout:
            entry.score -= self.policy.timeout_penalty
            return True
        return False

    def step(self):
        if self.state == TrainingState.RUNNING:
            for index in range(len(self.active) - 1, -1, -1):
                entry = self.active[index]
                if self._step_entry(entry):
                    self.track_scores[entry.genome.key].append(entry.score)
                    self.track_completions[entry.genome.key].append(entry.completed)
                    del self.active[index]
            if not self.active:
                self.track_index += 1
                if self.track_index < len(self.generation_tracks):
                    self._begin_track()
                else:
                    self._finish_generation()
        elif self.state == TrainingState.VALIDATING:
            self._step_validation()

    def advance(self, step_budget: int = 1):
        for _ in range(max(0, step_budget)):
            if self.state not in (TrainingState.RUNNING, TrainingState.VALIDATING):
                break
            self.step()

    def _finish_generation(self):
        best = None
        for key, genome in self.population.population.items():
            scores = self.track_scores[key]
            completions = self.track_completions.get(key, [])
            completed_count = sum(bool(value) for value in completions)
            suite_size = max(1, len(scores), len(self.generation_tracks))
            all_completed = completed_count == suite_size
            # Completion count is deliberately dominant.  Only after two
            # genomes complete the same number of required tracks do their
            # worst-track performance and lap speed decide the ranking.
            genome.fitness = (
                completed_count * self.policy.suite_track_completion_reward
                + (self.policy.suite_all_tracks_reward if all_completed else 0.0)
                + statistics.fmean(scores) * .25
                + min(scores)
            )
            if best is None or genome.fitness > best.fitness:
                best = genome
        self.population.reporters.post_evaluate(
            self.population.config, self.population.population, self.population.species, best
        )
        if self.population.best_genome is None or best.fitness > self.population.best_genome.fitness:
            self.population.best_genome = copy.deepcopy(best)
        self.completed_champion = copy.deepcopy(self.population.best_genome)
        self.candidate_champion = copy.deepcopy(best)
        self.population.population = self.population.reproduction.reproduce(
            self.population.config, self.population.species,
            self.population.config.pop_size, self.population.generation,
        )
        if not self.population.species.species:
            self.population.reporters.complete_extinction()
            if self.population.config.reset_on_extinction:
                self.population.population = self.population.reproduction.create_new(
                    self.population.config.genome_type,
                    self.population.config.genome_config,
                    self.population.config.pop_size,
                )
            else:
                raise neat.CompleteExtinctionException()
        self.population.species.speciate(
            self.population.config, self.population.population, self.population.generation
        )
        self.population.reporters.end_generation(
            self.population.config, self.population.population, self.population.species
        )
        self.population.generation += 1
        self.generation = self.population.generation
        self.completed_generations += 1
        self._begin_validation()

    def _begin_validation(self):
        self.validation_results = []
        self._validation_tracks = self._held_out_tracks()
        self._validation_index = 0
        self.state = TrainingState.VALIDATING
        self._begin_validation_track()

    def _begin_validation_track(self):
        track = self._validation_tracks[self._validation_index]
        self.runtime = create_track_runtime(track)
        car = Car(self.sprites[self.profile.skin], self.runtime,
                  self.car_stats[self.profile.skin])
        network = neat.nn.FeedForwardNetwork.create(
            self.completed_champion, self.population.config
        )
        self._validation_entry = EvaluationEntry(self.completed_champion, network, car)

    def _step_validation(self):
        entry = self._validation_entry
        track = self._validation_tracks[self._validation_index]
        car = entry.car
        output = entry.network.activate(car.sensors(self.runtime))
        car.step(max(range(4), key=output.__getitem__))
        finished, stalled, timeout = car.update_progress(self.runtime)
        crashed = car.crashed(self.runtime)
        if not (finished or stalled or timeout or crashed):
            return
        self.validation_results.append({
            "track_id": track.track_id,
            "seed": track.generation.get("seed"),
            "passed": bool(finished and not crashed),
            "elapsed": car.elapsed,
            "reason": "complete" if finished else (
                "collision" if crashed else "stalled" if stalled else "timeout"
            ),
        })
        self._validation_index += 1
        if self._validation_index >= len(self._validation_tracks):
            if self.champion_validated:
                self.state = TrainingState.VALIDATED
            else:
                self.state = TrainingState.RUNNING
                self._begin_generation()
        else:
            self._begin_validation_track()
