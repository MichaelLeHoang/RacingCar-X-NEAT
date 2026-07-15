"""Simulation-owned race state machine."""

from __future__ import annotations

import time
from enum import Enum

from racing_core import Car


class RaceState(str, Enum):
    PREPARING = "PREPARING"
    COUNTDOWN = "COUNTDOWN"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETE = "COMPLETE"
    CRASHED = "CRASHED"
    STALLED = "STALLED"
    TIMEOUT = "TIMEOUT"


TERMINAL_STATES = {
    RaceState.COMPLETE, RaceState.CRASHED, RaceState.STALLED, RaceState.TIMEOUT,
}


class RaceSession:
    COUNTDOWN_SECONDS = .8

    def __init__(self, runtime, level: int | None = None, clock=time.perf_counter):
        self.runtime = runtime
        self.level = level
        self.clock = clock
        self.state = RaceState.PREPARING
        self.model = None
        self.controller = None
        self.car: Car | None = None
        self.started_at: float | None = None
        self.elapsed_time = 0.0
        self.finished_elapsed_time: float | None = None
        self.countdown_remaining = self.COUNTDOWN_SECONDS
        self.new_personal_best = False

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def displayed_time(self) -> float:
        return (self.finished_elapsed_time if self.finished_elapsed_time is not None
                else self.elapsed_time)

    def accepts_model(self, model) -> bool:
        return getattr(model, "controller_version", None) == "five-sensor-v1"

    def accept_drop(self, model, sprite, controller) -> bool:
        if self.state != RaceState.PREPARING or not self.accepts_model(model):
            return False
        self.model = model
        self.controller = controller
        self.car = Car(sprite, self.runtime, model.car_stats)
        self.state = RaceState.COUNTDOWN
        self.countdown_remaining = self.COUNTDOWN_SECONDS
        return True

    def pause(self):
        if self.state == RaceState.RUNNING:
            self.state = RaceState.PAUSED

    def resume(self):
        if self.state == RaceState.PAUSED:
            self.state = RaceState.RUNNING

    def _finish(self, state: RaceState):
        self.state = state
        self.finished_elapsed_time = self.elapsed_time

    def update(self, dt: float):
        if self.state == RaceState.COUNTDOWN:
            self.countdown_remaining = max(0.0, self.countdown_remaining - dt)
            if self.countdown_remaining == 0:
                self.state = RaceState.RUNNING
                self.started_at = self.clock()
                self.model.attempts += 1
            return
        if self.state != RaceState.RUNNING or self.car is None:
            return
        self.elapsed_time += dt
        outputs = self.controller.activate(self.car.sensors(self.runtime))
        self.car.step(max(range(4), key=outputs.__getitem__))
        # Training uses the ordered gates returned by ``update_progress`` for
        # fitness and validation.  A player-facing race has the simpler,
        # visible contract: after the car has genuinely left the start area,
        # returning to the painted finish mask completes the attempt.  This
        # also prevents the car's initial placement on the line from winning.
        _, stalled, timeout = self.car.update_progress(self.runtime, dt)
        returned_to_finish = self.car.left_start and self.car.finish_overlap(self.runtime)
        if returned_to_finish:
            self._finish(RaceState.COMPLETE)
        elif self.car.crashed(self.runtime):
            self._finish(RaceState.CRASHED)
        elif stalled:
            self._finish(RaceState.STALLED)
        elif timeout:
            self._finish(RaceState.TIMEOUT)
