import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from campaign import campaign_tracks
from race_session import RaceSession, RaceState
from racing_core import Car, ModelRecord, create_track_runtime


class StraightController:
    def activate(self, inputs):
        return (0.0, 0.0, 1.0, 0.0)


class BrakeController:
    def activate(self, inputs):
        return (0.0, 0.0, 0.0, 1.0)


def blank_model():
    return ModelRecord("Test", "white", {}, 1, 0.0)


class RaceProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init(); pygame.display.set_mode((1, 1))
        cls.sprite = pygame.Surface((32, 64), pygame.SRCALPHA)
        pygame.draw.rect(cls.sprite, (255, 255, 255), (5, 2, 22, 60))

    def set_car_center(self, car, center):
        car.x = center[0] - car.image.get_width() / 2
        car.y = center[1] - car.image.get_height() / 2

    def test_finish_requires_every_ordered_gate(self):
        runtime = create_track_runtime(campaign_tracks()[1])
        car = Car(self.sprite, runtime)
        car.left_start = True
        self.set_car_center(car, runtime.start_center)
        finished, _, _ = car.update_progress(runtime)
        self.assertFalse(finished)
        self.assertFalse(car.finish_armed)
        for gate in runtime.gates:
            self.set_car_center(car, gate.center)
            car.update_progress(runtime)
        self.assertEqual(len(runtime.gates), car.next_checkpoint)
        self.set_car_center(car, runtime.start_center)
        finished, _, _ = car.update_progress(runtime)
        self.assertTrue(finished)

    def test_touching_finish_early_never_completes(self):
        runtime = create_track_runtime(campaign_tracks()[2])
        car = Car(self.sprite, runtime)
        car.left_start = True
        self.set_car_center(car, runtime.start_center)
        self.assertTrue(car.finish_overlap(runtime))
        self.assertFalse(car.update_progress(runtime)[0])

    def test_original_track_finish_mask_completes_without_approximate_gates(self):
        runtime = create_track_runtime(campaign_tracks()[0])
        self.assertGreater(runtime.finish_mask.count(), 0)
        car = Car(self.sprite, runtime)
        car.left_start = True
        self.set_car_center(car, runtime.start_center)
        self.assertTrue(car.finish_overlap(runtime))
        self.assertLess(car.next_checkpoint, len(runtime.gates))
        self.assertTrue(car.update_progress(runtime)[0])

    def test_play_completes_on_finish_mask_after_leaving_start(self):
        runtime = create_track_runtime(campaign_tracks()[2])
        session = RaceSession(runtime)
        session.accept_drop(blank_model(), self.sprite, BrakeController())
        session.update(.8)
        session.car.left_start = True
        self.set_car_center(session.car, runtime.start_center)
        self.assertLess(session.car.next_checkpoint, len(runtime.gates))
        session.update(1 / 60)
        self.assertEqual(RaceState.COMPLETE, session.state)
        self.assertEqual(session.elapsed_time, session.finished_elapsed_time)

    def test_play_does_not_complete_on_initial_finish_overlap(self):
        runtime = create_track_runtime(campaign_tracks()[1])
        session = RaceSession(runtime)
        session.accept_drop(blank_model(), self.sprite, BrakeController())
        session.update(.8)
        self.set_car_center(session.car, runtime.start_center)
        session.update(1 / 60)
        self.assertFalse(session.car.left_start)
        self.assertEqual(RaceState.RUNNING, session.state)

    def test_terminal_time_is_permanently_frozen(self):
        runtime = create_track_runtime(campaign_tracks()[1])
        session = RaceSession(runtime, 2, clock=lambda: 10.0)
        self.assertTrue(session.accept_drop(blank_model(), self.sprite, StraightController()))
        session.update(.8)
        self.assertEqual(RaceState.RUNNING, session.state)
        session.elapsed_time = 3.25
        session._finish(RaceState.TIMEOUT)
        for _ in range(100): session.update(1 / 60)
        self.assertEqual(3.25, session.displayed_time)

    def test_pause_and_restart_reset_all_attempt_state(self):
        runtime = create_track_runtime(campaign_tracks()[1])
        session = RaceSession(runtime)
        session.accept_drop(blank_model(), self.sprite, StraightController()); session.update(.8)
        session.elapsed_time = 2.0; session.pause(); session.update(5.0)
        self.assertEqual(2.0, session.elapsed_time)
        fresh = RaceSession(runtime)
        self.assertEqual(RaceState.PREPARING, fresh.state)
        self.assertEqual(0.0, fresh.elapsed_time)
        self.assertIsNone(fresh.car); self.assertIsNone(fresh.finished_elapsed_time)


if __name__ == "__main__":
    unittest.main()
