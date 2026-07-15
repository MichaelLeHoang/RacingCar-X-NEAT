import unittest

from campaign import CAMPAIGN_PATH, campaign_tracks
from track_generator import TrackGenerationConstraints, generate_track, generate_with_stats
from track_geometry import validate_track


class GeneratorTests(unittest.TestCase):
    def test_same_seed_is_identical(self):
        first = generate_track(987654, 6)
        second = generate_track(987654, 6)
        self.assertEqual(first.signature(), second.signature())
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.track_id, second.track_id)

    def test_different_seeds_produce_meaningful_variation(self):
        signatures = {generate_track(seed, 4).signature() for seed in range(20, 40)}
        self.assertGreaterEqual(len(signatures), 12)

    def test_generation_obeys_attempt_limit_and_valid_fallback(self):
        constraints = TrackGenerationConstraints(
            attempt_limit=1, node_limit_per_attempt=1,
            min_difficulty_score=9.9, max_difficulty_score=10.0,
        )
        track, stats = generate_with_stats(3, 1, constraints=constraints)
        self.assertTrue(validate_track(track).valid)
        self.assertLessEqual(stats.attempts, 1)
        self.assertTrue(stats.used_fallback)

    def test_two_hundred_representative_seeds_are_valid_and_bounded(self):
        for seed in range(200):
            difficulty = seed % 10 + 1
            track, stats = generate_with_stats(seed + 5000, difficulty)
            result = validate_track(track)
            self.assertTrue(result.valid, (seed, result.messages))
            self.assertLessEqual(stats.attempts, 200)
            self.assertEqual(len(track.tiles), len(result.path))

    def test_campaign_content_is_checked_in_and_stable(self):
        self.assertTrue(CAMPAIGN_PATH.exists())
        first = [track.signature() for track in campaign_tracks()]
        second = [track.signature() for track in campaign_tracks()]
        self.assertEqual(first, second)
        scores = [validate_track(track).metrics.difficulty_score for track in campaign_tracks()]
        self.assertTrue(all(a < b for a, b in zip(scores, scores[1:])))


if __name__ == "__main__":
    unittest.main()
