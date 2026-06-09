import unittest

from solver_lps.features.uwb.realtime.two_d.domain.position_calcul import estimate_position_least_squares


class PositionCalculTests(unittest.TestCase):
    def test_estimate_position_least_squares_recovers_known_point(self):
        anchors = {
            1: (0.0, 0.0),
            2: (100.0, 0.0),
            3: (0.0, 100.0),
        }
        target = (30.0, 40.0)
        distances = {
            aid: ((target[0] - ax) ** 2 + (target[1] - ay) ** 2) ** 0.5
            for aid, (ax, ay) in anchors.items()
        }
        result = estimate_position_least_squares(anchors, distances)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(target[0], result[0], places=5)
        self.assertAlmostEqual(target[1], result[1], places=5)


if __name__ == "__main__":
    unittest.main()

