import unittest

from solver_lps.features.ground.domain.projection import centered_screen_point, fit_bounds_to_rect, project_world_point


class _Rect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class ProjectionTests(unittest.TestCase):
    def test_centered_screen_point_projects_from_center(self):
        self.assertEqual((100, 50), centered_screen_point(0, 0, 1.0, 200, 100, 0, 0))
        self.assertEqual((110, 60), centered_screen_point(0, 0, 1.0, 200, 100, 10, 10))

    def test_fit_bounds_to_rect_returns_centered_projection(self):
        projection = fit_bounds_to_rect((0.0, 100.0, 0.0, 50.0), _Rect(10, 20, 200, 100))
        self.assertAlmostEqual(2.0, projection.scale)
        self.assertEqual((10, 20), project_world_point(projection, 0.0, 0.0))
        self.assertEqual((210, 120), project_world_point(projection, 100.0, 50.0))


if __name__ == "__main__":
    unittest.main()

