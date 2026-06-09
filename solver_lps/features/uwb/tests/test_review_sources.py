import unittest

from solver_lps.features.uwb.review.data.review_sources import (
    DEFAULT_UWB_TAG_REVIEW_PATH,
    UwbTagReviewSource,
)


class ReviewSourcesTests(unittest.TestCase):
    def test_default_tag_review_csv_loads(self):
        source = UwbTagReviewSource(DEFAULT_UWB_TAG_REVIEW_PATH)
        self.assertGreater(len(source.samples), 0)
        position = source.get_position_at(1.5)
        self.assertIsNotNone(position)
        self.assertEqual(3, len(position))


if __name__ == "__main__":
    unittest.main()

