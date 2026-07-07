from solver_lps.features.cv.review.analysis.full_court.domain.tracking_pipeline import build_tracking_frames


def build_cv_review_frames(csv_path: str, **overrides):
    return build_tracking_frames(csv_path, **overrides)
