from solver_lps.features.cv.review.analysis.full_court.domain.cv_based_analysis import build_cv_review_frames


class TrackingPipelineAnalysisRepository:
    def build_tracking_frames(self, csv_path: str, **overrides):
        return build_cv_review_frames(csv_path, **overrides)
