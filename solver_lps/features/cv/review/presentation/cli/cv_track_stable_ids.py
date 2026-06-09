if __package__ in (None, ""):
    import inspect
    import os
    import sys

    if "__file__" in globals():
        script_path = os.path.abspath(__file__)
    else:
        script_path = os.path.abspath(inspect.getsourcefile(lambda: 0) or sys.argv[0] or os.getcwd())
    package_root = os.path.abspath(os.path.join(os.path.dirname(script_path), "..", "..", "..", ".."))
    project_root = os.path.dirname(package_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from solver_lps.features.cv.review.domain.stable_id_tracker import *  # noqa: F401,F403
from solver_lps.features.cv.review.domain.stable_id_tracker import main


if __name__ == "__main__":
    raise SystemExit(main())

