if __package__ in (None, ""):
    import inspect
    import os
    import sys

    if "__file__" in globals():
        package_root = os.path.dirname(os.path.abspath(__file__))
    else:
        source_file = inspect.getsourcefile(lambda: 0) or sys.argv[0] or os.getcwd()
        package_root = os.path.dirname(os.path.abspath(source_file))
        if os.path.basename(package_root).lower() != "solver_lps":
            candidate = os.path.join(package_root, "solver_lps")
            if os.path.isdir(candidate):
                package_root = candidate
    project_root = os.path.dirname(package_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from solver_lps.presentation.pages.home_page import main


if __name__ == "__main__":
    raise SystemExit(main())
