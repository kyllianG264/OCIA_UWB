import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".solver_env"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
REQUIREMENTS_FILE = ROOT_DIR / "requirements-launcher.txt"
MAIN_FILE = ROOT_DIR / "main.py"

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "pygame": "pygame",
    "PIL": "pillow",
    "paho.mqtt.client": "paho-mqtt",
}


def _candidate_builder_pythons():
    seen = set()
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    user_profile = os.environ.get("USERPROFILE", "")
    candidates = [
        sys.executable,
        os.path.join(user_profile, ".venv", "Scripts", "python.exe"),
        str(ROOT_DIR.parent / "NeptuVisionS2" / ".venv" / "Scripts" / "python.exe"),
        os.path.join(local_appdata, "Programs", "Python", "Python312", "python.exe"),
        os.path.join(local_appdata, "Programs", "Python", "Python311", "python.exe"),
        os.path.join(user_profile, ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies", "python", "python.exe"),
        r"C:\Program Files\KiCad\10.0\bin\python.exe",
    ]
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(candidate))
        if candidate and normalized not in seen and os.path.exists(candidate):
            seen.add(normalized)
            yield candidate


def _run(command, *, cwd=None, capture_output=False, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=capture_output,
        env=env,
    )


def _venv_python_works():
    if not VENV_PYTHON.exists():
        return False
    probe = _run([str(VENV_PYTHON), "-c", "import sys; print(sys.executable)"], capture_output=True)
    return probe.returncode == 0


def _missing_imports(python_executable):
    probe_script = """
import importlib
import json
import sys

modules = json.loads(sys.argv[1])
missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
print(json.dumps(missing))
"""
    probe = _run(
        [
            str(python_executable),
            "-c",
            probe_script,
            json.dumps(list(REQUIRED_IMPORTS.keys())),
        ],
        capture_output=True,
        extra_env={"PYGAME_HIDE_SUPPORT_PROMPT": "1"},
    )
    if probe.returncode != 0:
        return list(REQUIRED_IMPORTS.items())
    try:
        output = (probe.stdout or "").strip().splitlines()
        missing_modules = set(json.loads(output[-1] if output else "[]"))
    except json.JSONDecodeError:
        return list(REQUIRED_IMPORTS.items())
    return [(module_name, package_name) for module_name, package_name in REQUIRED_IMPORTS.items() if module_name in missing_modules]


def _create_or_repair_venv():
    builder_python = None
    for candidate in _candidate_builder_pythons():
        probe = _run([candidate, "-c", "import ensurepip, venv"])
        if probe.returncode == 0:
            builder_python = candidate
            break
    if builder_python is None:
        raise SystemExit("Aucun interpreteur Python capable de creer un venv n'a ete trouve.")

    command = [builder_python, "-m", "venv", str(VENV_DIR), "--clear"]
    print(f"[solver-bootstrap] creation du venv: {' '.join(command)}")
    completed = _run(command, cwd=ROOT_DIR)
    if completed.returncode != 0:
        raise SystemExit("Echec de creation du venv local du solver.")


def _install_requirements():
    ensure_pip = _run([str(VENV_PYTHON), "-m", "ensurepip", "--upgrade"], cwd=ROOT_DIR)
    if ensure_pip.returncode != 0:
        raise SystemExit("Echec d'initialisation de pip dans le venv solver.")
    pip_upgrade = _run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], cwd=ROOT_DIR)
    if pip_upgrade.returncode != 0:
        raise SystemExit("Echec de mise a jour de pip dans le venv solver.")
    install = _run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], cwd=ROOT_DIR)
    if install.returncode != 0:
        raise SystemExit("Echec d'installation des dependances du solver.")


def ensure_solver_environment():
    if not _venv_python_works():
        _create_or_repair_venv()
    missing = _missing_imports(VENV_PYTHON)
    if missing:
        print("[solver-bootstrap] dependances manquantes: " + ", ".join(f"{module}->{package}" for module, package in missing))
        _install_requirements()
        missing = _missing_imports(VENV_PYTHON)
        if missing:
            raise SystemExit(
                "Le venv solver est present mais incomplet apres installation: "
                + ", ".join(f"{module}->{package}" for module, package in missing)
            )
    return str(VENV_PYTHON)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Bootstrap du solver LPS.")
    parser.add_argument("--check", action="store_true", help="Verifie/repare l'environnement puis quitte.")
    return parser.parse_known_args(argv)


def main(argv=None):
    args, passthrough = parse_args(argv)
    python_executable = ensure_solver_environment()
    if args.check:
        print(f"[solver-bootstrap] environnement OK: {python_executable}")
        return 0

    os.chdir(ROOT_DIR)
    sys.argv = [str(MAIN_FILE), *passthrough]
    with open(MAIN_FILE, "rb") as handle:
        code = compile(handle.read(), str(MAIN_FILE), "exec")
    globals_dict = {"__name__": "__main__", "__file__": str(MAIN_FILE), "__package__": None}
    exec(code, globals_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
