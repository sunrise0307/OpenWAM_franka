"""Use one Polymetis build for all native controller libraries."""

import os
from pathlib import Path
import resource
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYMETIS_ROOT = PROJECT_ROOT / "frankateleop/polymetis/polymetis"
BUILD_DIR = POLYMETIS_ROOT / "build_eef"
PYTHON_DIR = POLYMETIS_ROOT / "python"


def _prefixed_path(prefixes, existing):
    entries = list(prefixes)
    entries.extend(value for value in existing.split(os.pathsep) if value and value not in entries)
    return os.pathsep.join(entries)


def build_environment(current):
    """Select the build used by the local robot launch scripts."""
    required = (
        BUILD_DIR / "torch_isolation/libtorchrot.so",
        BUILD_DIR / "torch_isolation/libtorchscript_pinocchio.so",
        PYTHON_DIR / "polymetis/robot_interface.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("local Polymetis build is incomplete: %s" % ", ".join(missing))
    environment = dict(current)
    environment["POLYMETIS_BUILD_DIR"] = str(BUILD_DIR)
    environment["PYTHONPATH"] = _prefixed_path((str(PYTHON_DIR), str(PROJECT_ROOT)),
                                                  current.get("PYTHONPATH", ""))
    environment["LD_LIBRARY_PATH"] = _prefixed_path((
        str(BUILD_DIR / "torch_isolation/pinocchio_isolation"),
        str(BUILD_DIR / "torch_isolation"),
    ), current.get("LD_LIBRARY_PATH", ""))
    return environment


def bootstrap():
    """Restart before native imports so the dynamic loader sees the same build."""
    environment = build_environment(os.environ)
    keys = ("POLYMETIS_BUILD_DIR", "PYTHONPATH", "LD_LIBRARY_PATH")
    if any(os.environ.get(key) != environment[key] for key in keys):
        os.execvpe(sys.executable,
                   [sys.executable, "-m", "rot6d_deploy.main", *sys.argv[1:]],
                   environment)


def preflight_imports():
    """Contain a native import failure in a child process before robot access."""
    def disable_core_dump():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import polymetis; import torchcontrol; "
             "from rot6d_deploy.control.cartesian_policy import WatchdogCartesianImpedanceControl"],
            capture_output=True, text=True, timeout=20, check=False,
            preexec_fn=disable_core_dump,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Polymetis library import timed out") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[-1600:]
        raise RuntimeError("Polymetis library import failed in a separate process "
                           "(exit %d): %s" % (result.returncode, detail))
