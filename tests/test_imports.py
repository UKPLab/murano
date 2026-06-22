"""Import smoke tests for lightweight package entry points."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(SRC_ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return env


def test_importing_results_does_not_import_model():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import murano.results; print('murano.model' in sys.modules)",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=_subprocess_env(),
    )
    assert completed.stdout.strip() == "False"


def test_importing_murano_does_not_import_model():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import murano; print('murano.model' in sys.modules)",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=_subprocess_env(),
    )
    assert completed.stdout.strip() == "False"


def test_accessing_logits_step_does_not_import_model():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, murano; "
                "assert murano.Logits is not None; "
                "print('murano.model' in sys.modules)"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=_subprocess_env(),
    )
    assert completed.stdout.strip() == "False"


def test_quick_compliance_rate_does_not_import_model():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, murano; "
                "murano.compliance_rate(['Sure'], [\"I'm sorry\"]); "
                "print('murano.model' in sys.modules)"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=_subprocess_env(),
    )
    assert completed.stdout.strip() == "False"
