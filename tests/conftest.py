"""
Pytest configuration and fixtures.
"""

import pytest
import sys
from pathlib import Path
import torch

# Add src directory to path for imports to ensure murano is importable
_src_path = Path(__file__).parent.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from murano.model import MuranoModel

import os


@pytest.fixture(scope="session")
def murano_model():
    """
    Session-scoped fixture to load the MuranoModel (gpt2) once.
    This avoids reloading the model for every test function.
    """
    print("\nLoading model for testing (gpt2)...")
    model = MuranoModel.from_pretrained(
        "gpt2",
        device_map="auto",
        dispatch=False,
    )
    return model
