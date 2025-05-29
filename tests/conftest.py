import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def ROOT_DIR():
    """Return the project root directory."""
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / ".git").is_dir()
            or (parent / "pyproject.toml").is_file()
            or (parent / "setup.py").is_file()
        ):
            return parent
    raise RuntimeError("Project root not found.")
