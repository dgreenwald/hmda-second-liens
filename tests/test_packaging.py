import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_prediction_is_the_only_unconditional_runtime_dependency():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    assert project["dependencies"] == ["numpy>=1.24"]
    extras = project["optional-dependencies"]
    assert {Requirement(value).name for value in extras["train"]} == {
        "dgreenwald-py-tools",
        "python-dotenv",
        "scikit-learn",
        "scipy",
        "pandas",
        "matplotlib",
        "pyarrow",
        "joblib",
        "threadpoolctl",
    }
    assert {Requirement(value).name for value in extras["dev"]} == {
        "pytest",
        "ruff",
        "build",
    }
