"""Explicit wheel installation check; not collected by ordinary pytest.

Run through make test-lightweight-install in a full [train,dev] environment.
Temporary environments may download build dependencies and NumPy.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from hmda_seconds import portable_predict
from hmda_seconds.logistic_features import (
    FeatureSpecification,
    LogisticFeatureTransformer,
)


def write_synthetic_case(directory):
    """Use the full transformer as the oracle, with fabricated parameters only."""
    rng = np.random.default_rng(76)
    frame = pd.DataFrame(
        {
            "log_lti": rng.normal(size=80),
            "log_county_value_to_loan": rng.normal(size=80),
            "purchaser_type": np.tile(np.arange(10), 8),
            "loan_type": np.repeat(np.arange(1, 5), 20),
        }
    )
    transform = LogisticFeatureTransformer(
        FeatureSpecification("spline_lti", "purchaser_type")
    ).fit(frame)
    coefficients = rng.normal(size=len(transform.feature_names_))
    offset = -0.2
    shares = {"2000": 0.12, "2001": 0.3}
    annual = {
        year: {
            "mixture_share": share,
            "intercept": offset + np.log(share / (1 - share)),
        }
        for year, share in shares.items()
    }
    payload = {
        "format_version": 1,
        "specification": "spline_lti__purchaser_type",
        "category_levels": portable_predict.CATEGORIES,
        "feature_names": transform.feature_names_,
        "coefficients": coefficients.tolist(),
        "density_ratio_offset": offset,
        "knots": transform.knots_["log_lti"].tolist(),
        "raw_location": [
            transform.raw_location_[v] for v in portable_predict.CONTINUOUS
        ],
        "raw_scale": [transform.raw_scale_[v] for v in portable_predict.CONTINUOUS],
        "basis_location": transform.basis_location_["log_lti"].tolist(),
        "basis_scale": transform.basis_scale_["log_lti"].tolist(),
        "annual": annual,
    }
    years = [2000, 2001] * 40
    score = transform.transform(frame) @ coefficients
    case = {
        "inputs": frame.to_dict(orient="list"),
        "years": years,
        "scalar": expit(score + annual["2000"]["intercept"]).tolist(),
        "mixed": expit(
            score + np.array([annual[str(y)]["intercept"] for y in years])
        ).tolist(),
    }
    (directory / "model.json").write_text(json.dumps(payload, allow_nan=False))
    (directory / "case.json").write_text(json.dumps(case, allow_nan=False))


INSTALLED_CHECK = """
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import hmda_seconds.portable_predict as predictor

module_path = Path(predictor.__file__).resolve()
assert module_path.is_relative_to(Path(sys.prefix).resolve()), module_path
installed = {d.metadata["Name"].lower().replace("_", "-") for d in importlib.metadata.distributions()}
assert installed <= {"pip", "setuptools", "wheel", "numpy", "hmda-second-liens"}, installed
metadata = importlib.metadata.metadata("hmda-second-liens")
requirements = metadata.get_all("Requires-Dist") or []
assert [r for r in requirements if ";" not in r] == ["numpy>=1.24"], requirements
model = predictor.load_model("model.json")
case = json.loads(Path("case.json").read_text())
for mode, years in (("scalar", 2000), ("mixed", case["years"])):
    expected = np.asarray(case[mode])
    actual = model.predict_proba_second_lien(case["inputs"], year=years)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.predict(case["inputs"], year=years), np.where(expected >= .5, 2, 1))
print("Installed wheel verified: NumPy-only dependencies; scalar/mixed-year predictions match.")
"""


def main():
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="hmda-lightweight-") as temporary:
        work = Path(temporary)
        source = work / "source"
        source.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(repo / name, source / name)
        shutil.copytree(
            repo / "src",
            source / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(work / "dist"),
                str(source),
            ],
            check=True,
            cwd=work,
        )
        wheels = list((work / "dist").glob("*.whl"))
        assert len(wheels) == 1, wheels
        with zipfile.ZipFile(wheels[0]) as archive:
            for name in archive.namelist():
                path = Path(name)
                assert not {"data", "output"} & set(path.parts), name
                assert path.suffix not in {
                    ".pkl",
                    ".pickle",
                    ".parquet",
                    ".json",
                    ".csv",
                }, name
        environment = work / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        subprocess.run(
            [str(python), "-I", "-m", "pip", "install", str(wheels[0])],
            check=True,
            cwd=work,
        )
        subprocess.run([str(python), "-I", "-m", "pip", "check"], check=True, cwd=work)
        write_synthetic_case(work)
        subprocess.run([str(python), "-I", "-c", INSTALLED_CHECK], check=True, cwd=work)


if __name__ == "__main__":
    main()
