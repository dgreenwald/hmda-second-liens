"""Import an explicitly selected historical-results snapshot for distribution.

This release path accepts historical model-ID association without inventing missing
artifact digests. The ordinary portable exporter retains its stricter contract.
"""

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, mixture, portable_predict
from .density_ratio import artifacts
from .density_ratio.families.logistic import load_known_source_prior_model
from .portable_export import _atomic_write, portable_payload, serialize_payload


def _source(root, path, sources):
    path = path.resolve()
    relative = path.relative_to(root).as_posix()
    if "sync_backups" in path.relative_to(root).parts:
        raise ValueError("Backup results cannot supply the benchmark snapshot")
    content = path.read_bytes()
    sources[relative] = hashlib.sha256(content).hexdigest()
    return content


def _verify_predictions(model, payload):
    """Compare against the original fit using fabricated observations only."""
    knots = model.transformer.knots_["log_lti"]
    lti = [knots[0] - 10, *knots, knots[-1] + 10]
    frame = pd.DataFrame(
        [
            {
                "log_lti": x,
                "purchaser_type": purchaser,
                "loan_type": loan,
                "log_county_value_to_loan": county,
            }
            for x in lti
            for purchaser in range(10)
            for loan in range(1, 5)
            for county in (-3.0, 0.0, 3.0)
        ]
    )
    portable = portable_predict.PortableLogisticModel(payload)
    frame = frame[list(portable.required_columns)]
    ratio = model.log_ratio(frame)
    for year, row in payload["annual"].items():
        expected = mixture.adjusted_probability(ratio, row["mixture_share"])
        np.testing.assert_allclose(
            portable.predict_proba_second_lien(frame, year=int(year)),
            expected,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            portable.predict(frame, year=int(year)), np.where(expected >= 0.5, 2, 1)
        )


def snapshot_payloads(results_root: str | Path) -> dict[str, dict]:
    """Validate both current finalists and their complete annual snapshot."""
    if not str(results_root).strip():
        raise ValueError("An explicit results root is required")
    root = Path(results_root).expanduser().resolve()
    sources = {}
    annual_path = root / "tables/historical_model_family_annual.csv"
    annual = pd.read_csv(BytesIO(_source(root, annual_path, sources)))
    historical = {
        year: json.loads(
            _source(
                root, root / f"historical_model_family/shards/year_{year}.json", sources
            )
        )
        for year in config.APPLY_YEARS
    }
    payloads = {}
    for feature_set, specification, c in (
        ("core", portable_predict.SPECIFICATION, 0.1),
        ("hmda_only", portable_predict.HMDA_ONLY_SPECIFICATION, 1.0),
    ):
        model_sources = dict(sources)
        candidates = list(
            (root / "model_family_comparison/shards").glob(
                f"model_family_comparison_forward__logistic__{specification}__train_2004_2007__*.json"
            )
        )
        if len(candidates) != 1:
            raise ValueError(f"Expected one forward comparison shard for {feature_set}")
        comparison = json.loads(_source(root, candidates[0], model_sources))
        if comparison["fold"]["direction"] != "forward" or comparison["fold"][
            "train_years"
        ] != list(config.TRAIN_YEARS):
            raise ValueError("Benchmark comparison must use the final forward fit")
        if len(comparison["models"]) != 1:
            raise ValueError("Expected exactly one comparison model")
        record = comparison["models"][0]
        # Resolve only the canonical suffix; never read a cluster absolute path.
        filename = f"known_source_prior__{specification}__c_{str(c).rstrip('0').rstrip('.').replace('.', 'p')}__train_2004_2007.pkl"
        suffix = (
            Path("model_family_comparison/models/logistic") / specification / filename
        )
        if (
            not Path(record["artifact_path"])
            .as_posix()
            .endswith("/" + suffix.as_posix())
        ):
            raise ValueError("Unexpected forward artifact path")
        path = root / suffix
        _source(root, path, model_sources)
        _source(root, artifacts.metadata_path(path), model_sources)
        model = load_known_source_prior_model(path)
        metadata = artifacts.load_metadata(path)
        if (
            model.specification.name != specification
            or model.regularization_c != c
            or tuple(model.train_years) != tuple(config.TRAIN_YEARS)
            or model.transformer.specification != model.specification
            or record["model_id"] != model.model_id
            or record["configuration"] != metadata.configuration.to_dict()
        ):
            raise ValueError("Benchmark identity does not match comparison metadata")
        predictor_set = "unrestricted" if feature_set == "core" else "hmda_only"
        selected = annual.loc[
            annual["predictor_set"].eq(predictor_set)
            & annual["model_family"].eq("logistic")
        ]
        if len(selected) != 27 or set(selected["year"]) != set(config.APPLY_YEARS):
            raise ValueError(
                "Benchmark annual table must contain each year 1990–2016 once"
            )
        rows = []
        for year in config.APPLY_YEARS:
            shard = historical[year]
            if shard["year"] != year:
                raise ValueError("Historical shard year mismatch")
            matches = [
                row
                for row in shard["annual"]
                if row["predictor_set"] == predictor_set
                and row["model_family"] == "logistic"
            ]
            if len(matches) != 1:
                raise ValueError("Expected one historical row per benchmark and year")
            row = matches[0]
            csv_row = selected.loc[selected["year"].eq(year)].iloc[0]
            if (
                row["year"] != year
                or row["model_id"] != model.model_id
                or csv_row["model_id"] != model.model_id
            ):
                raise ValueError("Historical model identity mismatch")
            for item in (row, csv_row):
                if not item["optimizer_converged"] == True:
                    raise ValueError("Historical mixture optimization did not converge")
                if not isinstance(item["mixture_at_boundary"], (bool, np.bool_)):
                    raise TypeError("Invalid historical boundary diagnostic")
                if (
                    not np.isfinite(item["mixture_share"])
                    or not 0 < item["mixture_share"] < 1
                ):
                    raise ValueError("Invalid historical mixture share")
                digest = item.get("model_sha256")
                if (
                    digest is not None
                    and not pd.isna(digest)
                    and digest != metadata.payload_sha256
                ):
                    raise ValueError("Historical artifact digest mismatch")
            if (
                not np.isclose(
                    row["mixture_share"], csv_row["mixture_share"], rtol=0, atol=1e-12
                )
                or row["n_model_sample"] != csv_row["n_model_sample"]
                or row["mixture_at_boundary"] != csv_row["mixture_at_boundary"]
            ):
                raise ValueError("Historical table and shard disagree")
            rows.append(row)
        payload = portable_payload(
            model, metadata, pd.DataFrame(rows).set_index("year")
        )
        payload["provenance"].update(
            {
                "snapshot_association": "model_id_and_comparison_configuration",
                "annual_artifact_binding": "not_asserted; source snapshot predates required annual artifact digests",
                "source_files_sha256": model_sources,
            }
        )
        _verify_predictions(model, payload)
        payloads[feature_set] = payload
    return payloads


def bundle_logistic_models(
    results_root: str | Path,
    *,
    staging_dir: str | Path = config.MODEL_DIR / "benchmark_release",
    resource_dir: str | Path = Path(__file__).parent / "benchmarks",
) -> list[Path]:
    """Validate both benchmarks before staging and updating package resources."""
    payloads = snapshot_payloads(results_root)
    encoded = {name: serialize_payload(payload) for name, payload in payloads.items()}
    destinations = []
    for directory in (Path(staging_dir), Path(resource_dir)):
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in encoded.items():
            path = directory / f"{name}.json"
            _atomic_write(path, content)
            destinations.append(path)
    return destinations
