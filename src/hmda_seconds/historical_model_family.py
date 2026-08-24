"""Step 10 historical application of the four frozen model finalists."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import clean, config, plausibility
from .density_ratio import artifacts, evaluation
from .density_ratio.cluster import expand_job_paths
from .density_ratio.families.gradient_boosting import load_boosting_model
from .density_ratio.families.logistic import load_known_source_prior_model
from .density_ratio.shards import read_manifest, read_shard, shard_path

SCHEMA_VERSION = 1
MODEL_KEYS = (
    ("unrestricted", "logistic"),
    ("unrestricted", "boosting"),
    ("hmda_only", "logistic"),
    ("hmda_only", "boosting"),
)
QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)
QUANTILE_NAMES = ("q01", "q05", "q50", "q95", "q99")


def write_historical_manifest(
    path: str | Path,
    *,
    comparison_manifest: str | Path,
    years=tuple(config.APPLY_YEARS),
    selection_data_dir: str | Path = config.SELECTION_DATA_DIR,
    hmda_data_dir: str | Path = config.HMDA_DATA_DIR,
    output_root: str | Path = config.OUTPUT_DIR / "historical_model_family",
) -> Path:
    """Write the canonical immutable-year application manifest."""
    requested = tuple(int(year) for year in years)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Historical years must be nonempty and unique")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "comparison_manifest": str(Path(comparison_manifest).resolve()),
        "years": list(requested),
        "selection_data_dir": str(Path(selection_data_dir).resolve()),
        "hmda_data_dir": str(Path(hmda_data_dir).resolve()),
        "output_root": str(Path(output_root).resolve()),
    }
    return _atomic_json(Path(path), payload, replace=True)


def read_historical_manifest(path: str | Path) -> dict:
    """Read and validate a historical application manifest."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported historical manifest")
    years = payload.get("years")
    if not isinstance(years, list) or not years or len(set(years)) != len(years):
        raise ValueError("Historical manifest years must be nonempty and unique")
    for key in (
        "comparison_manifest",
        "selection_data_dir",
        "hmda_data_dir",
        "output_root",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"Historical manifest is missing {key}")
    return payload


def historical_shard_path(output_root: str | Path, year: int) -> Path:
    return Path(output_root) / "shards" / f"year_{int(year)}.json"


def load_finalists(comparison_manifest: str | Path) -> dict[tuple[str, str], object]:
    """Load the four persisted 2004--2007 comparison fits."""
    planned = [expand_job_paths(item) for item in read_manifest(comparison_manifest)]
    forward = [item for item in planned if item.fold.direction == "forward"]
    if len(forward) != 4:
        raise ValueError("Comparison manifest must contain four forward finalists")
    models = {}
    for item in forward:
        shard = read_shard(shard_path(item.job))
        if len(shard.models) != 1:
            raise ValueError("Each finalist shard must contain exactly one model")
        record = shard.models[0]
        metadata = artifacts.validate_existing_artifact(
            record.artifact_path, allow_legacy=False
        )
        if metadata is None or metadata.configuration != record.configuration:
            raise ValueError("Finalist artifact identity does not match its shard")
        predictor_set = (
            "hmda_only"
            if record.configuration.specification.startswith("hmda_only")
            else "unrestricted"
        )
        if record.configuration.family == "logistic":
            family = "logistic"
        elif record.configuration.family == "hist_gradient_boosting":
            family = "boosting"
        else:
            raise ValueError(
                f"Unsupported historical family {record.configuration.family}"
            )
        key = (predictor_set, family)
        if key in models:
            raise ValueError(f"Duplicate historical finalist {key}")
        loader = (
            load_known_source_prior_model
            if family == "logistic"
            else load_boosting_model
        )
        models[key] = loader(record.artifact_path)
    if set(models) != set(MODEL_KEYS):
        raise ValueError("Historical finalists are incomplete")
    return models


def evaluate_year(
    frame: pd.DataFrame,
    models: Mapping[tuple[str, str], object],
) -> dict[str, object]:
    """Score one transient annual frame and return aggregate-only records."""
    if frame.empty or frame["year"].nunique() != 1:
        raise ValueError("Historical frame must contain one nonempty year")
    year = int(frame["year"].iloc[0])
    reference = models[("unrestricted", "logistic")]
    annual = []
    for predictor_set, family in MODEL_KEYS:
        model = models[(predictor_set, family)]
        log_ratio = np.asarray(model.log_ratio(frame), dtype=float)
        adjusted = evaluation.adjust_log_ratio(log_ratio)
        probability = adjusted.probability
        estimate = adjusted.mixture_estimate
        row = {
            "year": year,
            "predictor_set": predictor_set,
            "model_family": family,
            "model_id": model.model_id,
            "n_model_sample": len(frame),
            "actual_second_share": _actual_share(frame),
            "mixture_share": estimate.share,
            "mean_probability": float(probability.mean()),
            "hard_share_050": float((probability >= 0.5).mean()),
            "optimizer_converged": estimate.optimizer_converged,
            "optimizer_iterations": estimate.optimizer_iterations,
            "mixture_at_boundary": estimate.at_boundary,
            "mean_log_likelihood": estimate.mean_log_likelihood,
            "em_converged": estimate.em_converged,
            "em_iterations": estimate.em_iterations,
            "mixture_em_difference": estimate.share - estimate.em_share,
            "probability_below_001": float((probability < 0.01).mean()),
            "probability_above_099": float((probability > 0.99).mean()),
        }
        row.update(_quantile_columns(log_ratio, "log_ratio"))
        row.update(_quantile_columns(probability, "probability"))
        actual = row["actual_second_share"]
        row["share_error"] = None if actual is None else estimate.share - actual
        row["absolute_share_error"] = (
            None if actual is None else abs(estimate.share - actual)
        )
        annual.append(row)
    continuous = continuous_support(frame, reference)
    categorical = categorical_support(frame)
    return {
        "schema_version": SCHEMA_VERSION,
        "year": year,
        "annual": annual,
        "continuous_support": continuous,
        "categorical_support": categorical,
    }


def continuous_support(frame: pd.DataFrame, reference_model) -> list[dict]:
    """Summarize primitive continuous variables on the frozen source scale."""
    transformer = reference_model.transformer
    rows = []
    for variable in config.CONTINUOUS_VARS:
        values = frame[variable].to_numpy(dtype=float)
        standardized = (
            values - transformer.raw_location_[variable]
        ) / transformer.raw_scale_[variable]
        row = {
            "year": int(frame["year"].iloc[0]),
            "variable": variable,
            "n": len(values),
            "standardized_mean": float(standardized.mean()),
            "standardized_sd": float(standardized.std(ddof=0)),
            "fraction_abs_z_gt_2": float((np.abs(standardized) > 2).mean()),
            "fraction_abs_z_gt_3": float((np.abs(standardized) > 3).mean()),
        }
        row.update(_quantile_columns(standardized, "standardized"))
        if variable in transformer.knots_:
            knots = transformer.knots_[variable]
            row["fraction_below_outer_knot"] = float((values < knots[0]).mean())
            row["fraction_above_outer_knot"] = float((values > knots[-1]).mean())
        else:
            row["fraction_below_outer_knot"] = None
            row["fraction_above_outer_knot"] = None
        rows.append(row)
    return rows


def categorical_support(frame: pd.DataFrame) -> list[dict]:
    """Report annual counts and shares for every canonical category level."""
    rows = []
    year = int(frame["year"].iloc[0])
    for variable in config.CATEGORY_VARS:
        values = frame[variable]
        for level in config.CATEGORY_LEVELS[variable]:
            count = int((values == level).sum())
            rows.append(
                {
                    "year": year,
                    "variable": variable,
                    "level": int(level),
                    "count": count,
                    "share": count / len(frame),
                }
            )
    return rows


def run_manifest_year(manifest: str | Path, job_index: int) -> Path:
    """Execute one manifest year, reusing an identical immutable shard."""
    payload = read_historical_manifest(manifest)
    years = payload["years"]
    if job_index < 0 or job_index >= len(years):
        raise IndexError("Historical job index is out of range")
    year = int(years[job_index])
    destination = historical_shard_path(payload["output_root"], year)
    models = load_finalists(payload["comparison_manifest"])
    if destination.exists():
        existing = json.loads(destination.read_text())
        _validate_year_shard(existing, year)
        expected_ids = {model.model_id for model in models.values()}
        observed_ids = {row["model_id"] for row in existing["annual"]}
        if observed_ids != expected_ids:
            raise FileExistsError(f"Conflicting historical shard {destination}")
        return destination
    selection_file = Path(payload["selection_data_dir"]) / f"hmda{year}.parquet"
    if selection_file.exists():
        frame = pd.read_parquet(selection_file)
    else:
        county_values = clean.build_county_value_panel(config.APPLY_YEARS)
        frame = plausibility.load_historical_application_year(
            year, Path(payload["hmda_data_dir"]), county_values
        )
    result = evaluate_year(frame, models)
    return _atomic_json(destination, result, replace=False)


def aggregate_historical(
    manifest: str | Path,
    *,
    output_dir: str | Path = config.TABLE_DIR,
    figure_dir: str | Path = config.FIGURE_DIR,
    result_root: str | Path | None = None,
) -> list[Path]:
    """Validate every year shard and publish deterministic aggregate outputs."""
    payload = read_historical_manifest(manifest)
    years = tuple(int(year) for year in payload["years"])
    if years != tuple(config.APPLY_YEARS):
        raise ValueError("Historical aggregation requires the frozen 1990--2016 years")
    resolved_result_root = (
        Path(payload["output_root"]) if result_root is None else Path(result_root)
    )
    shards = []
    for year in years:
        path = historical_shard_path(resolved_result_root, year)
        shard = json.loads(path.read_text())
        _validate_year_shard(shard, year)
        shards.append(shard)
    annual = pd.DataFrame(row for shard in shards for row in shard["annual"])
    continuous = pd.DataFrame(
        row for shard in shards for row in shard["continuous_support"]
    )
    categorical = pd.DataFrame(
        row for shard in shards for row in shard["categorical_support"]
    )
    paired = paired_family_results(annual)
    boundary = boundary_summary(annual)
    envelope = support_envelope(continuous)
    outputs = {
        "annual": annual.sort_values(
            ["year", "predictor_set", "model_family"]
        ).reset_index(drop=True),
        "paired": paired,
        "boundary": boundary,
        "continuous_support": continuous.sort_values(
            ["year", "variable"]
        ).reset_index(drop=True),
        "categorical_support": categorical.sort_values(
            ["year", "variable", "level"]
        ).reset_index(drop=True),
        "support_envelope": envelope,
    }
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    destinations = []
    for name, frame in outputs.items():
        destination = output_dir / f"historical_model_family_{name}.csv"
        _atomic_csv(frame, destination)
        destinations.append(destination)
    series = figure_dir / "historical_model_family_shares.pdf"
    differences = figure_dir / "historical_model_family_differences.pdf"
    render_shares(annual, series)
    render_differences(paired, differences)
    return [*destinations, series, differences]


def paired_family_results(annual: pd.DataFrame) -> pd.DataFrame:
    """Construct matched boosting-minus-logistic annual differences."""
    values = (
        annual.pivot(
            index=["year", "predictor_set"],
            columns="model_family",
            values=["mixture_share", "hard_share_050", "absolute_share_error"],
        )
        .sort_index()
    )
    if values[["mixture_share", "hard_share_050"]].isna().any().any():
        raise ValueError("Historical family cells are not one-to-one")
    values.columns = [f"{metric}_{family}" for metric, family in values.columns]
    result = values.reset_index()
    for metric in ("mixture_share", "hard_share_050", "absolute_share_error"):
        result[f"boosting_minus_logistic_{metric}"] = (
            result[f"{metric}_boosting"] - result[f"{metric}_logistic"]
        )
    return result


def boundary_summary(annual: pd.DataFrame) -> pd.DataFrame:
    """Report each finalist's 2003--2004 continuity and labeled error."""
    selected = annual.loc[annual["year"].isin((2003, 2004))]
    wide = selected.pivot(
        index=["predictor_set", "model_family"],
        columns="year",
        values="mixture_share",
    )
    if set(wide.columns) != {2003, 2004}:
        raise ValueError("Historical results must contain 2003 and 2004")
    result = wide.rename(
        columns={2003: "mixture_share_2003", 2004: "mixture_share_2004"}
    ).reset_index()
    result["mixture_share_2004_minus_2003"] = (
        result["mixture_share_2004"] - result["mixture_share_2003"]
    )
    actual = annual.loc[annual["year"] == 2004, "actual_second_share"].dropna()
    if actual.nunique() != 1:
        raise ValueError("Finalists do not share one observed 2004 share")
    result["actual_second_share_2004"] = float(actual.iloc[0])
    result["mixture_share_2004_minus_actual"] = (
        result["mixture_share_2004"] - result["actual_second_share_2004"]
    )
    return result


def support_envelope(continuous: pd.DataFrame) -> pd.DataFrame:
    """Compare pre-2004 continuous summaries with the 2004--2007 envelope."""
    metrics = [
        column
        for column in continuous.columns
        if column not in {"year", "variable", "n"}
    ]
    source = continuous.loc[continuous["year"].between(2004, 2007)]
    rows = []
    for _, row in continuous.loc[continuous["year"] < 2004].iterrows():
        reference = source.loc[source["variable"] == row["variable"]]
        for metric in metrics:
            value = row[metric]
            if pd.isna(value):
                continue
            lower = reference[metric].min()
            upper = reference[metric].max()
            position = "below" if value < lower else "above" if value > upper else "within"
            rows.append(
                {
                    "year": int(row["year"]),
                    "variable": row["variable"],
                    "metric": metric,
                    "value": value,
                    "source_annual_min": lower,
                    "source_annual_max": upper,
                    "envelope_position": position,
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "variable", "metric"]).reset_index(drop=True)


def render_shares(annual: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for axis, predictor_set in zip(axes, ("unrestricted", "hmda_only"), strict=True):
        selected = annual.loc[annual["predictor_set"] == predictor_set]
        for family, color in (("logistic", "C0"), ("boosting", "C1")):
            values = selected.loc[selected["model_family"] == family]
            axis.plot(values["year"], values["mixture_share"], color=color, label=family.title())
        actual = selected.loc[selected["model_family"] == "logistic"]
        labeled = actual["actual_second_share"].notna()
        axis.plot(actual.loc[labeled, "year"], actual.loc[labeled, "actual_second_share"], color="black", marker="o", markersize=3, label="Actual")
        axis.axvline(2003.5, color="0.4", linestyle="--", linewidth=0.8)
        axis.set_ylabel("Second-lien share")
        axis.set_title(predictor_set.replace("_", " ").title())
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Year")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)


def render_differences(paired: pd.DataFrame, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for predictor_set, color in (("unrestricted", "C0"), ("hmda_only", "C1")):
        values = paired.loc[paired["predictor_set"] == predictor_set]
        axis.plot(values["year"], values["boosting_minus_logistic_mixture_share"], color=color, marker="o", markersize=3, label=predictor_set.replace("_", " ").title())
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(2003.5, color="0.4", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Year")
    axis.set_ylabel("Boosting minus logistic mixture share")
    axis.legend(frameon=False)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)


def _validate_year_shard(shard: dict, year: int) -> None:
    if shard.get("schema_version") != SCHEMA_VERSION or shard.get("year") != year:
        raise ValueError(f"Historical shard has wrong identity for {year}")
    annual = pd.DataFrame(shard.get("annual", []))
    if len(annual) != 4 or set(zip(annual["predictor_set"], annual["model_family"])) != set(MODEL_KEYS):
        raise ValueError(f"Historical shard {year} does not contain four finalists")
    if annual["n_model_sample"].nunique() != 1:
        raise ValueError(f"Historical finalists use different samples in {year}")
    if annual["model_id"].nunique() != 4:
        raise ValueError(f"Historical shard {year} reuses a model identity")
    observed = annual["actual_second_share"].dropna()
    if not observed.empty and observed.nunique() != 1:
        raise ValueError(f"Historical finalists have different labels in {year}")
    if (
        not annual["optimizer_converged"].all()
        or not annual["em_converged"].all()
        or annual["mixture_at_boundary"].any()
    ):
        raise ValueError(f"Historical mixture optimization failed in {year}")
    if not np.allclose(annual["mean_probability"], annual["mixture_share"], atol=1e-7):
        raise ValueError(f"Historical probability means do not match shares in {year}")
    if (annual["mixture_em_difference"].abs() > 1e-7).any():
        raise ValueError(f"Historical optimizer and EM estimates differ in {year}")
    continuous = pd.DataFrame(shard.get("continuous_support", []))
    if len(continuous) != len(config.CONTINUOUS_VARS):
        raise ValueError(f"Historical continuous support is incomplete in {year}")
    if set(continuous["variable"]) != set(config.CONTINUOUS_VARS):
        raise ValueError(f"Historical continuous variables are wrong in {year}")
    if set(continuous["n"]) != {int(annual["n_model_sample"].iloc[0])}:
        raise ValueError(f"Historical support sample differs in {year}")
    expected_categories = sum(len(levels) for levels in config.CATEGORY_LEVELS.values())
    categorical = pd.DataFrame(shard.get("categorical_support", []))
    if len(categorical) != expected_categories:
        raise ValueError(f"Historical categorical support is incomplete in {year}")
    for variable, levels in config.CATEGORY_LEVELS.items():
        selected = categorical.loc[categorical["variable"] == variable]
        if set(selected["level"]) != set(levels):
            raise ValueError(f"Historical {variable} levels are wrong in {year}")
        if selected["count"].sum() != int(annual["n_model_sample"].iloc[0]):
            raise ValueError(f"Historical {variable} counts are incomplete in {year}")
        if not np.isclose(selected["share"].sum(), 1.0):
            raise ValueError(f"Historical {variable} shares are incomplete in {year}")


def _actual_share(frame: pd.DataFrame) -> float | None:
    if config.LABEL_VAR not in frame or not frame[config.LABEL_VAR].notna().any():
        return None
    return float((frame[config.LABEL_VAR].to_numpy() == config.SECOND_LIEN_CLASS).mean())


def _quantile_columns(values: np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{prefix} values must be finite and nonempty")
    return {
        f"{prefix}_{name}": float(value)
        for name, value in zip(QUANTILE_NAMES, np.quantile(array, QUANTILES), strict=True)
    }


def _atomic_json(path: Path, payload: dict, *, replace: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(values)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != values:
                    raise FileExistsError(f"Conflicting historical shard {path}")
        return path
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
