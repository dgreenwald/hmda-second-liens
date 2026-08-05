"""Reverse-temporal logistic specification and ridge selection."""

from __future__ import annotations

import copy
import pickle
import time
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from . import clean, config
from .logistic_features import (
    CENSUS_REGION_BY_STATE,
    FeatureSpecification,
    LogisticFeatureTransformer,
    core_specifications,
)

SELECTION_COLUMNS = [
    "year",
    config.LABEL_VAR,
    *config.CONTINUOUS_VARS,
    *config.CATEGORY_VARS,
    "state_code",
]


@dataclass(frozen=True)
class ReverseFold:
    """Four later training years and every available earlier validation year."""

    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]

    @property
    def train_start(self) -> int:
        return self.train_years[0]

    @property
    def train_end(self) -> int:
        return self.train_years[-1]


@dataclass
class SelectedLogisticModel:
    """Serializable fitted transformer and logistic classifier."""

    transformer: LogisticFeatureTransformer
    classifier: LogisticRegression
    specification: FeatureSpecification
    regularization_c: float

    def predict_proba_second_lien(self, df: pd.DataFrame) -> np.ndarray:
        features = self.transformer.transform(df)
        column = list(self.classifier.classes_).index(config.SECOND_LIEN_CLASS)
        return self.classifier.predict_proba(features)[:, column]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probability = self.predict_proba_second_lien(df)
        return np.where(
            probability >= 0.5,
            config.SECOND_LIEN_CLASS,
            config.FIRST_LIEN_CLASS,
        )


def reverse_folds(
    first_labeled_year: int = 2004,
    last_labeled_year: int = 2016,
    training_window: int = 4,
) -> list[ReverseFold]:
    """Create the frozen triangular reverse-validation design."""
    first_training_year = first_labeled_year + 1
    last_training_year = last_labeled_year - training_window + 1
    return [
        ReverseFold(
            train_years=tuple(range(start, start + training_window)),
            validation_years=tuple(range(first_labeled_year, start)),
        )
        for start in range(first_training_year, last_training_year + 1)
    ]


def load_selection_years(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    years: Iterable[int] = range(2004, 2017),
) -> dict[int, pd.DataFrame]:
    """Load the narrow cleaned frames used by every candidate."""
    data_dir = Path(data_dir)
    return {
        year: pd.read_parquet(data_dir / f"hmda{year}.parquet", columns=SELECTION_COLUMNS)
        for year in years
    }


def prepare_selection_data(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    years: Iterable[int] = range(2004, 2017),
    yearly_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Clean and persist narrow labeled-year frames for resumable selection."""
    years = list(years)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    county_values = clean.build_county_value_panel(config.APPLY_YEARS)
    rows = []
    for year in years:
        frame = clean.load_and_clean_year(
            year, county_values, yearly_dir=yearly_dir
        )
        output = data_dir / f"hmda{year}.parquet"
        frame[SELECTION_COLUMNS].to_parquet(output, index=False)
        rows.append(
            {
                "year": year,
                "n": len(frame),
                "n_first": int(
                    (frame[config.LABEL_VAR] == config.FIRST_LIEN_CLASS).sum()
                ),
                "n_second": int(
                    (frame[config.LABEL_VAR] == config.SECOND_LIEN_CLASS).sum()
                ),
                "output": str(output),
            }
        )
    return pd.DataFrame(rows)


def evaluate_candidate_grid(
    data_by_year: dict[int, pd.DataFrame],
    candidate_c: dict[FeatureSpecification, Iterable[float]],
    folds: Iterable[ReverseFold] | None = None,
    checkpoint_file: str | Path | None = None,
    n_jobs: int = config.LOGISTIC_SELECTION_JOBS,
) -> pd.DataFrame:
    """Fit each candidate/C pair and return all reverse-fold Brier cells.

    When ``checkpoint_file`` is supplied, completed cells are appended after
    every fold/specification and reused on a resumed run.
    """
    if folds is None:
        folds = reverse_folds()
    folds = list(folds)
    checkpoint_file = Path(checkpoint_file) if checkpoint_file else None
    if checkpoint_file is not None and checkpoint_file.exists():
        completed = pd.read_csv(checkpoint_file)
    else:
        completed = pd.DataFrame()
    rows = []

    for fold in folds:
        training = pd.concat(
            [data_by_year[year] for year in fold.train_years],
            ignore_index=True,
        )
        labels = training[config.LABEL_VAR].to_numpy()
        tasks = []
        for specification, c_values in candidate_c.items():
            c_values = sorted(set(c_values))
            missing_c = [
                regularization_c
                for regularization_c in c_values
                if not _fold_candidate_complete(
                    completed,
                    specification.name,
                    regularization_c,
                    fold,
                )
            ]
            if missing_c:
                tasks.append((specification, missing_c))

        with threadpool_limits(
            limits=config.LOGISTIC_SELECTION_THREADS_PER_JOB,
            user_api="blas",
        ), ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _evaluate_fold_specification,
                    training,
                    labels,
                    data_by_year,
                    fold,
                    specification,
                    missing_c,
                    completed,
                )
                for specification, missing_c in tasks
            ]
            for future in as_completed(futures):
                fold_rows = future.result()
                rows.extend(fold_rows)
                if checkpoint_file is not None and fold_rows:
                    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(fold_rows).to_csv(
                        checkpoint_file,
                        mode="a",
                        header=not checkpoint_file.exists(),
                        index=False,
                    )

    new = pd.DataFrame(rows)
    if completed.empty:
        return new
    combined = pd.concat([completed, new], ignore_index=True)
    return combined.drop_duplicates(
        [
            "specification",
            "regularization_c",
            "train_start",
            "validation_year",
        ],
        keep="last",
    ).reset_index(drop=True)


def _evaluate_fold_specification(
    training: pd.DataFrame,
    labels: np.ndarray,
    data_by_year: dict[int, pd.DataFrame],
    fold: ReverseFold,
    specification: FeatureSpecification,
    c_values: Iterable[float],
    completed: pd.DataFrame,
) -> list[dict]:
    transformer = LogisticFeatureTransformer(specification)
    features = transformer.fit_transform(training)
    models, fit_diagnostics = fit_regularization_path(features, labels, c_values)
    rows = []
    for validation_year in fold.validation_years:
        validation = data_by_year[validation_year]
        validation_features = transformer.transform(validation)
        validation_second = (
            validation[config.LABEL_VAR].to_numpy()
            == config.SECOND_LIEN_CLASS
        )
        for regularization_c, model in models.items():
            if _cell_complete(
                completed,
                specification.name,
                regularization_c,
                fold.train_start,
                validation_year,
            ):
                continue
            start = time.perf_counter()
            probability = model.predict_proba(validation_features)[
                :, list(model.classes_).index(config.SECOND_LIEN_CLASS)
            ]
            prediction_seconds = time.perf_counter() - start
            diagnostics = fit_diagnostics[regularization_c]
            rows.append(
                {
                    "specification": specification.name,
                    "continuous_form": specification.continuous_form,
                    "interactions": specification.interactions,
                    "geography": specification.geography or "none",
                    "regularization_c": regularization_c,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "validation_year": validation_year,
                    "horizon": fold.train_start - validation_year,
                    "n_validation": len(validation),
                    "brier_score": np.mean(
                        (probability - validation_second) ** 2
                    ),
                    "observed_second_share": validation_second.mean(),
                    "mean_predicted_second_share": probability.mean(),
                    "fit_seconds": diagnostics["fit_seconds"],
                    "prediction_seconds": prediction_seconds,
                    "n_iter": diagnostics["n_iter"],
                    "converged": diagnostics["converged"],
                }
            )
    return rows


def fit_regularization_path(
    features: np.ndarray,
    labels: np.ndarray,
    c_values: Iterable[float],
) -> tuple[dict[float, LogisticRegression], dict[float, dict]]:
    """Fit an ascending ridge path with warm starts and convergence recovery."""
    models = {}
    diagnostics = {}
    classifier = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        max_iter=config.LOGISTIC_SELECTION_MAX_ITER,
        solver=config.LOGISTIC_SELECTION_SOLVER,
        tol=config.LOGISTIC_SELECTION_TOL,
        warm_start=True,
    )
    for regularization_c in sorted(set(c_values)):
        classifier.set_params(C=regularization_c)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(features, labels)
        converged = not any(
            issubclass(warning.category, ConvergenceWarning) for warning in caught
        )
        if not converged:
            classifier.set_params(max_iter=4 * config.LOGISTIC_SELECTION_MAX_ITER)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(features, labels)
            converged = not any(
                issubclass(warning.category, ConvergenceWarning)
                for warning in caught
            )
        fit_seconds = time.perf_counter() - start
        models[regularization_c] = copy.deepcopy(classifier)
        diagnostics[regularization_c] = {
            "fit_seconds": fit_seconds,
            "n_iter": int(classifier.n_iter_.max()),
            "converged": converged,
        }
    return models, diagnostics


def aggregate_brier_cells(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average equally within horizon and then equally across horizons."""
    group = [
        "specification",
        "continuous_form",
        "interactions",
        "geography",
        "regularization_c",
        "horizon",
    ]
    by_horizon = (
        cells.groupby(group, as_index=False)
        .agg(
            mean_brier=("brier_score", "mean"),
            n_cells=("brier_score", "size"),
            min_brier=("brier_score", "min"),
            max_brier=("brier_score", "max"),
        )
        .sort_values(group)
    )
    candidate_group = group[:-1]
    summary = (
        by_horizon.groupby(candidate_group, as_index=False)
        .agg(
            selection_brier=("mean_brier", "mean"),
            n_horizons=("horizon", "nunique"),
            n_cells=("n_cells", "sum"),
        )
        .sort_values("selection_brier")
        .reset_index(drop=True)
    )
    return by_horizon, summary


def refinement_values(best_coarse_c: float) -> tuple[float, float]:
    """Return the two predeclared adjacent decades around a coarse winner."""
    exponent = np.log10(best_coarse_c)
    if not np.isclose(exponent, round(exponent)):
        raise ValueError("Coarse C must be an integer power of ten")
    return 10.0 ** (exponent - 1), 10.0 ** (exponent + 1)


def refinement_grid(
    coarse_summary: pd.DataFrame,
    specifications: Iterable[FeatureSpecification],
) -> dict[FeatureSpecification, tuple[float, float]]:
    """Build each specification's refinement grid from coarse results."""
    best = (
        coarse_summary.sort_values("selection_brier")
        .groupby("specification", as_index=False)
        .first()
        .set_index("specification")["regularization_c"]
    )
    return {
        specification: refinement_values(float(best.loc[specification.name]))
        for specification in specifications
    }


def fit_selected_model(
    training: pd.DataFrame,
    specification: FeatureSpecification,
    regularization_c: float,
) -> SelectedLogisticModel:
    """Refit a chosen specification once on the final 2004--2007 sample."""
    transformer = LogisticFeatureTransformer(specification)
    features = transformer.fit_transform(training)
    models, diagnostics = fit_regularization_path(
        features, training[config.LABEL_VAR].to_numpy(), [regularization_c]
    )
    if not diagnostics[regularization_c]["converged"]:
        raise RuntimeError("Selected logistic model did not converge")
    return SelectedLogisticModel(
        transformer=transformer,
        classifier=models[regularization_c],
        specification=specification,
        regularization_c=regularization_c,
    )


def save_selected_model(
    model: SelectedLogisticModel,
    output_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
) -> None:
    """Serialize a selected transformer/classifier bundle."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as file:
        pickle.dump(model, file)


def load_selected_model(
    input_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
) -> SelectedLogisticModel:
    """Load a trusted bundle written by :func:`save_selected_model`."""
    with Path(input_file).open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, SelectedLogisticModel):
        raise TypeError(f"Expected SelectedLogisticModel, got {type(model).__name__}")
    return model


def run_model_selection(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    model_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    include_geography: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the resumable coarse/refined core search and guarded challengers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_by_year = load_selection_years(data_dir=data_dir)
    specifications = core_specifications()

    coarse_file = output_dir / "logistic_selection_core_coarse_cells.csv"
    coarse_cells = evaluate_candidate_grid(
        data_by_year,
        {
            specification: config.LOGISTIC_SELECTION_COARSE_C
            for specification in specifications
        },
        checkpoint_file=coarse_file,
    )
    coarse_horizons, coarse_summary = aggregate_brier_cells(coarse_cells)
    coarse_horizons.to_csv(
        output_dir / "logistic_selection_core_coarse_horizons.csv", index=False
    )
    coarse_summary.to_csv(
        output_dir / "logistic_selection_core_coarse_summary.csv", index=False
    )

    refine_candidates = refinement_grid(coarse_summary, specifications)
    refinement_file = output_dir / "logistic_selection_core_refinement_cells.csv"
    refinement_cells = evaluate_candidate_grid(
        data_by_year,
        refine_candidates,
        checkpoint_file=refinement_file,
    )
    core_cells = pd.concat([coarse_cells, refinement_cells], ignore_index=True)
    core_cells = core_cells.drop_duplicates(
        [
            "specification",
            "regularization_c",
            "train_start",
            "validation_year",
        ]
    )
    core_horizons, core_summary = aggregate_brier_cells(core_cells)
    core_cells.to_csv(output_dir / "logistic_selection_core_cells.csv", index=False)
    core_horizons.to_csv(
        output_dir / "logistic_selection_core_horizons.csv", index=False
    )
    core_summary.to_csv(
        output_dir / "logistic_selection_core_summary.csv", index=False
    )

    selected_row = core_summary.iloc[0]
    selected_specification = FeatureSpecification(
        continuous_form=selected_row["continuous_form"],
        interactions=selected_row["interactions"],
    )
    selected_c = float(selected_row["regularization_c"])
    final_training = pd.concat(
        [data_by_year[year] for year in config.TRAIN_YEARS],
        ignore_index=True,
    )
    selected_model = fit_selected_model(
        final_training, selected_specification, selected_c
    )
    save_selected_model(selected_model, model_file)
    decision = pd.DataFrame(
        [
            {
                **selected_row.to_dict(),
                "model_file": str(model_file),
                "training_years": "2004-2007",
                "selection_metric": "raw_brier_equal_horizon_weight",
            }
        ]
    )
    decision.to_csv(
        output_dir / "logistic_selection_decision.csv", index=False
    )

    results = {
        "core_cells": core_cells,
        "core_horizons": core_horizons,
        "core_summary": core_summary,
        "decision": decision,
    }
    if include_geography:
        results.update(
            run_geographic_challengers(
                data_by_year,
                selected_specification,
                selected_c,
                core_cells,
                output_dir,
            )
        )
    return results


def run_geographic_challengers(
    data_by_year: dict[int, pd.DataFrame],
    selected_specification: FeatureSpecification,
    selected_c: float,
    core_cells: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Tune state/region challengers without automatically selecting them."""
    output_dir = Path(output_dir)
    specifications = [
        FeatureSpecification(
            selected_specification.continuous_form,
            selected_specification.interactions,
            geography=geography,
        )
        for geography in ("region", "state")
    ]
    coarse_cells = evaluate_candidate_grid(
        data_by_year,
        {
            specification: config.LOGISTIC_SELECTION_COARSE_C
            for specification in specifications
        },
        checkpoint_file=output_dir
        / "logistic_selection_geography_coarse_cells.csv",
    )
    _, coarse_summary = aggregate_brier_cells(coarse_cells)
    refinement_cells = evaluate_candidate_grid(
        data_by_year,
        refinement_grid(coarse_summary, specifications),
        checkpoint_file=output_dir
        / "logistic_selection_geography_refinement_cells.csv",
    )
    cells = pd.concat([coarse_cells, refinement_cells], ignore_index=True)
    cells = cells.drop_duplicates(
        [
            "specification",
            "regularization_c",
            "train_start",
            "validation_year",
        ]
    )
    horizons, summary = aggregate_brier_cells(cells)
    comparison = geographic_incremental_brier(
        core_cells, selected_specification.name, selected_c, cells, summary
    )
    support = geographic_support(data_by_year)
    best_candidates = {
        FeatureSpecification(
            row.continuous_form,
            row.interactions,
            geography=row.geography,
        ): float(row.regularization_c)
        for row in summary.sort_values("selection_brier")
        .groupby("geography", as_index=False)
        .first()
        .itertuples()
    }
    coefficients = coefficient_stability(data_by_year, best_candidates)

    outputs = {
        "geography_cells": cells,
        "geography_horizons": horizons,
        "geography_summary": summary,
        "geography_incremental_brier": comparison,
        "geography_support": support,
        "geography_coefficients": coefficients,
    }
    for name, frame in outputs.items():
        frame.to_csv(
            output_dir / f"logistic_selection_{name}.csv", index=False
        )
    return outputs


def run_spline_purchaser_challenger(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_dir: str | Path = config.TABLE_DIR,
) -> dict[str, pd.DataFrame]:
    """Evaluate purchaser-specific LTI spline shapes against the selected core."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = pd.read_csv(output_dir / "logistic_selection_decision.csv").iloc[0]
    core_cells = pd.read_csv(output_dir / "logistic_selection_core_cells.csv")
    core_specification = str(decision["specification"])
    core_c = float(decision["regularization_c"])
    challenger = FeatureSpecification(
        continuous_form="spline_lti",
        interactions="purchaser_type_spline_lti",
    )
    data_by_year = load_selection_years(data_dir=data_dir)

    prefix = "logistic_selection_spline_purchaser"
    coarse_cells = evaluate_candidate_grid(
        data_by_year,
        {challenger: config.LOGISTIC_SELECTION_COARSE_C},
        checkpoint_file=output_dir / f"{prefix}_coarse_cells.csv",
    )
    coarse_horizons, coarse_summary = aggregate_brier_cells(coarse_cells)
    coarse_horizons.to_csv(
        output_dir / f"{prefix}_coarse_horizons.csv", index=False
    )
    coarse_summary.to_csv(
        output_dir / f"{prefix}_coarse_summary.csv", index=False
    )

    refinement_cells = evaluate_candidate_grid(
        data_by_year,
        refinement_grid(coarse_summary, [challenger]),
        checkpoint_file=output_dir / f"{prefix}_refinement_cells.csv",
    )
    cells = pd.concat([coarse_cells, refinement_cells], ignore_index=True)
    cells = cells.drop_duplicates(
        [
            "specification",
            "regularization_c",
            "train_start",
            "validation_year",
        ]
    )
    horizons, summary = aggregate_brier_cells(cells)
    best = summary.iloc[0]
    core = core_cells.loc[
        (core_cells["specification"] == core_specification)
        & np.isclose(core_cells["regularization_c"], core_c),
        ["train_start", "validation_year", "horizon", "brier_score"],
    ].rename(columns={"brier_score": "core_brier"})
    challenger_cells = cells.loc[
        (cells["specification"] == best["specification"])
        & np.isclose(cells["regularization_c"], best["regularization_c"]),
        ["train_start", "validation_year", "brier_score"],
    ].rename(columns={"brier_score": "challenger_brier"})
    comparison = core.merge(
        challenger_cells,
        on=["train_start", "validation_year"],
        validate="one_to_one",
    )
    comparison["brier_difference_from_core"] = (
        comparison["challenger_brier"] - comparison["core_brier"]
    )
    comparison["challenger_specification"] = best["specification"]
    comparison["challenger_c"] = best["regularization_c"]
    decision_summary = pd.DataFrame(
        [
            {
                "core_specification": core_specification,
                "core_c": core_c,
                "core_selection_brier": float(decision["selection_brier"]),
                "challenger_specification": best["specification"],
                "challenger_c": float(best["regularization_c"]),
                "challenger_selection_brier": float(best["selection_brier"]),
                "brier_difference_from_core": (
                    float(best["selection_brier"])
                    - float(decision["selection_brier"])
                ),
                "n_cells_challenger_improves": int(
                    (comparison["brier_difference_from_core"] < 0).sum()
                ),
                "n_cells": len(comparison),
            }
        ]
    )
    outputs = {
        "cells": cells,
        "horizons": horizons,
        "summary": summary,
        "comparison_cells": comparison,
        "comparison": decision_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{prefix}_{name}.csv", index=False)
    return outputs


def geographic_incremental_brier(
    core_cells: pd.DataFrame,
    core_specification: str,
    core_c: float,
    geography_cells: pd.DataFrame,
    geography_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each best geographic challenger cell with the selected core."""
    core = core_cells.loc[
        (core_cells["specification"] == core_specification)
        & np.isclose(core_cells["regularization_c"], core_c),
        ["train_start", "validation_year", "horizon", "brier_score"],
    ].rename(columns={"brier_score": "core_brier"})
    best = (
        geography_summary.sort_values("selection_brier")
        .groupby("geography", as_index=False)
        .first()
    )
    parts = []
    for row in best.itertuples():
        challenger = geography_cells.loc[
            (geography_cells["specification"] == row.specification)
            & np.isclose(
                geography_cells["regularization_c"], row.regularization_c
            ),
            ["train_start", "validation_year", "brier_score"],
        ].rename(columns={"brier_score": "geography_brier"})
        comparison = core.merge(
            challenger,
            on=["train_start", "validation_year"],
            validate="one_to_one",
        )
        comparison["geography"] = row.geography
        comparison["specification"] = row.specification
        comparison["regularization_c"] = row.regularization_c
        comparison["brier_difference_from_core"] = (
            comparison["geography_brier"] - comparison["core_brier"]
        )
        parts.append(comparison)
    return pd.concat(parts, ignore_index=True)


def geographic_support(
    data_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Report state and region loan counts in every labeled year."""
    rows = []
    for year, frame in data_by_year.items():
        state = pd.to_numeric(frame["state_code"], errors="coerce")
        region = state.map(CENSUS_REGION_BY_STATE)
        for geography, values in (("state", state), ("region", region)):
            counts = values.value_counts(dropna=False)
            rows.extend(
                {
                    "year": year,
                    "geography": geography,
                    "level": level,
                    "n": int(count),
                    "share": count / len(frame),
                }
                for level, count in counts.items()
            )
    return pd.DataFrame(rows)


def coefficient_stability(
    data_by_year: dict[int, pd.DataFrame],
    candidates: dict[FeatureSpecification, float],
) -> pd.DataFrame:
    """Refit best geographic challengers and record coefficients by window."""
    rows = []
    for fold in reverse_folds():
        training = pd.concat(
            [data_by_year[year] for year in fold.train_years],
            ignore_index=True,
        )
        for specification, regularization_c in candidates.items():
            selected = fit_selected_model(
                training, specification, regularization_c
            )
            rows.append(
                {
                    "specification": specification.name,
                    "geography": specification.geography,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "feature": "intercept",
                    "coefficient": float(selected.classifier.intercept_[0]),
                }
            )
            rows.extend(
                {
                    "specification": specification.name,
                    "geography": specification.geography,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "feature": feature,
                    "coefficient": float(coefficient),
                }
                for feature, coefficient in zip(
                    selected.transformer.feature_names_,
                    selected.classifier.coef_[0],
                    strict=True,
                )
            )
    return pd.DataFrame(rows)


def _fold_candidate_complete(
    completed: pd.DataFrame,
    specification: str,
    regularization_c: float,
    fold: ReverseFold,
) -> bool:
    return all(
        _cell_complete(
            completed,
            specification,
            regularization_c,
            fold.train_start,
            validation_year,
        )
        for validation_year in fold.validation_years
    )


def _cell_complete(
    completed: pd.DataFrame,
    specification: str,
    regularization_c: float,
    train_start: int,
    validation_year: int,
) -> bool:
    if completed.empty:
        return False
    match = (
        (completed["specification"] == specification)
        & np.isclose(completed["regularization_c"], regularization_c)
        & (completed["train_start"] == train_start)
        & (completed["validation_year"] == validation_year)
    )
    return bool(match.any())
