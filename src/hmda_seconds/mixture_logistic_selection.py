"""Mixture-adjusted reselection of logistic features and ridge strength."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration, config, mixture, model_selection
from .density_ratio import adapters, artifacts, evaluation
from .density_ratio import folds as temporal_folds
from .logistic_features import (
    FeatureSpecification,
    LogisticFeatureTransformer,
    core_specifications,
)

CHALLENGER = FeatureSpecification(
    "spline_lti", "purchaser_type_spline_lti"
)


def candidate_specifications() -> list[FeatureSpecification]:
    """Return the original core grid plus the focused spline challenger."""
    return [*core_specifications(), CHALLENGER]


def fit_candidate_path(
    training: pd.DataFrame,
    specification: FeatureSpecification,
    c_values: Iterable[float],
) -> tuple[dict[float, mixture.KnownSourcePriorModel], dict[float, dict]]:
    """Fit and retain every equal-prior ridge value on one transformed fold."""
    transformer = LogisticFeatureTransformer(specification)
    features = transformer.fit_transform(training)
    labels = training[config.LABEL_VAR].to_numpy()
    y_second = labels == config.SECOND_LIEN_CLASS
    weights = mixture.equal_source_prior_weights(training, y_second)
    counts = artifacts.training_counts(
        training,
        label_var=config.LABEL_VAR,
        first_lien_class=config.FIRST_LIEN_CLASS,
        second_lien_class=config.SECOND_LIEN_CLASS,
    )
    classifiers, diagnostics = model_selection.fit_regularization_path(
        features, labels, c_values, sample_weight=weights
    )
    train_years = tuple(sorted(pd.unique(training["year"])))
    models = {
        regularization_c: mixture.KnownSourcePriorModel(
            transformer=transformer,
            ratio=mixture.classifier_ratio_variant(
                "known_source_prior",
                features,
                y_second,
                classifier,
            ),
            fit_diagnostics=diagnostics[regularization_c],
            specification=specification,
            regularization_c=regularization_c,
            train_years=train_years,
            n_training=counts[0],
            n_first_lien=counts[1],
            n_second_lien=counts[2],
        )
        for regularization_c, classifier in classifiers.items()
    }
    return models, diagnostics


def evaluate_grid(
    data_by_year: dict[int, pd.DataFrame],
    folds: Iterable[temporal_folds.TemporalFold],
    candidate_c: dict[FeatureSpecification, Iterable[float]],
    cells: pd.DataFrame,
    checkpoint_file: str | Path,
    model_dir: str | Path,
) -> pd.DataFrame:
    """Evaluate missing candidate cells while saving every fitted model."""
    checkpoint_file = Path(checkpoint_file)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        training = None
        for specification, values in candidate_c.items():
            c_values = sorted(set(values))
            missing_c = [
                value
                for value in c_values
                if not _candidate_complete(
                    cells, specification.name, value, fold
                )
            ]
            if not missing_c:
                continue
            models = {}
            diagnostics = {}
            need_fit = []
            for regularization_c in missing_c:
                path = mixture.known_source_prior_model_path(
                    fold.train_years,
                    specification,
                    regularization_c,
                    model_dir,
                )
                if path.exists():
                    models[regularization_c] = (
                        mixture.load_known_source_prior_model(path)
                    )
                    diagnostics[regularization_c] = models[
                        regularization_c
                    ].fit_diagnostics
                else:
                    need_fit.append(regularization_c)
            if need_fit:
                if training is None:
                    training = pd.concat(
                        [data_by_year[year] for year in fold.train_years],
                        ignore_index=True,
                    )
                print(
                    f"Fitting {specification.name} C={need_fit} "
                    f"on {fold.train_start}-{fold.train_end}",
                    flush=True,
                )
                fitted, fitted_diagnostics = fit_candidate_path(
                    training, specification, need_fit
                )
                models.update(fitted)
                diagnostics.update(fitted_diagnostics)
                for regularization_c, fitted_model in fitted.items():
                    mixture.save_known_source_prior_model(
                        fitted_model,
                        mixture.known_source_prior_model_path(
                            fold.train_years,
                            specification,
                            regularization_c,
                            model_dir,
                        ),
                    )
            for regularization_c in missing_c:
                model = models[regularization_c]
                for validation_year in fold.validation_years:
                    if _cell_present(
                        cells,
                        specification.name,
                        regularization_c,
                        fold.train_start,
                        validation_year,
                    ):
                        continue
                    row = evaluate_target(
                        model,
                        data_by_year[validation_year],
                        fold,
                        diagnostics[regularization_c],
                    )
                    cells = _upsert_cell(cells, row, checkpoint_file)
    return cells


def evaluate_target(
    fitted: mixture.KnownSourcePriorModel,
    target: pd.DataFrame,
    fold: temporal_folds.TemporalFold,
    fit_diagnostics: dict,
) -> pd.DataFrame:
    """Estimate a target share and score one adjusted candidate cell."""
    evaluated = evaluation.evaluate_target(
        adapters.adapt_known_source_prior_model(fitted),
        target,
        fold,
        label_var=config.LABEL_VAR,
        second_lien_class=config.SECOND_LIEN_CLASS,
    )
    result = evaluated.result
    estimate = evaluated.mixture_estimate
    return pd.DataFrame(
        [
            {
                "specification": fitted.specification.name,
                "continuous_form": fitted.specification.continuous_form,
                "interactions": fitted.specification.interactions,
                "regularization_c": fitted.regularization_c,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_year": result.target_year,
                "horizon": result.horizon,
                "n_validation": result.n_observations,
                "brier_score": result.brier_score,
                "log_loss": result.log_loss,
                "actual_second_share": result.actual_second_share,
                "mixture_share": result.mixture_share,
                "mixture_share_error": (
                    result.mixture_share - result.actual_second_share
                ),
                "fit_seconds": fit_diagnostics.get("fit_seconds", np.nan),
                "prediction_seconds": evaluated.evaluation_seconds,
                "n_iter": fit_diagnostics.get("n_iter", np.nan),
                "converged": fit_diagnostics.get("converged", True),
                "optimizer_converged": estimate.optimizer_converged,
                "mixture_at_boundary": estimate.at_boundary,
                "mixture_em_difference": estimate.share - estimate.em_share,
            }
        ]
    )


def aggregate_candidates(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average candidate metrics within, then equally across horizons."""
    keys = [
        "specification",
        "continuous_form",
        "interactions",
        "regularization_c",
    ]
    horizons = (
        cells.groupby([*keys, "horizon"], as_index=False)
        .agg(
            mean_brier=("brier_score", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_share_error=("mixture_share_error", "mean"),
            n_cells=("validation_year", "size"),
        )
        .sort_values(["specification", "regularization_c", "horizon"])
    )
    summary = (
        horizons.groupby(keys, as_index=False)
        .agg(
            selection_brier=("mean_brier", "mean"),
            selection_log_loss=("mean_log_loss", "mean"),
            selection_share_error=("mean_share_error", "mean"),
            n_horizons=("horizon", "nunique"),
            n_cells=("n_cells", "sum"),
        )
        .sort_values("selection_brier")
        .reset_index(drop=True)
    )
    return horizons, summary


def select_screen_survivors(
    screen_summary: pd.DataFrame,
    incumbent: FeatureSpecification,
    n_survivors: int = config.MIXTURE_LOGISTIC_SCREEN_SURVIVORS,
) -> list[FeatureSpecification]:
    """Keep the incumbent and the best distinct screened specifications."""
    if n_survivors < 1:
        raise ValueError("n_survivors must be positive")
    best_by_spec = (
        screen_summary.sort_values("selection_brier")
        .groupby("specification", as_index=False)
        .first()
        .sort_values("selection_brier")
    )
    by_name = {
        specification.name: specification
        for specification in candidate_specifications()
    }
    if incumbent.name not in set(best_by_spec["specification"]):
        raise ValueError("Incumbent is absent from the screen")
    survivors = [incumbent]
    for name in best_by_spec["specification"]:
        specification = by_name[str(name)]
        if specification not in survivors:
            survivors.append(specification)
        if len(survivors) == n_survivors:
            break
    return survivors


def run_mixture_logistic_selection(
    data_dir: str | Path = config.SELECTION_DATA_DIR,
    output_dir: str | Path = config.TABLE_DIR,
    model_dir: str | Path = config.MIXTURE_LOGISTIC_SELECTION_MODEL_DIR,
    incumbent_file: str | Path = config.SELECTED_LOGISTIC_MODEL_FILE,
    final_model_file: str | Path = config.MIXTURE_SELECTED_LOGISTIC_MODEL_FILE,
) -> dict[str, pd.DataFrame]:
    """Run the frozen staged mixture-native logistic reselection."""
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_by_year = model_selection.load_selection_years(data_dir=data_dir)
    incumbent = model_selection.load_selected_model(incumbent_file)
    folds = list(reversed(temporal_folds.reverse_folds()))
    checkpoint_file = output_dir / "mixture_logistic_selection_checkpoint.csv"
    cells = _read(checkpoint_file)

    specifications = candidate_specifications()
    screen_fold = folds[0]
    coarse_grid = {
        specification: config.LOGISTIC_SELECTION_COARSE_C
        for specification in specifications
    }
    cells = evaluate_grid(
        data_by_year,
        [screen_fold],
        coarse_grid,
        cells,
        checkpoint_file,
        model_dir,
    )
    spec_names = {specification.name for specification in specifications}
    screen_cells = cells.loc[
        (cells["train_start"] == screen_fold.train_start)
        & cells["specification"].isin(spec_names)
        & cells["regularization_c"].isin(config.LOGISTIC_SELECTION_COARSE_C)
    ]
    _, screen_summary = aggregate_candidates(screen_cells)
    survivors = select_screen_survivors(
        screen_summary, incumbent.specification
    )

    survivor_coarse = {
        specification: config.LOGISTIC_SELECTION_COARSE_C
        for specification in survivors
    }
    cells = evaluate_grid(
        data_by_year,
        folds,
        survivor_coarse,
        cells,
        checkpoint_file,
        model_dir,
    )
    survivor_names = {specification.name for specification in survivors}
    full_coarse_cells = cells.loc[
        cells["specification"].isin(survivor_names)
        & cells["regularization_c"].isin(config.LOGISTIC_SELECTION_COARSE_C)
    ]
    _, coarse_summary = aggregate_candidates(full_coarse_cells)
    best_coarse = (
        coarse_summary.sort_values("selection_brier")
        .groupby("specification", as_index=False)
        .first()
        .set_index("specification")["regularization_c"]
    )
    refinement = {
        specification: model_selection.refinement_values(
            float(best_coarse.loc[specification.name])
        )
        for specification in survivors
    }
    cells = evaluate_grid(
        data_by_year,
        folds,
        refinement,
        cells,
        checkpoint_file,
        model_dir,
    )
    eligible_c = {
        specification.name: {
            *config.LOGISTIC_SELECTION_COARSE_C,
            *refinement[specification],
        }
        for specification in survivors
    }
    eligible = pd.concat(
        [
            cells.loc[
                (cells["specification"] == name)
                & cells["regularization_c"].isin(values)
            ]
            for name, values in eligible_c.items()
        ],
        ignore_index=True,
    ).drop_duplicates(
        ["specification", "regularization_c", "train_start", "validation_year"]
    )
    horizons, summary = aggregate_candidates(eligible)
    winner_row = summary.iloc[0]
    winner_spec = next(
        specification
        for specification in survivors
        if specification.name == winner_row["specification"]
    )
    winner_c = float(winner_row["regularization_c"])
    winner_cells = eligible.loc[
        (eligible["specification"] == winner_spec.name)
        & np.isclose(eligible["regularization_c"], winner_c)
    ]
    comparison = compare_estimators(winner_cells, output_dir)

    final_years = tuple(config.TRAIN_YEARS)
    final_model_file = Path(final_model_file)
    if final_model_file.exists():
        final_model = mixture.load_known_source_prior_model(final_model_file)
    else:
        final_training = pd.concat(
            [data_by_year[year] for year in final_years], ignore_index=True
        )
        final_model = mixture.fit_known_source_prior_model(
            final_training,
            winner_spec,
            winner_c,
            model_file=final_model_file,
        )
    if (
        final_model.specification != winner_spec
        or not np.isclose(final_model.regularization_c, winner_c)
        or final_model.train_years != final_years
    ):
        raise RuntimeError("Cached final mixture logistic model is not the winner")
    forward_metrics = evaluate_forward(
        final_model, data_by_year, output_dir / "mixture_logistic_forward_metrics.csv"
    )
    forward_summary = _simple_summary(forward_metrics)

    survivor_table = pd.DataFrame(
        [
            {
                "specification": specification.name,
                "incumbent_forced": specification == incumbent.specification,
            }
            for specification in survivors
        ]
    )
    decision = pd.DataFrame(
        [
            {
                **winner_row.to_dict(),
                "screen_train_start": screen_fold.train_start,
                "n_screen_specifications": len(specifications),
                "n_survivors": len(survivors),
                "incumbent_specification": incumbent.specification.name,
                "incumbent_c": incumbent.regularization_c,
                "model_file": str(final_model_file),
                "selection_metric": "mixture_adjusted_brier_equal_horizon_weight",
            }
        ]
    )
    outputs = {
        "screen_summary": screen_summary,
        "survivors": survivor_table,
        "cells": eligible,
        "horizons": horizons,
        "summary": summary,
        "comparison": comparison,
        "forward_metrics": forward_metrics,
        "forward_summary": forward_summary,
        "decision": decision,
    }
    for name, frame in outputs.items():
        frame.to_csv(
            output_dir / f"mixture_logistic_selection_{name}.csv", index=False
        )
    return outputs


def compare_estimators(
    winner_cells: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Join the reselected winner to the three existing mixture estimators."""
    keys = ["train_start", "validation_year", "horizon"]
    result = winner_cells[[*keys, "brier_score"]].rename(
        columns={"brier_score": "reselected_logistic_brier"}
    )
    sources = (
        (
            "existing_logistic_brier",
            output_dir / "mixture_calibration_reverse_metrics.csv",
        ),
        ("boosting_brier", output_dir / "boosting_reverse_metrics.csv"),
        ("forest_brier", output_dir / "rf_mixture_reverse_metrics.csv"),
    )
    for name, path in sources:
        other = pd.read_csv(path)[[*keys, "brier_score"]].rename(
            columns={"brier_score": name}
        )
        result = result.merge(other, on=keys, validate="one_to_one")
        result[f"reselected_minus_{name}"] = (
            result["reselected_logistic_brier"] - result[name]
        )
    return result


def evaluate_forward(
    model: mixture.KnownSourcePriorModel,
    data_by_year: dict[int, pd.DataFrame],
    checkpoint_file: str | Path,
) -> pd.DataFrame:
    """Apply the final mixture-native winner to every forward year."""
    checkpoint_file = Path(checkpoint_file)
    metrics = _read(checkpoint_file)
    fold = temporal_folds.forward_fold(model.train_years, config.VALIDATE_YEARS)
    for year in fold.target_years:
        if not metrics.empty and bool((metrics["validation_year"] == year).any()):
            continue
        target = data_by_year[year]
        evaluated = evaluation.evaluate_target(
            adapters.adapt_known_source_prior_model(model),
            target,
            fold,
            label_var=config.LABEL_VAR,
            second_lien_class=config.SECOND_LIEN_CLASS,
        )
        row = pd.DataFrame(
            [
                {
                    "evaluation_design": "forward_robustness",
                    "specification": model.specification.name,
                    "regularization_c": model.regularization_c,
                    "train_start": min(model.train_years),
                    "train_end": max(model.train_years),
                    "validation_year": year,
                    "horizon": fold.horizon_for(year),
                    **evaluation.probability_metrics(
                        target[config.LABEL_VAR].to_numpy()
                        == config.SECOND_LIEN_CLASS,
                        evaluated.probability,
                    ),
                    "mixture_share": evaluated.result.mixture_share,
                    "share_optimizer_converged": (
                        evaluated.result.optimizer_converged
                    ),
                    "share_at_boundary": evaluated.result.mixture_at_boundary,
                }
            ]
        )
        metrics = _replace_forward(metrics, row, checkpoint_file)
    return metrics


def _simple_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **{
                    column: metrics[column].mean()
                    for column in calibration.METRIC_COLUMNS
                },
                "n_cells": len(metrics),
                "weighting": "equal_across_validation_years",
            }
        ]
    )


def _candidate_complete(
    cells: pd.DataFrame,
    specification: str,
    regularization_c: float,
    fold: temporal_folds.TemporalFold,
) -> bool:
    return all(
        _cell_present(
            cells,
            specification,
            regularization_c,
            fold.train_start,
            year,
        )
        for year in fold.validation_years
    )


def _cell_present(
    cells: pd.DataFrame,
    specification: str,
    regularization_c: float,
    train_start: int,
    validation_year: int,
) -> bool:
    if cells.empty:
        return False
    return bool(
        (
            (cells["specification"] == specification)
            & np.isclose(cells["regularization_c"], regularization_c)
            & (cells["train_start"] == train_start)
            & (cells["validation_year"] == validation_year)
        ).any()
    )


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _upsert_cell(
    existing: pd.DataFrame, new: pd.DataFrame, path: Path
) -> pd.DataFrame:
    row = new.iloc[0]
    if not existing.empty:
        keep = ~(
            (existing["specification"] == row["specification"])
            & np.isclose(existing["regularization_c"], row["regularization_c"])
            & (existing["train_start"] == row["train_start"])
            & (existing["validation_year"] == row["validation_year"])
        )
        existing = existing.loc[keep]
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined


def _replace_forward(
    existing: pd.DataFrame, new: pd.DataFrame, path: Path
) -> pd.DataFrame:
    year = int(new["validation_year"].iloc[0])
    if not existing.empty:
        existing = existing.loc[existing["validation_year"] != year]
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined
