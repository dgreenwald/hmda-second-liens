from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from hmda_seconds import mixture
from hmda_seconds.density_ratio import evaluation, folds


@dataclass
class CountingRatioModel:
    model_id: str = "counting__train_2005_2006"
    train_years: tuple[int, ...] = (2005, 2006)
    calls: int = field(default=0, init=False)

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return frame["score"].to_numpy(dtype=float)


def test_shared_evaluator_calls_model_once_and_matches_primitives():
    model = CountingRatioModel()
    fold = folds.temporal_fold(model.train_years, (2004,), direction="reverse")
    target = pd.DataFrame(
        {
            "year": [2004] * 8,
            "lien_status": [1, 2, 1, 2, 1, 2, 1, 2],
            "score": [-2.0, -1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0],
        }
    )

    evaluated = evaluation.evaluate_target(
        model,
        target,
        fold,
        label_var="lien_status",
        second_lien_class=2,
    )
    direct_share = mixture.estimate_mixture_share(target["score"].to_numpy())
    direct_probability = mixture.adjusted_probability(
        target["score"].to_numpy(), direct_share.share
    )

    assert model.calls == 1
    assert evaluated.mixture_estimate.share == pytest.approx(direct_share.share)
    assert evaluated.probability == pytest.approx(direct_probability)
    assert evaluated.result.mean_probability == pytest.approx(
        direct_probability.mean()
    )
    assert evaluated.result.hard_share_050 == pytest.approx(
        (direct_probability >= 0.5).mean()
    )
    assert evaluated.result.fold_id == fold.fold_id
    assert evaluated.result.horizon == 1
    assert evaluated.metrics == evaluation.evaluate_sample(
        target["lien_status"].to_numpy() == 2, direct_probability
    )


def test_metric_record_combines_canonical_metrics_and_cell_fields():
    y_second = np.array([False, True, False, True])
    probability = np.array([0.1, 0.2, 0.7, 0.8])

    record = evaluation.metric_record(
        y_second,
        probability,
        metadata={"evaluation_design": "forward_robustness"},
        additional={"mixture_share": 0.4},
    )

    assert record["n"] == 4
    assert record["evaluation_design"] == "forward_robustness"
    assert record["mixture_share"] == pytest.approx(0.4)


def test_metric_record_rejects_field_collisions():
    with pytest.raises(ValueError, match="fields overlap: n"):
        evaluation.metric_record(
            np.array([False, True]),
            np.array([0.2, 0.8]),
            metadata={"n": 2},
        )


def test_shared_evaluator_rejects_misaligned_model_and_target_years():
    model = CountingRatioModel()
    fold = folds.temporal_fold(model.train_years, (2004,), direction="reverse")
    target = pd.DataFrame(
        {"year": [2003, 2003], "lien_status": [1, 2], "score": [-1.0, 1.0]}
    )

    with pytest.raises(ValueError, match="not a target"):
        evaluation.evaluate_target(
            model,
            target,
            fold,
            label_var="lien_status",
            second_lien_class=2,
        )

    model.train_years = (2006, 2007)
    target["year"] = 2004
    with pytest.raises(ValueError, match="source years"):
        evaluation.evaluate_target(
            model,
            target,
            fold,
            label_var="lien_status",
            second_lien_class=2,
        )


def test_merge_cell_metrics_preserves_schema_and_computes_differences():
    keys = {
        "train_start": [2005, 2006],
        "validation_year": [2004, 2005],
        "horizon": [1, 1],
    }
    primary = pd.DataFrame({**keys, "score": [0.12, 0.20], "ignored": [1, 2]})
    baseline = pd.DataFrame({**keys, "loss": [0.10, 0.25]})
    alternative = pd.DataFrame({**keys, "metric": [0.11, 0.18]})

    result = evaluation.merge_cell_metrics(
        primary,
        primary_metric="score",
        primary_output="challenger_brier",
        comparisons={
            "baseline_brier": (baseline, "loss"),
            "alternative_brier": (alternative, "metric"),
        },
        difference_columns={
            "baseline_brier": "challenger_minus_baseline",
            "alternative_brier": "challenger_minus_alternative",
        },
    )

    assert list(result) == [
        "train_start",
        "validation_year",
        "horizon",
        "challenger_brier",
        "baseline_brier",
        "alternative_brier",
        "challenger_minus_baseline",
        "challenger_minus_alternative",
    ]
    assert result["challenger_minus_baseline"].tolist() == pytest.approx(
        [0.02, -0.05]
    )
    assert result["challenger_minus_alternative"].tolist() == pytest.approx(
        [0.01, 0.02]
    )


def test_merge_cell_metrics_requires_matching_names_and_unique_cells():
    primary = pd.DataFrame(
        {
            "train_start": [2005],
            "validation_year": [2004],
            "horizon": [1],
            "score": [0.12],
        }
    )
    duplicate = pd.DataFrame(
        {
            "train_start": [2005, 2005],
            "validation_year": [2004, 2004],
            "horizon": [1, 1],
            "score": [0.10, 0.11],
        }
    )

    with pytest.raises(ValueError, match="must have the same names"):
        evaluation.merge_cell_metrics(
            primary,
            primary_metric="score",
            primary_output="primary",
            comparisons={"other": (duplicate, "score")},
            difference_columns={},
        )
    with pytest.raises(pd.errors.MergeError):
        evaluation.merge_cell_metrics(
            primary,
            primary_metric="score",
            primary_output="primary",
            comparisons={"other": (duplicate, "score")},
            difference_columns={"other": "difference"},
        )
