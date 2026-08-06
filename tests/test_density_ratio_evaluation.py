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
