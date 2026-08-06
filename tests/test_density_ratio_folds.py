from collections import Counter

import pytest

from hmda_seconds import config, model_selection
from hmda_seconds.density_ratio import folds
from hmda_seconds.density_ratio.protocols import TemporalFold


def test_reverse_design_has_exact_windows_cells_and_horizons():
    design = folds.reverse_folds()

    assert len(design) == 9
    assert all(isinstance(fold, TemporalFold) for fold in design)
    assert [fold.train_start for fold in design] == list(range(2005, 2014))
    assert design[0].train_years == (2005, 2006, 2007, 2008)
    assert design[0].target_years == (2004,)
    assert design[-1].train_years == (2013, 2014, 2015, 2016)
    assert design[-1].target_years == tuple(range(2004, 2013))
    assert sum(len(fold.target_years) for fold in design) == 45

    horizon_counts = Counter(
        horizon for fold in design for horizon in fold.horizons
    )
    assert horizon_counts == {horizon: 10 - horizon for horizon in range(1, 10)}
    assert all(fold.direction == "reverse" for fold in design)
    assert all(
        set(fold.train_years).isdisjoint(fold.target_years) for fold in design
    )


def test_forward_design_uses_configured_release_source_and_targets():
    fold = folds.forward_fold(config.TRAIN_YEARS, config.VALIDATE_YEARS)

    assert fold.fold_id == "forward__train_2004_2007__target_2008_2016"
    assert fold.train_years == tuple(config.TRAIN_YEARS)
    assert fold.target_years == tuple(config.VALIDATE_YEARS)
    assert fold.horizons == tuple(range(1, 10))
    assert fold.horizon_for(2016) == 9


def test_legacy_model_selection_api_delegates_to_shared_design():
    shared = folds.reverse_folds()
    legacy = model_selection.reverse_folds()

    assert [fold.to_dict() for fold in legacy] == [
        fold.to_dict() for fold in shared
    ]
    constructed = model_selection.ReverseFold((2005, 2006), (2004,))
    assert constructed.direction == "reverse"
    assert constructed.horizon_for(2004) == 1


def test_explicit_fold_rejects_interleaved_years_and_wrong_horizons():
    with pytest.raises(ValueError, match="Cannot infer direction"):
        folds.temporal_fold((2005, 2007), (2004, 2006))
    with pytest.raises(ValueError, match="horizons do not match"):
        TemporalFold(
            fold_id="bad",
            train_years=(2005, 2006),
            target_years=(2004,),
            direction="reverse",
            horizons=(2,),
        )
