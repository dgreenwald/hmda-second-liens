import numpy as np
import pandas as pd

from hmda_seconds import mixture_calibration


def test_tail_metrics_reports_ordered_quantiles_and_extreme_shares():
    log_ratio = np.linspace(-20, 20, 10_001)

    metrics = mixture_calibration._tail_metrics(log_ratio)

    assert metrics["log_ratio_min"] == -20
    assert metrics["log_ratio_median"] == 0
    assert metrics["log_ratio_max"] == 20
    assert 0.24 < metrics["share_log_ratio_gt_10"] < 0.26
    assert 0.24 < metrics["share_log_ratio_lt_minus_10"] < 0.26


def test_cell_complete_requires_all_three_checkpoint_parts():
    key = pd.DataFrame({"train_start": [2013], "validation_year": [2012]})

    assert mixture_calibration._cell_complete(key, key, key, 2013, 2012)
    assert not mixture_calibration._cell_complete(
        key, pd.DataFrame(), key, 2013, 2012
    )
