import numpy as np
import pandas as pd
import pytest

from hmda_seconds import logistic_features, model_selection


def test_reverse_folds_form_triangular_backward_design():
    folds = model_selection.reverse_folds()

    assert len(folds) == 9
    assert folds[0].train_years == (2005, 2006, 2007, 2008)
    assert folds[0].validation_years == (2004,)
    assert folds[-1].train_years == (2013, 2014, 2015, 2016)
    assert folds[-1].validation_years == tuple(range(2004, 2013))
    assert sum(len(fold.validation_years) for fold in folds) == 45


def test_aggregation_weights_horizons_equally():
    cells = pd.DataFrame(
        {
            "specification": ["a", "a", "a"],
            "continuous_form": ["linear"] * 3,
            "interactions": ["none"] * 3,
            "geography": ["none"] * 3,
            "regularization_c": [1.0] * 3,
            "horizon": [1, 1, 2],
            "brier_score": [0.1, 0.3, 0.8],
        }
    )

    by_horizon, summary = model_selection.aggregate_brier_cells(cells)

    assert by_horizon.set_index("horizon").loc[1, "mean_brier"] == pytest.approx(
        0.2
    )
    assert summary["selection_brier"].item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("coarse", "expected"),
    [(1e-4, (1e-5, 1e-3)), (1e-2, (1e-3, 1e-1)), (1.0, (0.1, 10.0))],
)
def test_refinement_values_add_adjacent_decades(coarse, expected):
    assert model_selection.refinement_values(coarse) == pytest.approx(expected)


def test_candidate_grid_returns_every_validation_cell(training_frame):
    frame_2004 = training_frame.copy()
    frame_2004["year"] = 2004
    frame_2005 = training_frame.copy()
    frame_2005["year"] = 2005
    specification = logistic_features.FeatureSpecification("linear", "none")
    fold = model_selection.ReverseFold((2005,), (2004,))

    cells = model_selection.evaluate_candidate_grid(
        {2004: frame_2004, 2005: frame_2005},
        {specification: [0.1, 1.0]},
        folds=[fold],
    )

    assert len(cells) == 2
    assert set(cells["regularization_c"]) == {0.1, 1.0}
    assert (cells["horizon"] == 1).all()
    assert cells["brier_score"].between(0.0, 1.0).all()
    assert cells["converged"].all()


def test_selected_model_round_trip_predictions(training_frame, tmp_path):
    specification = logistic_features.FeatureSpecification("linear", "none")
    selected = model_selection.fit_selected_model(training_frame, specification, 1.0)

    probability = selected.predict_proba_second_lien(training_frame)
    prediction = selected.predict(training_frame)

    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert set(prediction) <= {1, 2}

    output = tmp_path / "selected.pkl"
    model_selection.save_selected_model(selected, output)
    loaded = model_selection.load_selected_model(output)
    assert loaded.predict(training_frame).tolist() == prediction.tolist()


def test_candidate_grid_resumes_from_checkpoint(training_frame, tmp_path):
    frame_2004 = training_frame.assign(year=2004)
    frame_2005 = training_frame.assign(year=2005)
    specification = logistic_features.FeatureSpecification("linear", "none")
    fold = model_selection.ReverseFold((2005,), (2004,))
    checkpoint = tmp_path / "cells.csv"
    data = {2004: frame_2004, 2005: frame_2005}

    first = model_selection.evaluate_candidate_grid(
        data,
        {specification: [1.0]},
        folds=[fold],
        checkpoint_file=checkpoint,
    )
    second = model_selection.evaluate_candidate_grid(
        data,
        {specification: [1.0]},
        folds=[fold],
        checkpoint_file=checkpoint,
    )

    assert len(first) == len(second) == 1
