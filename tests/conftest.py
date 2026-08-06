import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def training_frame():
    """A small synthetic frame with the schema expected by the estimators.

    log_lti is constructed to cleanly separate the two classes so tests can
    check the classifier actually learns something, not just that it runs.
    """
    rng = np.random.default_rng(0)
    n = 200
    lien_status = rng.integers(1, 3, size=n)  # 1 or 2

    log_lti = np.where(
        lien_status == 1,
        rng.normal(0.5, 0.3, size=n),
        rng.normal(-0.5, 0.3, size=n),
    )
    log_county_value_to_loan = rng.normal(0.0, 1.0, size=n)

    return pd.DataFrame(
        {
            "year": rng.integers(2004, 2008, size=n),
            "log_lti": log_lti,
            "log_county_value_to_loan": log_county_value_to_loan,
            "purchaser_type": rng.integers(0, 4, size=n),
            "loan_type": rng.integers(1, 5, size=n),
            "has_edit_status": rng.integers(0, 2, size=n).astype(bool),
            "loan_below_10k": rng.integers(0, 2, size=n).astype(bool),
            "lien_status": lien_status,
        }
    )
