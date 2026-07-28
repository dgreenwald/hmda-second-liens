from hmda_seconds import config


def test_train_validate_years_disjoint():
    assert set(config.TRAIN_YEARS).isdisjoint(config.VALIDATE_YEARS)


def test_train_and_validate_years_within_apply_years():
    assert set(config.TRAIN_YEARS) | set(config.VALIDATE_YEARS) <= set(config.APPLY_YEARS)


def test_train_years_match_reliable_lien_status_window():
    assert list(config.TRAIN_YEARS) == [2004, 2005, 2006, 2007]


def test_feature_lists_nonempty():
    assert config.CONTINUOUS_VARS
    assert config.CATEGORY_VARS
    assert config.LABEL_VAR == "lien_status"


def test_train_test_split_sizes_sum_to_one():
    assert config.TRAIN_SIZE + config.TEST_SIZE == 1.0
