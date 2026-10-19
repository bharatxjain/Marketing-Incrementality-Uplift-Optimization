import numpy as np
import pandas as pd
import pytest

from src import config, data


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


def test_schema_and_arms(raw):
    assert raw.shape == (64000, 12)
    assert set(raw["segment"]) == set(config.ARMS)


def test_missing_column_fails_clearly(raw):
    with pytest.raises(data.DataValidationError, match="Missing required columns"):
        data.validate(raw.drop(columns=["spend"]))


def test_unknown_treatment_label_fails(raw):
    bad = raw.head(100).copy()
    bad.loc[0, "segment"] = "SMS"
    with pytest.raises(data.DataValidationError, match="Unknown treatment labels"):
        data.validate(bad)


def test_no_outcomes_in_features(raw):
    X = data.build_features(raw)
    assert not set(X.columns) & set(data.POST_TREATMENT + ["segment"])
    assert list(X.columns) == data.FEATURE_COLUMNS
    assert X.notna().all().all()


def test_features_stable_on_subsets(raw):
    sub = raw[raw["channel"] == "Web"].head(50)
    assert list(data.build_features(sub).columns) == data.FEATURE_COLUMNS


def test_splits_reproducible_and_stratified(raw):
    a = data.assign_splits(raw, seed=config.SEED)
    b = data.assign_splits(raw, seed=config.SEED)
    assert (a == b).all()
    shares = a.value_counts(normalize=True)
    for name, frac in config.SPLIT_FRACTIONS.items():
        assert abs(shares[name] - frac) < 0.005
    tab = pd.crosstab(a, raw["segment"], normalize="columns")
    assert np.allclose(tab.to_numpy(), tab.to_numpy().mean(axis=1, keepdims=True), atol=0.005)


def test_folds_cover_dev_only(raw):
    split = data.assign_splits(raw)
    folds = data.assign_folds(raw, split == "dev")
    assert (folds[split == "test"] == -1).all()
    dev_folds = folds[split == "dev"]
    assert set(dev_folds) == set(range(config.N_FOLDS))
    assert dev_folds.value_counts(normalize=True).between(0.19, 0.21).all()


def test_campaign_arrays(raw):
    X, t, y = data.campaign_arrays(raw, "womens", "spend")
    assert len(X) == len(t) == len(y)
    assert set(np.unique(t)) == {0, 1}
    assert (raw.loc[X.index, "segment"].isin([config.CONTROL, "Womens E-Mail"])).all()
