"""Data ingestion, validation, feature construction and splitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

REQUIRED_COLUMNS = [
    "recency", "history_segment", "history", "mens", "womens", "zip_code",
    "newbie", "channel", "segment", "visit", "conversion", "spend",
]
PRE_TREATMENT = ["recency", "history_segment", "history", "mens", "womens",
                 "zip_code", "newbie", "channel"]
POST_TREATMENT = ["visit", "conversion", "spend"]

# Fixed category levels so one-hot columns are identical across any subset.
# "Surburban" is the spelling used in the original file.
ZIP_LEVELS = ["Rural", "Surburban", "Urban"]
CHANNEL_LEVELS = ["Multichannel", "Phone", "Web"]
HISTORY_SEGMENT_LEVELS = [
    "1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500",
    "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +",
]

FEATURE_COLUMNS = [
    "recency", "history", "log_history", "mens", "womens", "newbie",
    "zip_Rural", "zip_Surburban", "zip_Urban",
    "channel_Multichannel", "channel_Phone", "channel_Web",
]


class DataValidationError(ValueError):
    """Raised when the raw data violates the documented data contract."""


def find_data_file(candidates: list[Path] | None = None) -> Path:
    for path in candidates or config.RAW_DATA_CANDIDATES:
        if Path(path).exists():
            return Path(path)
    raise FileNotFoundError(
        "Hillstrom CSV not found. Place it at data/raw/Kevin_Hillstrom_MineThatData.csv "
        f"(download: {config.RAW_DATA_URL})."
    )


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the data contract. Returns the frame unchanged or raises."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        cols = df.columns[df.isna().any()].tolist()
        raise DataValidationError(f"Unexpected missing values in {cols}")

    bad_arms = set(df["segment"].unique()) - set(config.ARMS)
    if bad_arms:
        raise DataValidationError(f"Unknown treatment labels: {sorted(bad_arms)}")
    absent = set(config.ARMS) - set(df["segment"].unique())
    if absent:
        raise DataValidationError(f"Treatment arms absent from data: {sorted(absent)}")

    for col in ["mens", "womens", "newbie", "visit", "conversion"]:
        if not df[col].isin([0, 1]).all():
            raise DataValidationError(f"{col} must be binary 0/1")
    if (df["spend"] < 0).any() or (df["history"] < 0).any():
        raise DataValidationError("spend and history must be non-negative")
    if not df["recency"].between(1, 12).all():
        raise DataValidationError("recency must be in 1..12 months")
    if ((df["spend"] > 0) & (df["conversion"] == 0)).any():
        raise DataValidationError("positive spend without a conversion")
    if ((df["conversion"] == 1) & (df["visit"] == 0)).any():
        raise DataValidationError("conversion without a visit")
    if not set(df["zip_code"]).issubset(ZIP_LEVELS):
        raise DataValidationError("unexpected zip_code levels")
    if not set(df["channel"]).issubset(CHANNEL_LEVELS):
        raise DataValidationError("unexpected channel levels")
    if not set(df["history_segment"]).issubset(HISTORY_SEGMENT_LEVELS):
        raise DataValidationError("unexpected history_segment levels")
    return df


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load and validate the raw Hillstrom CSV (the file itself is never modified)."""
    df = pd.read_csv(path or find_data_file())
    df = validate(df)
    df = df.reset_index(drop=True)
    df.index.name = "customer_id"
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix built only from pre-treatment columns.

    ``history_segment`` is a deterministic binning of ``history`` and is dropped
    to avoid redundant features.
    """
    X = pd.DataFrame(index=df.index)
    X["recency"] = df["recency"].astype(float)
    X["history"] = df["history"].astype(float)
    X["log_history"] = np.log1p(df["history"].astype(float))
    for col in ["mens", "womens", "newbie"]:
        X[col] = df[col].astype(float)
    for lvl in ZIP_LEVELS:
        X[f"zip_{lvl}"] = (df["zip_code"] == lvl).astype(float)
    for lvl in CHANNEL_LEVELS:
        X[f"channel_{lvl}"] = (df["channel"] == lvl).astype(float)
    assert list(X.columns) == FEATURE_COLUMNS
    leaked = set(X.columns) & set(POST_TREATMENT + ["segment"])
    if leaked:
        raise DataValidationError(f"Post-treatment columns in features: {leaked}")
    return X


def assign_splits(df: pd.DataFrame, seed: int = config.SEED,
                  fractions: dict[str, float] | None = None) -> pd.Series:
    """Stratified partition labels (default dev/test) for the whole frame.

    Stratification is by treatment arm x conversion so that every partition has
    all three arms and a comparable number of (rare) conversions. A customer is
    in the same partition for every campaign comparison.
    """
    fractions = fractions or config.SPLIT_FRACTIONS
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    labels = pd.Series(index=df.index, dtype=object)
    names = list(fractions)
    cum = np.cumsum([fractions[n] for n in names])
    for _, idx in df.groupby(["segment", "conversion"]).groups.items():
        idx = np.asarray(idx)
        perm = rng.permutation(idx)
        cuts = np.round(cum * len(perm)).astype(int)
        start = 0
        for name, stop in zip(names, cuts):
            labels.loc[perm[start:stop]] = name
            start = stop
    return labels.astype("category")


def assign_folds(df: pd.DataFrame, mask, n_folds: int = config.N_FOLDS,
                 seed: int = config.SEED) -> pd.Series:
    """Stratified cross-fitting fold ids (0..n_folds-1) for rows in ``mask``; -1 elsewhere."""
    fracs = {f: 1 / n_folds for f in range(n_folds)}
    folds = pd.Series(-1, index=df.index)
    sub = df.loc[np.asarray(mask)]
    folds.loc[sub.index] = assign_splits(sub, seed=seed + 1, fractions=fracs).astype(int)
    return folds


def campaign_arrays(df: pd.DataFrame, campaign: str, outcome: str = config.PRIMARY_OUTCOME,
                    X: pd.DataFrame | None = None):
    """Return (X, t, y) for one campaign arm vs the no-email control."""
    label = config.CAMPAIGNS[campaign]
    mask = df["segment"].isin([label, config.CONTROL])
    sub = df.loc[mask]
    X = build_features(sub) if X is None else X.loc[mask]
    t = (sub["segment"] == label).astype(int).to_numpy()
    y = sub[outcome].astype(float).to_numpy()
    return X, t, y


def arm_shares(df: pd.DataFrame) -> dict[str, float]:
    """Empirical assignment probabilities, used as the known propensities."""
    return df["segment"].value_counts(normalize=True).reindex(config.ARMS).to_dict()
