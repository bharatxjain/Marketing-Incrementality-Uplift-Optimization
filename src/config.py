"""Frozen configuration for the primary experiment.

Every notebook, test and the dashboard import from here so that the treatment
definition, split seed and economic assumptions are recorded in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
# Candidate locations for the raw CSV, checked in order. The first one that
# exists is used. The original file name from the 2008 challenge is kept.
RAW_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "raw" / "Kevin_Hillstrom_MineThatData.csv",
    PROJECT_ROOT / "Data" / "Kevin_Hillstrom_MineThatData.csv",
    PROJECT_ROOT / "data" / "raw" / "hillstrom.csv",
]
RAW_DATA_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# ----------------------------------------------------------------------------
# Experiment design (verified against the data in notebook 01)
# ----------------------------------------------------------------------------
CONTROL = "No E-Mail"
CAMPAIGNS = {"womens": "Womens E-Mail", "mens": "Mens E-Mail"}
ARMS = [CONTROL, CAMPAIGNS["mens"], CAMPAIGNS["womens"]]
# Primary campaign for the headline analysis; the other is a replication.
PRIMARY_CAMPAIGN = "womens"

OUTCOMES = ["visit", "conversion", "spend"]
# Outcome used for targeting economics. Spend is in dollars, so incremental
# spend times margin is incremental contribution without a conversion-value
# assumption.
PRIMARY_OUTCOME = "spend"

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
# 80% development set (model fitting + 5-fold cross-fitted model selection),
# 20% test set touched only once, in notebook 04, after the pipeline is frozen.
SPLIT_FRACTIONS = {"dev": 0.8, "test": 0.2}
N_FOLDS = 5
ROBUSTNESS_SEEDS = [7, 11, 23, 101, 2024]

# ----------------------------------------------------------------------------
# Business assumptions (USER-SUPPLIED, not estimated from the data)
# ----------------------------------------------------------------------------
# Contribution margin: share of incremental spend kept after cost of goods.
MARGIN = 0.35
# Fully loaded cost per email contact: send cost + creative amortisation +
# an allowance for unsubscribe / fatigue risk.
COST_PER_CONTACT = 0.10
# Population used to express results in business units.
POPULATION_SIZE = 100_000
# Default budget for the constrained scenario (dollars for POPULATION_SIZE).
DEFAULT_BUDGET = 3_000.0

COVERAGE_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
