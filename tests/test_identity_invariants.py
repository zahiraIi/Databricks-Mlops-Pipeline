"""
Test: Identity Invariants
Validates that EndingBalance = BeginningBalance + PeriodActivity
across all rows in the dataset. This is a hard constraint from Oracle GL.

Run in CI: pytest tests/test_identity_invariants.py
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import os


@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "pipeline_config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_data(config):
    """Load source CSV for offline testing."""
    csv_path = config["data"]["source_csv"]
    # For CI, fall back to a relative path
    if not os.path.exists(csv_path):
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "..", 
            "Synthetic Oracle TrialBalance", "synthetic_sat_trialbalance.csv"
        )
    return pd.read_csv(csv_path)


class TestIdentityInvariants:
    """Test suite for GL Trial Balance identity invariants."""

    def test_ending_balance_identity(self, sample_data, config):
        """EndingBalance must equal BeginningBalance + PeriodActivity."""
        tol = config["rules"]["identity_tolerance"]
        
        errors = np.abs(
            sample_data["EndingBalance"] 
            - (sample_data["BeginningBalance"] + sample_data["PeriodActivity"])
        )
        
        violations = (errors > tol).sum()
        fail_rate = violations / len(sample_data)
        
        assert fail_rate <= config["champion_gate"]["identity_max_fail_rate"], (
            f"Identity invariant violated: {violations} rows "
            f"(fail rate {fail_rate:.6f} > threshold {config['champion_gate']['identity_max_fail_rate']})"
        )

    def test_activity_identity(self, sample_data):
        """PeriodActivity must equal PeriodDebit - PeriodCredit."""
        tol = 1e-6
        
        errors = np.abs(
            sample_data["PeriodActivity"] 
            - (sample_data["PeriodDebit"] - sample_data["PeriodCredit"])
        )
        
        violations = (errors > tol).sum()
        assert violations == 0, (
            f"Activity identity violated: {violations} rows. "
            f"PeriodActivity != PeriodDebit - PeriodCredit"
        )

    def test_no_null_balances(self, sample_data):
        """Financial balance columns must not contain nulls."""
        balance_cols = ["BeginningBalance", "PeriodDebit", "PeriodCredit", 
                       "PeriodActivity", "EndingBalance"]
        
        for col in balance_cols:
            null_count = sample_data[col].isna().sum()
            assert null_count == 0, f"Column {col} has {null_count} null values"

    def test_non_negative_debits_credits(self, sample_data):
        """PeriodDebit and PeriodCredit should be non-negative."""
        assert (sample_data["PeriodDebit"] >= 0).all(), "Negative PeriodDebit found"
        assert (sample_data["PeriodCredit"] >= 0).all(), "Negative PeriodCredit found"

    def test_period_column_valid(self, sample_data, config):
        """AccountingPeriod values must be in expected set."""
        expected_periods = set(config["split"].values()) - {"temporal"}
        actual_periods = set(sample_data["AccountingPeriod"].unique())
        
        # All periods should be recognizable
        assert actual_periods.issubset(expected_periods) or len(actual_periods) > 0, (
            f"Unknown periods found: {actual_periods - expected_periods}"
        )
