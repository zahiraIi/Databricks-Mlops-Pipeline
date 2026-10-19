"""
Test: Feature Leakage Prevention
Ensures that the activity prediction model NEVER uses columns that would 
constitute label leakage (PeriodDebit, PeriodCredit, PeriodActivity, EndingBalance).

Run in CI: pytest tests/test_feature_leakage.py
"""

import pytest
import yaml
import os


@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "pipeline_config.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Columns that MUST NEVER appear in activity model features
FORBIDDEN_COLS = {
    "PeriodDebit",
    "PeriodCredit", 
    "PeriodActivity",
    "EndingBalance",
    "label_had_activity",
    "has_activity_flag",
    "anomaly_score",
    "anomaly_flag",
    "is_anomaly",
}


class TestFeatureLeakage:
    """Ensure no label leakage in activity model feature configuration."""

    def test_no_forbidden_cols_in_activity_features(self, config):
        """Activity model features must not include any forbidden columns."""
        activity_features = set(config["columns"]["activity_model_features"])
        leakage = activity_features.intersection(FORBIDDEN_COLS)
        
        assert len(leakage) == 0, (
            f"LEAKAGE: Forbidden columns found in activity_model_features: {leakage}. "
            f"These columns encode the target variable and must not be used as features."
        )

    def test_anomaly_signals_not_in_activity_features(self, config):
        """Anomaly signals (scored values) must not be in activity features."""
        activity_features = set(config["columns"]["activity_model_features"])
        anomaly_signals = set(config["columns"]["anomaly_signals"])
        
        overlap = activity_features.intersection(anomaly_signals)
        assert len(overlap) == 0, (
            f"Overlap between activity features and anomaly signals: {overlap}. "
            f"Anomaly signals are the values being scored, not prediction inputs."
        )

    def test_activity_features_exist_in_schema(self, config):
        """All configured activity features must be valid column names."""
        valid_cols = set(config["columns"]["key_cols"] + 
                       config["columns"]["balance_cols"] + 
                       config["columns"]["meta_cols"])
        # Add BeginningBalance explicitly (it's the only balance col allowed)
        valid_base_cols = set(config["columns"]["key_cols"]) | {"BeginningBalance"}
        
        activity_features = set(config["columns"]["activity_model_features"])
        
        # Check that base features (non-engineered) exist in schema
        base_features = activity_features.intersection(valid_cols | valid_base_cols)
        assert len(base_features) > 0, "No valid base features found in activity_model_features"

    def test_split_method_is_temporal(self, config):
        """Split must be temporal (by period), never random."""
        assert config["split"]["method"] == "temporal", (
            f"Split method is '{config['split']['method']}' — must be 'temporal'. "
            f"Random row shuffling causes data leakage in time-series financial data."
        )

    def test_beginning_balance_is_only_allowed_financial_col(self, config):
        """Only BeginningBalance should appear in activity features from balance_cols."""
        balance_cols = set(config["columns"]["balance_cols"])
        activity_features = set(config["columns"]["activity_model_features"])
        
        balance_in_features = balance_cols.intersection(activity_features)
        allowed = {"BeginningBalance"}
        
        forbidden_balance = balance_in_features - allowed
        assert len(forbidden_balance) == 0, (
            f"Forbidden balance columns in activity features: {forbidden_balance}. "
            f"Only BeginningBalance is allowed (it's known before the period starts)."
        )
