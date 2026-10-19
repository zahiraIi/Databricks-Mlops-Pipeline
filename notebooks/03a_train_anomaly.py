# Databricks notebook source
# MAGIC %md
# MAGIC # 03a — Train Anomaly Model (Isolation Forest + Rules)
# MAGIC 
# MAGIC The anomaly model scores PeriodActivity/EndingBalance signals (not features that define
# MAGIC the label). Rules layer flags pass through as hard anomalies. IF provides soft scoring.
# MAGIC 
# MAGIC Evaluation uses proxy labels: rules flags + top-5% extremes + spiked synthetics.
# MAGIC 
# COMMAND ----------

import yaml
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
import mlflow.sklearn

import os as _os
_nb_dir = _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in dir() else _os.getcwd()
_config_candidates = [
    _os.path.join(_nb_dir, "..", "config", "pipeline_config.yml"),
    _os.path.join(_nb_dir, "config", "pipeline_config.yml"),
    "/Workspace/Users/user@example.com/mlops-demo-repo/config/pipeline_config.yml",
]
_config_path = next(p for p in _config_candidates if _os.path.exists(p))
with open(_config_path, "r") as f:
    cfg = yaml.safe_load(f)

CATALOG = cfg["catalog"]
SCHEMA = cfg["schema"]
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['feature_table']}"
MODEL_CFG = cfg["anomaly_model"]

mlflow.set_registry_uri("databricks-uc")

print(f"Feature table: {FEATURE_TABLE}")
print(f"Model: {MODEL_CFG['name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data — anomaly signals + rules flags

# COMMAND ----------

# Load feature table
df_spark = spark.table(FEATURE_TABLE)

# For anomaly model: use train + validate periods for fitting
df_fit = df_spark.filter(F.col("split").isin("train", "validate")).toPandas()
df_test = df_spark.filter(F.col("split") == "test").toPandas()

print(f"Fit rows: {len(df_fit)}, Test rows: {len(df_test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define anomaly signal columns

# COMMAND ----------

from pyspark.sql import functions as F

ANOMALY_SIGNAL_COLS = [
    "PeriodActivity",
    "EndingBalance",
    "balance_change_ratio",
    "peer_group_z_score",
    "activity_to_balance_ratio",
]

# Filter to rows with activity (dormant rows are not anomalous by definition)
df_fit_active = df_fit[df_fit["PeriodActivity"] != 0].copy()
df_test_active = df_test[df_test["PeriodActivity"] != 0].copy()

# Also keep full set for scoring (dormant rows get score = 0)
print(f"Active rows for fitting: {len(df_fit_active)}")
print(f"Active rows for test: {len(df_test_active)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Isolation Forest

# COMMAND ----------

X_fit = df_fit_active[ANOMALY_SIGNAL_COLS].fillna(0).replace([np.inf, -np.inf], 999.0)
X_test = df_test_active[ANOMALY_SIGNAL_COLS].fillna(0).replace([np.inf, -np.inf], 999.0)

# Scale signals
scaler = StandardScaler()
X_fit_scaled = scaler.fit_transform(X_fit)
X_test_scaled = scaler.transform(X_test)

# Train IF
iso_forest = IsolationForest(
    n_estimators=MODEL_CFG["params"]["n_estimators"],
    contamination=MODEL_CFG["params"]["contamination"],
    max_samples=MODEL_CFG["params"]["max_samples"],
    random_state=MODEL_CFG["params"]["random_state"],
)

with mlflow.start_run(run_name="anomaly_isolation_forest") as run:
    iso_forest.fit(X_fit_scaled)
    
    # Score: lower = more anomalous (sklearn convention: -1=anomaly, 1=normal)
    scores_test = iso_forest.decision_function(X_test_scaled)
    predictions_test = iso_forest.predict(X_test_scaled)  # -1 or 1
    
    mlflow.log_params(MODEL_CFG["params"])
    mlflow.log_metric("n_fit_samples", len(X_fit_scaled))
    mlflow.log_metric("n_test_samples", len(X_test_scaled))
    mlflow.log_metric("n_anomalies_predicted", int((predictions_test == -1).sum()))
    mlflow.log_metric("anomaly_rate", float((predictions_test == -1).mean()))
    
    print(f"Anomalies predicted in test: {(predictions_test == -1).sum()} / {len(predictions_test)}")
    
    anomaly_run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate proxy labels for evaluation
# MAGIC 
# MAGIC Since anomaly detection is unsupervised, we evaluate against:
# MAGIC 1. Rules-flagged rows (hard anomalies from QA step)
# MAGIC 2. Top-5% most extreme movements
# MAGIC 3. Spiked synthetic anomalies

# COMMAND ----------

# Proxy label 1: Rules flags (from QA step — already in feature table)
rules_flagged = df_test_active["flag_identity_break"] | df_test_active["flag_sign_flip"] | df_test_active["flag_orphan_activity"]

# Proxy label 2: Top 5% extreme balance_change_ratio
extreme_threshold = df_test_active["balance_change_ratio"].abs().quantile(0.95)
extreme_flagged = df_test_active["balance_change_ratio"].abs() >= extreme_threshold

# Proxy label 3: Spike synthetic anomalies (inject extreme values into test)
np.random.seed(42)
n_spikes = max(5, int(len(df_test_active) * 0.05))
spike_indices = np.random.choice(len(X_test_scaled), size=n_spikes, replace=False)
X_test_spiked = X_test_scaled.copy()
X_test_spiked[spike_indices] = X_test_spiked[spike_indices] * 10  # 10x amplification

scores_spiked = iso_forest.decision_function(X_test_spiked)
spiked_labels = np.zeros(len(X_test_spiked))
spiked_labels[spike_indices] = 1  # these are the injected anomalies

# Combined proxy label (union)
proxy_labels = (rules_flagged.values.astype(int) | extreme_flagged.values.astype(int))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate: Precision@K

# COMMAND ----------

def precision_at_k(scores, labels, k):
    """Lower score = more anomalous for IF decision_function."""
    ranked_indices = np.argsort(scores)[:k]  # most anomalous first
    return labels[ranked_indices].mean()

with mlflow.start_run(run_id=anomaly_run_id):
    for k in MODEL_CFG["evaluation"]["precision_at_k"]:
        if k <= len(scores_test):
            p_at_k = precision_at_k(scores_test, proxy_labels, k)
            mlflow.log_metric(f"precision_at_{k}", p_at_k)
            print(f"Precision@{k}: {p_at_k:.3f}")
        else:
            print(f"Skipping precision@{k} — only {len(scores_test)} test samples")
    
    # Spiked evaluation
    p_at_k_spiked = precision_at_k(scores_spiked, spiked_labels, n_spikes)
    mlflow.log_metric("precision_at_k_spiked", p_at_k_spiked)
    print(f"\nSpiked precision@{n_spikes}: {p_at_k_spiked:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log model artifact

# COMMAND ----------

import mlflow.sklearn

with mlflow.start_run(run_id=anomaly_run_id):
    # Log the scaler + model as a pipeline
    from sklearn.pipeline import Pipeline
    anomaly_pipeline = Pipeline([
        ("scaler", scaler),
        ("isolation_forest", iso_forest),
    ])
    
    mlflow.sklearn.log_model(
        anomaly_pipeline,
        artifact_path="anomaly_model",
        registered_model_name=f"{CATALOG}.{SCHEMA}.{cfg['registry']['anomaly_model_name']}",
    )
    
    print(f"Model logged and registered: {cfg['registry']['anomaly_model_name']}")
    print(f"Run ID: {anomaly_run_id}")
