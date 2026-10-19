# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Monitor Metrics & Retrain Trigger
# MAGIC 
# MAGIC Tracks:
# MAGIC 1. Feature drift (PSI on key features)
# MAGIC 2. Prediction distribution shift
# MAGIC 3. Model staleness (periods since last retrain)
# MAGIC 
# MAGIC Triggers retrain if any threshold is breached.
# MAGIC 
# COMMAND ----------

import yaml
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from datetime import datetime

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
MONITOR_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['monitor_metrics_table']}"
ACTIVITY_PREDS_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['activity_predictions_table']}"

MON_CFG = cfg["monitoring"]

print(f"PSI threshold: {MON_CFG['psi_threshold']}")
print(f"Prediction shift threshold: {MON_CFG['prediction_shift_threshold']}")
print(f"Staleness trigger: {MON_CFG['staleness_periods']} periods")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Population Stability Index (PSI)
# MAGIC 
# MAGIC Compares feature distributions between training data and latest scoring data.
# MAGIC PSI > 0.20 indicates significant distribution shift.

# COMMAND ----------

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index between two distributions."""
    # Bin the expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    if len(breakpoints) < 3:
        return 0.0  # Not enough variation to compute PSI
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Avoid division by zero
    expected_pct = (expected_counts + 1) / (expected_counts.sum() + bins)
    actual_pct = (actual_counts + 1) / (actual_counts.sum() + bins)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute PSI on key features

# COMMAND ----------

df_features = spark.table(FEATURE_TABLE)

# Training data (reference distribution)
df_train = df_features.filter(F.col("split") == "train").toPandas()

# Latest scored data (current distribution)
df_latest = df_features.filter(F.col("split") == "test").toPandas()

# Features to monitor for drift
MONITOR_FEATURES = [
    "BeginningBalance",
    "abs_beginning_balance",
    "log_abs_balance",
    "peer_group_mean_balance",
    "balance_quartile",
]

psi_results = {}
for feat in MONITOR_FEATURES:
    if feat in df_train.columns and feat in df_latest.columns:
        expected = df_train[feat].dropna().values
        actual = df_latest[feat].dropna().values
        if len(expected) > 10 and len(actual) > 10:
            psi_results[feat] = calculate_psi(expected, actual)

print("=== PSI Results ===")
for feat, psi in psi_results.items():
    status = "DRIFT" if psi > MON_CFG["psi_threshold"] else "OK"
    print(f"  {feat}: {psi:.4f} [{status}]")

max_psi = max(psi_results.values()) if psi_results else 0.0
drift_triggered = max_psi > MON_CFG["psi_threshold"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction Distribution Shift

# COMMAND ----------

# Compare positive prediction rate across periods
df_preds = spark.table(ACTIVITY_PREDS_TABLE).toPandas()

if len(df_preds) > 0:
    current_positive_rate = df_preds["predicted_active"].mean()
    # Compare to the training label rate
    train_positive_rate = df_train["label_had_activity"].mean()
    
    shift = abs(current_positive_rate - train_positive_rate) / max(train_positive_rate, 0.001)
    prediction_shift_triggered = shift > MON_CFG["prediction_shift_threshold"]
    
    print(f"\n=== Prediction Distribution ===")
    print(f"  Training positive rate: {train_positive_rate:.4f}")
    print(f"  Current positive rate:  {current_positive_rate:.4f}")
    print(f"  Relative shift: {shift:.4f} (threshold: {MON_CFG['prediction_shift_threshold']})")
    print(f"  Status: {'SHIFT DETECTED' if prediction_shift_triggered else 'OK'}")
else:
    prediction_shift_triggered = False
    print("No predictions found — skipping shift check.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Staleness Check

# COMMAND ----------

from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

ACTIVITY_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{cfg['registry']['activity_model_name']}"

# Check when the Champion was last updated
try:
    champion_version = client.get_model_version_by_alias(ACTIVITY_MODEL_NAME, "Champion")
    last_updated = champion_version.creation_timestamp / 1000  # ms to seconds
    days_since_update = (datetime.now().timestamp() - last_updated) / 86400
    
    # Approximate periods since update (assume monthly periods)
    periods_since_update = days_since_update / 30
    staleness_triggered = periods_since_update > MON_CFG["staleness_periods"]
    
    print(f"\n=== Model Staleness ===")
    print(f"  Last model update: {datetime.fromtimestamp(last_updated).isoformat()}")
    print(f"  Days since update: {days_since_update:.1f}")
    print(f"  Approx periods: {periods_since_update:.1f} (threshold: {MON_CFG['staleness_periods']})")
    print(f"  Status: {'STALE' if staleness_triggered else 'OK'}")
except Exception as e:
    print(f"  No Champion alias found — model may not be promoted yet. ({e})")
    staleness_triggered = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write monitoring metrics to Delta

# COMMAND ----------

monitor_record = {
    "timestamp": datetime.now().isoformat(),
    "max_psi": float(max_psi),
    "drift_triggered": drift_triggered,
    "prediction_shift": float(shift) if "shift" in dir() else 0.0,
    "prediction_shift_triggered": prediction_shift_triggered,
    "staleness_triggered": staleness_triggered,
    "retrain_recommended": drift_triggered or prediction_shift_triggered or staleness_triggered,
}

# Add per-feature PSI
for feat, psi in psi_results.items():
    monitor_record[f"psi_{feat}"] = float(psi)

df_monitor = spark.createDataFrame([monitor_record])

(
    df_monitor.write
    .format("delta")
    .mode("append")
    .saveAsTable(MONITOR_TABLE)
)

print(f"\nMonitor metrics appended to: {MONITOR_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrain Decision

# COMMAND ----------

retrain = drift_triggered or prediction_shift_triggered or staleness_triggered

print("\n" + "="*50)
print("MONITORING SUMMARY")
print("="*50)
print(f"  Feature drift (PSI):      {'TRIGGERED' if drift_triggered else 'OK'}")
print(f"  Prediction shift:         {'TRIGGERED' if prediction_shift_triggered else 'OK'}")
print(f"  Model staleness:          {'TRIGGERED' if staleness_triggered else 'OK'}")
print(f"  ---")
print(f"  RETRAIN RECOMMENDED: {'YES' if retrain else 'NO'}")
print("="*50)

if retrain:
    # Set task value for downstream job orchestration
    try:
        dbutils.jobs.taskValues.set(key="retrain_triggered", value=True)
        print("\nTask value set: retrain_triggered = True")
        print("Downstream retrain job will be triggered.")
    except:
        print("\n(Not running in job context — retrain trigger logged to Delta only)")
else:
    print("\nNo retraining needed at this time.")
