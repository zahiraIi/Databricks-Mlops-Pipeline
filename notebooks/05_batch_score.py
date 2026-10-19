# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Batch Score (Both Models)
# MAGIC 
# MAGIC Loads Champion versions of both models, scores the latest period,
# MAGIC and writes results to gold Delta tables for dashboard consumption.
# MAGIC 
# MAGIC Output: `tb_anomaly_scores` + `tb_activity_predictions`
# MAGIC 
# COMMAND ----------

import yaml
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

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
ANOMALY_SCORES_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['anomaly_scores_table']}"
ACTIVITY_PREDS_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['activity_predictions_table']}"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

ANOMALY_MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.{cfg['registry']['anomaly_model_name']}@Champion"
ACTIVITY_MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.{cfg['registry']['activity_model_name']}@Champion"

print(f"Scoring with:")
print(f"  Anomaly model: {ANOMALY_MODEL_URI}")
print(f"  Activity model: {ACTIVITY_MODEL_URI}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load latest period data

# COMMAND ----------

# Score the test period (in production, this would be the latest incoming period)
df_spark = spark.table(FEATURE_TABLE)

# Get the latest period to score
latest_period = df_spark.select(F.max("period_ordinal")).collect()[0][0]
df_score = df_spark.filter(F.col("period_ordinal") == latest_period)

print(f"Scoring period ordinal: {latest_period}")
print(f"Rows to score: {df_score.count()}")

df_score_pd = df_score.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score: Anomaly Model

# COMMAND ----------

ANOMALY_SIGNAL_COLS = [
    "PeriodActivity",
    "EndingBalance",
    "balance_change_ratio",
    "peer_group_z_score",
    "activity_to_balance_ratio",
]

# Load champion anomaly model (sklearn pipeline: scaler + IF)
anomaly_model = mlflow.sklearn.load_model(ANOMALY_MODEL_URI)

# Only score active rows (dormant = not anomalous by definition)
active_mask = df_score_pd["PeriodActivity"] != 0
X_anomaly = df_score_pd.loc[active_mask, ANOMALY_SIGNAL_COLS].fillna(0).replace([np.inf, -np.inf], 999.0)

# Score (decision_function: lower = more anomalous)
anomaly_scores = np.full(len(df_score_pd), 0.0)  # dormant rows get 0
anomaly_predictions = np.full(len(df_score_pd), 1)  # dormant = normal

if len(X_anomaly) > 0:
    scores_active = anomaly_model.decision_function(X_anomaly)
    preds_active = anomaly_model.predict(X_anomaly)
    anomaly_scores[active_mask] = scores_active
    anomaly_predictions[active_mask] = preds_active

df_score_pd["anomaly_score"] = anomaly_scores
df_score_pd["anomaly_flag"] = (anomaly_predictions == -1).astype(int)

# Include rules flags as hard anomalies
df_score_pd["rules_flag"] = (
    df_score_pd["flag_identity_break"].astype(int) |
    df_score_pd["flag_sign_flip"].astype(int) |
    df_score_pd["flag_orphan_activity"].astype(int)
)

# Combined: rules OR IF flagged
df_score_pd["is_anomaly"] = ((df_score_pd["anomaly_flag"] == 1) | (df_score_pd["rules_flag"] == 1)).astype(int)

print(f"Anomalies flagged: {df_score_pd['is_anomaly'].sum()} / {len(df_score_pd)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score: Activity Model

# COMMAND ----------

ACTIVITY_FEATURES = [
    "BeginningBalance", "Company", "Branch", "Account",
    "CostCenter", "Product", "Intercompany",
    "period_ordinal", "balance_quartile",
    "peer_group_mean_balance", "peer_group_std_balance",
    "peer_group_count", "historical_activity_rate",
    "abs_beginning_balance", "log_abs_balance", "is_zero_balance",
]

# Load champion activity model
activity_model = mlflow.lightgbm.load_model(ACTIVITY_MODEL_URI)

X_activity = df_score_pd[ACTIVITY_FEATURES]
activity_probs = activity_model.predict(X_activity)
activity_preds = (activity_probs >= 0.5).astype(int)

df_score_pd["activity_probability"] = activity_probs
df_score_pd["predicted_active"] = activity_preds

print(f"Predicted active: {activity_preds.sum()} / {len(activity_preds)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write anomaly scores to gold table

# COMMAND ----------

KEY_COLS = cfg["columns"]["key_cols"]

anomaly_output_cols = KEY_COLS + [
    "AccountingPeriod", "BeginningBalance", "PeriodActivity", "EndingBalance",
    "anomaly_score", "anomaly_flag", "rules_flag", "is_anomaly",
    "flag_identity_break", "flag_sign_flip", "flag_orphan_activity",
]

df_anomaly_out = spark.createDataFrame(df_score_pd[anomaly_output_cols])
df_anomaly_out = df_anomaly_out.withColumn("_scored_at", F.current_timestamp())

(
    df_anomaly_out.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(ANOMALY_SCORES_TABLE)
)

print(f"Anomaly scores written: {ANOMALY_SCORES_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write activity predictions to gold table

# COMMAND ----------

activity_output_cols = KEY_COLS + [
    "AccountingPeriod", "BeginningBalance",
    "activity_probability", "predicted_active", "label_had_activity",
]

df_activity_out = spark.createDataFrame(df_score_pd[activity_output_cols])
df_activity_out = df_activity_out.withColumn("_scored_at", F.current_timestamp())

(
    df_activity_out.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(ACTIVITY_PREDS_TABLE)
)

print(f"Activity predictions written: {ACTIVITY_PREDS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring Summary

# COMMAND ----------

print("=== BATCH SCORING COMPLETE ===")
print(f"Period scored: {df_score_pd['AccountingPeriod'].iloc[0]}")
print(f"Total rows scored: {len(df_score_pd)}")
print(f"Anomalies flagged: {df_score_pd['is_anomaly'].sum()}")
print(f"  - Rules flags: {df_score_pd['rules_flag'].sum()}")
print(f"  - IF flags: {df_score_pd['anomaly_flag'].sum()}")
print(f"Predicted active next period: {activity_preds.sum()}")
