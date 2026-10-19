# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Evaluate & Champion Gate
# MAGIC 
# MAGIC Promotion criteria (ALL must pass):
# MAGIC 1. Identity QA fail rate ≈ 0
# MAGIC 2. Activity model beats "always inactive" baseline
# MAGIC 3. Anomaly model precision@k on review set is acceptable
# MAGIC 
# MAGIC Only promotes models to "Champion" alias if all gates pass.
# MAGIC 
# COMMAND ----------

import yaml
import mlflow
from mlflow.tracking import MlflowClient

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
GATE = cfg["champion_gate"]

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

ANOMALY_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{cfg['registry']['anomaly_model_name']}"
ACTIVITY_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{cfg['registry']['activity_model_name']}"

print("=== CHAMPION GATE EVALUATION ===")
print(f"Anomaly model: {ANOMALY_MODEL_NAME}")
print(f"Activity model: {ACTIVITY_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate 1: Identity QA Fail Rate ≈ 0

# COMMAND ----------

from pyspark.sql import functions as F

silver_table = f"{CATALOG}.{SCHEMA}.{cfg['data']['silver_qa_table']}"
df_qa = spark.table(silver_table)

total_rows = df_qa.count()
identity_fails = df_qa.filter(F.col("flag_identity_break")).count()
fail_rate = identity_fails / total_rows

gate1_passed = fail_rate <= GATE["identity_max_fail_rate"]

print(f"Gate 1 - Identity QA:")
print(f"  Fail rate: {fail_rate:.6f} (threshold: {GATE['identity_max_fail_rate']})")
print(f"  Result: {'PASS' if gate1_passed else 'FAIL'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate 2: Activity Model Beats Baseline

# COMMAND ----------

# Get latest version of activity model
activity_versions = client.search_model_versions(f"name='{ACTIVITY_MODEL_NAME}'")
if not activity_versions:
    raise ValueError(f"No model versions found for {ACTIVITY_MODEL_NAME}")

latest_activity = sorted(activity_versions, key=lambda v: int(v.version), reverse=True)[0]
activity_run = client.get_run(latest_activity.run_id)

activity_f1 = activity_run.data.metrics.get("test_f1", 0)
baseline_f1 = activity_run.data.metrics.get("baseline_f1", 0)
pr_auc = activity_run.data.metrics.get("test_pr_auc", 0)
recall_minority = activity_run.data.metrics.get("test_recall_minority", 0)

gate2_f1 = (activity_f1 - baseline_f1) >= cfg["activity_model"]["evaluation"]["min_f1_above_baseline"]
gate2_prauc = pr_auc >= cfg["activity_model"]["evaluation"]["min_pr_auc"]
gate2_recall = recall_minority >= cfg["activity_model"]["evaluation"]["min_recall_minority"]
gate2_passed = gate2_f1 and gate2_prauc and gate2_recall

print(f"Gate 2 - Activity Model:")
print(f"  F1: {activity_f1:.4f} (baseline: {baseline_f1:.4f}, lift: {activity_f1 - baseline_f1:.4f}, min lift: {cfg['activity_model']['evaluation']['min_f1_above_baseline']})")
print(f"  PR-AUC: {pr_auc:.4f} (min: {cfg['activity_model']['evaluation']['min_pr_auc']})")
print(f"  Recall minority: {recall_minority:.4f} (min: {cfg['activity_model']['evaluation']['min_recall_minority']})")
print(f"  Result: {'PASS' if gate2_passed else 'FAIL'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate 3: Anomaly Model Precision@K

# COMMAND ----------

anomaly_versions = client.search_model_versions(f"name='{ANOMALY_MODEL_NAME}'")
if not anomaly_versions:
    raise ValueError(f"No model versions found for {ANOMALY_MODEL_NAME}")

latest_anomaly = sorted(anomaly_versions, key=lambda v: int(v.version), reverse=True)[0]
anomaly_run = client.get_run(latest_anomaly.run_id)

precision_at_50 = anomaly_run.data.metrics.get("precision_at_50", 0)
min_precision = GATE["anomaly_min_precision_at_50"]

gate3_passed = precision_at_50 >= min_precision

print(f"Gate 3 - Anomaly Model:")
print(f"  Precision@50: {precision_at_50:.4f} (min: {min_precision})")
print(f"  Result: {'PASS' if gate3_passed else 'FAIL'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Gate Decision

# COMMAND ----------

all_passed = gate1_passed and gate2_passed and gate3_passed

print("\n" + "="*50)
print("CHAMPION GATE SUMMARY")
print("="*50)
print(f"  Gate 1 (Identity QA):    {'PASS' if gate1_passed else 'FAIL'}")
print(f"  Gate 2 (Activity model): {'PASS' if gate2_passed else 'FAIL'}")
print(f"  Gate 3 (Anomaly model):  {'PASS' if gate3_passed else 'FAIL'}")
print(f"  ---")
print(f"  OVERALL: {'PROMOTE TO CHAMPION' if all_passed else 'BLOCKED — DO NOT PROMOTE'}")
print("="*50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote or block

# COMMAND ----------

if all_passed:
    # Set alias "Champion" on both models
    client.set_registered_model_alias(ANOMALY_MODEL_NAME, "Champion", latest_anomaly.version)
    client.set_registered_model_alias(ACTIVITY_MODEL_NAME, "Champion", latest_activity.version)
    
    print(f"\n✓ Promoted {ANOMALY_MODEL_NAME} v{latest_anomaly.version} → Champion")
    print(f"✓ Promoted {ACTIVITY_MODEL_NAME} v{latest_activity.version} → Champion")
else:
    print("\n✗ Models NOT promoted. Review gate failures above.")
    print("  Previous Champion (if any) remains active for scoring.")
    
    # Do NOT raise — allow pipeline to continue for debugging
    # but set a widget/task value that downstream can check
    dbutils.jobs.taskValues.set(key="champion_gate_passed", value=False)
