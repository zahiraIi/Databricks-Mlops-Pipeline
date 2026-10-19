# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Clean & Identity QA (Rules Layer)
# MAGIC Hard checks first — cheap, explainable, finance-friendly.
# MAGIC 
# MAGIC Rules applied:
# MAGIC 1. **Identity invariant**: EndingBalance = BeginningBalance + PeriodActivity
# MAGIC 2. **Sign flip detection**: Unexpected sign changes with small relative activity
# MAGIC 3. **Orphan activity**: Non-zero activity where BeginBal=0 AND few peers have activity
# MAGIC 
# COMMAND ----------

import yaml
from pyspark.sql import functions as F, Window

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
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['bronze_table']}"
SILVER_QA_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['silver_qa_table']}"

IDENTITY_TOL = cfg["rules"]["identity_tolerance"]
ORPHAN_PEER_PCT = cfg["rules"]["orphan_activity"]["peer_activity_pct_threshold"]
MIN_PEER_SIZE = cfg["rules"]["orphan_activity"]["min_peer_group_size"]

print(f"Reading from: {BRONZE_TABLE}")
print(f"Writing to: {SILVER_QA_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load bronze data

# COMMAND ----------

df = spark.table(BRONZE_TABLE)
total_rows = df.count()
print(f"Total rows in bronze: {total_rows}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule 1: Identity Invariant Check
# MAGIC `EndingBalance = BeginningBalance + PeriodActivity` (within tolerance)

# COMMAND ----------

df_qa = df.withColumn(
    "identity_error",
    F.abs(F.col("EndingBalance") - (F.col("BeginningBalance") + F.col("PeriodActivity")))
).withColumn(
    "flag_identity_break",
    F.when(F.col("identity_error") > IDENTITY_TOL, True).otherwise(False)
)

identity_fails = df_qa.filter(F.col("flag_identity_break")).count()
print(f"Identity failures: {identity_fails} / {total_rows} ({identity_fails/total_rows*100:.4f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule 2: Sign Flip Detection
# MAGIC Flag if sign(EndBal) != sign(BegBal) AND |Activity| < |BegBal| * 0.5
# MAGIC (A flip caused by small activity relative to balance is suspicious)

# COMMAND ----------

df_qa = df_qa.withColumn(
    "flag_sign_flip",
    F.when(
        (F.col("BeginningBalance") != 0) &
        (F.signum(F.col("EndingBalance")) != F.signum(F.col("BeginningBalance"))) &
        (F.abs(F.col("PeriodActivity")) < F.abs(F.col("BeginningBalance")) * 0.5),
        True
    ).otherwise(False)
)

sign_flips = df_qa.filter(F.col("flag_sign_flip")).count()
print(f"Suspicious sign flips: {sign_flips}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule 3: Orphan Activity Detection
# MAGIC Non-zero activity where BeginningBalance = 0 AND < 5% of the peer group
# MAGIC (same Account) has activity. Peer group = accounts with same Account code.

# COMMAND ----------

# Define peer group window (by Account code within the same period)
peer_window = Window.partitionBy("AccountingPeriod", "Account")

df_qa = df_qa.withColumn(
    "has_activity", 
    F.when(F.col("PeriodActivity") != 0, 1).otherwise(0)
).withColumn(
    "peer_count", F.count("*").over(peer_window)
).withColumn(
    "peer_active_count", F.sum("has_activity").over(peer_window)
).withColumn(
    "peer_active_pct", F.col("peer_active_count") / F.col("peer_count")
).withColumn(
    "flag_orphan_activity",
    F.when(
        (F.col("BeginningBalance") == 0) &
        (F.col("PeriodActivity") != 0) &
        (F.col("peer_active_pct") < ORPHAN_PEER_PCT) &
        (F.col("peer_count") >= MIN_PEER_SIZE),
        True
    ).otherwise(False)
)

orphans = df_qa.filter(F.col("flag_orphan_activity")).count()
print(f"Orphan activity flags: {orphans}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combine flags and write silver QA table

# COMMAND ----------

df_qa = df_qa.withColumn(
    "qa_flag_count",
    F.col("flag_identity_break").cast("int") +
    F.col("flag_sign_flip").cast("int") +
    F.col("flag_orphan_activity").cast("int")
).withColumn(
    "qa_passed", F.col("qa_flag_count") == 0
).withColumn(
    "_qa_timestamp", F.current_timestamp()
)

# Select final columns
output_cols = [c for c in df.columns] + [
    "identity_error",
    "flag_identity_break",
    "flag_sign_flip",
    "flag_orphan_activity",
    "qa_flag_count",
    "qa_passed",
    "_qa_timestamp",
]

df_silver = df_qa.select(output_cols)

(
    df_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_QA_TABLE)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## QA Summary

# COMMAND ----------

print("=== QA SUMMARY ===")
print(f"Total rows:          {total_rows}")
print(f"Identity breaks:     {identity_fails}")
print(f"Sign flips:          {sign_flips}")
print(f"Orphan activity:     {orphans}")
print(f"Rows passing QA:     {df_silver.filter(F.col('qa_passed')).count()}")
print(f"Rows with any flag:  {df_silver.filter(~F.col('qa_passed')).count()}")

# Fail the pipeline if identity failure rate exceeds champion gate threshold
fail_rate = identity_fails / total_rows
assert fail_rate < cfg["champion_gate"]["identity_max_fail_rate"], \
    f"PIPELINE HALT: Identity fail rate {fail_rate:.6f} exceeds threshold {cfg['champion_gate']['identity_max_fail_rate']}"

print("\n✓ Identity QA gate PASSED — proceeding to feature engineering.")
