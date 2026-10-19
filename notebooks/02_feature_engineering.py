# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Feature Engineering
# MAGIC 
# MAGIC Produces TWO feature sets:
# MAGIC 1. **Activity model features** — leakage-safe (NO PeriodDebit/Credit/Activity/Ending)
# MAGIC 2. **Anomaly model signals** — uses activity/ending as the signal being scored
# MAGIC 
# MAGIC Split: by AccountingPeriod (temporal), never random.
# MAGIC 
# COMMAND ----------

import yaml
from pyspark.sql import functions as F, Window
from pyspark.sql.types import IntegerType

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
SILVER_QA_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['silver_qa_table']}"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['feature_table']}"

SPLIT_CFG = cfg["split"]
KEY_COLS = cfg["columns"]["key_cols"]

print(f"Reading from: {SILVER_QA_TABLE}")
print(f"Writing to: {FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load silver data (QA-passed rows)

# COMMAND ----------

df = spark.table(SILVER_QA_TABLE).filter(F.col("qa_passed"))
print(f"QA-passed rows: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Period ordinal (temporal encoding)

# COMMAND ----------

# Map period strings to ordinals for temporal features
period_order = {
    SPLIT_CFG["train_period"]: 1,
    SPLIT_CFG["validate_period"]: 2,
    SPLIT_CFG["test_period"]: 3,
}

period_map_expr = F.create_map([F.lit(x) for pair in period_order.items() for x in pair])

df = df.withColumn("period_ordinal", period_map_expr[F.col("AccountingPeriod")])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Activity model features (LEAKAGE-SAFE)
# MAGIC 
# MAGIC Uses ONLY: BeginningBalance + dimension codes + engineered features.
# MAGIC Does NOT use: PeriodDebit, PeriodCredit, PeriodActivity, EndingBalance.

# COMMAND ----------

# --- Balance quartile (within-period ranking) ---
period_window = Window.partitionBy("AccountingPeriod")

df = df.withColumn(
    "balance_quartile",
    F.ntile(4).over(period_window.orderBy("BeginningBalance"))
)

# --- Peer group stats (same Account code) ---
peer_window = Window.partitionBy("AccountingPeriod", "Account")

df = df.withColumn(
    "peer_group_mean_balance",
    F.avg("BeginningBalance").over(peer_window)
).withColumn(
    "peer_group_std_balance",
    F.stddev("BeginningBalance").over(peer_window)
).withColumn(
    "peer_group_count",
    F.count("*").over(peer_window)
)

# --- Historical activity rate (how often does this account combo have activity?) ---
# For train period, this is 0 (no history). For later periods, use prior info.
account_window = Window.partitionBy(*KEY_COLS).orderBy("period_ordinal").rowsBetween(
    Window.unboundedPreceding, -1
)

df = df.withColumn(
    "has_activity_flag",
    F.when(F.col("PeriodActivity") != 0, 1).otherwise(0)
).withColumn(
    "historical_activity_rate",
    F.coalesce(F.avg("has_activity_flag").over(account_window), F.lit(0.0))
)

# --- Balance magnitude features ---
df = df.withColumn(
    "abs_beginning_balance", F.abs(F.col("BeginningBalance"))
).withColumn(
    "log_abs_balance",
    F.when(F.abs(F.col("BeginningBalance")) > 0, 
            F.log1p(F.abs(F.col("BeginningBalance"))))
    .otherwise(0.0)
).withColumn(
    "is_zero_balance",
    F.when(F.col("BeginningBalance") == 0, 1).otherwise(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Anomaly model signals (NOT features — these are what we score)

# COMMAND ----------

# These are the signals the anomaly model evaluates (it flags unusual values here)
df = df.withColumn(
    "balance_change_ratio",
    F.when(F.col("BeginningBalance") != 0,
            F.col("PeriodActivity") / F.abs(F.col("BeginningBalance")))
    .otherwise(F.when(F.col("PeriodActivity") != 0, F.lit(999.0)).otherwise(0.0))
).withColumn(
    "peer_group_z_score",
    F.when(
        (F.col("peer_group_std_balance").isNotNull()) & (F.col("peer_group_std_balance") > 0),
        (F.col("EndingBalance") - F.col("peer_group_mean_balance")) / F.col("peer_group_std_balance")
    ).otherwise(0.0)
).withColumn(
    "activity_to_balance_ratio",
    F.when(F.col("BeginningBalance") != 0,
            F.abs(F.col("PeriodActivity")) / F.abs(F.col("BeginningBalance")))
    .otherwise(F.when(F.col("PeriodActivity") != 0, F.lit(999.0)).otherwise(0.0))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Label column for activity model

# COMMAND ----------

df = df.withColumn(
    "label_had_activity",
    F.when(F.col("PeriodActivity") != 0, 1).otherwise(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assign split column (temporal)

# COMMAND ----------

df = df.withColumn(
    "split",
    F.when(F.col("AccountingPeriod") == SPLIT_CFG["train_period"], "train")
     .when(F.col("AccountingPeriod") == SPLIT_CFG["validate_period"], "validate")
     .when(F.col("AccountingPeriod") == SPLIT_CFG["test_period"], "test")
     .otherwise("score")
)

df.groupBy("split").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write feature table

# COMMAND ----------

(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FEATURE_TABLE)
)

print(f"Feature table written: {FEATURE_TABLE}")
print(f"Total rows: {df.count()}")
print(f"Total columns: {len(df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leakage validation

# COMMAND ----------

# CRITICAL: Verify no leakage columns in activity model feature set
FORBIDDEN_IN_ACTIVITY = {"PeriodDebit", "PeriodCredit", "PeriodActivity", "EndingBalance",
                          "label_had_activity", "has_activity_flag"}

activity_features = set(cfg["columns"]["activity_model_features"])
leakage = activity_features.intersection(FORBIDDEN_IN_ACTIVITY)

assert len(leakage) == 0, f"LEAKAGE DETECTED: {leakage} in activity model features!"
print("✓ Leakage check PASSED — no forbidden columns in activity feature set.")
