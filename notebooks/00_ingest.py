# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Ingest: CSV → Delta Bronze
# MAGIC Reads the synthetic trial balance CSV and writes to a Delta bronze table.
# MAGIC Append-only design; partitioned by AccountingPeriod.
# MAGIC 
# COMMAND ----------

import yaml
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Load config
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
SOURCE_CSV = cfg["data"]["source_csv"]
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.{cfg['data']['bronze_table']}"

print(f"Source: {SOURCE_CSV}")
print(f"Target: {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read CSV with explicit schema

# COMMAND ----------

schema = StructType([
    StructField("AccountingPeriod", StringType(), True),
    StructField("Ledger", StringType(), True),
    StructField("Company", IntegerType(), True),
    StructField("Branch", IntegerType(), True),
    StructField("Account", IntegerType(), True),
    StructField("CostCenter", IntegerType(), True),
    StructField("Product", IntegerType(), True),
    StructField("Intercompany", IntegerType(), True),
    StructField("BeginningBalance", DoubleType(), True),
    StructField("PeriodDebit", DoubleType(), True),
    StructField("PeriodCredit", DoubleType(), True),
    StructField("PeriodActivity", DoubleType(), True),
    StructField("EndingBalance", DoubleType(), True),
    StructField("CurrencyCode", StringType(), True),
    StructField("LoadDate", StringType(), True),
    StructField("LoadEndDate", StringType(), True),
    StructField("CurrentRecordFlag", IntegerType(), True),
    StructField("FileName", StringType(), True),
    StructField("FileDate", StringType(), True),
])

df_raw = (
    spark.read
    .option("header", "true")
    .schema(schema)
    .csv(SOURCE_CSV)
)

print(f"Rows read: {df_raw.count()}")
df_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add ingestion metadata and write to bronze

# COMMAND ----------

df_bronze = (
    df_raw
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.lit(SOURCE_CSV))
)

# Create or append to bronze table
(
    df_bronze.write
    .format("delta")
    .mode("overwrite")  # First load; switch to append for incremental
    .partitionBy("AccountingPeriod")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE)
)

print(f"Bronze table written: {BRONZE_TABLE}")
spark.sql(f"SELECT AccountingPeriod, COUNT(*) as row_count FROM {BRONZE_TABLE} GROUP BY 1 ORDER BY 1").show()
