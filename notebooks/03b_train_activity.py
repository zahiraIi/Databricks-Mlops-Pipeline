# Databricks notebook source
# MAGIC %md
# MAGIC # 03b — Train Activity Prediction Model (LightGBM)
# MAGIC 
# MAGIC Binary classifier: will a given account have non-zero PeriodActivity next period?
# MAGIC 
# MAGIC Key design decisions:
# MAGIC - Split by AccountingPeriod (temporal), never random
# MAGIC - Features: BeginningBalance + dims ONLY (no PeriodDebit/Credit/Activity/Ending = leakage)
# MAGIC - Baseline: "always inactive" (97.6% accuracy) — must beat this
# MAGIC - scale_pos_weight handles the 40:1 class imbalance
# MAGIC 
# COMMAND ----------

import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc, 
    classification_report, accuracy_score, recall_score
)

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
MODEL_CFG = cfg["activity_model"]

mlflow.set_registry_uri("databricks-uc")

print(f"Feature table: {FEATURE_TABLE}")
print(f"Model: {MODEL_CFG['name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load features (leakage-safe subset only)

# COMMAND ----------

from pyspark.sql import functions as F

df_spark = spark.table(FEATURE_TABLE)

# Activity model features — strictly NO period activity/debit/credit/ending
ACTIVITY_FEATURES = [
    "BeginningBalance",
    "Company",
    "Branch",
    "Account",
    "CostCenter",
    "Product",
    "Intercompany",
    "period_ordinal",
    "balance_quartile",
    "peer_group_mean_balance",
    "peer_group_std_balance",
    "peer_group_count",
    "historical_activity_rate",
    "abs_beginning_balance",
    "log_abs_balance",
    "is_zero_balance",
]

LABEL_COL = "label_had_activity"
SPLIT_COL = "split"

# Select only what we need
select_cols = ACTIVITY_FEATURES + [LABEL_COL, SPLIT_COL]
df_pd = df_spark.select(select_cols).toPandas()

print(f"Total rows: {len(df_pd)}")
print(f"Features: {len(ACTIVITY_FEATURES)}")
print(f"\nLabel distribution:")
print(df_pd[LABEL_COL].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Temporal split (train on period 4, validate on 5, test on 6)

# COMMAND ----------

df_train = df_pd[df_pd[SPLIT_COL] == "train"]
df_val = df_pd[df_pd[SPLIT_COL] == "validate"]
df_test = df_pd[df_pd[SPLIT_COL] == "test"]

X_train, y_train = df_train[ACTIVITY_FEATURES], df_train[LABEL_COL]
X_val, y_val = df_val[ACTIVITY_FEATURES], df_val[LABEL_COL]
X_test, y_test = df_test[ACTIVITY_FEATURES], df_test[LABEL_COL]

print(f"Train: {len(X_train)} rows, {y_train.mean()*100:.1f}% positive")
print(f"Val:   {len(X_val)} rows, {y_val.mean()*100:.1f}% positive")
print(f"Test:  {len(X_test)} rows, {y_test.mean()*100:.1f}% positive")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train LightGBM with early stopping on validation

# COMMAND ----------

with mlflow.start_run(run_name="activity_lightgbm") as run:
    
    # LightGBM datasets
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    params = {
        "objective": MODEL_CFG["params"]["objective"],
        "metric": MODEL_CFG["params"]["metric"],
        "max_depth": MODEL_CFG["params"]["max_depth"],
        "learning_rate": MODEL_CFG["params"]["learning_rate"],
        "scale_pos_weight": MODEL_CFG["params"]["scale_pos_weight"],
        "num_leaves": MODEL_CFG["params"]["num_leaves"],
        "min_child_samples": MODEL_CFG["params"]["min_child_samples"],
        "seed": MODEL_CFG["params"]["random_state"],
        "verbose": -1,
    }
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=MODEL_CFG["params"]["n_estimators"],
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50),
        ],
    )
    
    # --- Evaluate on TEST set ---
    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Metrics
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Baseline: always predict inactive
    baseline_accuracy = 1 - y_test.mean()
    baseline_f1 = f1_score(y_test, np.zeros_like(y_test), zero_division=0)
    
    # Log
    mlflow.log_params(params)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_recall_minority", test_recall)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_pr_auc", pr_auc)
    mlflow.log_metric("baseline_accuracy", baseline_accuracy)
    mlflow.log_metric("baseline_f1", baseline_f1)
    mlflow.log_metric("f1_lift_over_baseline", test_f1 - baseline_f1)
    mlflow.log_metric("best_iteration", model.best_iteration)
    
    print("=== TEST RESULTS ===")
    print(f"Model F1:      {test_f1:.4f}")
    print(f"Baseline F1:   {baseline_f1:.4f}")
    print(f"F1 Lift:       {test_f1 - baseline_f1:.4f}")
    print(f"PR-AUC:        {pr_auc:.4f}")
    print(f"Recall (pos):  {test_recall:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['inactive', 'active'])}")
    
    activity_run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance

# COMMAND ----------

import matplotlib.pyplot as plt

importance = pd.DataFrame({
    "feature": ACTIVITY_FEATURES,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)

print("Top 10 features by gain:")
print(importance.head(10).to_string(index=False))

with mlflow.start_run(run_id=activity_run_id):
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.head(15).plot.barh(x="feature", y="importance", ax=ax)
    ax.set_title("LightGBM Feature Importance (Gain)")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register model to Unity Catalog

# COMMAND ----------

with mlflow.start_run(run_id=activity_run_id):
    mlflow.lightgbm.log_model(
        model,
        artifact_path="activity_model",
        registered_model_name=f"{CATALOG}.{SCHEMA}.{cfg['registry']['activity_model_name']}",
    )
    
    print(f"Model registered: {cfg['registry']['activity_model_name']}")
    print(f"Run ID: {activity_run_id}")
