# Databricks-Mlops-Pipeline

CI/CD MLOps pipeline for synthetic general ledger trial balance anomaly detection and account activity prediction.

## Architecture

```
ingest → clean/identity QA → feature table
   ├─ train_anomaly (IF + rules) → evaluate → register
   └─ train_activity (LightGBM) → evaluate → register
→ batch_score (both) → monitor metrics → retrain trigger
```

## Project Structure

```
mlops-demo/
├── config/
│   └── pipeline_config.yml       # Central configuration
├── notebooks/
│   ├── 00_ingest.py              # CSV → Delta bronze
│   ├── 01_clean_identity_qa.py   # Rules layer (hard checks)
│   ├── 02_feature_engineering.py # Feature table (leakage-safe)
│   ├── 03a_train_anomaly.py      # Isolation Forest + rules
│   ├── 03b_train_activity.py     # LightGBM binary classifier
│   ├── 04_evaluate_and_register.py  # Champion gate → UC registry
│   ├── 05_batch_score.py         # Score both → gold Delta
│   └── 06_monitor_retrain.py     # PSI drift + retrain trigger
├── tests/
│   ├── test_identity_invariants.py
│   └── test_feature_leakage.py
├── resources/
│   ├── job_train.yml             # Training job (multi-task DAG)
│   └── job_score.yml             # Scoring job (scheduled)
├── databricks.yml                # DAB bundle root
├── requirements.txt
└── README.md
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Rules layer first | Hard checks (identity, sign flips, orphans) are cheap, explainable, finance-friendly |
| Split by AccountingPeriod | Temporal integrity — never random row shuffle on financial time-series |
| No leakage in activity model | Features use only BeginningBalance + dims; PeriodDebit/Credit/Activity/Ending are excluded |
| Proxy labels for IF eval | Isolation Forest is unsupervised — evaluate via rules flags, top-5% extremes, spiked synthetics |
| Batch scoring to Delta | Trial balance anomaly review = table + dashboard, not a low-latency API |
| Champion gate | Promote only if: identity fail ≈ 0, activity beats baseline, anomaly precision@k acceptable |

## Delta Tables (`main.mlops_demo`)

| Table | Layer | Purpose |
|-------|-------|---------|
| `tb_bronze` | Bronze | Raw ingested data |
| `tb_silver_qa` | Silver | QA-flagged data with rules results |
| `tb_features` | Silver | Engineered features + labels + split |
| `tb_anomaly_scores` | Gold | Anomaly model output |
| `tb_activity_predictions` | Gold | Activity model output |
| `tb_monitor_metrics` | Gold | PSI, drift, staleness tracking |

## UC Models

| Model | Type | Registry Path |
|-------|------|---------------|
| Anomaly | IsolationForest pipeline | `main.mlops_demo` |
| Activity | LightGBM | `main.mlops_demo` |

## CI/CD (Declarative Automation Bundle)

```bash
# Deploy to dev
databricks bundle deploy --target dev

# Run training pipeline
databricks bundle run tb_train --target dev

# Deploy to prod (after PR merge + approval)
databricks bundle deploy --target prod
```

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Data Source

Synthetic 5,000-row dataset modeled on a general ledger trial balance:
- 3 periods (Apr–Jun 2026)
- 19 columns, same schema as a general ledger trial balance
- Integrity enforced: `EndingBalance = BeginningBalance + PeriodActivity`
- Scales to large tables

## Monitoring & Retraining

- **PSI > 0.20** on key features → retrain triggered
- **Prediction shift > 15%** in positive rate → retrain triggered  
- **Model staleness > 3 periods** → retrain triggered
- Monitor metrics appended to Delta for historical tracking
