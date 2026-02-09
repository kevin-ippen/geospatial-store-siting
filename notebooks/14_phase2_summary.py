# Databricks notebook source
# MAGIC %md
# MAGIC # 14 - Phase 2 Summary & Validation (Phase 2)
# MAGIC
# MAGIC Comprehensive validation of all Phase 2 outputs: feature table, model, scored locations, endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

client = MlflowClient()
all_passed = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Feature Table Validation

# COMMAND ----------

print("=" * 60)
print("FEATURE TABLE: gold.location_features")
print("=" * 60)

features_df = spark.table(f"{GOLD}.location_features")
total = features_df.count()
existing = features_df.filter(F.col("site_type") == "existing").count()
candidates = features_df.filter(F.col("site_type") == "candidate").count()

print(f"  Total rows:      {total:,}")
print(f"  Existing stores: {existing:,}")
print(f"  Candidates:      {candidates:,}")

if total < 5000:
    print(f"  WARN: Expected ~5,350 rows, got {total}")
    all_passed = False
else:
    print(f"  PASS: Row count OK")

# Null check on numeric features
null_cols = []
for col_name in NUMERIC_FEATURES:
    if col_name in features_df.columns:
        nulls = features_df.filter(F.col(col_name).isNull()).count()
        if nulls > 0:
            null_cols.append(f"{col_name}({nulls})")

if null_cols:
    print(f"  WARN: Nulls in: {', '.join(null_cols)}")
else:
    print(f"  PASS: No nulls in numeric features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model Validation

# COMMAND ----------

print("\n" + "=" * 60)
print(f"MODEL: {REGISTERED_MODEL_NAME}")
print("=" * 60)

try:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    print(f"  Model version:   {latest.version}")
    print(f"  Status:          {latest.status}")
    print(f"  Run ID:          {latest.run_id}")

    # Get metrics from the training run
    run = client.get_run(latest.run_id)
    metrics = run.data.metrics
    print(f"  Test RMSE:       ${metrics.get('test_rmse', 0):,.0f}")
    print(f"  Test MAE:        ${metrics.get('test_mae', 0):,.0f}")
    print(f"  Test R2:         {metrics.get('test_r2', 0):.4f}")
    print(f"  Test MAPE:       {metrics.get('test_mape', 0):.2%}")

    r2 = metrics.get("test_r2", 0)
    mape = metrics.get("test_mape", 1)
    if r2 >= MIN_R2:
        print(f"  PASS: R2 ({r2:.4f}) >= {MIN_R2}")
    else:
        print(f"  FAIL: R2 ({r2:.4f}) < {MIN_R2}")
        all_passed = False

    if mape <= MAX_MAPE:
        print(f"  PASS: MAPE ({mape:.2%}) <= {MAX_MAPE:.0%}")
    else:
        print(f"  FAIL: MAPE ({mape:.2%}) > {MAX_MAPE:.0%}")
        all_passed = False

except Exception as e:
    print(f"  FAIL: Could not load model — {e}")
    all_passed = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Scored Locations Validation

# COMMAND ----------

print("\n" + "=" * 60)
print("SCORED LOCATIONS: gold.scored_locations")
print("=" * 60)

try:
    scored_df = spark.table(f"{GOLD}.scored_locations")
    scored_count = scored_df.count()
    print(f"  Total scored:    {scored_count:,}")

    # Tier distribution
    tier_dist = scored_df.groupBy("score_tier").agg(
        F.count("*").alias("count"),
        F.avg("predicted_annual_sales").alias("avg_sales"),
    ).orderBy("score_tier").collect()

    print(f"\n  Tier Distribution:")
    for row in tier_dist:
        print(f"    Tier {row.score_tier}: {row['count']:>5,} sites | Avg: ${row.avg_sales:>12,.0f}")

    # Metro distribution
    metro_dist = scored_df.groupBy("metro").agg(
        F.count("*").alias("count"),
        F.countDistinct(F.when(F.col("score_tier") == "A", F.col("site_id"))).alias("tier_a"),
    ).orderBy("metro").collect()

    print(f"\n  Metro Distribution:")
    for row in metro_dist:
        print(f"    {row.metro:12s}: {row['count']:>5,} sites | Tier A: {row.tier_a:>4,}")

    # Prediction range check
    stats = scored_df.agg(
        F.min("predicted_annual_sales").alias("min_pred"),
        F.max("predicted_annual_sales").alias("max_pred"),
        F.stddev("predicted_annual_sales").alias("std_pred"),
    ).collect()[0]

    print(f"\n  Prediction range: ${stats.min_pred:,.0f} - ${stats.max_pred:,.0f}")
    print(f"  Prediction std:   ${stats.std_pred:,.0f}")

    if stats.std_pred < 10000:
        print(f"  WARN: Very low variance — model may not be discriminating well")
    else:
        print(f"  PASS: Predictions show healthy variance")

    if scored_count < 4000:
        print(f"  WARN: Expected ~5,000 scored, got {scored_count}")
        all_passed = False
    else:
        print(f"  PASS: Scored count OK")

except Exception as e:
    print(f"  FAIL: Could not load scored locations — {e}")
    all_passed = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Serving Endpoint Status

# COMMAND ----------

print("\n" + "=" * 60)
print(f"ENDPOINT: {ENDPOINT_NAME}")
print("=" * 60)

try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
    state = endpoint.state.ready
    print(f"  State:           {state}")

    if str(state) == "READY":
        print(f"  PASS: Endpoint is ready")
    else:
        print(f"  WARN: Endpoint not ready — state: {state}")

except Exception as e:
    print(f"  WARN: Could not check endpoint — {e}")
    print(f"  (This is OK if endpoint deployment is still in progress)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature Importance (Top 10)

# COMMAND ----------

print("\n" + "=" * 60)
print("TOP 10 FEATURES BY SHAP IMPORTANCE")
print("=" * 60)

try:
    run = client.get_run(latest.run_id)
    # Download SHAP importance artifact
    local_path = mlflow.artifacts.download_artifacts(f"runs:/{latest.run_id}/shap_importance.csv")
    import pandas as pd
    shap_imp = pd.read_csv(local_path).head(10)
    for _, row in shap_imp.iterrows():
        bar = "#" * int(row.mean_abs_shap / shap_imp.mean_abs_shap.max() * 30)
        print(f"  {row.feature:35s} ${row.mean_abs_shap:>10,.0f}  {bar}")
except Exception as e:
    print(f"  Could not load SHAP importance: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Overall Result

# COMMAND ----------

print("\n" + "=" * 60)
if all_passed:
    print("PHASE 2 VALIDATION: ALL CHECKS PASSED")
    print("\nReady for Phase 3 (Databricks App)!")
    print(f"\nGold tables available:")
    print(f"  - {GOLD}.location_features ({total:,} rows)")
    print(f"  - {GOLD}.scored_locations ({scored_count:,} rows)")
    print(f"  - {GOLD}.model_feature_columns ({len(NUMERIC_FEATURES)} features)")
    print(f"\nModel: {REGISTERED_MODEL_NAME}")
    print(f"Endpoint: {ENDPOINT_NAME}")
else:
    print("PHASE 2 VALIDATION: SOME CHECKS NEED ATTENTION")
    print("Review warnings above before proceeding to Phase 3.")
print("=" * 60)
