# Databricks notebook source
# MAGIC %md
# MAGIC # 13 - Batch Score Candidate Locations (Phase 2)
# MAGIC
# MAGIC Scores all ~5,000 candidate locations using the trained model.
# MAGIC Computes SHAP explanations, ranks by predicted sales, and assigns score tiers.
# MAGIC
# MAGIC **Input**: `gold.location_features` (candidates) + registered model
# MAGIC **Output**: `gold.scored_locations`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import mlflow
import xgboost as xgb
import shap
import json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Candidate Features

# COMMAND ----------

features_df = spark.table(f"{GOLD}.location_features").filter(F.col("site_type") == "candidate")
candidate_count = features_df.count()
print(f"Candidate locations to score: {candidate_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Model & Feature Columns

# COMMAND ----------

# Load the registered XGBoost model (not the pyfunc — we need direct access for batch SHAP)
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
latest_version = max(model_versions, key=lambda v: int(v.version))
model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version.version}"

model = mlflow.xgboost.load_model(model_uri)
print(f"Loaded model: {REGISTERED_MODEL_NAME} v{latest_version.version}")

# Load feature columns
feature_cols_df = spark.table(f"{GOLD}.model_feature_columns").orderBy("feature_index").toPandas()
feature_cols = feature_cols_df["feature_name"].tolist()
print(f"Feature columns: {len(feature_cols)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prepare Features for Scoring

# COMMAND ----------

# Convert to pandas for scoring
candidates_pdf = features_df.toPandas()

# Store identifiers for later
site_ids = candidates_pdf["site_id"].values
h3_res8s = candidates_pdf["h3_res8"].values
metros = candidates_pdf["metro"].values
lats = candidates_pdf["latitude"].values
lons = candidates_pdf["longitude"].values

# One-hot encode categoricals to match training
candidates_encoded = pd.get_dummies(
    candidates_pdf, columns=["property_type", "metro"], prefix=["prop", "metro"], dtype=float
)

# Align columns with training features (add missing, drop extra)
X = pd.DataFrame(0, index=range(len(candidates_encoded)), columns=feature_cols)
for col in feature_cols:
    if col in candidates_encoded.columns:
        X[col] = candidates_encoded[col].values

# Defensive: clean columns that may have array-like string values
for col in feature_cols:
    if col in X.columns and X[col].dtype == object:
        X[col] = pd.to_numeric(X[col].astype(str).str.strip("[]"), errors="coerce")

X = X.astype(float).fillna(0.0)
print(f"Scoring matrix: {X.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Score All Candidates

# COMMAND ----------

predictions = model.predict(X)

print(f"Scored {len(predictions)} candidates")
print(f"Predicted sales range: ${predictions.min():,.0f} - ${predictions.max():,.0f}")
print(f"Predicted sales mean: ${predictions.mean():,.0f}")
print(f"Predicted sales median: ${np.median(predictions):,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compute SHAP Values for All Candidates

# COMMAND ----------

print("Computing SHAP values (this may take a minute)...")
_shap_ok = False
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Extract top-5 SHAP features per candidate
    top5_shap_list = []
    for i in range(len(predictions)):
        pairs = sorted(zip(feature_cols, shap_values[i]), key=lambda x: abs(x[1]), reverse=True)[:5]
        top5_shap_list.append(json.dumps({k: round(float(v), 2) for k, v in pairs}))

    _shap_base = round(float(explainer.expected_value), 2)
    _shap_ok = True
    print(f"SHAP computed for {len(top5_shap_list)} candidates")
except Exception as e:
    print(f"WARNING: SHAP failed ({type(e).__name__}: {e}). Using XGBoost feature importance as fallback.")
    # Fallback: use XGBoost native feature importance
    importance = model.get_booster().get_score(importance_type="gain")
    top5_shap_list = []
    for i in range(len(predictions)):
        top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_shap_list.append(json.dumps({k: round(float(v), 2) for k, v in top5}))
    _shap_base = float(predictions.mean())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Build Scored Results with Tiers

# COMMAND ----------

# Create pandas result
results_pdf = pd.DataFrame({
    "site_id": site_ids,
    "h3_res8": h3_res8s,
    "metro": metros,
    "latitude": lats,
    "longitude": lons,
    "predicted_annual_sales": np.round(predictions, 2),
    "shap_base_value": _shap_base,
    "shap_top5": top5_shap_list,
})

# Compute percentile rank (higher predicted sales = higher percentile)
results_pdf["percentile_rank"] = results_pdf["predicted_annual_sales"].rank(pct=True).round(4)

# Assign tiers
def assign_tier(pct):
    if pct >= 0.90:
        return "A"
    elif pct >= 0.70:
        return "B"
    elif pct >= 0.40:
        return "C"
    else:
        return "D"

results_pdf["score_tier"] = results_pdf["percentile_rank"].apply(assign_tier)

# Summary by tier
print("Score Tier Distribution:")
print("=" * 45)
for tier in ["A", "B", "C", "D"]:
    tier_data = results_pdf[results_pdf["score_tier"] == tier]
    print(f"  Tier {tier}: {len(tier_data):>5,} sites | Avg predicted: ${tier_data['predicted_annual_sales'].mean():>12,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Top Sites Per Metro

# COMMAND ----------

print("\nTop 5 Sites Per Metro:")
print("=" * 70)
for metro in sorted(results_pdf["metro"].unique()):
    metro_top = results_pdf[results_pdf["metro"] == metro].nlargest(5, "predicted_annual_sales")
    print(f"\n{metro}:")
    for _, row in metro_top.iterrows():
        shap_dict = json.loads(row["shap_top5"])
        top_driver = list(shap_dict.keys())[0] if shap_dict else "N/A"
        print(f"  {row['site_id']:20s} ${row['predicted_annual_sales']:>12,.0f}  (Tier {row['score_tier']})  Top driver: {top_driver}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sanity Check: Top-Scored Sites Should Have Good Fundamentals

# COMMAND ----------

# Join top sites back to features to verify
top_50 = results_pdf.nlargest(50, "predicted_annual_sales")
top_50_ids = top_50["site_id"].tolist()

top_features = features_df.filter(F.col("site_id").isin(top_50_ids))
bottom_50_ids = results_pdf.nsmallest(50, "predicted_annual_sales")["site_id"].tolist()
bottom_features = features_df.filter(F.col("site_id").isin(bottom_50_ids))

print("Top 50 vs Bottom 50 Sites — Key Feature Averages:")
print("=" * 60)
for feat in ["population_1ring", "median_income_1ring", "max_daily_traffic_1ring", "competitor_count_1ring"]:
    if feat in [f.name for f in features_df.schema.fields]:
        top_avg = top_features.agg(F.avg(feat)).collect()[0][0]
        bot_avg = bottom_features.agg(F.avg(feat)).collect()[0][0]
        print(f"  {feat:35s}  Top50: {top_avg:>12,.1f}  Bot50: {bot_avg:>12,.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Write to Gold

# COMMAND ----------

scored_df = spark.createDataFrame(results_pdf)

table_name = f"{GOLD}.scored_locations"
scored_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"Saved {scored_df.count()} scored locations to {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {table_name} ORDER BY predicted_annual_sales DESC LIMIT 20"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Batch scored all candidate locations:
# MAGIC - **Candidates scored**: ~5,000
# MAGIC - **SHAP explanations**: Top-5 features per site
# MAGIC - **Tiers**: A (top 10%), B (10-30%), C (30-60%), D (bottom 40%)
# MAGIC - **Output**: `gold.scored_locations`
