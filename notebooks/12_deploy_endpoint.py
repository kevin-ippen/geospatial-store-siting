# Databricks notebook source
# MAGIC %md
# MAGIC # 12 - Deploy Model Serving Endpoint (Phase 2)
# MAGIC
# MAGIC Deploys the site scoring model as a custom pyfunc serving endpoint that returns
# MAGIC both predicted annual sales AND top-5 SHAP feature explanations per request.
# MAGIC
# MAGIC **Input**: Registered model `{catalog}.models.site_scoring`
# MAGIC **Output**: Serving endpoint `qsr-site-scoring`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import json
import time
from mlflow.models import infer_signature
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Define Custom PyFunc Wrapper
# MAGIC
# MAGIC Wraps XGBoost model + SHAP explainer into a single pyfunc that returns
# MAGIC predictions with explanations.

# COMMAND ----------

class SiteScoringPyfunc(mlflow.pyfunc.PythonModel):
    """Custom pyfunc that returns predictions with SHAP explanations."""

    def load_context(self, context):
        import xgboost as xgb

        # Load the XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(context.artifacts["xgb_model"])
        self.feature_names = self.model.feature_names

        # Try SHAP, fall back to native importance
        self._shap_ok = False
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            self._shap_ok = True
        except Exception:
            # Fallback: use XGBoost native feature importance
            self._importance = self.model.get_score(importance_type="gain")

    def predict(self, context, model_input, params=None):
        import xgboost as xgb

        # Ensure model_input is a DataFrame with correct columns
        if isinstance(model_input, pd.DataFrame):
            df = model_input
        else:
            df = pd.DataFrame(model_input)

        # Make predictions
        dmatrix = xgb.DMatrix(df, feature_names=self.feature_names)
        predictions = self.model.predict(dmatrix)

        # Build response with predictions + top-5 explanations
        results = []
        for i, pred in enumerate(predictions):
            if self._shap_ok:
                shap_values = self.explainer.shap_values(df.iloc[[i]])
                shap_pairs = list(zip(self.feature_names, shap_values[0]))
                top5 = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)[:5]
                base_val = round(float(self.explainer.expected_value), 2)
            else:
                top5 = sorted(self._importance.items(), key=lambda x: x[1], reverse=True)[:5]
                base_val = round(float(predictions.mean()), 2)

            results.append({
                "predicted_annual_sales": round(float(pred), 2),
                "shap_base_value": base_val,
                "shap_top5": json.dumps({k: round(float(v), 2) for k, v in top5}),
            })

        return pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Trained Model & Re-Log as Custom PyFunc

# COMMAND ----------

# Get the latest version of the registered model
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
latest_version = max(model_versions, key=lambda v: int(v.version))
source_run_id = latest_version.run_id
print(f"Source model: {REGISTERED_MODEL_NAME} v{latest_version.version} (run: {source_run_id})")

# Download the XGBoost model artifact
source_model_uri = f"runs:/{source_run_id}/model/model.xgb"
local_model_path = mlflow.artifacts.download_artifacts(f"runs:/{source_run_id}/model")

# Find the actual model file
import os
xgb_model_path = None
for root, dirs, files in os.walk(local_model_path):
    for f in files:
        if f.endswith(".xgb") or f == "model.xgb":
            xgb_model_path = os.path.join(root, f)
            break
    # Also check for the default xgboost model file
    if xgb_model_path is None:
        for f in files:
            if f == "model" or "xgboost" in f.lower():
                xgb_model_path = os.path.join(root, f)
                break

# If no explicit file found, use mlflow's native loading to save it
if xgb_model_path is None:
    import xgboost as xgb
    import tempfile
    loaded_model = mlflow.xgboost.load_model(f"runs:/{source_run_id}/model")
    _tmpdir = tempfile.mkdtemp()
    xgb_model_path = os.path.join(_tmpdir, "site_scoring_model.xgb")
    loaded_model.save_model(xgb_model_path)

print(f"XGBoost model path: {xgb_model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Sample Input for Signature

# COMMAND ----------

# Load feature columns
feature_cols_df = spark.table(f"{GOLD}.model_feature_columns").orderBy("feature_index").toPandas()
feature_cols = feature_cols_df["feature_name"].tolist()

# Create a sample input from actual feature data
sample_input = spark.table(f"{GOLD}.location_features") \
    .filter(F.col("site_type") == "existing") \
    .limit(3) \
    .toPandas()

# One-hot encode to match training (need same columns)
sample_encoded = pd.get_dummies(sample_input, columns=["property_type", "metro"], prefix=["prop", "metro"], dtype=float)
if "format" in sample_encoded.columns:
    sample_encoded = pd.get_dummies(sample_encoded, columns=["format"], prefix=["fmt"], dtype=float)

# Keep only the feature columns, fill missing with 0
sample_X = pd.DataFrame(0, index=range(len(sample_encoded)), columns=feature_cols)
for col in feature_cols:
    if col in sample_encoded.columns:
        sample_X[col] = sample_encoded[col].values

print(f"Sample input shape: {sample_X.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log Custom PyFunc Model

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run(run_name="site_scoring_pyfunc_v1") as run:
    # Define output signature
    output_schema = pd.DataFrame({
        "predicted_annual_sales": [0.0],
        "shap_base_value": [0.0],
        "shap_top5": ["{}"],
    })
    signature = infer_signature(sample_X, output_schema)

    pyfunc_model_info = mlflow.pyfunc.log_model(
        artifact_path="pyfunc_model",
        python_model=SiteScoringPyfunc(),
        artifacts={"xgb_model": xgb_model_path},
        pip_requirements=[
            "xgboost>=2.0",
            "shap>=0.43",
            "pandas>=2.0",
            "numpy>=1.24",
        ],
        signature=signature,
        registered_model_name=f"{REGISTERED_MODEL_NAME}_pyfunc",
    )

    pyfunc_run_id = run.info.run_id
    print(f"Custom pyfunc logged: run_id={pyfunc_run_id}")
    print(f"Registered as: {REGISTERED_MODEL_NAME}_pyfunc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy Serving Endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

# Get latest pyfunc model version
pyfunc_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}_pyfunc'")
pyfunc_latest = max(pyfunc_versions, key=lambda v: int(v.version))

served_entities = [
    ServedEntityInput(
        entity_name=f"{REGISTERED_MODEL_NAME}_pyfunc",
        entity_version=pyfunc_latest.version,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]

# Check if endpoint already exists
existing_endpoints = [ep.name for ep in w.serving_endpoints.list()]

if ENDPOINT_NAME in existing_endpoints:
    print(f"Updating existing endpoint: {ENDPOINT_NAME}")
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_entities=served_entities,
    )
else:
    print(f"Creating new endpoint: {ENDPOINT_NAME}")
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=served_entities),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Wait for Endpoint Ready

# COMMAND ----------

print(f"Waiting for endpoint '{ENDPOINT_NAME}' to become ready...")

max_wait = 900  # 15 minutes
poll_interval = 30
elapsed = 0

while elapsed < max_wait:
    endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
    state = endpoint.state.ready
    config_state = endpoint.state.config_update

    print(f"  [{elapsed}s] ready={state}, config_update={config_state}")

    if str(state) == "READY":
        print(f"\nEndpoint '{ENDPOINT_NAME}' is READY")
        break

    time.sleep(poll_interval)
    elapsed += poll_interval
else:
    print(f"\nWARNING: Endpoint not ready after {max_wait}s — check Databricks UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Endpoint

# COMMAND ----------

import requests

# Get workspace URL and token
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Prepare test payload
test_payload = {"dataframe_records": sample_X.head(3).to_dict(orient="records")}

response = requests.post(
    f"{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=test_payload,
    timeout=60,
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print("\nTest predictions:")
    predictions = result.get("predictions", result.get("dataframe_records", []))
    for i, pred in enumerate(predictions):
        print(f"  Site {i+1}: ${pred.get('predicted_annual_sales', 'N/A'):,.0f}")
        shap_str = pred.get("shap_top5", "{}")
        if isinstance(shap_str, str):
            shap_dict = json.loads(shap_str)
        else:
            shap_dict = shap_str
        for feat, val in list(shap_dict.items())[:3]:
            sign = "+" if val > 0 else ""
            print(f"    {feat}: {sign}${val:,.0f}")
    print("\nEndpoint test PASSED")
else:
    print(f"Error: {response.text}")
    raise Exception(f"Endpoint test failed with status {response.status_code}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Deployed model serving endpoint:
# MAGIC - **Endpoint**: `qsr-site-scoring`
# MAGIC - **Model**: Custom pyfunc wrapping XGBoost + SHAP
# MAGIC - **Returns**: Predicted annual sales + top-5 SHAP feature explanations
# MAGIC - **Config**: Small CPU, scale-to-zero enabled
