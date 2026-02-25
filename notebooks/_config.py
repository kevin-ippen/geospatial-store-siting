# Databricks notebook source
# MAGIC %md
# MAGIC # Shared Configuration
# MAGIC
# MAGIC Run this notebook using `%run ./_config` at the top of other notebooks.
# MAGIC All values are configurable via job parameters (widgets) or `databricks.yml` variables.

# COMMAND ----------

# --- Parameterized Config (reads from job widgets, falls back to defaults) ---

def _widget(name, default):
    """Read a widget value set by job base_parameters, or return default."""
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default

# Unity Catalog
CATALOG = _widget("catalog", "qsr_siting")

# Demo mode: when true, data generation notebooks create synthetic data
DEMO_MODE = _widget("demo_mode", "true") == "true"
DEMO_METRO = _widget("demo_metro", "Chicago")

# Medallion schemas
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"
MODELS_SCHEMA = "models"

# Fully qualified schema paths
BRONZE = f"{CATALOG}.{BRONZE_SCHEMA}"
SILVER = f"{CATALOG}.{SILVER_SCHEMA}"
GOLD = f"{CATALOG}.{GOLD_SCHEMA}"
MODELS = f"{CATALOG}.{MODELS_SCHEMA}"

# Volume paths
RAW_VOLUME = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/raw"
CHECKPOINT_VOLUME = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/checkpoints"

# H3 resolutions for multi-scale analysis
H3_RES_MARKET = 7   # ~1.2km - market-level overview
H3_RES_TRADE = 8    # ~460m - trade area analysis (primary)
H3_RES_SITE = 9     # ~174m - site-level precision

# Metro definitions with bounding boxes (lat_min, lat_max, lon_min, lon_max)
ALL_METROS = {
    "Chicago": {"bounds": (41.6, 42.1, -88.0, -87.5), "center": (41.85, -87.75)},
    "Dallas": {"bounds": (32.6, 33.1, -97.0, -96.5), "center": (32.85, -96.75)},
    "Phoenix": {"bounds": (33.2, 33.8, -112.3, -111.8), "center": (33.5, -112.05)},
    "Atlanta": {"bounds": (33.6, 34.0, -84.6, -84.2), "center": (33.8, -84.4)},
    "Denver": {"bounds": (39.5, 40.0, -105.1, -104.7), "center": (39.75, -104.9)},
}

# Metro-specific income multipliers (reflects cost-of-living differences)
METRO_INCOME_MULTIPLIER = {
    "Chicago": 1.05,
    "Dallas": 0.95,
    "Phoenix": 0.90,
    "Atlanta": 0.95,
    "Denver": 1.15,
}

# Huff gravity model parameters
HUFF_BETA = 2.0            # distance decay exponent (higher = more local)
HUFF_BETA_RANGE = (1.0, 3.0, 0.25)  # (min, max, step) for calibration grid search
HUFF_CANNIB_RADIUS = 3.0   # miles — max distance for cannibalization analysis

# In demo mode, restrict to a single metro for faster generation
if DEMO_MODE:
    METROS = {DEMO_METRO: ALL_METROS[DEMO_METRO]}
else:
    METROS = ALL_METROS

# QSR competitor brands with target location counts (per metro)
COMPETITOR_BRANDS = {
    "McDonald's": {"count": 80, "category": "QSR_Burger", "drive_thru_pct": 0.85},
    "Burger King": {"count": 50, "category": "QSR_Burger", "drive_thru_pct": 0.80},
    "Wendy's": {"count": 40, "category": "QSR_Burger", "drive_thru_pct": 0.80},
    "Taco Bell": {"count": 36, "category": "QSR_Mexican", "drive_thru_pct": 0.75},
    "Chick-fil-A": {"count": 24, "category": "QSR_Chicken", "drive_thru_pct": 0.90},
    "Chipotle": {"count": 30, "category": "Fast_Casual", "drive_thru_pct": 0.30},
    "Five Guys": {"count": 16, "category": "Fast_Casual", "drive_thru_pct": 0.10},
}

# --- Phase 2: ML & Feature Engineering ---

# MLflow experiment
MLFLOW_EXPERIMENT = f"/{CATALOG}/site-scoring-model"

# Model serving
ENDPOINT_NAME = _widget("model_endpoint_name", "qsr-site-scoring")
REGISTERED_MODEL_NAME = f"{CATALOG}.{MODELS_SCHEMA}.site_scoring"

# Feature columns used by the model (order matters for serving)
DEMOGRAPHIC_FEATURES = [
    "population_1ring", "median_income_1ring", "pct_target_demo_1ring",
    "daytime_pop_1ring", "pct_college_1ring",
]
TRAFFIC_FEATURES = [
    "max_daily_traffic_1ring", "avg_transit_score_1ring", "total_pedestrian_index_1ring",
]
COMPETITION_FEATURES = [
    "competitor_count_1ring", "competitor_count_3ring",
    "nearest_competitor_dist", "nearest_same_category_dist", "competitive_intensity",
]
POI_FEATURES = [
    "retail_anchor_count_1ring", "office_poi_count_1ring",
    "school_count_2ring", "total_foot_traffic_1ring",
]
PROPERTY_FEATURES = [
    "drive_thru_capable_flag", "parking_spaces", "square_feet", "rent_per_sqft",
]
DERIVED_FEATURES = [
    "trade_area_quality", "cannibalization_risk", "market_saturation",
    "huff_market_share", "huff_expected_demand",
]

# All numeric features for model input
NUMERIC_FEATURES = (
    DEMOGRAPHIC_FEATURES + TRAFFIC_FEATURES + COMPETITION_FEATURES +
    POI_FEATURES + PROPERTY_FEATURES + DERIVED_FEATURES
)

# Categorical features (will be one-hot encoded)
CATEGORICAL_FEATURES = ["property_type", "metro"]

# Daypart demand features
DAYPART_COLUMNS = [
    "breakfast_score", "lunch_score", "dinner_score",
    "late_night_score", "weekend_multiplier",
]

# Model quality gates (relaxed for synthetic demo data with ~350 training samples)
MIN_R2 = 0.35
MAX_MAPE = 0.30
CV_FOLDS = 5  # k-fold cross-validation for confidence intervals on model quality

print(f"Config loaded: CATALOG={CATALOG}, DEMO_MODE={DEMO_MODE}, METROS={list(METROS.keys())}")
