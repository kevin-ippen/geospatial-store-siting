# Databricks notebook source
# MAGIC %md
# MAGIC # Schema Validator
# MAGIC
# MAGIC Checks that all required bronze tables exist and have the expected columns.
# MAGIC Run this before the ML pipeline when using your own data (`demo_mode = false`).

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

REQUIRED_TABLES = {
    f"{BRONZE}.demographics": [
        "h3_index", "metro", "latitude", "longitude", "population",
        "median_income", "median_age", "pct_college_educated",
    ],
    f"{BRONZE}.traffic": [
        "h3_index", "avg_daily_traffic", "transit_score", "pedestrian_index",
    ],
    f"{BRONZE}.competitors": [
        "competitor_id", "brand", "category", "latitude", "longitude",
        "h3_res8", "metro", "estimated_annual_sales", "drive_thru",
    ],
    f"{BRONZE}.poi": [
        "poi_id", "name", "category", "latitude", "longitude",
        "h3_res8", "metro", "foot_traffic_index",
    ],
    f"{BRONZE}.existing_stores": [
        "store_id", "store_name", "latitude", "longitude", "h3_res8",
        "metro", "format", "annual_sales", "transactions_per_day",
    ],
    f"{BRONZE}.locations": [
        "location_id", "latitude", "longitude", "h3_res8", "metro",
        "property_type", "square_feet", "parking_spaces",
        "drive_thru_capable", "rent_per_sqft",
    ],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Tables

# COMMAND ----------

all_passed = True
for table_name, required_cols in REQUIRED_TABLES.items():
    try:
        df = spark.table(table_name)
        actual_cols = set(df.columns)
        missing = [c for c in required_cols if c not in actual_cols]
        count = df.count()

        if missing:
            print(f"WARN  {table_name}: missing columns {missing}")
            all_passed = False
        elif count == 0:
            print(f"WARN  {table_name}: exists but has 0 rows")
            all_passed = False
        else:
            print(f"  OK  {table_name}: {count:,} rows, all required columns present")
    except Exception as e:
        print(f"FAIL  {table_name}: {e}")
        all_passed = False

# COMMAND ----------

if all_passed:
    print("\nAll schema checks passed.")
else:
    msg = "Schema validation failed. See warnings above."
    print(f"\n{msg}")
    dbutils.notebook.exit(msg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Minimum Sample Size Guidance
# MAGIC
# MAGIC The ML pipeline needs sufficient data to train a meaningful model.
# MAGIC This section checks whether your data meets minimum thresholds and
# MAGIC provides guidance when it doesn't.

# COMMAND ----------

SAMPLE_SIZE_THRESHOLDS = {
    f"{BRONZE}.existing_stores": {
        "min_recommended": 300,
        "min_viable": 100,
        "guidance": (
            "The site scoring model trains on existing store revenue as labels. "
            "With fewer than 300 stores, the model may overfit to your specific portfolio. "
            "With fewer than 100, XGBoost will not have enough signal to learn meaningful "
            "feature-label relationships.\n"
            "  - 100-300 stores: Model will work but expect high CV variance (R² ± 0.15+)\n"
            "  - 300-500 stores: Good fit for single-market demos\n"
            "  - 500+ stores: Production-grade training set\n"
            "  - 1000+ stores: Can support metro-specific sub-models"
        ),
    },
    f"{BRONZE}.demographics": {
        "min_recommended": 2000,
        "min_viable": 500,
        "guidance": (
            "Demographics drive population, income, and age features. Fewer than 2,000 "
            "hexagons means sparse trade area features — k-ring aggregations may have "
            "many zero-population rings.\n"
            "  - 500-2000 hexagons: Adequate for single-metro analysis\n"
            "  - 2000-10000 hexagons: Good for multi-metro comparison\n"
            "  - 10000+ hexagons: Full metro coverage at H3 resolution 8"
        ),
    },
    f"{BRONZE}.competitors": {
        "min_recommended": 200,
        "min_viable": 50,
        "guidance": (
            "Competitor locations drive cannibalization risk and competitive intensity "
            "features. Fewer than 200 competitors means the Huff gravity model has sparse "
            "supply points, leading to overestimated market share.\n"
            "  - Source: SafeGraph, Foursquare, or web scraping of brand location pages"
        ),
    },
}

print(f"\n{'='*60}")
print("Sample Size Assessment")
print(f"{'='*60}\n")

size_warnings = []
for table_name, thresholds in SAMPLE_SIZE_THRESHOLDS.items():
    try:
        count = spark.table(table_name).count()
        if count >= thresholds["min_recommended"]:
            print(f"  ✓ {table_name}: {count:,} rows (recommended: {thresholds['min_recommended']:,}+)")
        elif count >= thresholds["min_viable"]:
            print(f"  ⚠ {table_name}: {count:,} rows (viable but below recommended {thresholds['min_recommended']:,})")
            print(f"    {thresholds['guidance']}")
            size_warnings.append(table_name)
        else:
            print(f"  ✗ {table_name}: {count:,} rows (below minimum viable {thresholds['min_viable']:,})")
            print(f"    {thresholds['guidance']}")
            size_warnings.append(table_name)
    except Exception:
        pass  # table doesn't exist — already caught in schema validation

if size_warnings:
    print(f"\n  ⚠ {len(size_warnings)} table(s) below recommended size — model quality may be limited")
else:
    print(f"\n  ✓ All tables meet recommended sample sizes")

print(f"{'='*60}\n")
