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
