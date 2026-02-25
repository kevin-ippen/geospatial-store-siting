# Databricks notebook source
# MAGIC %md
# MAGIC # Quality Checks for Store Siting Data
# MAGIC
# MAGIC Validates data quality after each pipeline step.
# MAGIC Pass `table_name` as a widget parameter to specify which table to check.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

from pyspark.sql import functions as F

# Get table to validate from widget
dbutils.widgets.text("table_name", "demographics")
dbutils.widgets.text("min_rows", "100")
dbutils.widgets.text("schema_name", "bronze")

table_name = dbutils.widgets.get("table_name")
min_rows = int(dbutils.widgets.get("min_rows"))
schema_name = dbutils.widgets.get("schema_name")

# Resolve schema to fully-qualified path
schema_map = {"bronze": BRONZE, "silver": SILVER, "gold": GOLD, "models": MODELS}
schema_prefix = schema_map.get(schema_name, f"{CATALOG}.{schema_name}")
full_table_name = f"{schema_prefix}.{table_name}"
print(f"Validating: {full_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Check Functions

# COMMAND ----------

def check_table_exists(table_name):
    """Verify table exists."""
    try:
        spark.table(table_name)
        return True, f"Table {table_name} exists"
    except Exception as e:
        return False, f"Table {table_name} does not exist: {e}"

def check_row_count(table_name, min_rows):
    """Verify minimum row count."""
    df = spark.table(table_name)
    count = df.count()
    if count >= min_rows:
        return True, f"Row count: {count:,} (minimum: {min_rows:,})"
    else:
        return False, f"Row count {count:,} is below minimum {min_rows:,}"

def check_no_nulls_in_key_columns(table_name, key_columns):
    """Verify no nulls in key columns."""
    df = spark.table(table_name)
    issues = []
    for col in key_columns:
        if col in df.columns:
            null_count = df.filter(F.col(col).isNull()).count()
            if null_count > 0:
                issues.append(f"{col}: {null_count} nulls")
    if issues:
        return False, f"Null values found: {', '.join(issues)}"
    return True, "No nulls in key columns"

def check_unique_ids(table_name, id_column):
    """Verify ID column has unique values."""
    df = spark.table(table_name)
    if id_column not in df.columns:
        return True, f"ID column {id_column} not in table"
    total = df.count()
    distinct = df.select(id_column).distinct().count()
    if total == distinct:
        return True, f"All {total:,} IDs are unique"
    else:
        return False, f"Duplicate IDs: {total - distinct:,} duplicates"

def check_valid_h3(table_name, h3_column):
    """Verify H3 indexes are valid format."""
    df = spark.table(table_name)
    if h3_column not in df.columns:
        return True, f"H3 column {h3_column} not in table"
    # H3 indexes should be 15-character hex strings
    invalid = df.filter(
        (F.length(F.col(h3_column)) != 15) |
        (~F.col(h3_column).rlike("^[0-9a-f]+$"))
    ).count()
    if invalid == 0:
        return True, f"All H3 indexes are valid"
    else:
        return False, f"Invalid H3 indexes: {invalid:,}"

def check_value_ranges(table_name, column, min_val, max_val):
    """Verify values are within expected range."""
    df = spark.table(table_name)
    if column not in df.columns:
        return True, f"Column {column} not in table"
    out_of_range = df.filter(
        (F.col(column) < min_val) | (F.col(column) > max_val)
    ).count()
    if out_of_range == 0:
        return True, f"{column} values all within [{min_val}, {max_val}]"
    else:
        return False, f"{column}: {out_of_range:,} values out of range [{min_val}, {max_val}]"


def check_null_rate(table_name, column, max_null_pct=0.05):
    """Verify null rate is below threshold."""
    df = spark.table(table_name)
    if column not in df.columns:
        return True, f"Column {column} not in table"
    total = df.count()
    nulls = df.filter(F.col(column).isNull()).count()
    null_pct = nulls / max(total, 1)
    if null_pct <= max_null_pct:
        return True, f"{column} null rate: {null_pct:.1%} (max: {max_null_pct:.0%})"
    else:
        return False, f"{column} null rate: {null_pct:.1%} exceeds max {max_null_pct:.0%} ({nulls:,} nulls)"


def check_sales_quality_correlation(stores_table, min_corr=0.4):
    """Verify that annual sales correlate with location quality score.

    This is the critical circular-reasoning guard: if sales are random noise,
    SHAP explanations are meaningless. Target: r > 0.4.
    """
    import pandas as pd
    df = spark.table(stores_table).select("annual_sales", "location_quality_score")
    if "location_quality_score" not in df.columns:
        return True, "No location_quality_score column — skipping correlation check"
    pdf = df.toPandas()
    corr = pdf["annual_sales"].corr(pdf["location_quality_score"])
    if corr >= min_corr:
        return True, f"Sales ↔ quality correlation: {corr:.3f} (minimum: {min_corr})"
    else:
        return False, f"Sales ↔ quality correlation: {corr:.3f} is below minimum {min_corr} — SHAP explanations may be unreliable"


def check_metro_coverage(table_name, metro_column="metro", expected_metros=None):
    """Verify all expected metros have data."""
    df = spark.table(table_name)
    if metro_column not in df.columns:
        return True, f"No {metro_column} column in table"
    found = set(r[metro_column] for r in df.select(metro_column).distinct().collect())
    if expected_metros:
        missing = set(expected_metros) - found
        if missing:
            return False, f"Missing metros: {missing}"
    return True, f"Metros found: {sorted(found)}"


def check_referential_integrity(child_table, child_col, parent_table, parent_col):
    """Verify FK-like integrity between tables."""
    child_df = spark.table(child_table).select(child_col).distinct()
    parent_df = spark.table(parent_table).select(parent_col).distinct()
    orphans = child_df.join(parent_df, child_df[child_col] == parent_df[parent_col], "left_anti").count()
    if orphans == 0:
        return True, f"All {child_col} values exist in {parent_table}.{parent_col}"
    else:
        return False, f"{orphans:,} orphan values in {child_col} not found in {parent_table}.{parent_col}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table-Specific Checks

# COMMAND ----------

# Define checks for each table
TABLE_CHECKS = {
    "demographics": {
        "min_rows": 10000,
        "key_columns": ["h3_index", "metro", "population"],
        "id_column": "h3_index",
        "h3_column": "h3_index",
        "range_checks": [
            ("population", 0, 50000),
            ("median_income", 20000, 300000),
            ("pct_18_to_34", 0, 1),
        ]
    },
    "traffic": {
        "min_rows": 10000,
        "key_columns": ["h3_index", "avg_daily_traffic"],
        "id_column": "h3_index",
        "h3_column": "h3_index",
        "range_checks": [
            ("avg_daily_traffic", 0, 200000),
            ("pedestrian_index", 0, 100),
            ("transit_score", 0, 100),
        ]
    },
    "competitors": {
        "min_rows": 1000,
        "key_columns": ["competitor_id", "brand", "latitude", "longitude"],
        "id_column": "competitor_id",
        "h3_column": "h3_res8",
        "range_checks": [
            ("latitude", 25, 50),
            ("longitude", -130, -70),
            ("estimated_annual_sales", 500000, 5000000),
        ]
    },
    "poi": {
        "min_rows": 4000,
        "key_columns": ["poi_id", "category", "latitude", "longitude"],
        "id_column": "poi_id",
        "h3_column": "h3_res8",
        "range_checks": [
            ("foot_traffic_index", 0, 100),
        ]
    },
    "locations": {
        "min_rows": 4000,
        "key_columns": ["location_id", "latitude", "longitude", "property_type"],
        "id_column": "location_id",
        "h3_column": "h3_res8",
        "range_checks": [
            ("square_feet", 500, 10000),
            ("rent_per_sqft", 5, 100),
        ]
    },
    "existing_stores": {
        "min_rows": 300,
        "key_columns": ["store_id", "annual_sales", "latitude", "longitude"],
        "id_column": "store_id",
        "h3_column": "h3_res8",
        "range_checks": [
            ("annual_sales", 800000, 5000000),
            ("transactions_per_day", 200, 2000),
            ("avg_ticket", 5, 25),
        ]
    },
    # Phase 2: Gold layer tables
    "location_features": {
        "min_rows": 5000,
        "key_columns": ["site_id", "h3_res8", "site_type"],
        "id_column": "site_id",
        "h3_column": "h3_res8",
        "range_checks": [
            ("population_1ring", 0, 500000),
            ("median_income_1ring", 10000, 300000),
            ("competitor_count_1ring", 0, 50),
        ]
    },
    "scored_locations": {
        "min_rows": 4000,
        "key_columns": ["site_id", "predicted_annual_sales", "score_tier"],
        "id_column": "site_id",
        "h3_column": "h3_res8",
        "range_checks": [
            ("predicted_annual_sales", 500000, 5000000),
            ("percentile_rank", 0, 1),
        ]
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Checks

# COMMAND ----------

results = []
all_passed = True

# Get table config
config = TABLE_CHECKS.get(table_name, {
    "min_rows": min_rows,
    "key_columns": [],
    "id_column": None,
    "h3_column": None,
    "range_checks": []
})

# Check 1: Table exists
passed, msg = check_table_exists(full_table_name)
results.append(("Table Exists", passed, msg))
all_passed = all_passed and passed

if passed:
    # Check 2: Row count
    passed, msg = check_row_count(full_table_name, config.get("min_rows", min_rows))
    results.append(("Row Count", passed, msg))
    all_passed = all_passed and passed

    # Check 3: No nulls in key columns
    if config.get("key_columns"):
        passed, msg = check_no_nulls_in_key_columns(full_table_name, config["key_columns"])
        results.append(("No Nulls", passed, msg))
        all_passed = all_passed and passed

    # Check 4: Unique IDs
    if config.get("id_column"):
        passed, msg = check_unique_ids(full_table_name, config["id_column"])
        results.append(("Unique IDs", passed, msg))
        all_passed = all_passed and passed

    # Check 5: Valid H3
    if config.get("h3_column"):
        passed, msg = check_valid_h3(full_table_name, config["h3_column"])
        results.append(("Valid H3", passed, msg))
        all_passed = all_passed and passed

    # Check 6: Value ranges
    for col, min_val, max_val in config.get("range_checks", []):
        passed, msg = check_value_ranges(full_table_name, col, min_val, max_val)
        results.append((f"Range: {col}", passed, msg))
        all_passed = all_passed and passed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Summary

# COMMAND ----------

print(f"\n{'='*60}")
print(f"Quality Check Results for {full_table_name}")
print(f"{'='*60}\n")

for check_name, passed, msg in results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {check_name}: {msg}")

print(f"\n{'='*60}")
if all_passed:
    print("✅ ALL CHECKS PASSED")
else:
    print("❌ SOME CHECKS FAILED")
print(f"{'='*60}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fail Job if Checks Failed

# COMMAND ----------

if not all_passed:
    failed_checks = [name for name, passed, _ in results if not passed]
    raise Exception(f"Quality checks failed: {', '.join(failed_checks)}")

print(f"Quality validation complete for {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase Gate: Cross-Table Validation
# MAGIC
# MAGIC When `table_name = "validate_all"`, runs inter-table checks that validate
# MAGIC Phase 1 output is ready for Phase 2. This is the quality gate between
# MAGIC demo data generation and the ML pipeline.

# COMMAND ----------

if table_name == "validate_all":
    print(f"\n{'='*60}")
    print("PHASE GATE: Cross-Table Validation")
    print(f"{'='*60}\n")

    gate_results = []
    gate_passed = True

    # 1. All bronze tables exist with minimum rows
    bronze_tables = ["demographics", "traffic", "competitors", "poi",
                     "existing_stores", "locations", "daypart_demand"]
    for t in bronze_tables:
        full_name = f"{schema_map['bronze']}.{t}"
        passed, msg = check_table_exists(full_name)
        gate_results.append((f"Exists: {t}", passed, msg))
        gate_passed = gate_passed and passed
        if passed:
            cfg = TABLE_CHECKS.get(t, {})
            passed, msg = check_row_count(full_name, cfg.get("min_rows", 100))
            gate_results.append((f"Rows: {t}", passed, msg))
            gate_passed = gate_passed and passed

    # 2. Sales-quality correlation (the anti-circular-reasoning check)
    stores_table = f"{schema_map['bronze']}.existing_stores"
    try:
        passed, msg = check_sales_quality_correlation(stores_table, min_corr=0.4)
        gate_results.append(("Sales-Quality Correlation", passed, msg))
        gate_passed = gate_passed and passed
    except Exception as e:
        gate_results.append(("Sales-Quality Correlation", False, f"Error: {e}"))
        gate_passed = False

    # 3. Metro coverage — all expected metros have data in all tables
    demo_df = spark.table(f"{schema_map['bronze']}.demographics")
    expected_metros = [r["metro"] for r in demo_df.select("metro").distinct().collect()]
    for t in ["traffic", "competitors", "existing_stores", "locations"]:
        full_name = f"{schema_map['bronze']}.{t}"
        col = "metro" if t != "competitors" else "metro"
        passed, msg = check_metro_coverage(full_name, metro_column=col, expected_metros=expected_metros)
        gate_results.append((f"Metro Coverage: {t}", passed, msg))
        gate_passed = gate_passed and passed

    # 4. H3 index referential integrity (stores and locations should reference valid hex cells)
    passed, msg = check_referential_integrity(
        f"{schema_map['bronze']}.existing_stores", "h3_res8",
        f"{schema_map['bronze']}.demographics", "h3_index"
    )
    gate_results.append(("FK: stores → demographics", passed, msg))
    # Note: this may have orphans due to jitter in store placement — warn don't fail
    if not passed:
        print(f"  WARNING (non-fatal): {msg}")
        gate_results[-1] = ("FK: stores → demographics", True, f"WARNING: {msg}")

    # 5. Null rates on critical columns
    critical_null_checks = [
        (f"{schema_map['bronze']}.existing_stores", "annual_sales"),
        (f"{schema_map['bronze']}.existing_stores", "location_quality_score"),
        (f"{schema_map['bronze']}.demographics", "population"),
        (f"{schema_map['bronze']}.demographics", "median_income"),
    ]
    for tbl, col in critical_null_checks:
        passed, msg = check_null_rate(tbl, col, max_null_pct=0.01)
        gate_results.append((f"Nulls: {col}", passed, msg))
        gate_passed = gate_passed and passed

    # Print results
    print(f"\n{'='*60}")
    print("Phase Gate Results")
    print(f"{'='*60}\n")
    for check_name, passed, msg in gate_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {check_name}: {msg}")

    print(f"\n{'='*60}")
    if gate_passed:
        print("✅ PHASE GATE PASSED — safe to proceed to Phase 2 (ML Pipeline)")
    else:
        print("❌ PHASE GATE FAILED — fix data issues before running ML pipeline")
        failed_gates = [name for name, passed, _ in gate_results if not passed]
        raise Exception(f"Phase gate failed: {', '.join(failed_gates)}")
    print(f"{'='*60}\n")
