# Databricks notebook source
# MAGIC %md
# MAGIC # 10 - Feature Engineering (Phase 2)
# MAGIC
# MAGIC Builds `gold.location_features` using H3 k-ring spatial aggregation.
# MAGIC Features are computed for **both** existing stores (training) and candidate locations (scoring)
# MAGIC to guarantee train/serve feature parity.
# MAGIC
# MAGIC **Input**: bronze.demographics, bronze.traffic, bronze.competitors, bronze.poi, bronze.locations, bronze.existing_stores
# MAGIC **Output**: `gold.location_features` (~5,350 rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import h3
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build Unified Site Table
# MAGIC
# MAGIC Union existing stores and candidate locations into a single frame so features are computed identically.

# COMMAND ----------

existing = spark.table(f"{BRONZE}.existing_stores").select(
    F.col("store_id").alias("site_id"),
    F.col("h3_res8"),
    F.col("metro"),
    F.col("latitude"),
    F.col("longitude"),
    F.col("format").alias("property_type"),
    F.col("square_feet"),
    # existing_stores doesn't have parking_spaces or rent_per_sqft — fill with medians later
    F.lit(None).cast("int").alias("parking_spaces"),
    F.lit(None).cast("double").alias("rent_per_sqft"),
    F.col("drive_thru_pct").cast("double"),
    F.lit("existing").alias("site_type"),
)

candidates = spark.table(f"{BRONZE}.locations").select(
    F.col("location_id").alias("site_id"),
    F.col("h3_res8"),
    F.col("metro"),
    F.col("latitude"),
    F.col("longitude"),
    F.col("property_type"),
    F.col("square_feet"),
    F.col("parking_spaces"),
    F.col("rent_per_sqft"),
    F.when(F.col("drive_thru_capable"), 1.0).otherwise(0.0).alias("drive_thru_pct"),
    F.lit("candidate").alias("site_type"),
)

all_sites = existing.unionByName(candidates)
site_count = all_sites.count()
print(f"Unified site table: {site_count} rows ({all_sites.filter(F.col('site_type')=='existing').count()} existing + {all_sites.filter(F.col('site_type')=='candidate').count()} candidates)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register H3 K-Ring UDF

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2b. Pre-Compute K-Ring Expansions (Driver-Side)
# MAGIC
# MAGIC Serverless doesn't support Python UDFs in join ON clauses or .cache().
# MAGIC We compute H3 ring expansions on the driver in pure Python (~5K sites x ~37 neighbors = manageable)
# MAGIC then create Spark DataFrames from the result.

# COMMAND ----------

demographics = spark.table(f"{BRONZE}.demographics")
traffic = spark.table(f"{BRONZE}.traffic")

# Collect site H3 indexes to driver (small: ~5,350 rows)
site_h3_list = all_sites.select("site_id", "h3_res8").collect()
print(f"Computing ring expansions for {len(site_h3_list)} sites on driver...")

# Compute all k-ring expansions in pure Python
ring_data = {1: [], 2: [], 3: []}
for row in site_h3_list:
    sid = row.site_id
    h3_idx = row.h3_res8
    if h3_idx is None:
        continue
    for k in [1, 2, 3]:
        neighbors = h3.grid_disk(h3_idx, k)
        for n in neighbors:
            ring_data[k].append((sid, n))

# Create Spark DataFrames from driver-computed data (no UDFs involved)
from pyspark.sql.types import StructType, StructField

ring_schema = StructType([
    StructField("site_id", StringType(), False),
    StructField("neighbor_h3", StringType(), False),
])

sites_ring1 = spark.createDataFrame(ring_data[1], ring_schema)
sites_ring2 = spark.createDataFrame(ring_data[2], ring_schema)
sites_ring3 = spark.createDataFrame(ring_data[3], ring_schema)

print(f"Ring-1: {len(ring_data[1])} rows | Ring-2: {len(ring_data[2])} rows | Ring-3: {len(ring_data[3])} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Demographic Features (k=1 ring)

# COMMAND ----------

# Join demographics on the neighbor hexes (sites_ring1 is already cached with plain columns)
demo_joined = sites_ring1.join(
    demographics.select(
        F.col("h3_index").alias("neighbor_h3"),
        "population", "median_income", "pct_18_to_34", "pct_college_educated",
    ),
    on="neighbor_h3",
    how="inner",
)

# Aggregate demographic features per site
demo_features = demo_joined.groupBy("site_id").agg(
    F.sum("population").alias("population_1ring"),
    # Weighted average income (by population)
    (F.sum(F.col("median_income") * F.col("population")) / F.sum("population")).alias("median_income_1ring"),
    # Weighted average target demo (18-34)
    (F.sum(F.col("pct_18_to_34") * F.col("population")) / F.sum("population")).alias("pct_target_demo_1ring"),
    # Weighted college %
    (F.sum(F.col("pct_college_educated") * F.col("population")) / F.sum("population")).alias("pct_college_1ring"),
)

print(f"Demographic features: {demo_features.count()} sites")
display(demo_features.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Daytime Population (demographics + traffic commute flows)

# COMMAND ----------

# Need commute data from traffic table for daytime population
traffic_for_daytime = traffic.select(
    F.col("h3_index").alias("neighbor_h3"),
    "commute_inflow", "commute_outflow",
)

demo_traffic_joined = sites_ring1.join(
    demographics.select(F.col("h3_index").alias("neighbor_h3"), "population"),
    on="neighbor_h3", how="inner",
).join(
    traffic_for_daytime, on="neighbor_h3", how="inner",
)

daytime_features = demo_traffic_joined.groupBy("site_id").agg(
    F.sum(
        F.col("population") + F.coalesce(F.col("commute_inflow"), F.lit(0)) - F.coalesce(F.col("commute_outflow"), F.lit(0))
    ).alias("daytime_pop_1ring"),
)

print(f"Daytime pop features: {daytime_features.count()} sites")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Traffic Features (k=1 ring)

# COMMAND ----------

traffic_joined = sites_ring1.join(
    traffic.select(
        F.col("h3_index").alias("neighbor_h3"),
        "avg_daily_traffic", "transit_score", "pedestrian_index",
    ),
    on="neighbor_h3",
    how="inner",
)

traffic_features = traffic_joined.groupBy("site_id").agg(
    F.max("avg_daily_traffic").alias("max_daily_traffic_1ring"),
    F.avg("transit_score").alias("avg_transit_score_1ring"),
    F.sum("pedestrian_index").alias("total_pedestrian_index_1ring"),
)

print(f"Traffic features: {traffic_features.count()} sites")
display(traffic_features.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Competition Features (k=1 and k=3 rings)

# COMMAND ----------

competitors = spark.table(f"{BRONZE}.competitors")

# --- k=1 ring competitor count ---
comp_ring1 = sites_ring1.join(
    competitors.select(F.col("h3_res8").alias("neighbor_h3"), "competitor_id"),
    on="neighbor_h3",
    how="left",
)
comp_count_1 = comp_ring1.groupBy("site_id").agg(
    F.countDistinct("competitor_id").alias("competitor_count_1ring"),
)

# --- k=3 ring competitor count (ring already materialized above) ---
comp_ring3 = sites_ring3.join(
    competitors.select(F.col("h3_res8").alias("neighbor_h3"), "competitor_id"),
    on="neighbor_h3",
    how="left",
)
comp_count_3 = comp_ring3.groupBy("site_id").agg(
    F.countDistinct("competitor_id").alias("competitor_count_3ring"),
)

# --- Nearest competitor distance (haversine) ---
# Cross-join sites with all competitors and compute distance, then take MIN
# For ~5K sites x ~1.4K competitors = ~7M rows — manageable

def haversine_miles_expr(lat1_col, lon1_col, lat2_col, lon2_col):
    """Haversine distance in miles using native Spark SQL functions (no Python UDF)."""
    dlat = F.radians(lat2_col - lat1_col)
    dlon = F.radians(lon2_col - lon1_col)
    a = (
        F.pow(F.sin(dlat / 2), 2)
        + F.cos(F.radians(lat1_col)) * F.cos(F.radians(lat2_col)) * F.pow(F.sin(dlon / 2), 2)
    )
    return F.lit(3959.0) * F.lit(2.0) * F.asin(F.sqrt(a))

# Filter competitors to same metro for efficiency
site_comp_cross = all_sites.select("site_id", "metro", "latitude", "longitude").join(
    competitors.select(
        F.col("competitor_id"), F.col("latitude").alias("comp_lat"),
        F.col("longitude").alias("comp_lon"), F.col("category"), F.col("metro"),
    ),
    on="metro",
    how="inner",
)

site_comp_dist = site_comp_cross.withColumn(
    "dist_miles", haversine_miles_expr(F.col("latitude"), F.col("longitude"), F.col("comp_lat"), F.col("comp_lon"))
)

# Nearest any competitor + nearest same-category (QSR_Burger as proxy for "direct")
nearest_any = site_comp_dist.groupBy("site_id").agg(
    F.min("dist_miles").alias("nearest_competitor_dist"),
)

nearest_same_cat = site_comp_dist.filter(F.col("category") == "QSR_Burger").groupBy("site_id").agg(
    F.min("dist_miles").alias("nearest_same_category_dist"),
)

# Competitive intensity: gravity model SUM(1/dist^2) for competitors within ~3mi
comp_gravity = site_comp_dist.filter(F.col("dist_miles") <= 3.0).withColumn(
    "gravity", F.lit(1.0) / (F.col("dist_miles") * F.col("dist_miles") + F.lit(0.01))  # +0.01 to avoid div-by-zero
)
comp_intensity = comp_gravity.groupBy("site_id").agg(
    F.sum("gravity").alias("competitive_intensity"),
)

# Combine all competition features
competition_features = comp_count_1.join(comp_count_3, on="site_id", how="outer") \
    .join(nearest_any, on="site_id", how="outer") \
    .join(nearest_same_cat, on="site_id", how="outer") \
    .join(comp_intensity, on="site_id", how="outer")

print(f"Competition features: {competition_features.count()} sites")
display(competition_features.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. POI Features (k=1 and k=2 rings)

# COMMAND ----------

poi = spark.table(f"{BRONZE}.poi")

# k=1 ring POI features
poi_ring1 = sites_ring1.join(
    poi.select(
        F.col("h3_res8").alias("neighbor_h3"),
        "poi_id", "category", "size_category", "foot_traffic_index",
    ),
    on="neighbor_h3",
    how="left",
)

poi_features_1 = poi_ring1.groupBy("site_id").agg(
    F.countDistinct(F.when(F.col("size_category") == "anchor", F.col("poi_id"))).alias("retail_anchor_count_1ring"),
    F.countDistinct(F.when(F.col("category") == "Office", F.col("poi_id"))).alias("office_poi_count_1ring"),
    F.sum(F.coalesce(F.col("foot_traffic_index"), F.lit(0.0))).alias("total_foot_traffic_1ring"),
)

# k=2 ring for schools (ring already materialized above)
school_ring2 = sites_ring2.join(
    poi.select(F.col("h3_res8").alias("neighbor_h3"), "poi_id", "category"),
    on="neighbor_h3",
    how="left",
)

school_features = school_ring2.groupBy("site_id").agg(
    F.countDistinct(F.when(F.col("category") == "School", F.col("poi_id"))).alias("school_count_2ring"),
)

poi_features = poi_features_1.join(school_features, on="site_id", how="outer")

print(f"POI features: {poi_features.count()} sites")
display(poi_features.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Property Features (from source tables directly)

# COMMAND ----------

# Property features come directly from the all_sites table
property_features = all_sites.select(
    "site_id",
    F.when(F.col("drive_thru_pct") > 0, 1).otherwise(0).alias("drive_thru_capable_flag"),
    "parking_spaces",
    "square_feet",
    "rent_per_sqft",
    "property_type",
    "metro",
)

# Fill nulls for existing stores (they don't have parking/rent)
# Use metro-level medians from candidate locations
median_vals = candidates.groupBy("metro").agg(
    F.percentile_approx("parking_spaces", 0.5).alias("median_parking"),
    F.percentile_approx("rent_per_sqft", 0.5).alias("median_rent"),
)

property_features = property_features.join(median_vals, on="metro", how="left")
property_features = property_features.withColumn(
    "parking_spaces", F.coalesce(
        F.col("parking_spaces"),
        F.when(F.isnan("median_parking"), F.lit(0)).otherwise(F.col("median_parking")).cast("int")
    )
).withColumn(
    "rent_per_sqft", F.coalesce(
        F.col("rent_per_sqft"),
        F.when(F.isnan("median_rent"), F.lit(0.0)).otherwise(F.col("median_rent"))
    )
).drop("median_parking", "median_rent", "metro")

print(f"Property features: {property_features.count()} sites")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Derived Features

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cannibalization Risk
# MAGIC Distance-weighted proximity to our own existing stores.

# COMMAND ----------

existing_stores = spark.table(f"{BRONZE}.existing_stores").select(
    F.col("store_id").alias("own_store_id"),
    F.col("latitude").alias("own_lat"),
    F.col("longitude").alias("own_lon"),
    F.col("metro"),
)

# Cross-join all sites with existing stores (same metro)
own_store_cross = all_sites.select("site_id", "metro", "latitude", "longitude").join(
    existing_stores, on="metro", how="inner",
).filter(
    # Exclude self-joins for existing stores
    F.col("site_id") != F.col("own_store_id")
)

own_store_dist = own_store_cross.withColumn(
    "own_dist_miles", haversine_miles_expr(F.col("latitude"), F.col("longitude"), F.col("own_lat"), F.col("own_lon"))
)

# Cannibalization risk: sum of 1/dist for own stores within 3 miles
cannib = own_store_dist.filter(F.col("own_dist_miles") <= 3.0).withColumn(
    "cannib_weight", F.lit(1.0) / (F.col("own_dist_miles") + F.lit(0.1))
)
cannib_risk = cannib.groupBy("site_id").agg(
    F.sum("cannib_weight").alias("cannibalization_risk"),
)

print(f"Cannibalization features: {cannib_risk.count()} sites")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Assemble Final Feature Table

# COMMAND ----------

# Start with all_sites base
base = all_sites.select("site_id", "h3_res8", "metro", "latitude", "longitude", "site_type")

# Join all feature groups
features = (
    base
    .join(demo_features, on="site_id", how="left")
    .join(daytime_features, on="site_id", how="left")
    .join(traffic_features, on="site_id", how="left")
    .join(competition_features, on="site_id", how="left")
    .join(poi_features, on="site_id", how="left")
    .join(property_features, on="site_id", how="left")
    .join(cannib_risk, on="site_id", how="left")
)

# Fill remaining nulls with 0 for numeric features (e.g., no competitors nearby = 0)
for col_name in NUMERIC_FEATURES:
    if col_name in features.columns:
        features = features.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0.0)))

# Compute derived features that need other features
features = features.withColumn(
    "trade_area_quality",
    (
        F.col("median_income_1ring") / F.lit(100000) * F.lit(0.3)
        + F.col("max_daily_traffic_1ring") / F.lit(100000) * F.lit(0.3)
        + F.col("population_1ring") / F.lit(50000) * F.lit(0.4)
    )
).withColumn(
    "market_saturation",
    F.when(F.col("population_1ring") > 0,
           F.col("competitor_count_3ring") / F.col("population_1ring") * F.lit(10000))
    .otherwise(F.lit(0.0))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Validate & Save

# COMMAND ----------

total = features.count()
existing_count = features.filter(F.col("site_type") == "existing").count()
candidate_count = features.filter(F.col("site_type") == "candidate").count()

print(f"Total features: {total}")
print(f"  Existing stores: {existing_count}")
print(f"  Candidate sites: {candidate_count}")

# Check for nulls in key feature columns
null_report = []
for col_name in NUMERIC_FEATURES:
    if col_name in features.columns:
        nulls = features.filter(F.col(col_name).isNull()).count()
        if nulls > 0:
            null_report.append(f"  {col_name}: {nulls} nulls")

if null_report:
    print("\nNull report:")
    for line in null_report:
        print(line)
else:
    print("\nNo nulls in any numeric feature columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Distributions

# COMMAND ----------

display(
    features.select(NUMERIC_FEATURES).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Correlations with Sales (Existing Stores Only)

# COMMAND ----------

# Join sales for correlation check
stores_with_sales = features.filter(F.col("site_type") == "existing").join(
    spark.table(f"{BRONZE}.existing_stores").select("store_id", "annual_sales"),
    features.site_id == F.col("store_id"),
    how="inner",
)

print("Feature correlations with annual_sales (existing stores):")
print("=" * 55)
for feat in NUMERIC_FEATURES:
    if feat in stores_with_sales.columns:
        corr = stores_with_sales.stat.corr(feat, "annual_sales")
        if corr is None or corr != corr:  # NaN check
            print(f"  {feat:35s}  N/A (constant or null)")
            continue
        bar = "+" * int(abs(corr) * 30)
        sign = "+" if corr >= 0 else "-"
        print(f"  {feat:35s} {sign}{abs(corr):.3f} {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Gold

# COMMAND ----------

table_name = f"{GOLD}.location_features"

features.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"Saved {features.count()} rows to {table_name}")

print("Done.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {table_name} LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Built `gold.location_features` with unified features for existing stores + candidate locations:
# MAGIC - **Demographic**: population, income, target demo, college %, daytime pop (k=1 ring)
# MAGIC - **Traffic**: max daily traffic, transit score, pedestrian index (k=1 ring)
# MAGIC - **Competition**: counts at k=1/k=3, nearest dist, gravity intensity
# MAGIC - **POI**: anchors, offices, schools, foot traffic
# MAGIC - **Property**: drive-thru, parking, sqft, rent
# MAGIC - **Derived**: trade area quality, cannibalization risk, market saturation
