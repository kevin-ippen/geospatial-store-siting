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
# MAGIC ## 9b. Calibrate Huff β from Observed Sales
# MAGIC
# MAGIC Instead of hardcoding β = 2.0, we grid-search β ∈ [1.0, 3.0] to find the value
# MAGIC that minimizes MSE between Huff-predicted demand and actual store sales.
# MAGIC The fitted β is logged to MLflow and used for all downstream Huff computations.

# COMMAND ----------

import mlflow
from pyspark.sql import Window as W

# --- Huff β calibration: grid search over existing stores ---
# For each candidate β, compute Huff expected demand for existing stores,
# then pick the β that best correlates with actual sales.

_existing_for_cal = spark.table(f"{BRONZE}.existing_stores").select(
    F.col("store_id"), F.col("annual_sales"), F.col("metro"),
    F.col("latitude"), F.col("longitude"),
    F.col("square_feet"), F.col("drive_thru_pct"),
).toPandas()

# Pre-collect supply points and trade area data for calibration
_supply_pdf = all_supply.select("supply_id", "supply_lat", "supply_lon", "attractiveness", "metro").toPandas()

# Collect demographics for hex-level demand
_demo_cal = spark.table(f"{BRONZE}.demographics").select(
    "h3_index", "population", "latitude", "longitude", "metro"
).toPandas()

import math as _math

def _haversine_py(lat1, lon1, lat2, lon2):
    return _math.sqrt(((lat1 - lat2) * 69.0) ** 2
                      + ((lon1 - lon2) * 69.0 * _math.cos(_math.radians(lat1))) ** 2)

beta_min, beta_max, beta_step = HUFF_BETA_RANGE
beta_candidates = np.arange(beta_min, beta_max + beta_step, beta_step)
beta_results = []

DAILY_SPEND_PER_CAPITA = 12.0

print(f"Calibrating Huff β over {len(beta_candidates)} values [{beta_min}, {beta_max}]...")
for beta_val in beta_candidates:
    predicted_demands = []
    actual_sales = []
    for _, store in _existing_for_cal.iterrows():
        s_lat, s_lon, s_metro = store["latitude"], store["longitude"], store["metro"]
        s_sqft = float(store["square_feet"] or 2000)
        s_dt = float(store["drive_thru_pct"] or 0)
        s_attract = s_sqft * (1.0 + (0.3 if s_dt > 0 else 0.0))

        # Trade area: hexes within ~1.5 miles
        metro_hexes = _demo_cal[_demo_cal["metro"] == s_metro]
        nearby_hexes = metro_hexes[
            metro_hexes.apply(lambda h: _haversine_py(s_lat, s_lon, h["latitude"], h["longitude"]) <= 1.5, axis=1)
        ]
        if len(nearby_hexes) == 0:
            continue

        # Supply points in this metro
        metro_supply = _supply_pdf[_supply_pdf["metro"] == s_metro]

        total_huff_demand = 0.0
        for _, hx in nearby_hexes.iterrows():
            h_lat, h_lon, h_pop = hx["latitude"], hx["longitude"], hx["population"]
            # Site gravity
            d_site = max(_haversine_py(h_lat, h_lon, s_lat, s_lon), 0.05)
            site_grav = s_attract / (d_site ** beta_val)
            # Total supply gravity
            total_grav = 0.0
            for _, sp in metro_supply.iterrows():
                d_sp = max(_haversine_py(h_lat, h_lon, sp["supply_lat"], sp["supply_lon"]), 0.05)
                if d_sp <= HUFF_CANNIB_RADIUS:
                    total_grav += sp["attractiveness"] / (d_sp ** beta_val)
            total_grav += site_grav  # include self
            huff_prob = site_grav / total_grav if total_grav > 0 else 0
            total_huff_demand += h_pop * DAILY_SPEND_PER_CAPITA * 365 * huff_prob

        predicted_demands.append(total_huff_demand)
        actual_sales.append(store["annual_sales"])

    if len(predicted_demands) > 10:
        pred_arr = np.array(predicted_demands)
        actual_arr = np.array(actual_sales)
        mse = np.mean((pred_arr - actual_arr) ** 2)
        corr = np.corrcoef(pred_arr, actual_arr)[0, 1]
        beta_results.append({"beta": round(beta_val, 2), "mse": mse, "corr": corr})
        print(f"  β={beta_val:.2f}  MSE={mse:.2e}  corr={corr:.3f}")

# Pick best β by highest correlation (more robust than MSE with synthetic data)
best_beta_row = max(beta_results, key=lambda r: r["corr"])
CALIBRATED_HUFF_BETA = best_beta_row["beta"]

print(f"\nCalibrated Huff β = {CALIBRATED_HUFF_BETA} (corr={best_beta_row['corr']:.3f})")
print(f"  Default was β = {HUFF_BETA}")

# Log to MLflow
try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="huff_beta_calibration", nested=True):
        mlflow.log_param("beta_range", f"{beta_min}-{beta_max}")
        mlflow.log_param("beta_step", beta_step)
        mlflow.log_metric("calibrated_beta", CALIBRATED_HUFF_BETA)
        mlflow.log_metric("best_correlation", best_beta_row["corr"])
        mlflow.log_metric("best_mse", best_beta_row["mse"])
    print(f"  Logged calibrated β to MLflow experiment: {MLFLOW_EXPERIMENT}")
except Exception as e:
    print(f"  MLflow logging skipped: {e}")

# Use calibrated β for all downstream Huff computations
HUFF_BETA = CALIBRATED_HUFF_BETA

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9c. Huff Gravity Model Features
# MAGIC
# MAGIC Computes probabilistic market share using the Huff model with **calibrated β**:
# MAGIC `P(hex_i → site_j) = (Attract_j / dist_ij^β) / Σ_k(Attract_k / dist_ik^β)`
# MAGIC
# MAGIC **Outputs**: `huff_market_share` (average probability across trade area) and
# MAGIC `huff_expected_demand` (annual demand captured from surrounding population).

# All supply points: existing stores + competitors (each has location, sqft, drive-thru)
supply_stores = spark.table(f"{BRONZE}.existing_stores").select(
    F.col("store_id").alias("supply_id"),
    F.col("latitude").alias("supply_lat"),
    F.col("longitude").alias("supply_lon"),
    F.col("square_feet").alias("supply_sqft"),
    F.col("drive_thru_pct").alias("supply_dt"),
    F.col("metro"),
)
supply_competitors = competitors.select(
    F.col("competitor_id").alias("supply_id"),
    F.col("latitude").alias("supply_lat"),
    F.col("longitude").alias("supply_lon"),
    F.lit(2200).alias("supply_sqft"),  # assumed avg for competitors
    F.when(F.col("drive_thru"), 0.7).otherwise(0.0).alias("supply_dt"),
    F.col("metro"),
)
all_supply = supply_stores.unionByName(supply_competitors)

# Attractiveness = sqft * (1 + drive_thru_flag * 0.3)
all_supply = all_supply.withColumn(
    "attractiveness",
    F.col("supply_sqft") * (F.lit(1.0) + F.when(F.col("supply_dt") > 0, F.lit(0.3)).otherwise(F.lit(0.0)))
)

# For each site, expand to k=2 ring hexagons (trade area)
# Then for each hex in the trade area, compute Huff probabilities

# Step 1: Get site trade area hexagons with population
site_trade = sites_ring2.join(
    demographics.select(
        F.col("h3_index").alias("neighbor_h3"),
        "population", "latitude", "longitude",
    ),
    on="neighbor_h3",
    how="inner",
).select(
    "site_id", "neighbor_h3",
    F.col("population").alias("hex_pop"),
    F.col("latitude").alias("hex_lat"),
    F.col("longitude").alias("hex_lon"),
)

# Step 2: For each hex in each site's trade area, find all supply points within HUFF_CANNIB_RADIUS
# We need the site's attractiveness too — get it from all_sites
site_attract = all_sites.select(
    "site_id", "metro",
    F.col("square_feet").alias("site_sqft"),
    F.col("drive_thru_pct").alias("site_dt"),
    F.col("latitude").alias("site_lat"),
    F.col("longitude").alias("site_lon"),
).withColumn(
    "site_attractiveness",
    F.coalesce(F.col("site_sqft"), F.lit(2000)) * (F.lit(1.0) + F.when(F.col("site_dt") > 0, F.lit(0.3)).otherwise(F.lit(0.0)))
)

# Join trade area hexes with site attractiveness
trade_with_site = site_trade.join(
    site_attract.select("site_id", "metro", "site_lat", "site_lon", "site_attractiveness"),
    on="site_id",
    how="inner",
)

# Compute distance from each trade-area hex to the focal site
trade_with_site = trade_with_site.withColumn(
    "site_hex_dist",
    F.greatest(haversine_miles_expr(F.col("hex_lat"), F.col("hex_lon"), F.col("site_lat"), F.col("site_lon")), F.lit(0.05))
)

# Compute site gravity for each hex: attract / dist^beta
trade_with_site = trade_with_site.withColumn(
    "site_gravity", F.col("site_attractiveness") / F.pow(F.col("site_hex_dist"), F.lit(HUFF_BETA))
)

# Step 3: For each hex, sum gravity of ALL supply points within radius
# Cross-join hexes with supply points (same metro), compute distance + gravity
hex_supply = trade_with_site.select(
    "site_id", "neighbor_h3", "hex_pop", "hex_lat", "hex_lon", "metro", "site_gravity"
).join(
    all_supply.select("supply_id", "supply_lat", "supply_lon", "attractiveness", F.col("metro").alias("s_metro")),
    F.col("metro") == F.col("s_metro"),
    how="inner",
).drop("s_metro")

hex_supply = hex_supply.withColumn(
    "supply_dist",
    F.greatest(haversine_miles_expr(F.col("hex_lat"), F.col("hex_lon"), F.col("supply_lat"), F.col("supply_lon")), F.lit(0.05))
).filter(
    F.col("supply_dist") <= HUFF_CANNIB_RADIUS
).withColumn(
    "supply_gravity", F.col("attractiveness") / F.pow(F.col("supply_dist"), F.lit(HUFF_BETA))
)

# Sum all supply gravity per (site_id, hex)
total_gravity_per_hex = hex_supply.groupBy("site_id", "neighbor_h3", "hex_pop", "site_gravity").agg(
    F.sum("supply_gravity").alias("total_supply_gravity"),
)

# Huff probability = site_gravity / (site_gravity + total_supply_gravity)
huff_probs = total_gravity_per_hex.withColumn(
    "huff_prob",
    F.col("site_gravity") / (F.col("site_gravity") + F.col("total_supply_gravity"))
)

# Expected demand: population * $12/day avg spending * 365 * huff_prob
DAILY_SPEND_PER_CAPITA = 12.0
huff_features = huff_probs.groupBy("site_id").agg(
    F.avg("huff_prob").alias("huff_market_share"),
    F.sum(F.col("hex_pop") * F.lit(DAILY_SPEND_PER_CAPITA) * F.lit(365) * F.col("huff_prob")).alias("huff_expected_demand"),
)

print(f"Huff features: {huff_features.count()} sites")
display(huff_features.describe())

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
    .join(huff_features, on="site_id", how="left")
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
# MAGIC - **Derived**: trade area quality, cannibalization risk, market saturation, Huff market share, Huff expected demand
