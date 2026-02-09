# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Seed Demo Data
# MAGIC
# MAGIC Creates catalog/schemas and generates synthetic bronze-layer data for the demo metro.
# MAGIC Skip this notebook when `demo_mode = false` (customer brings their own data).
# MAGIC
# MAGIC **Output tables** (all in `{catalog}.bronze`):
# MAGIC - `demographics` — H3-indexed census-style data (~3K rows per metro)
# MAGIC - `traffic` — Vehicle/pedestrian/transit activity per H3 hex
# MAGIC - `competitors` — QSR competitor restaurant locations (~280 per metro)
# MAGIC - `poi` — Points of interest (retail, office, school, etc.) (~1,100 per metro)
# MAGIC - `existing_stores` — Current store locations with performance metrics (~70 per metro)
# MAGIC - `locations` — Candidate real-estate sites for new stores (~1,000 per metro)
# MAGIC - `daypart_demand` — Time-of-day demand scores per H3 hex

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

if not DEMO_MODE:
    print("DEMO_MODE is false — skipping demo data generation.")
    print(f"Ensure your data exists in {CATALOG}.bronze.* tables.")
    print("Run 00_validate_schema.py to verify.")
    dbutils.notebook.exit("SKIPPED — demo_mode=false")

# COMMAND ----------

import h3
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    BooleanType, DateType, TimestampType
)

fake = Faker()
Faker.seed(42)
now = datetime.now()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Catalog & Schemas

# COMMAND ----------

for schema in [BRONZE_SCHEMA, SILVER_SCHEMA, GOLD_SCHEMA, MODELS_SCHEMA]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
    print(f"  schema: {CATALOG}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate H3 Hexagons

# COMMAND ----------

def generate_hexagons(metro_name, bounds, center):
    """Generate H3 hexagons covering a metro's bounding box."""
    hexagons = set()
    lat_step = (bounds[1] - bounds[0]) / 80
    lon_step = (bounds[3] - bounds[2]) / 80
    for lat in np.arange(bounds[0], bounds[1], lat_step):
        for lon in np.arange(bounds[2], bounds[3], lon_step):
            hexagons.add(h3.latlng_to_cell(lat, lon, H3_RES_TRADE))
    rows = []
    for hx in hexagons:
        lat, lon = h3.cell_to_latlng(hx)
        dist = np.sqrt((lat - center[0])**2 + (lon - center[1])**2)
        rows.append({"h3_index": hx, "metro": metro_name, "lat": lat, "lon": lon, "dist": dist})
    return rows

all_hexagons = []
for name, info in METROS.items():
    all_hexagons.extend(generate_hexagons(name, info["bounds"], info["center"]))

print(f"Generated {len(all_hexagons)} hexagons across {list(METROS.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Demographics

# COMMAND ----------

demo_rows = []
for h in all_hexagons:
    np.random.seed(hash(h["h3_index"]) % (2**32))
    d = h["dist"]
    is_urban = d < 0.15
    is_inner_sub = 0.15 <= d < 0.25

    pop = int(np.clip(np.exp(np.random.normal(
        np.log(8000 if is_urban else 4000 if is_inner_sub else 1500),
        0.6 if is_urban else 0.5 if is_inner_sub else 0.7
    )), 100, 20000))

    income = np.clip(
        65000 * np.random.uniform(0.7 if is_urban else 1.0 if is_inner_sub else 0.8,
                                   1.1 if is_urban else 1.5 if is_inner_sub else 1.3)
        + np.random.normal(0, 10000), 30000, 200000
    )

    pct_u18 = np.random.uniform(0.12 if is_urban else 0.22, 0.22 if is_urban else 0.32)
    pct_1834 = np.random.uniform(0.30 if is_urban else 0.15, 0.45 if is_urban else 0.25)
    pct_3554 = np.random.uniform(0.20 if is_urban else 0.28, 0.30 if is_urban else 0.38)
    pct_55 = np.clip(1.0 - pct_u18 - pct_1834 - pct_3554, 0.1, 0.4)
    med_age = np.random.uniform(28 if is_urban else 35, 36 if is_urban else 45)
    pct_college = np.clip(0.20 + ((income - 30000) / 170000) * 0.50 + np.random.normal(0, 0.05), 0.15, 0.75)
    pct_renter = np.random.uniform(0.50 if is_urban else 0.15, 0.80 if is_urban else 0.45)
    unemp = np.clip(0.08 - ((income - 30000) / 170000) * 0.05 + np.random.normal(0, 0.01), 0.02, 0.15)

    demo_rows.append((
        h["h3_index"], H3_RES_TRADE, h["metro"], h["lat"], h["lon"],
        pop, int(pop / np.random.uniform(2.2, 2.8)), round(income, 2), round(med_age, 1),
        round(pct_u18, 4), round(pct_1834, 4), round(pct_3554, 4), round(pct_55, 4),
        round(pct_college, 4), round(pct_renter, 4), round(unemp, 4), now
    ))

demo_schema = StructType([
    StructField("h3_index", StringType()), StructField("resolution", IntegerType()),
    StructField("metro", StringType()), StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()), StructField("population", IntegerType()),
    StructField("households", IntegerType()), StructField("median_income", DoubleType()),
    StructField("median_age", DoubleType()), StructField("pct_under_18", DoubleType()),
    StructField("pct_18_to_34", DoubleType()), StructField("pct_35_to_54", DoubleType()),
    StructField("pct_over_55", DoubleType()), StructField("pct_college_educated", DoubleType()),
    StructField("pct_renter", DoubleType()), StructField("unemployment_rate", DoubleType()),
    StructField("updated_at", TimestampType()),
])
demographics_df = spark.createDataFrame(demo_rows, demo_schema)
demographics_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{BRONZE}.demographics")
print(f"demographics: {demographics_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Traffic

# COMMAND ----------

traffic_rows = []
for h in all_hexagons:
    np.random.seed(hash(h["h3_index"] + "_traffic") % (2**32))
    d = h["dist"]
    is_urban = d < 0.15
    base_traffic = 25000 if is_urban else 15000 if d < 0.25 else 8000
    daily = int(np.clip(np.random.normal(base_traffic, base_traffic * 0.3), 500, 80000))
    peak = int(daily * np.random.uniform(0.08, 0.14))
    ped = round(np.clip(np.random.normal(70 if is_urban else 40, 15), 5, 100), 1)
    transit = round(np.clip(np.random.normal(75 if is_urban else 40, 15), 0, 100), 1)
    inflow = int(np.random.uniform(200, 5000) if is_urban else np.random.uniform(50, 1500))
    outflow = int(np.random.uniform(200, 5000) if is_urban else np.random.uniform(50, 1500))
    is_emp = "Y" if inflow > outflow * 1.3 else "N"
    traffic_rows.append((h["h3_index"], H3_RES_TRADE, daily, peak, ped, transit, inflow, outflow, is_emp, now))

traffic_schema = StructType([
    StructField("h3_index", StringType()), StructField("resolution", IntegerType()),
    StructField("avg_daily_traffic", IntegerType()), StructField("peak_hour_traffic", IntegerType()),
    StructField("pedestrian_index", DoubleType()), StructField("transit_score", DoubleType()),
    StructField("commute_inflow", IntegerType()), StructField("commute_outflow", IntegerType()),
    StructField("is_employment_center", StringType()), StructField("updated_at", TimestampType()),
])
traffic_df = spark.createDataFrame(traffic_rows, traffic_schema)
traffic_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{BRONZE}.traffic")
print(f"traffic: {traffic_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Competitors

# COMMAND ----------

comp_rows = []
for metro_name, metro_info in METROS.items():
    bounds = metro_info["bounds"]
    for brand, cfg in COMPETITOR_BRANDS.items():
        for i in range(cfg["count"]):
            lat = np.random.uniform(bounds[0], bounds[1])
            lon = np.random.uniform(bounds[2], bounds[3])
            h3_8 = h3.latlng_to_cell(lat, lon, 8)
            h3_9 = h3.latlng_to_cell(lat, lon, 9)
            sales = round(np.random.normal(3_300_000, 400_000), 2)
            dt = np.random.random() < cfg["drive_thru_pct"]
            opened = fake.date_between(start_date="-15y", end_date="-1y")
            comp_rows.append((
                f"{brand[:3].upper()}-{metro_name[:3].upper()}-{i:04d}",
                brand, cfg["category"], float(lat), float(lon), h3_8, h3_9,
                metro_name, sales, dt, opened, now
            ))

comp_schema = StructType([
    StructField("competitor_id", StringType()), StructField("brand", StringType()),
    StructField("category", StringType()), StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()), StructField("h3_res8", StringType()),
    StructField("h3_res9", StringType()), StructField("metro", StringType()),
    StructField("estimated_annual_sales", DoubleType()), StructField("drive_thru", BooleanType()),
    StructField("opened_date", DateType()), StructField("updated_at", TimestampType()),
])
comp_df = spark.createDataFrame(comp_rows, comp_schema)
comp_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{BRONZE}.competitors")
print(f"competitors: {comp_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Points of Interest

# COMMAND ----------

POI_TYPES = {
    "Retail": {"subs": ["Shopping Mall", "Big Box Store", "Strip Mall", "Convenience Store"], "count": 200},
    "Office": {"subs": ["Office Park", "Office Tower", "Small Office"], "count": 180},
    "School": {"subs": ["Elementary School", "Middle School", "High School", "University"], "count": 120},
    "Entertainment": {"subs": ["Movie Theater", "Gym/Fitness", "Stadium", "Bowling Alley"], "count": 100},
    "Grocery": {"subs": ["Supermarket", "Specialty Grocery", "Discount Grocery"], "count": 100},
    "Healthcare": {"subs": ["Clinic", "Medical Center", "Hospital"], "count": 80},
}
SIZES = ["small", "medium", "large", "anchor"]

poi_rows = []
for metro_name, metro_info in METROS.items():
    bounds = metro_info["bounds"]
    for cat, cfg in POI_TYPES.items():
        for i in range(cfg["count"]):
            lat = np.random.uniform(bounds[0], bounds[1])
            lon = np.random.uniform(bounds[2], bounds[3])
            sub = np.random.choice(cfg["subs"])
            sz = np.random.choice(SIZES, p=[0.3, 0.35, 0.25, 0.1])
            ft = round(np.random.uniform(20, 100), 1)
            poi_rows.append((
                f"POI-{metro_name[:3].upper()}-{cat[:3].upper()}-{i:04d}",
                fake.company() if cat in ("Retail", "Office") else f"{sub} - {fake.city_suffix()}",
                cat, sub, float(lat), float(lon),
                h3.latlng_to_cell(lat, lon, 8), metro_name, sz, ft, now
            ))

poi_schema = StructType([
    StructField("poi_id", StringType()), StructField("name", StringType()),
    StructField("category", StringType()), StructField("subcategory", StringType()),
    StructField("latitude", DoubleType()), StructField("longitude", DoubleType()),
    StructField("h3_res8", StringType()), StructField("metro", StringType()),
    StructField("size_category", StringType()), StructField("foot_traffic_index", DoubleType()),
    StructField("updated_at", TimestampType()),
])
poi_df = spark.createDataFrame(poi_rows, poi_schema)
poi_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{BRONZE}.poi")
print(f"poi: {poi_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Existing Stores

# COMMAND ----------

FORMATS = ["traditional", "express", "drive_thru_only"]
FORMAT_WEIGHTS = [0.70, 0.15, 0.15]
STORES_PER_METRO = 70

store_rows = []
for metro_name, metro_info in METROS.items():
    bounds = metro_info["bounds"]
    for i in range(STORES_PER_METRO):
        lat = np.random.uniform(bounds[0] + 0.05, bounds[1] - 0.05)
        lon = np.random.uniform(bounds[2] + 0.05, bounds[3] - 0.05)
        fmt = np.random.choice(FORMATS, p=FORMAT_WEIGHTS)
        sqft = int(np.random.normal(2200 if fmt == "traditional" else 1400, 300))
        sales = round(np.random.normal(2_000_000, 250_000), 2)
        txns = int(sales / 365 / np.random.uniform(10.5, 12.0))
        ticket = round(sales / (txns * 365), 2)
        dt_pct = round(np.random.uniform(0.55, 0.75) if fmt != "express" else 0.0, 2)
        del_pct = round(np.random.uniform(0.15, 0.35), 2)
        quality = round(np.random.uniform(20, 60), 1)
        store_rows.append((
            f"STORE-{metro_name[:3].upper()}-{i:04d}",
            f"Store #{i+1} - {metro_name}",
            float(lat), float(lon),
            h3.latlng_to_cell(lat, lon, 8), h3.latlng_to_cell(lat, lon, 9),
            metro_name, fake.date_between(start_date="-20y", end_date="-1y"),
            fmt, sqft, sales, txns, ticket, dt_pct, del_pct, quality, now
        ))

store_schema = StructType([
    StructField("store_id", StringType()), StructField("store_name", StringType()),
    StructField("latitude", DoubleType()), StructField("longitude", DoubleType()),
    StructField("h3_res8", StringType()), StructField("h3_res9", StringType()),
    StructField("metro", StringType()), StructField("opened_date", DateType()),
    StructField("format", StringType()), StructField("square_feet", IntegerType()),
    StructField("annual_sales", DoubleType()), StructField("transactions_per_day", IntegerType()),
    StructField("avg_ticket", DoubleType()), StructField("drive_thru_pct", DoubleType()),
    StructField("delivery_pct", DoubleType()), StructField("location_quality_score", DoubleType()),
    StructField("updated_at", TimestampType()),
])
store_df = spark.createDataFrame(store_rows, store_schema)
store_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{BRONZE}.existing_stores")
print(f"existing_stores: {store_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Candidate Locations

# COMMAND ----------

PROPERTY_TYPES = ["standalone_pad", "freestanding_drive_thru", "strip_mall_endcap", "urban_inline", "strip_mall_inline"]
PROP_WEIGHTS = [0.20, 0.25, 0.20, 0.20, 0.15]
LOCATIONS_PER_METRO = 1000

loc_rows = []
for metro_name, metro_info in METROS.items():
    bounds = metro_info["bounds"]
    state_map = {"Chicago": "IL", "Dallas": "TX", "Phoenix": "AZ", "Atlanta": "GA", "Denver": "CO"}
    st = state_map.get(metro_name, "XX")
    for i in range(LOCATIONS_PER_METRO):
        lat = np.random.uniform(bounds[0], bounds[1])
        lon = np.random.uniform(bounds[2], bounds[3])
        ptype = np.random.choice(PROPERTY_TYPES, p=PROP_WEIGHTS)
        dt_cap = ptype in ("standalone_pad", "freestanding_drive_thru")
        sqft = int(np.random.normal(2500 if dt_cap else 1800, 400))
        parking = int(np.random.uniform(5, 40) if dt_cap else np.random.uniform(0, 10))
        rent = round(np.random.uniform(18, 55), 2)
        loc_rows.append((
            f"LOC-{metro_name[:3].upper()}-{i:04d}",
            float(lat), float(lon),
            h3.latlng_to_cell(lat, lon, 7), h3.latlng_to_cell(lat, lon, 8), h3.latlng_to_cell(lat, lon, 9),
            fake.street_address(), fake.city(), st, fake.zipcode_in_state(st),
            metro_name, ptype, sqft, parking, dt_cap, rent, now
        ))

loc_schema = StructType([
    StructField("location_id", StringType()), StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()), StructField("h3_res7", StringType()),
    StructField("h3_res8", StringType()), StructField("h3_res9", StringType()),
    StructField("address", StringType()), StructField("city", StringType()),
    StructField("state", StringType()), StructField("zip_code", StringType()),
    StructField("metro", StringType()), StructField("property_type", StringType()),
    StructField("square_feet", IntegerType()), StructField("parking_spaces", IntegerType()),
    StructField("drive_thru_capable", BooleanType()), StructField("rent_per_sqft", DoubleType()),
    StructField("created_at", TimestampType()),
])
loc_df = spark.createDataFrame(loc_rows, loc_schema)
loc_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").partitionBy("state").saveAsTable(f"{BRONZE}.locations")
print(f"locations: {loc_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Daypart Demand

# COMMAND ----------

# Reload demographics + traffic for scoring
demographics = spark.table(f"{BRONZE}.demographics")
traffic = spark.table(f"{BRONZE}.traffic")
poi_counts = spark.table(f"{BRONZE}.poi").groupBy(F.col("h3_res8").alias("h3_index")).agg(
    F.sum(F.when(F.col("category") == "Office", 1).otherwise(0)).alias("office_count"),
    F.sum(F.when(F.col("category") == "Retail", 1).otherwise(0)).alias("retail_count"),
    F.sum(F.when(F.col("category") == "School", 1).otherwise(0)).alias("school_count"),
    F.sum(F.when(F.col("category").isin("Entertainment", "Grocery"), 1).otherwise(0)).alias("ent_count"),
    F.count("*").alias("total_poi"),
)

base = (
    demographics.select("h3_index", "metro", "population", "median_income", "pct_college_educated", "median_age")
    .join(traffic.select("h3_index", "avg_daily_traffic", "transit_score", "pedestrian_index"), "h3_index", "left")
    .join(poi_counts, "h3_index", "left")
    .fillna(0, ["avg_daily_traffic", "transit_score", "pedestrian_index", "office_count", "retail_count", "school_count", "ent_count"])
)

# Percentile-rank normalize
norm_cols = ["population", "median_income", "pct_college_educated", "avg_daily_traffic", "transit_score",
             "pedestrian_index", "office_count", "retail_count", "ent_count"]
for c in norm_cols:
    base = base.withColumn(f"{c}_p", F.percent_rank().over(Window.orderBy(F.col(c))))

daypart = base.select(
    "h3_index", "metro",
    F.round(F.least(F.lit(100), F.col("avg_daily_traffic_p")*35 + F.col("population_p")*25 + F.col("transit_score_p")*20 + F.col("pedestrian_index_p")*10 + F.rand()*10), 1).alias("breakfast_score"),
    F.round(F.least(F.lit(100), F.col("office_count_p")*30 + F.col("population_p")*25 + F.col("retail_count_p")*20 + F.col("avg_daily_traffic_p")*15 + F.rand()*10), 1).alias("lunch_score"),
    F.round(F.least(F.lit(100), F.col("population_p")*30 + F.col("median_income_p")*25 + F.col("pct_college_educated_p")*15 + F.col("retail_count_p")*15 + F.rand()*15), 1).alias("dinner_score"),
    F.round(F.least(F.lit(100), F.col("pct_college_educated_p")*35 + F.col("ent_count_p")*25 + F.col("pedestrian_index_p")*15 + F.rand()*15), 1).alias("late_night_score"),
    F.round(F.greatest(F.lit(0.8), F.least(F.lit(1.5), F.lit(0.9) + F.col("retail_count_p")*0.25 + F.col("population_p")*0.15 + F.col("ent_count_p")*0.15 + F.rand()*0.05)), 2).alias("weekend_multiplier"),
)
daypart.write.format("delta").mode("overwrite").saveAsTable(f"{BRONZE}.daypart_demand")
print(f"daypart_demand: {daypart.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

tables = ["demographics", "traffic", "competitors", "poi", "existing_stores", "locations", "daypart_demand"]
for t in tables:
    cnt = spark.table(f"{BRONZE}.{t}").count()
    print(f"  {BRONZE}.{t:20s} {cnt:>8,} rows")

print(f"\nDemo data seeded for: {list(METROS.keys())}")
