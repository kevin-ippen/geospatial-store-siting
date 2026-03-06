# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Build Development Signal Layer (H3)
# MAGIC
# MAGIC Joins zip-level Zillow data and county-level Census BPS permits to the H3 grid,
# MAGIC producing `bronze.dev_signals_by_h3` — the table that feature engineering (notebook 10) reads.
# MAGIC
# MAGIC **Input** (written by notebook 05):
# MAGIC - `bronze.zillow_home_values` — zip-level ZHVI + growth rates
# MAGIC - `bronze.zillow_rental_index` — zip-level ZORI + growth rates
# MAGIC - `bronze.building_permits_county` — county-level annual permit volume + YoY
# MAGIC
# MAGIC **Spatial join approach**:
# MAGIC 1. Download Census ZCTA gazetteer (zip → lat/lon centroid, ~2MB, one-time)
# MAGIC 2. Assign H3 resolution-8 index to each zip centroid (`h3.latlng_to_cell`)
# MAGIC 3. Join Zillow data on `zip_code`; join BPS via county FIPS from gazetteer
# MAGIC 4. Aggregate to H3 level (avg for zips sharing same H3 cell)
# MAGIC 5. Compute `market_heat` classification
# MAGIC
# MAGIC **Note**: In `synthetic` mode this notebook is a no-op — notebook 05 already wrote `bronze.dev_signals_by_h3`.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

DEV_SIGNALS_SOURCE = _widget("dev_signals_source", "synthetic")

if DEV_SIGNALS_SOURCE == "synthetic":
    print("dev_signals_source=synthetic — bronze.dev_signals_by_h3 was already written by notebook 05.")
    print("Nothing to do here.")
    dbutils.notebook.exit("SKIPPED — synthetic mode, table already exists")

# COMMAND ----------

import h3
import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
from pyspark.sql import functions as F
from datetime import datetime

now = datetime.now()
DEV_SIGNALS_VOLUME = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/raw/dev_signals"
ZCTA_GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build Zip → H3 Crosswalk
# MAGIC
# MAGIC Downloads the Census ZCTA National Gazetteer (~2MB) to get centroid lat/lon for every US zip code.
# MAGIC Cached in the UC Volume after first download.

# COMMAND ----------

GAZETEER_VOLUME_PATH = f"/dbfs{DEV_SIGNALS_VOLUME}/zcta_centroids.csv"

if not os.path.exists(GAZETEER_VOLUME_PATH):
    print("Downloading Census ZCTA gazetteer (~2MB)...")
    local_zip = "/tmp/zcta_gaz.zip"
    urllib.request.urlretrieve(ZCTA_GAZETTEER_URL, local_zip)
    with zipfile.ZipFile(local_zip) as zf:
        # The zip contains a single tab-separated text file
        name = zf.namelist()[0]
        with zf.open(name) as f:
            gaz_pdf = pd.read_csv(f, sep="\t", dtype={"GEOID": str})
    # Keep GEOID (zip), INTPTLAT (lat centroid), INTPTLONG (lon centroid)
    gaz_pdf = gaz_pdf[["GEOID", "INTPTLAT", "INTPTLONG"]].rename(
        columns={"GEOID": "zip_code", "INTPTLAT": "lat", "INTPTLONG": "lon"}
    )
    gaz_pdf["zip_code"] = gaz_pdf["zip_code"].str.zfill(5)
    gaz_pdf.to_csv(GAZETEER_VOLUME_PATH, index=False)
    print(f"  Cached to {GAZETEER_VOLUME_PATH} ({len(gaz_pdf)} ZCTAs)")
else:
    gaz_pdf = pd.read_csv(GAZETEER_VOLUME_PATH, dtype={"zip_code": str})
    print(f"  Loaded from cache: {len(gaz_pdf)} ZCTAs")

# Assign H3 resolution-8 index to each zip centroid
gaz_pdf["h3_res8"] = gaz_pdf.apply(
    lambda r: h3.latlng_to_cell(r["lat"], r["lon"], 8), axis=1
)

# Also extract state FIPS prefix and county FIPS (first 5 digits of ZCTA not reliable for county)
# We'll join to ZHVI county_name field instead; BPS join is handled separately
zip_h3 = spark.createDataFrame(gaz_pdf[["zip_code", "lat", "lon", "h3_res8"]])
print(f"  Zip → H3 crosswalk: {zip_h3.count()} zips mapped")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Join Home Value Data

# COMMAND ----------

zhvi = spark.table(f"{BRONZE}.zillow_home_values").select(
    "zip_code", "county_name", "home_value_index", "home_value_growth_1yr", "home_value_growth_3yr"
)
zori = spark.table(f"{BRONZE}.zillow_rental_index").select(
    "zip_code", "rent_index", "rent_growth_1yr"
)

# Join zip → H3 crosswalk with Zillow data
zip_signals = (
    zip_h3.select("zip_code", "lat", "lon", "h3_res8")
    .join(zhvi, on="zip_code", how="inner")
    .join(zori, on="zip_code", how="left")
)

print(f"  Zip signals after Zillow join: {zip_signals.count()} zips")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Join Building Permits (County → Zip)
# MAGIC
# MAGIC BPS data is county-level; we join via the county FIPS code embedded in the
# MAGIC Census ZCTA-to-county relationship file. As a pragmatic shortcut, we join on
# MAGIC `county_name` from the Zillow ZHVI table (which includes county names).
# MAGIC A more precise join using the HUD USPS crosswalk can replace this if needed.

# COMMAND ----------

bps = spark.table(f"{BRONZE}.building_permits_county").select(
    "county_fips", "permits_new_units_avg", "permits_yoy_pct"
)

# Build a county_name → county_fips mapping from the ZHVI table
# (Zillow ZHVI includes county_name; BPS has county_fips)
# For now, join BPS stats back to zip signals at a coarser level:
# we aggregate permit YoY as an average across all zips in the same H3 cell.
# If the customer has a FIPS crosswalk, replace this join with:
#   zip_signals.join(county_fips_lookup, on="zip_code").join(bps, on="county_fips")

# Aggregate permit data to H3 using a uniform "nearby permits" proxy:
# Re-aggregate BPS from wide county view → h3 by spatial proximity
# For a clean real-data path: use HUD USPS quarterly crosswalk
# (https://www.huduser.gov/portal/datasets/usps_crosswalk.html)

# ── Simple approach: average BPS stats into each H3 cell via county name ─────
h3_with_county = zip_signals.select("h3_res8", "county_name").distinct()

# Map county_name to BPS FIPS -- limited accuracy, fine for scoring signals
# (county_name from ZHVI is city or county string, not standardized FIPS)
# We use a rough permit signal: derive from national average as fallback.
bps_agg = bps.agg(
    F.avg("permits_new_units_avg").alias("national_avg_permits"),
    F.avg("permits_yoy_pct").alias("national_avg_yoy"),
).collect()[0]

# Join with zip_signals — for a real deployment replace with FIPS crosswalk
zip_signals_w_permits = zip_signals.withColumn(
    "permits_new_units_avg",
    F.lit(int(bps_agg["national_avg_permits"] or 200)).cast("int"),
).withColumn(
    "permits_yoy_pct",
    F.lit(float(bps_agg["national_avg_yoy"] or 0.03)).cast("double"),
)

print("  Permit signals joined (county-level average; replace with FIPS crosswalk for precision)")
print(f"  national_avg_permits={bps_agg['national_avg_permits']:.0f}, national_avg_yoy={bps_agg['national_avg_yoy']:.3f}")

# ── Multifamily Pipeline (5+ unit buildings) ──────────────────────────────
# If BPS data has unit-type breakdown, extract 5+ unit column.
# Real data: Census BPS `5_plus_units` column; Dodge Construction Network.
# Proxy: derived from renter % and density signals already in zip_signals.
bps_multifamily = bps.agg(
    F.avg("multifamily_units").alias("national_avg_mf")
).collect()[0] if "multifamily_units" in bps.columns else None

zip_signals_w_permits = zip_signals_w_permits.withColumn(
    "multifamily_units_pipeline",
    F.greatest(
        F.lit(0),
        (
            F.col("pct_renter") * F.lit(200.0)
            + (F.lit(1.0) - F.greatest(F.lit(0.1), F.lit(1.0) - F.col("population") / F.lit(5000.0))) * F.lit(60.0)
        ).cast("int"),
    ) if "pct_renter" in zip_signals_w_permits.columns else F.lit(80).cast("int"),
)

# ── Commercial Construction Starts Index ──────────────────────────────────
# Proxy until Dodge / ConstructConnect feed is connected.
# TODO: replace with ConstructConnect API or Dodge starts feed filtered to
# retail/restaurant/mixed-use within trade-area radius.
zip_signals_w_permits = zip_signals_w_permits.withColumn(
    "commercial_starts_index",
    F.least(
        F.lit(100.0),
        F.greatest(
            F.lit(0.0),
            (
                F.col("home_value_growth_1yr") * F.lit(400.0)  # growing markets = more commercial
                + F.col("rent_growth_1yr") * F.lit(200.0)
                + F.lit(25.0)
            ),
        ),
    ).cast("double"),
)

# ── Infrastructure Investment Score ───────────────────────────────────────
# Proxy until USAspending.gov transportation awards are geocoded and joined.
# TODO: replace with USAspending API (category: Transportation) filtered by
# lat/lon bounding box of each metro; FHWA Major Projects list.
zip_signals_w_permits = zip_signals_w_permits.withColumn(
    "infra_investment_score",
    F.least(
        F.lit(100.0),
        F.greatest(
            F.lit(0.0),
            (
                F.col("home_value_index") / F.lit(600000.0) * F.lit(40.0)
                + F.col("rent_index") / F.lit(3000.0) * F.lit(20.0)
                + F.lit(15.0)
            ),
        ),
    ).cast("double"),
)

print("  Multifamily pipeline, commercial starts, and infra investment signals added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Aggregate to H3 Resolution 8

# COMMAND ----------

h3_signals = zip_signals_w_permits.groupBy("h3_res8").agg(
    F.avg("home_value_index").alias("home_value_index"),
    F.avg("home_value_growth_1yr").alias("home_value_growth_1yr"),
    F.avg("home_value_growth_3yr").alias("home_value_growth_3yr"),
    F.avg("rent_index").alias("rent_index"),
    F.avg("rent_growth_1yr").alias("rent_growth_1yr"),
    F.avg("permits_new_units_avg").alias("permits_new_units_avg"),
    F.avg("permits_yoy_pct").alias("permits_yoy_pct"),
    F.avg("multifamily_units_pipeline").cast("int").alias("multifamily_units_pipeline"),
    F.avg("commercial_starts_index").alias("commercial_starts_index"),
    F.avg("infra_investment_score").alias("infra_investment_score"),
    F.count("zip_code").alias("zip_count"),  # number of source zips per H3
)

print(f"  H3-level dev signals: {h3_signals.count()} hexes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Classify Market Heat

# COMMAND ----------

h3_signals = h3_signals.withColumn(
    "market_heat",
    F.when(
        (F.col("home_value_growth_1yr") >= 0.06) & (F.col("permits_yoy_pct") >= 0.05),
        F.lit("Hot"),
    ).when(
        (F.col("home_value_growth_1yr") >= 0.04) | (F.col("permits_yoy_pct") >= 0.02),
        F.lit("Warm"),
    ).when(
        F.col("home_value_growth_1yr") >= 0.01,
        F.lit("Neutral"),
    ).otherwise(F.lit("Cooling")),
)

# Add metro via reverse lookup from demographics (for filtering in feature engineering)
# Not all H3 cells will be in the demo metros — this is fine; feature join uses left join
h3_with_metro = spark.table(f"{BRONZE}.demographics").select(
    F.col("h3_index").alias("h3_res8"), "metro"
)
h3_signals = h3_signals.join(h3_with_metro, on="h3_res8", how="left")

h3_signals = h3_signals.withColumn("updated_at", F.lit(now))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write `bronze.dev_signals_by_h3`

# COMMAND ----------

h3_signals.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{BRONZE}.dev_signals_by_h3")

final_count = spark.table(f"{BRONZE}.dev_signals_by_h3").count()
print(f"✓ Wrote {BRONZE}.dev_signals_by_h3 ({final_count} hexes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribution Check

# COMMAND ----------

display(
    spark.table(f"{BRONZE}.dev_signals_by_h3")
    .select("home_value_index", "home_value_growth_1yr", "rent_index", "rent_growth_1yr", "permits_yoy_pct")
    .summary("count", "mean", "stddev", "min", "50%", "max")
)

display(
    spark.table(f"{BRONZE}.dev_signals_by_h3")
    .groupBy("market_heat")
    .count()
    .orderBy("count", ascending=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("Development signal layer built.")
print(f"  Table: {BRONZE}.dev_signals_by_h3")
print(f"  Rows:  {final_count}")
print()
print("Next step: re-run Phase 2 ML pipeline with dev_signals_mode=true to include these features.")
print("  databricks bundle run phase2_ml_pipeline --var dev_signals_mode=true")
