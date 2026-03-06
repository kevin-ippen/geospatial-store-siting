# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Ingest Development Signals
# MAGIC
# MAGIC Brings in real-estate development signals from three public data sources:
# MAGIC
# MAGIC | Source | Dataset | Granularity |
# MAGIC |--------|---------|-------------|
# MAGIC | Zillow Research | ZHVI — Home Value Index | Zip code |
# MAGIC | Zillow Research | ZORI — Observed Rent Index | Zip code |
# MAGIC | Census Bureau | Building Permit Survey (BPS) | County (FIPS) |
# MAGIC
# MAGIC **Three ingestion modes** controlled by `dev_signals_source` widget:
# MAGIC
# MAGIC | Mode | When to use | Output |
# MAGIC |------|-------------|--------|
# MAGIC | `synthetic` (default) | Demo / first run — no external data required | Writes `bronze.dev_signals_by_h3` directly from existing `bronze.demographics` |
# MAGIC | `volume` | Customer pre-placed CSVs in `/Volumes/{catalog}/bronze/raw/dev_signals/` | 3 bronze tables consumed by notebook 06 |
# MAGIC | `download` | Fetch live from Zillow Research public URLs + Census BPS API | 3 bronze tables consumed by notebook 06 |
# MAGIC
# MAGIC **Downstream**: notebook `06_build_dev_signal_layer` joins zip/county tables to H3 grid (only needed for `volume` / `download` modes).

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

DEV_SIGNALS_SOURCE = _widget("dev_signals_source", "synthetic")

print(f"Config: catalog={CATALOG}, dev_signals_source={DEV_SIGNALS_SOURCE}")

# COMMAND ----------

import h3
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
)
from datetime import datetime

now = datetime.now()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mode: Synthetic
# MAGIC
# MAGIC Derives spatially coherent development signals from existing `bronze.demographics`.
# MAGIC No external data needed — safe for demos and first-time setup.
# MAGIC
# MAGIC Writes `bronze.dev_signals_by_h3` directly (notebook 06 is skipped in this mode).

# COMMAND ----------

if DEV_SIGNALS_SOURCE == "synthetic":

    demo = spark.table(f"{BRONZE}.demographics").select(
        F.col("h3_index").alias("h3_res8"),
        "metro",
        "latitude",
        "longitude",
        "median_income",
        "pct_renter",
        "population",
    )

    # ── Home Value Index (Zillow ZHVI proxy) ──────────────────────────────────
    # Home value ≈ income × affordability ratio (3.5–6.5×), urban core higher
    # Growth ≈ 3–8% YoY, inversely correlated with current value (catch-up markets)
    dev = demo.withColumn(
        "_income_norm", F.col("median_income") / F.lit(80000.0)
    ).withColumn(
        "home_value_index",
        (
            F.col("median_income") * (F.lit(4.0) + F.col("_income_norm") * F.lit(1.5))
            + (F.rand(seed=42) - F.lit(0.5)) * F.lit(40000.0)
        ).cast("double"),
    ).withColumn(
        # Growth higher where values are lower (value-add / growth markets)
        "home_value_growth_1yr",
        (
            F.lit(0.055)
            - (F.col("_income_norm") - F.lit(1.0)) * F.lit(0.01)
            + (F.rand(seed=43) - F.lit(0.5)) * F.lit(0.03)
        ).cast("double"),
    ).withColumn(
        "home_value_growth_3yr",
        (F.col("home_value_growth_1yr") * F.lit(0.9) + (F.rand(seed=44) - F.lit(0.5)) * F.lit(0.02)).cast("double"),
    )

    # ── Rent Index (Zillow ZORI proxy) ────────────────────────────────────────
    # Monthly rent ≈ income × 30% / 12 for renter markets, adjusted by renter %
    dev = dev.withColumn(
        "rent_index",
        (
            F.col("median_income") * F.lit(0.30) / F.lit(12.0)
            * (F.lit(0.8) + F.col("pct_renter") * F.lit(0.5))
            + (F.rand(seed=45) - F.lit(0.5)) * F.lit(150.0)
        ).cast("double"),
    ).withColumn(
        "rent_growth_1yr",
        (
            F.lit(0.040)
            + F.col("pct_renter") * F.lit(0.02)
            + (F.rand(seed=46) - F.lit(0.5)) * F.lit(0.025)
        ).cast("double"),
    )

    # ── Building Permit Activity (Census BPS proxy) ───────────────────────────
    # New construction higher in lower-density, higher-income suburban areas
    dev = dev.withColumn(
        "_density_factor",
        F.greatest(F.lit(0.1), F.lit(1.0) - F.col("population") / F.lit(5000.0)),
    ).withColumn(
        "permits_new_units_avg",
        (
            F.col("_density_factor") * F.col("_income_norm") * F.lit(120.0)
            + F.rand(seed=47) * F.lit(80.0)
        ).cast("int"),
    ).withColumn(
        "permits_yoy_pct",
        (
            F.col("_density_factor") * F.lit(0.08)
            + (F.rand(seed=48) - F.lit(0.5)) * F.lit(0.15)
        ).cast("double"),
    )

    # ── Multifamily Pipeline (Census BPS 5+ unit proxy) ──────────────────────
    # Units in 5+ unit buildings authorized within the next 1-2 years.
    # Distinct from total permits (which includes single-family): multifamily drives
    # daytime + evening foot traffic density far more than equivalent SFH count.
    # Real data: Census BPS columns `5_plus_units`; Dodge Construction Network.
    dev = dev.withColumn(
        "multifamily_units_pipeline",
        F.greatest(
            F.lit(0),
            (
                F.col("pct_renter") * F.lit(200.0)
                + (F.lit(1.0) - F.col("_density_factor")) * F.col("_income_norm") * F.lit(60.0)
                + F.rand(seed=51) * F.lit(80.0)
            ).cast("int"),
        ),
    )

    # ── Commercial Construction Starts Index (Dodge / ConstructConnect proxy) ─
    # 0–100 index scoring new commercial construction activity (retail, restaurant,
    # mixed-use) near the hex cell. A new anchor tenant or mixed-use node signals
    # future co-tenancy opportunity and trade area expansion.
    # Real data: Dodge Construction Network starts feed; ConstructConnect project leads;
    # BuildCentral; or city open-data commercial permit extracts.
    dev = dev.withColumn(
        "commercial_starts_index",
        F.least(
            F.lit(100.0),
            F.greatest(
                F.lit(0.0),
                (
                    F.col("_density_factor") * F.col("_income_norm") * F.lit(65.0)
                    + F.col("pct_renter") * F.lit(15.0)
                    + (F.rand(seed=50) - F.lit(0.3)) * F.lit(30.0)
                ),
            ),
        ).cast("double"),
    )

    # ── Infrastructure Investment Score (USAspending / FHWA proxy) ────────────
    # 0–100 score for active federal infrastructure investment within 2 miles:
    # new interchanges, transit stations, road rebuilds, port upgrades.
    # These create medium-term access and foot-traffic uplift.
    # Real data: USAspending.gov awards API (Transportation category, filtered by
    # geography); FHWA Major Projects list; USDOT Build America financed projects;
    # state DOT STIP/TIP plans.
    dev = dev.withColumn(
        "infra_investment_score",
        F.least(
            F.lit(100.0),
            F.greatest(
                F.lit(0.0),
                (
                    (F.lit(1.0) - F.col("_density_factor")) * F.lit(40.0)
                    + F.col("_income_norm") * F.lit(20.0)
                    + F.rand(seed=52) * F.lit(25.0)
                ),
            ),
        ).cast("double"),
    )

    # ── Market Heat Classification ─────────────────────────────────────────────
    dev = dev.withColumn(
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

    dev_signals = dev.select(
        "h3_res8",
        "metro",
        "home_value_index",
        "home_value_growth_1yr",
        "home_value_growth_3yr",
        "rent_index",
        "rent_growth_1yr",
        "permits_new_units_avg",
        "permits_yoy_pct",
        "multifamily_units_pipeline",
        "commercial_starts_index",
        "infra_investment_score",
        "market_heat",
        F.lit(now).alias("updated_at"),
    )

    dev_signals.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{BRONZE}.dev_signals_by_h3"
    )

    row_count = spark.table(f"{BRONZE}.dev_signals_by_h3").count()
    print(f"✓ Synthetic dev signals written: {row_count} rows → {BRONZE}.dev_signals_by_h3")
    print("  Notebook 06 is not needed in synthetic mode.")
    dbutils.notebook.exit(f"OK — synthetic, {row_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mode: Volume or Download
# MAGIC
# MAGIC Reads real data from Zillow Research and Census BPS.
# MAGIC Produces 3 intermediate bronze tables consumed by notebook 06.

# COMMAND ----------

# Volume paths for pre-placed CSVs
DEV_SIGNALS_VOLUME = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/raw/dev_signals"

# Zillow Research public CSV URLs (as of 2025 — verify current URLs at zillow.com/research/data/)
ZHVI_URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
ZORI_URL = "https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_month.csv"

# Census BPS — downloadable from: https://www.census.gov/construction/bps/
# Format: annual CSV at county level (residential permits).
# Manual download required; save to Volume as building_permits_county.csv
BPS_VOLUME_PATH = f"{DEV_SIGNALS_VOLUME}/building_permits_county.csv"

# COMMAND ----------

import os
import urllib.request
import pandas as pd

def _ensure_volume_dir():
    dbutils.fs.mkdirs(DEV_SIGNALS_VOLUME)

def _read_or_download_zillow(url: str, local_name: str) -> pd.DataFrame:
    """Read from UC Volume if present, otherwise download from URL."""
    local_path = f"/tmp/{local_name}"
    volume_path = f"{DEV_SIGNALS_VOLUME}/{local_name}"

    if DEV_SIGNALS_SOURCE == "volume":
        # Read from Volume only — fail if not present
        try:
            return pd.read_csv(f"/dbfs{volume_path}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found at {volume_path}.\n"
                f"Download from: {url}\n"
                f"Then upload to the volume: {DEV_SIGNALS_VOLUME}/"
            )
    else:
        # download mode
        print(f"  Downloading {local_name}...")
        urllib.request.urlretrieve(url, local_path)
        return pd.read_csv(local_path)

_ensure_volume_dir()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Zillow ZHVI — Home Value Index

# COMMAND ----------

zhvi_pdf = _read_or_download_zillow(ZHVI_URL, "zillow_zhvi.csv")

print(f"  ZHVI raw shape: {zhvi_pdf.shape}")

# Zillow CSV is wide-format: columns = [RegionID, SizeRank, RegionName, RegionType, StateName, State, City, Metro, CountyName, <date_cols...>]
# We need the two most recent month columns for 1yr/3yr growth rate calculation
date_cols = sorted([c for c in zhvi_pdf.columns if c[:2] in ("19", "20") and len(c) == 10])
if len(date_cols) < 36:
    raise ValueError(f"ZHVI CSV has fewer than 36 monthly columns; got {len(date_cols)}")

col_latest = date_cols[-1]
col_1yr_ago = date_cols[-13]     # 12 months back
col_3yr_ago = date_cols[-37]     # 36 months back

zhvi_slim = zhvi_pdf[["RegionName", "StateName", "Metro", "CountyName", col_latest, col_1yr_ago, col_3yr_ago]].copy()
zhvi_slim.columns = ["zip_code", "state", "metro", "county_name", "home_value_index", "hvi_1yr_ago", "hvi_3yr_ago"]
zhvi_slim["zip_code"] = zhvi_slim["zip_code"].astype(str).str.zfill(5)
zhvi_slim = zhvi_slim.dropna(subset=["home_value_index"])

# Compute growth rates
zhvi_slim["home_value_growth_1yr"] = (
    zhvi_slim["home_value_index"] - zhvi_slim["hvi_1yr_ago"]
) / zhvi_slim["hvi_1yr_ago"].replace(0, float("nan"))

zhvi_slim["home_value_growth_3yr"] = (
    (zhvi_slim["home_value_index"] / zhvi_slim["hvi_3yr_ago"].replace(0, float("nan"))) ** (1 / 3) - 1
)

zhvi_slim = zhvi_slim.drop(columns=["hvi_1yr_ago", "hvi_3yr_ago"])
zhvi_slim["updated_at"] = now

zhvi_spark = spark.createDataFrame(zhvi_slim)
zhvi_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{BRONZE}.zillow_home_values"
)
print(f"✓ Wrote {BRONZE}.zillow_home_values ({zhvi_slim.shape[0]} zip codes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Zillow ZORI — Observed Rent Index

# COMMAND ----------

zori_pdf = _read_or_download_zillow(ZORI_URL, "zillow_zori.csv")

print(f"  ZORI raw shape: {zori_pdf.shape}")

date_cols_r = sorted([c for c in zori_pdf.columns if c[:2] in ("19", "20") and len(c) == 10])
if len(date_cols_r) < 13:
    raise ValueError(f"ZORI CSV has fewer than 13 monthly columns; got {len(date_cols_r)}")

col_latest_r = date_cols_r[-1]
col_1yr_ago_r = date_cols_r[-13]

zori_slim = zori_pdf[["RegionName", col_latest_r, col_1yr_ago_r]].copy()
zori_slim.columns = ["zip_code", "rent_index", "rent_1yr_ago"]
zori_slim["zip_code"] = zori_slim["zip_code"].astype(str).str.zfill(5)
zori_slim = zori_slim.dropna(subset=["rent_index"])

zori_slim["rent_growth_1yr"] = (
    zori_slim["rent_index"] - zori_slim["rent_1yr_ago"]
) / zori_slim["rent_1yr_ago"].replace(0, float("nan"))

zori_slim = zori_slim.drop(columns=["rent_1yr_ago"])
zori_slim["updated_at"] = now

zori_spark = spark.createDataFrame(zori_slim)
zori_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{BRONZE}.zillow_rental_index"
)
print(f"✓ Wrote {BRONZE}.zillow_rental_index ({zori_slim.shape[0]} zip codes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Census BPS — Building Permits by County
# MAGIC
# MAGIC **Download instructions** (manual step, one-time setup):
# MAGIC 1. Go to: https://www.census.gov/construction/bps/
# MAGIC 2. Download the annual county-level CSV for the most recent 5 years
# MAGIC 3. Upload each file to: `/Volumes/{catalog}/bronze/raw/dev_signals/`
# MAGIC 4. Name files: `bps_2019.csv`, `bps_2020.csv`, ... `bps_2023.csv`
# MAGIC
# MAGIC **Columns expected**: `state_fips`, `county_fips`, `county_name`, `total_units` (or `1_unit`, `2_units`, `3_4_units`, `5_plus_units`)
# MAGIC
# MAGIC If BPS file is not found, synthetic permit data will be generated as a fallback.

# COMMAND ----------

# Try to load BPS from Volume; if missing, generate synthetic county-level permits
_bps_dfs = []
for yr in range(2019, 2024):
    bps_path = f"/dbfs{DEV_SIGNALS_VOLUME}/bps_{yr}.csv"
    if os.path.exists(bps_path):
        _bdf = pd.read_csv(bps_path)
        _bdf["year"] = yr
        _bps_dfs.append(_bdf)
        print(f"  Loaded BPS {yr}: {len(_bdf)} counties")

if _bps_dfs:
    bps_pdf = pd.concat(_bps_dfs, ignore_index=True)

    # Normalize column names (Census BPS uses varying column schemas)
    bps_pdf.columns = [c.lower().strip().replace(" ", "_") for c in bps_pdf.columns]

    # Compute total_units from unit-type columns if needed
    unit_cols = [c for c in bps_pdf.columns if "unit" in c and c != "total_units"]
    if "total_units" not in bps_pdf.columns and unit_cols:
        bps_pdf["total_units"] = bps_pdf[unit_cols].sum(axis=1)

    # Build FIPS code + compute YoY and annual average
    bps_pdf["county_fips"] = (
        bps_pdf["state_fips"].astype(str).str.zfill(2)
        + bps_pdf["county_fips"].astype(str).str.zfill(3)
    )

    permits_pivot = bps_pdf.pivot_table(
        index="county_fips", columns="year", values="total_units", aggfunc="sum"
    ).reset_index()
    permits_pivot.columns.name = None

    yr_cols = [c for c in permits_pivot.columns if str(c).isdigit()]
    permits_pivot["permits_new_units_avg"] = permits_pivot[yr_cols].mean(axis=1).astype(int)
    permits_pivot["permits_yoy_pct"] = (
        (permits_pivot[yr_cols[-1]] - permits_pivot[yr_cols[-2]]) / permits_pivot[yr_cols[-2]].replace(0, float("nan"))
    )
    permits_pivot = permits_pivot[["county_fips", "permits_new_units_avg", "permits_yoy_pct"]]
    permits_pivot["updated_at"] = now

    bps_spark = spark.createDataFrame(permits_pivot)
else:
    print("  BPS files not found — generating synthetic county-level permits")
    # Generate synthetic permits for counties represented in Zillow data
    county_from_zhvi = zhvi_slim[["county_name", "state"]].drop_duplicates()
    county_from_zhvi["county_fips"] = (
        pd.util.hash_array(county_from_zhvi["county_name"].values) % 90000 + 10000
    ).astype(str)
    county_from_zhvi["permits_new_units_avg"] = np.random.default_rng(42).integers(50, 800, size=len(county_from_zhvi))
    county_from_zhvi["permits_yoy_pct"] = np.random.default_rng(43).uniform(-0.15, 0.20, size=len(county_from_zhvi))
    county_from_zhvi["updated_at"] = now
    bps_spark = spark.createDataFrame(county_from_zhvi[["county_fips", "permits_new_units_avg", "permits_yoy_pct", "updated_at"]])

bps_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{BRONZE}.building_permits_county"
)
print(f"✓ Wrote {BRONZE}.building_permits_county ({bps_spark.count()} counties)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"Dev signal ingestion complete (source={DEV_SIGNALS_SOURCE})")
print(f"  {BRONZE}.zillow_home_values   → {spark.table(f'{BRONZE}.zillow_home_values').count()} zip codes")
print(f"  {BRONZE}.zillow_rental_index  → {spark.table(f'{BRONZE}.zillow_rental_index').count()} zip codes")
print(f"  {BRONZE}.building_permits_county → {spark.table(f'{BRONZE}.building_permits_county').count()} counties")
print()
print("Next step: run notebook 06_build_dev_signal_layer to produce bronze.dev_signals_by_h3")
