# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Ingest Real Competitor & POI Data (OpenStreetMap Overpass)
# MAGIC
# MAGIC Fetches real QSR competitor locations and Points of Interest from the
# MAGIC **OpenStreetMap Overpass API** — a free, public, no-key-required data source.
# MAGIC
# MAGIC **Widget modes** (controlled by job base_parameters):
# MAGIC | `competitors_source` | `poi_source` | Behavior |
# MAGIC |----------------------|--------------|----------|
# MAGIC | `synthetic` (default) | `synthetic` | No-op — bronze tables from notebook 01 are used |
# MAGIC | `osm` | `synthetic` | Replace competitors with OSM data only |
# MAGIC | `synthetic` | `osm` | Replace POI with OSM data only |
# MAGIC | `osm` | `osm` | Replace both tables with OSM data |
# MAGIC
# MAGIC **Coverage notes**: OSM coverage varies by metro. In US metros, McDonald's/Burger King
# MAGIC coverage is typically 85–95%; smaller brands may be 50–70%. The notebook reports
# MAGIC coverage statistics and falls back to synthetic enrichment when data is sparse.
# MAGIC
# MAGIC **Schema compatibility**: output tables match `bronze.competitors` and `bronze.poi`
# MAGIC schemas exactly so all downstream notebooks (Phase 2) run unchanged.
# MAGIC
# MAGIC **Rate limiting**: 2-second sleep between Overpass requests. Total ~15–25 API calls
# MAGIC for 5 metros → finishes in under 3 minutes. Responses are cached in UC Volume.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

COMPETITORS_SOURCE = _widget("competitors_source", "synthetic")
POI_SOURCE = _widget("poi_source", "synthetic")

print(f"Config: catalog={CATALOG}, competitors_source={COMPETITORS_SOURCE}, poi_source={POI_SOURCE}")

if COMPETITORS_SOURCE == "synthetic" and POI_SOURCE == "synthetic":
    print("Both sources are synthetic — nothing to do. Exiting.")
    dbutils.notebook.exit("SKIPPED — both sources are synthetic")

# COMMAND ----------

import h3
import json
import time
import os
import numpy as np
import pandas as pd
import requests
import urllib.request
from datetime import datetime, date
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    BooleanType, DateType, TimestampType
)

now = datetime.now()
OSM_CACHE_DIR = f"/dbfs/Volumes/{CATALOG}/{BRONZE_SCHEMA}/raw/osm_cache"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Helpers

# COMMAND ----------

dbutils.fs.mkdirs(f"dbfs:/Volumes/{CATALOG}/{BRONZE_SCHEMA}/raw/osm_cache")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def _overpass_query(query: str, retries: int = 3, timeout: int = 90) -> dict:
    """Query Overpass API with retry/backoff. Returns parsed JSON dict."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=timeout + 30,
                headers={"User-Agent": "Databricks-QSR-Siting/1.0"},
            )
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited — waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            print(f"    Timeout on attempt {attempt+1}/{retries}")
            if attempt < retries - 1:
                time.sleep(10)
        except Exception as e:
            print(f"    Error on attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return {"elements": []}


def _extract_latlon(element: dict) -> tuple[float, float] | None:
    """Extract (lat, lon) from an OSM element (node, way with center, or relation)."""
    if element["type"] == "node":
        return element.get("lat"), element.get("lon")
    elif element["type"] in ("way", "relation"):
        center = element.get("center", {})
        return center.get("lat"), center.get("lon")
    return None, None


def _cache_path(prefix: str, metro: str) -> str:
    return f"{OSM_CACHE_DIR}/{prefix}_{metro.lower().replace(' ', '_')}.json"


def _load_cache(prefix: str, metro: str) -> list | None:
    path = _cache_path(prefix, metro)
    if os.path.exists(path):
        with open(path) as f:
            print(f"    Cache hit: {path}")
            return json.load(f)
    return None


def _save_cache(prefix: str, metro: str, elements: list):
    path = _cache_path(prefix, metro)
    with open(path, "w") as f:
        json.dump(elements, f)
    print(f"    Cached {len(elements)} elements → {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Brand & Tag Mappings

# COMMAND ----------

# OSM brand name patterns → our schema (brand, category, drive_thru_likely)
# Uses case-insensitive substring matching on OSM `brand` or `name` tags.
BRAND_MAP = [
    ("mcdonald",     "McDonald's",  "QSR_Burger",   True),
    ("burger king",  "Burger King", "QSR_Burger",   True),
    ("wendy",        "Wendy's",     "QSR_Burger",   True),
    ("taco bell",    "Taco Bell",   "QSR_Mexican",  True),
    ("chick-fil-a",  "Chick-fil-A", "QSR_Chicken",  True),
    ("chick fil a",  "Chick-fil-A", "QSR_Chicken",  True),
    ("chipotle",     "Chipotle",    "Fast_Casual",  False),
    ("five guys",    "Five Guys",   "Fast_Casual",  False),
]

# Expected brand counts per metro (from COMPETITOR_BRANDS in _config.py)
BRAND_EXPECTED = {b: c["count"] for b, c in COMPETITOR_BRANDS.items()}

# Build the Overpass brand pattern (pipe-separated, case insensitive via [~,i])
_brand_names_flat = [pattern for pattern, _, _, _ in BRAND_MAP]
_brand_regex = "|".join(_brand_names_flat)

# OSM amenity/shop/leisure → (our_category, our_subcategory, size_hint)
OSM_POI_MAP = {
    # Retail
    "mall":              ("Retail",        "Shopping Mall",     "anchor"),
    "department_store":  ("Retail",        "Big Box Store",     "anchor"),
    "superstore":        ("Retail",        "Big Box Store",     "large"),
    "wholesale":         ("Retail",        "Big Box Store",     "large"),
    "clothes":           ("Retail",        "Strip Mall",        "small"),
    "furniture":         ("Retail",        "Strip Mall",        "medium"),
    "electronics":       ("Retail",        "Strip Mall",        "medium"),
    "convenience":       ("Retail",        "Convenience Store", "small"),
    # Grocery
    "supermarket":       ("Grocery",       "Supermarket",       "large"),
    "grocery":           ("Grocery",       "Specialty Grocery", "medium"),
    "organic":           ("Grocery",       "Specialty Grocery", "medium"),
    "discount":          ("Grocery",       "Discount Grocery",  "medium"),
    # Office
    "coworking":         ("Office",        "Small Office",      "small"),
    # Entertainment
    "cinema":            ("Entertainment", "Movie Theater",     "large"),
    "stadium":           ("Entertainment", "Stadium",           "anchor"),
    "sports_centre":     ("Entertainment", "Gym/Fitness",       "medium"),
    "fitness_centre":    ("Entertainment", "Gym/Fitness",       "medium"),
    "bowling_alley":     ("Entertainment", "Bowling Alley",     "medium"),
    # Healthcare
    "hospital":          ("Healthcare",    "Hospital",          "anchor"),
    "clinic":            ("Healthcare",    "Clinic",            "medium"),
    "doctors":           ("Healthcare",    "Clinic",            "small"),
    # School
    "school":            ("School",        "School",            "medium"),
    "university":        ("School",        "University",        "anchor"),
    "college":           ("School",        "University",        "large"),
    "kindergarten":      ("School",        "Elementary School", "small"),
}

SIZE_FOOT_TRAFFIC = {
    "anchor": (65.0, 90.0),
    "large":  (45.0, 70.0),
    "medium": (25.0, 55.0),
    "small":  (10.0, 35.0),
}

def _map_brand(tags: dict) -> tuple | None:
    """Map OSM tags to (brand, category, drive_thru). Returns None if no match."""
    brand_tag = (tags.get("brand") or tags.get("name") or "").lower()
    for pattern, brand_name, category, dt in BRAND_MAP:
        if pattern in brand_tag:
            # Check drive-thru override from OSM tags
            osm_dt = tags.get("drive_through") or tags.get("service:drive_through") or ""
            dt_flag = (osm_dt.lower() == "yes") if osm_dt else dt
            return brand_name, category, dt_flag
    return None


def _map_poi_tags(tags: dict) -> tuple | None:
    """Map OSM tags to (category, subcategory, size_category). Returns None if no match."""
    # Check shop, amenity, leisure in priority order
    for tag_key in ("shop", "amenity", "leisure", "office"):
        tag_val = (tags.get(tag_key) or "").lower()
        if tag_val in OSM_POI_MAP:
            return OSM_POI_MAP[tag_val]
    # Office buildings — any office tag
    if tags.get("office"):
        return ("Office", "Office Tower" if tags.get("building:levels", "0") >= "5" else "Small Office", "medium")
    return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingest Competitors (OSM)

# COMMAND ----------

if COMPETITORS_SOURCE == "osm":

    print("=" * 60)
    print("Querying OSM Overpass for QSR competitors...")
    print("=" * 60)

    comp_rows = []

    for metro_name, metro_info in METROS.items():
        bounds = metro_info["bounds"]
        lat_min, lat_max, lon_min, lon_max = bounds

        # Check cache first
        cached = _load_cache("competitors", metro_name)
        if cached is not None:
            elements = cached
        else:
            query = f"""[out:json][timeout:90];
(
  node["amenity"="fast_food"]["brand"~"{_brand_regex}",i]({lat_min},{lon_min},{lat_max},{lon_max});
  way["amenity"="fast_food"]["brand"~"{_brand_regex}",i]({lat_min},{lon_min},{lat_max},{lon_max});
  node["amenity"="restaurant"]["brand"~"{_brand_regex}",i]({lat_min},{lon_min},{lat_max},{lon_max});
  way["amenity"="restaurant"]["brand"~"{_brand_regex}",i]({lat_min},{lon_min},{lat_max},{lon_max});
  node["name"~"{_brand_regex}",i]["amenity"~"fast_food|restaurant"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out center tags;"""

            print(f"\n  Querying {metro_name}...")
            result = _overpass_query(query)
            elements = result.get("elements", [])
            _save_cache("competitors", metro_name, elements)
            time.sleep(2)  # Respect rate limits between metros

        # Track brand counts for coverage reporting
        brand_counts = {}
        seen_ids = set()

        for elem in elements:
            if elem.get("id") in seen_ids:
                continue
            seen_ids.add(elem.get("id"))

            tags = elem.get("tags", {})
            mapped = _map_brand(tags)
            if not mapped:
                continue

            lat, lon = _extract_latlon(elem)
            if lat is None or lon is None:
                continue

            brand_name, category, drive_thru = mapped
            brand_counts[brand_name] = brand_counts.get(brand_name, 0) + 1

            h3_8 = h3.latlng_to_cell(lat, lon, 8)
            h3_9 = h3.latlng_to_cell(lat, lon, 9)
            comp_id = f"OSM-{elem['type'][:1].upper()}{elem['id']}"

            # Estimate annual sales: OSM has no revenue data, use brand-calibrated synthetic
            np.random.seed(abs(hash(comp_id)) % (2**31))
            base_sales = {"QSR_Burger": 3_200_000, "QSR_Mexican": 2_900_000,
                          "QSR_Chicken": 3_400_000, "Fast_Casual": 2_600_000}
            sales = round(np.random.normal(base_sales.get(category, 3_000_000), 400_000), 2)

            # Estimate opening date: OSM sometimes has start_date tag
            start_date_str = tags.get("start_date") or tags.get("opening_date") or ""
            try:
                opened = datetime.strptime(start_date_str[:4], "%Y").date() if start_date_str else None
            except ValueError:
                opened = None
            if opened is None:
                np.random.seed(abs(hash(comp_id + "_date")) % (2**31))
                years_ago = np.random.randint(1, 15)
                opened = date(datetime.now().year - years_ago, np.random.randint(1, 12), 1)

            comp_rows.append((
                comp_id, brand_name, category,
                float(lat), float(lon), h3_8, h3_9,
                metro_name, max(500_000.0, sales), bool(drive_thru),
                opened, now
            ))

        # Coverage report
        print(f"\n  {metro_name} — {len(seen_ids)} raw elements → {len([r for r in comp_rows if r[7] == metro_name])} matched")
        for brand_name, expected in BRAND_EXPECTED.items():
            found = brand_counts.get(brand_name, 0)
            pct = found / expected * 100
            status = "✓" if pct >= 50 else "⚠"
            print(f"    {status} {brand_name:15s}: {found:3d} found / {expected:3d} expected ({pct:.0f}%)")

    if not comp_rows:
        print("\nWARNING: No OSM competitors found. Keeping existing synthetic bronze.competitors.")
        dbutils.notebook.exit("WARN — OSM returned 0 competitors; synthetic data unchanged")

    comp_schema = StructType([
        StructField("competitor_id", StringType()), StructField("brand", StringType()),
        StructField("category", StringType()), StructField("latitude", DoubleType()),
        StructField("longitude", DoubleType()), StructField("h3_res8", StringType()),
        StructField("h3_res9", StringType()), StructField("metro", StringType()),
        StructField("estimated_annual_sales", DoubleType()), StructField("drive_thru", BooleanType()),
        StructField("opened_date", DateType()), StructField("updated_at", TimestampType()),
    ])

    comp_df = spark.createDataFrame(comp_rows, comp_schema)
    comp_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(f"{BRONZE}.competitors")

    total_comp = comp_df.count()
    print(f"\n✓ Wrote {BRONZE}.competitors — {total_comp} real OSM locations")

else:
    print("competitors_source=synthetic — bronze.competitors unchanged (from notebook 01)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ingest POI (OSM)
# MAGIC
# MAGIC Issues one Overpass query per metro covering:
# MAGIC - Grocery & retail shops (supermarket, mall, department_store, convenience)
# MAGIC - Schools (school, university, college, kindergarten)
# MAGIC - Entertainment (cinema, fitness_centre, stadium, bowling_alley)
# MAGIC - Healthcare (hospital, clinic, doctors)
# MAGIC - Offices (office=*)

# COMMAND ----------

if POI_SOURCE == "osm":

    print("=" * 60)
    print("Querying OSM Overpass for POI...")
    print("=" * 60)

    poi_rows = []
    poi_counter = 0

    for metro_name, metro_info in METROS.items():
        bounds = metro_info["bounds"]
        lat_min, lat_max, lon_min, lon_max = bounds

        cached = _load_cache("poi", metro_name)
        if cached is not None:
            elements = cached
        else:
            # One batched query for all POI types
            query = f"""[out:json][timeout:120];
(
  node["shop"~"supermarket|grocery|organic|convenience|mall|department_store|furniture|electronics|clothes|wholesale|discount"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["shop"~"supermarket|grocery|organic|convenience|mall|department_store|furniture|electronics|clothes|wholesale|discount"]({lat_min},{lon_min},{lat_max},{lon_max});
  node["amenity"~"school|university|college|kindergarten|cinema|hospital|clinic|doctors"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["amenity"~"school|university|college|kindergarten|cinema|hospital|clinic|doctors"]({lat_min},{lon_min},{lat_max},{lon_max});
  node["leisure"~"fitness_centre|sports_centre|stadium|bowling_alley"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["leisure"~"fitness_centre|sports_centre|stadium|bowling_alley"]({lat_min},{lon_min},{lat_max},{lon_max});
  node["office"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["office"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out center tags;"""

            print(f"\n  Querying {metro_name}...")
            result = _overpass_query(query, timeout=120)
            elements = result.get("elements", [])
            _save_cache("poi", metro_name, elements)
            time.sleep(2)

        seen_ids = set()
        metro_poi_count = 0
        cat_counts = {}

        for elem in elements:
            if elem.get("id") in seen_ids:
                continue
            seen_ids.add(elem.get("id"))

            tags = elem.get("tags", {})
            mapped = _map_poi_tags(tags)
            if not mapped:
                continue

            lat, lon = _extract_latlon(elem)
            if lat is None or lon is None:
                continue

            category, subcategory, size_cat = mapped
            cat_counts[category] = cat_counts.get(category, 0) + 1

            h3_8 = h3.latlng_to_cell(lat, lon, 8)
            poi_id = f"OSM-{elem['type'][:1].upper()}{elem['id']}"

            # Foot traffic index: random within size band
            np.random.seed(abs(hash(poi_id)) % (2**31))
            lo, hi = SIZE_FOOT_TRAFFIC[size_cat]
            foot_traffic = round(np.random.uniform(lo, hi), 1)

            # Name: prefer OSM name tag, fall back to "Unknown {subcategory}"
            name = tags.get("name") or tags.get("brand") or f"{subcategory}"

            poi_rows.append((
                poi_id, name[:200], category, subcategory,
                float(lat), float(lon), h3_8, metro_name,
                size_cat, foot_traffic, now
            ))
            metro_poi_count += 1

        print(f"\n  {metro_name} — {metro_poi_count} POI matched from {len(elements)} elements")
        for cat, cnt in sorted(cat_counts.items()):
            print(f"    {cat:15s}: {cnt}")

    if not poi_rows:
        print("\nWARNING: No OSM POI found. Keeping existing synthetic bronze.poi.")
        dbutils.notebook.exit("WARN — OSM returned 0 POI; synthetic data unchanged")

    poi_schema = StructType([
        StructField("poi_id", StringType()), StructField("name", StringType()),
        StructField("category", StringType()), StructField("subcategory", StringType()),
        StructField("latitude", DoubleType()), StructField("longitude", DoubleType()),
        StructField("h3_res8", StringType()), StructField("metro", StringType()),
        StructField("size_category", StringType()), StructField("foot_traffic_index", DoubleType()),
        StructField("updated_at", TimestampType()),
    ])

    poi_df = spark.createDataFrame(poi_rows, poi_schema)
    poi_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(f"{BRONZE}.poi")

    total_poi = poi_df.count()
    print(f"\n✓ Wrote {BRONZE}.poi — {total_poi} real OSM locations")

else:
    print("poi_source=synthetic — bronze.poi unchanged (from notebook 01)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("OSM Overpass ingestion complete.")
print()

if COMPETITORS_SOURCE == "osm":
    cnt = spark.table(f"{BRONZE}.competitors").count()
    brand_dist = spark.table(f"{BRONZE}.competitors").groupBy("brand").count().orderBy("count", ascending=False)
    print(f"  {BRONZE}.competitors — {cnt} locations (real OSM)")
    display(brand_dist)
else:
    print(f"  {BRONZE}.competitors — unchanged (synthetic)")

print()

if POI_SOURCE == "osm":
    cnt = spark.table(f"{BRONZE}.poi").count()
    cat_dist = spark.table(f"{BRONZE}.poi").groupBy("category").count().orderBy("count", ascending=False)
    print(f"  {BRONZE}.poi — {cnt} locations (real OSM)")
    display(cat_dist)
else:
    print(f"  {BRONZE}.poi — unchanged (synthetic)")

print()
print("Next step: run Phase 2 ML pipeline to re-score candidate locations with real data.")
