"""
QSR Site Selection Intelligence — Databricks App Backend

FastAPI backend serving:
- Scored locations from gold.scored_locations
- Feature data from gold.location_features
- Real-time scoring via model serving endpoint
- SHAP explanations for selected sites
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
import math
from typing import Optional

import httpx
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sql import connect
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CATALOG = os.getenv("CATALOG", "qsr_siting")
SCHEMA = os.getenv("SCHEMA", "gold")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT", "qsr-site-scoring")
SQL_WAREHOUSE_ID = os.getenv("SQL_WAREHOUSE_ID", "")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# Populated from database at startup
METROS: list[str] = []
TIERS = ["A", "B", "C", "D"]

# Features used for similarity and what-if scoring
SIMILARITY_FEATURES = [
    "population_1ring", "median_income_1ring", "pct_target_demo_1ring",
    "daytime_pop_1ring", "max_daily_traffic_1ring", "avg_transit_score_1ring",
    "competitor_count_1ring", "competitor_count_3ring", "nearest_competitor_dist",
    "competitive_intensity", "retail_anchor_count_1ring", "school_count_2ring",
    "trade_area_quality", "cannibalization_risk", "market_saturation",
    "huff_market_share", "huff_expected_demand",
]

# Huff model parameters (must match _config.py)
HUFF_BETA = 2.0
HUFF_CANNIB_RADIUS = 3.0

# ---------------------------------------------------------------------------
# Globals (initialised in lifespan)
# ---------------------------------------------------------------------------
_wc: WorkspaceClient | None = None
_http: httpx.AsyncClient | None = None
_host: str = ""
_warehouse_id: str = ""


def _get_sql_connection():
    """Create a new SQL connection using the app's service principal."""
    cfg = _wc.config
    # Extract bearer token from SDK auth headers (more reliable than credentials_provider)
    headers = cfg.authenticate()
    token = headers["Authorization"].split(" ", 1)[1]
    return connect(
        server_hostname=cfg.host.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{_warehouse_id}",
        access_token=token,
    )


def _run_sql(query: str, params: dict | None = None) -> list[dict]:
    """Execute a SQL query and return rows as dicts."""
    try:
        with _get_sql_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                cols = [d[0] for d in cursor.description]
                return [dict(zip(cols, row)) for row in cursor.fetchall()]
    except Exception as e:
        print(f"SQL ERROR: {type(e).__name__}: {e}\nQuery: {query[:200]}")
        raise


# ---------------------------------------------------------------------------
# Lifespan — discover warehouse + warmup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _wc, _http, _host, _warehouse_id, METROS

    _wc = WorkspaceClient()
    _host = _wc.config.host
    _http = httpx.AsyncClient(timeout=30)
    _warehouse_id = SQL_WAREHOUSE_ID

    # Verify auth works
    try:
        headers = _wc.config.authenticate()
        print(f"Auth OK | token prefix: {headers.get('Authorization', '')[:20]}...")
    except Exception as e:
        print(f"Auth WARNING: {type(e).__name__}: {e}")

    # Discover available metros from data
    try:
        rows = await asyncio.to_thread(
            _run_sql,
            f"SELECT DISTINCT metro FROM {CATALOG}.{SCHEMA}.scored_locations ORDER BY metro",
        )
        METROS = [r["metro"] for r in rows]
        print(f"Discovered metros: {METROS}")
    except Exception as e:
        METROS = ["Chicago"]
        print(f"Metro discovery failed ({e}), defaulting to {METROS}")

    print(f"App started | host={_host} | warehouse={_warehouse_id}")
    yield
    await _http.aclose()


app = FastAPI(title="QSR Site Selection Intelligence", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html") as f:
        return f.read()


# ---------------------------------------------------------------------------
# API: Scored Locations
# ---------------------------------------------------------------------------
@app.get("/api/scored-locations")
async def get_scored_locations(
    metro: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    limit: int = Query(500, le=5000),
):
    """Return scored candidate locations with optional metro/tier filter."""
    conditions = []
    params = {}
    if metro and metro in METROS:
        conditions.append("metro = :metro")
        params["metro"] = metro
    if tier and tier in TIERS:
        conditions.append("score_tier = :tier")
        params["tier"] = tier

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT site_id, h3_res8, metro, latitude, longitude,
               predicted_annual_sales, percentile_rank, score_tier, shap_top5
        FROM {CATALOG}.{SCHEMA}.scored_locations
        {where}
        ORDER BY predicted_annual_sales DESC
        LIMIT {limit}
    """

    rows = await asyncio.to_thread(_run_sql, query, params or None)
    return {"locations": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# API: Site Detail (features + SHAP)
# ---------------------------------------------------------------------------
@app.get("/api/site/{site_id}")
async def get_site_detail(site_id: str):
    """Return full feature detail for a single site, including Huff gravity metrics."""
    query = f"""
        SELECT f.*,
               s.predicted_annual_sales, s.percentile_rank,
               s.score_tier, s.shap_top5,
               COALESCE(f.huff_market_share, 0) AS huff_market_share,
               COALESCE(f.huff_expected_demand, 0) AS huff_expected_demand
        FROM {CATALOG}.{SCHEMA}.location_features f
        JOIN {CATALOG}.{SCHEMA}.scored_locations s ON f.site_id = s.site_id
        WHERE f.site_id = :site_id
    """
    rows = await asyncio.to_thread(_run_sql, query, {"site_id": site_id})
    if not rows:
        raise HTTPException(404, "Site not found")
    return rows[0]


# ---------------------------------------------------------------------------
# API: Existing Stores (for map layer)
# ---------------------------------------------------------------------------
@app.get("/api/existing-stores")
async def get_existing_stores(metro: Optional[str] = Query(None)):
    """Return existing store locations with performance data."""
    params = {}
    where = ""
    if metro and metro in METROS:
        where = "WHERE metro = :metro"
        params["metro"] = metro

    query = f"""
        SELECT store_id, store_name, latitude, longitude, metro,
               annual_sales, format, location_quality_score
        FROM {CATALOG}.bronze.existing_stores
        {where}
        ORDER BY annual_sales DESC
    """
    rows = await asyncio.to_thread(_run_sql, query, params or None)
    return {"stores": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# API: Metro Summary Statistics
# ---------------------------------------------------------------------------
@app.get("/api/metro-summary")
async def get_metro_summary():
    """Return aggregate stats per metro."""
    query = f"""
        SELECT metro,
               COUNT(*) AS total_candidates,
               SUM(CASE WHEN score_tier = 'A' THEN 1 ELSE 0 END) AS tier_a,
               SUM(CASE WHEN score_tier = 'B' THEN 1 ELSE 0 END) AS tier_b,
               AVG(predicted_annual_sales) AS avg_predicted_sales,
               MAX(predicted_annual_sales) AS max_predicted_sales,
               AVG(latitude) AS avg_lat,
               AVG(longitude) AS avg_lon
        FROM {CATALOG}.{SCHEMA}.scored_locations
        GROUP BY metro
        ORDER BY avg_predicted_sales DESC
    """
    rows = await asyncio.to_thread(_run_sql, query)
    return {"metros": rows}


# ---------------------------------------------------------------------------
# API: Score a Location (real-time via model endpoint)
# ---------------------------------------------------------------------------
@app.post("/api/score")
async def score_location(request: Request):
    """Score a location in real-time via the model serving endpoint."""
    body = await request.json()
    features = body.get("features", {})

    headers = _wc.config.authenticate()
    headers["Content-Type"] = "application/json"

    resp = await _http.post(
        f"{_host}/serving-endpoints/{MODEL_ENDPOINT}/invocations",
        headers=headers,
        json={"dataframe_records": [features]},
    )
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)

    return resp.json()


# ---------------------------------------------------------------------------
# API: Feature Importance (model-level)
# ---------------------------------------------------------------------------
@app.get("/api/feature-importance")
async def get_feature_importance():
    """Return top features from the model (from gold.model_feature_columns)."""
    query = f"""
        SELECT feature_name, feature_index
        FROM {CATALOG}.{SCHEMA}.model_feature_columns
        ORDER BY feature_index
    """
    rows = await asyncio.to_thread(_run_sql, query)
    return {"features": rows}


# ---------------------------------------------------------------------------
# API: Batch Compare Sites
# ---------------------------------------------------------------------------
@app.post("/api/sites/compare")
async def compare_sites(request: Request):
    """Return full details for multiple sites in one query."""
    body = await request.json()
    site_ids = body.get("site_ids", [])
    if not site_ids or len(site_ids) > 4:
        raise HTTPException(400, "Provide 1-4 site_ids")

    placeholders = ", ".join(f":id{i}" for i in range(len(site_ids)))
    params = {f"id{i}": sid for i, sid in enumerate(site_ids)}

    query = f"""
        SELECT f.*, s.predicted_annual_sales, s.percentile_rank,
               s.score_tier, s.shap_top5
        FROM {CATALOG}.{SCHEMA}.location_features f
        JOIN {CATALOG}.{SCHEMA}.scored_locations s ON f.site_id = s.site_id
        WHERE f.site_id IN ({placeholders})
    """
    rows = await asyncio.to_thread(_run_sql, query, params)
    return {"sites": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# API: Competitors (for map layer)
# ---------------------------------------------------------------------------
@app.get("/api/competitors")
async def get_competitors(metro: Optional[str] = Query(None)):
    """Return competitor locations with brand/category."""
    params = {}
    where = ""
    if metro and metro in METROS:
        where = "WHERE metro = :metro"
        params["metro"] = metro

    query = f"""
        SELECT competitor_id, brand, category, latitude, longitude,
               drive_thru, metro
        FROM {CATALOG}.bronze.competitors
        {where}
        ORDER BY brand
    """
    rows = await asyncio.to_thread(_run_sql, query, params or None)
    return {"competitors": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# API: Points of Interest (for map layer)
# ---------------------------------------------------------------------------
@app.get("/api/poi")
async def get_poi(
    metro: Optional[str] = Query(None),
    poi_type: Optional[str] = Query(None),
):
    """Return POI locations with optional type filter."""
    conditions = []
    params = {}
    if metro and metro in METROS:
        conditions.append("metro = :metro")
        params["metro"] = metro
    if poi_type:
        conditions.append("category = :poi_type")
        params["poi_type"] = poi_type

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT poi_id, category AS poi_type, name, latitude, longitude, metro
        FROM {CATALOG}.bronze.poi
        {where}
        LIMIT 3000
    """
    rows = await asyncio.to_thread(_run_sql, query, params or None)
    return {"poi": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# API: Heatmap Data (H3 hexagons)
# ---------------------------------------------------------------------------
@app.get("/api/heatmap/{metric}")
async def get_heatmap(
    metric: str,
    metro: Optional[str] = Query(None),
):
    """Return H3 cell values for heatmap overlay."""
    if not metro or metro not in METROS:
        raise HTTPException(400, "Metro filter required for heatmaps")

    valid_metrics = ["demand", "income", "traffic", "competition"]
    if metric not in valid_metrics:
        raise HTTPException(400, f"Invalid metric. Use: {valid_metrics}")

    if metric == "competition":
        # Competitors already have h3_res8 column
        query = f"""
            SELECT h3_res8 AS h3_index,
                   COUNT(*) AS value,
                   AVG(latitude) AS lat, AVG(longitude) AS lon
            FROM {CATALOG}.bronze.competitors
            WHERE metro = :metro
            GROUP BY h3_res8
            ORDER BY value DESC
            LIMIT 2000
        """
    elif metric == "traffic":
        # Traffic table has no metro column — join with demographics to filter
        query = f"""
            SELECT t.h3_index, t.avg_daily_traffic AS value,
                   d.latitude AS lat, d.longitude AS lon
            FROM {CATALOG}.bronze.traffic t
            JOIN {CATALOG}.bronze.demographics d ON t.h3_index = d.h3_index
            WHERE d.metro = :metro
            ORDER BY t.avg_daily_traffic DESC
            LIMIT 2000
        """
    else:
        # Demographics table has latitude/longitude columns directly
        col = "population" if metric == "demand" else "median_income"
        query = f"""
            SELECT h3_index, {col} AS value,
                   latitude AS lat, longitude AS lon
            FROM {CATALOG}.bronze.demographics
            WHERE metro = :metro
            ORDER BY {col} DESC
            LIMIT 2000
        """

    rows = await asyncio.to_thread(_run_sql, query, {"metro": metro})
    return {"cells": rows, "metric": metric, "count": len(rows)}


# ---------------------------------------------------------------------------
# API: Cannibalization Impact
# ---------------------------------------------------------------------------
@app.get("/api/cannibalization/{site_id}")
async def get_cannibalization(site_id: str):
    """Estimate sales cannibalization using Huff gravity model.

    For each nearby existing store, computes demand reallocation:
    1. Gets all supply points (stores + competitors) in the overlap zone
    2. Computes Huff attractiveness = sqft * (1 + drive_thru * 0.3)
    3. Before new site: store captures share based on its gravity vs all others
    4. After new site: gravity is redistributed, reducing existing store share
    5. Impact = store_sales * (share_before - share_after) / share_before
    """
    # Get candidate location + its features
    site_query = f"""
        SELECT s.latitude, s.longitude, s.metro,
               f.square_feet AS site_sqft,
               f.drive_thru_capable_flag AS site_dt,
               COALESCE(f.huff_expected_demand, 0) AS huff_expected_demand
        FROM {CATALOG}.{SCHEMA}.scored_locations s
        JOIN {CATALOG}.{SCHEMA}.location_features f ON s.site_id = f.site_id
        WHERE s.site_id = :site_id
    """
    site_rows = await asyncio.to_thread(_run_sql, site_query, {"site_id": site_id})
    if not site_rows:
        raise HTTPException(404, "Site not found")

    site = site_rows[0]
    site_sqft = float(site.get("site_sqft") or 2000)
    site_dt = float(site.get("site_dt") or 0)
    new_site_attract = site_sqft * (1.0 + (0.3 if site_dt > 0 else 0.0))

    # Find nearby existing stores (within HUFF_CANNIB_RADIUS miles)
    delta_lat = HUFF_CANNIB_RADIUS / 69.0
    delta_lon = HUFF_CANNIB_RADIUS / (69.0 * math.cos(math.radians(site["latitude"])))

    store_query = f"""
        SELECT store_id, store_name, latitude, longitude, annual_sales,
               format, square_feet, drive_thru_pct
        FROM {CATALOG}.bronze.existing_stores
        WHERE metro = :metro
          AND latitude BETWEEN :lat_min AND :lat_max
          AND longitude BETWEEN :lon_min AND :lon_max
    """
    store_params = {
        "metro": site["metro"],
        "lat_min": site["latitude"] - delta_lat,
        "lat_max": site["latitude"] + delta_lat,
        "lon_min": site["longitude"] - delta_lon,
        "lon_max": site["longitude"] + delta_lon,
    }
    stores = await asyncio.to_thread(_run_sql, store_query, store_params)

    # Also get competitors in the overlap zone for gravity denominator
    comp_query = f"""
        SELECT competitor_id, latitude, longitude, drive_thru
        FROM {CATALOG}.bronze.competitors
        WHERE metro = :metro
          AND latitude BETWEEN :lat_min AND :lat_max
          AND longitude BETWEEN :lon_min AND :lon_max
    """
    competitors_nearby = await asyncio.to_thread(_run_sql, comp_query, store_params)

    def _haversine(lat1, lon1, lat2, lon2):
        """Haversine distance in miles."""
        return math.sqrt(
            ((lat1 - lat2) * 69.0) ** 2
            + ((lon1 - lon2) * 69.0 * math.cos(math.radians(lat1))) ** 2
        )

    def _gravity(attract, dist):
        """Huff gravity = attractiveness / dist^beta."""
        d = max(dist, 0.05)  # floor to avoid division by zero
        return attract / (d ** HUFF_BETA)

    # Build supply point list (existing stores + competitors, but NOT the new site)
    supply_points = []
    for s in stores:
        sqft = float(s.get("square_feet") or 2000)
        dt = float(s.get("drive_thru_pct") or 0)
        supply_points.append({
            "id": s["store_id"],
            "lat": s["latitude"], "lon": s["longitude"],
            "attract": sqft * (1.0 + (0.3 if dt > 0 else 0.0)),
        })
    for c in competitors_nearby:
        supply_points.append({
            "id": c["competitor_id"],
            "lat": c["latitude"], "lon": c["longitude"],
            "attract": 2200 * (1.0 + (0.3 if c.get("drive_thru") else 0.0)),
        })

    # For each existing store, estimate Huff share before/after the new site
    impacts = []
    total_impact = 0
    for store in stores:
        dist_to_new = _haversine(store["latitude"], store["longitude"],
                                  site["latitude"], site["longitude"])
        if dist_to_new > HUFF_CANNIB_RADIUS:
            continue

        store_sqft = float(store.get("square_feet") or 2000)
        store_dt = float(store.get("drive_thru_pct") or 0)
        store_attract = store_sqft * (1.0 + (0.3 if store_dt > 0 else 0.0))

        # Compute gravity sums at the store's location (proxy for its trade area centroid)
        total_gravity_before = 0.0
        store_own_gravity = 0.0
        for sp in supply_points:
            d = _haversine(store["latitude"], store["longitude"], sp["lat"], sp["lon"])
            if d <= HUFF_CANNIB_RADIUS:
                g = _gravity(sp["attract"], d)
                total_gravity_before += g
                if sp["id"] == store["store_id"]:
                    store_own_gravity = g

        # After: add the new site's gravity
        new_site_gravity = _gravity(new_site_attract,
                                     _haversine(store["latitude"], store["longitude"],
                                                site["latitude"], site["longitude"]))
        total_gravity_after = total_gravity_before + new_site_gravity

        # Huff share change
        share_before = store_own_gravity / total_gravity_before if total_gravity_before > 0 else 0
        share_after = store_own_gravity / total_gravity_after if total_gravity_after > 0 else 0

        if share_before > 0:
            impact_pct = (share_before - share_after) / share_before
        else:
            impact_pct = 0

        impacted_sales = store["annual_sales"] * impact_pct
        total_impact += impacted_sales

        impacts.append({
            "store_id": store["store_id"],
            "store_name": store["store_name"],
            "latitude": store["latitude"],
            "longitude": store["longitude"],
            "distance_mi": round(dist_to_new, 2),
            "current_sales": store["annual_sales"],
            "impact_pct": round(impact_pct * 100, 1),
            "impacted_sales": round(impacted_sales),
            "huff_share_before": round(share_before * 100, 1),
            "huff_share_after": round(share_after * 100, 1),
        })

    impacts.sort(key=lambda x: x["distance_mi"])
    return {
        "site_id": site_id,
        "method": "huff_gravity",
        "huff_beta": HUFF_BETA,
        "impacts": impacts,
        "total_impacted_sales": round(total_impact),
        "stores_affected": len(impacts),
        "new_site_attractiveness": round(new_site_attract),
    }


# ---------------------------------------------------------------------------
# API: Similar Existing Sites
# ---------------------------------------------------------------------------
@app.get("/api/similar-sites/{site_id}")
async def get_similar_sites(site_id: str, top_n: int = Query(5, le=10)):
    """Find existing stores most similar to a candidate site."""
    feat_cols = ", ".join(SIMILARITY_FEATURES)

    # Get candidate features
    candidate_query = f"""
        SELECT site_id, {feat_cols}
        FROM {CATALOG}.{SCHEMA}.location_features
        WHERE site_id = :site_id
    """
    cand_rows = await asyncio.to_thread(_run_sql, candidate_query, {"site_id": site_id})
    if not cand_rows:
        raise HTTPException(404, "Site not found")

    candidate = cand_rows[0]

    # Get all existing store features + sales
    store_query = f"""
        SELECT f.site_id AS store_id, e.store_name, e.annual_sales, e.metro, e.format,
               e.latitude, e.longitude,
               {', '.join(f'f.{c}' for c in SIMILARITY_FEATURES)}
        FROM {CATALOG}.{SCHEMA}.location_features f
        JOIN {CATALOG}.bronze.existing_stores e ON f.site_id = e.store_id
    """
    stores = await asyncio.to_thread(_run_sql, store_query)

    # Compute cosine similarity in Python (small dataset, ~350 rows)
    def cosine_sim(a: dict, b: dict) -> float:
        dot = sum_a2 = sum_b2 = 0.0
        for feat in SIMILARITY_FEATURES:
            va = float(a.get(feat, 0) or 0)
            vb = float(b.get(feat, 0) or 0)
            dot += va * vb
            sum_a2 += va * va
            sum_b2 += vb * vb
        denom = math.sqrt(sum_a2) * math.sqrt(sum_b2)
        return dot / denom if denom > 0 else 0.0

    # Find shared traits (features within 20% of each other)
    def shared_traits(a: dict, b: dict) -> list[str]:
        traits = []
        labels = {
            "population_1ring": "population",
            "median_income_1ring": "income",
            "max_daily_traffic_1ring": "traffic",
            "competitor_count_1ring": "competition",
            "trade_area_quality": "trade area quality",
        }
        for feat, label in labels.items():
            va = float(a.get(feat, 0) or 0)
            vb = float(b.get(feat, 0) or 0)
            if va > 0 and abs(va - vb) / va < 0.20:
                traits.append(label)
        return traits

    scored = []
    for store in stores:
        sim = cosine_sim(candidate, store)
        scored.append({
            "store_id": store["store_id"],
            "store_name": store["store_name"],
            "annual_sales": store["annual_sales"],
            "metro": store["metro"],
            "format": store["format"],
            "latitude": store["latitude"],
            "longitude": store["longitude"],
            "similarity": round(sim * 100, 1),
            "shared_traits": shared_traits(candidate, store),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return {"site_id": site_id, "similar": scored[:top_n]}


# ---------------------------------------------------------------------------
# API: Full Scoring Features (for what-if analysis)
# ---------------------------------------------------------------------------
@app.get("/api/site/{site_id}/scoring-features")
async def get_scoring_features(site_id: str):
    """Return full model-ready feature vector for what-if analysis."""
    query = f"""
        SELECT f.*, m.feature_name
        FROM {CATALOG}.{SCHEMA}.location_features f
        CROSS JOIN {CATALOG}.{SCHEMA}.model_feature_columns m
        WHERE f.site_id = :site_id
        ORDER BY m.feature_index
    """
    # Simpler: just get features + column order
    feat_query = f"""
        SELECT * FROM {CATALOG}.{SCHEMA}.location_features
        WHERE site_id = :site_id
    """
    col_query = f"""
        SELECT feature_name FROM {CATALOG}.{SCHEMA}.model_feature_columns
        ORDER BY feature_index
    """
    features, columns = await asyncio.gather(
        asyncio.to_thread(_run_sql, feat_query, {"site_id": site_id}),
        asyncio.to_thread(_run_sql, col_query),
    )
    if not features:
        raise HTTPException(404, "Site not found")

    return {
        "features": features[0],
        "column_order": [c["feature_name"] for c in columns],
    }


# ---------------------------------------------------------------------------
# API: Confidence Intervals (bootstrap approximation)
# ---------------------------------------------------------------------------
@app.get("/api/confidence/{site_id}")
async def get_confidence(site_id: str):
    """Return confidence intervals based on similar existing stores."""
    feat_cols = ", ".join(SIMILARITY_FEATURES)

    # Get candidate features
    cand_query = f"""
        SELECT {feat_cols}
        FROM {CATALOG}.{SCHEMA}.location_features
        WHERE site_id = :site_id
    """
    cand_rows = await asyncio.to_thread(_run_sql, cand_query, {"site_id": site_id})
    if not cand_rows:
        raise HTTPException(404, "Site not found")

    candidate = cand_rows[0]

    # Get all existing store features + actual sales
    store_query = f"""
        SELECT f.site_id, e.annual_sales,
               {', '.join(f'f.{c}' for c in SIMILARITY_FEATURES)}
        FROM {CATALOG}.{SCHEMA}.location_features f
        JOIN {CATALOG}.bronze.existing_stores e ON f.site_id = e.store_id
    """
    stores = await asyncio.to_thread(_run_sql, store_query)

    # Compute similarity and pick top 15 most similar
    def feature_dist(a, b):
        total = 0
        for feat in SIMILARITY_FEATURES:
            va = float(a.get(feat, 0) or 0)
            vb = float(b.get(feat, 0) or 0)
            max_val = max(abs(va), abs(vb), 1)
            total += ((va - vb) / max_val) ** 2
        return math.sqrt(total)

    scored = [(s, feature_dist(candidate, s)) for s in stores]
    scored.sort(key=lambda x: x[1])
    top_similar = scored[:15]

    sales = sorted([s[0]["annual_sales"] for s in top_similar])
    n = len(sales)

    if n == 0:
        return {"p10": 0, "p25": 0, "p50": 0, "p75": 0, "p90": 0, "sample_size": 0}

    def percentile(data, pct):
        k = (len(data) - 1) * pct / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    return {
        "p10": round(percentile(sales, 10)),
        "p25": round(percentile(sales, 25)),
        "p50": round(percentile(sales, 50)),
        "p75": round(percentile(sales, 75)),
        "p90": round(percentile(sales, 90)),
        "sample_size": n,
    }


# ---------------------------------------------------------------------------
# API: Daypart Demand
# ---------------------------------------------------------------------------
@app.get("/api/daypart/{site_id}")
async def get_daypart(site_id: str):
    """Return daypart demand scores for a site's trade area."""
    query = f"""
        WITH site_hex AS (
            SELECT h3_res8 FROM {CATALOG}.{SCHEMA}.scored_locations
            WHERE site_id = :site_id
        )
        SELECT
            AVG(d.breakfast_score) AS breakfast,
            AVG(d.lunch_score) AS lunch,
            AVG(d.dinner_score) AS dinner,
            AVG(d.late_night_score) AS late_night,
            AVG(d.weekend_multiplier) AS weekend_multiplier
        FROM {CATALOG}.bronze.daypart_demand d
        WHERE d.h3_index IN (
            SELECT h3_res8 FROM site_hex
            UNION ALL
            SELECT explode(h3_kring(h3_res8, 1)) FROM site_hex
        )
    """
    rows = await asyncio.to_thread(_run_sql, query, {"site_id": site_id})
    if not rows or rows[0]["breakfast"] is None:
        return {"breakfast": 0, "lunch": 0, "dinner": 0, "late_night": 0, "weekend_multiplier": 1.0}
    return rows[0]


# ---------------------------------------------------------------------------
# API: Flag Site for Field Visit (write-back action)
# ---------------------------------------------------------------------------
@app.post("/api/flag-site")
async def flag_site(request: Request):
    """Flag a candidate site for field visit — writes to flagged_sites table.

    This is the operational write-back that transforms the app from analytical
    dashboard to decision tool. A real estate analyst can review the ML score,
    SHAP explanation, and cannibalization analysis, then flag promising sites
    for physical inspection.
    """
    body = await request.json()
    site_id = body.get("site_id")
    flagged_by = body.get("flagged_by", "analyst")
    notes = body.get("notes", "")
    priority = body.get("priority", "normal")

    if not site_id:
        raise HTTPException(400, "site_id is required")
    if priority not in ("high", "normal", "low"):
        raise HTTPException(400, "priority must be high, normal, or low")

    # Ensure flagged_sites table exists
    create_query = f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.flagged_sites (
            site_id STRING,
            flagged_by STRING,
            flagged_at TIMESTAMP,
            priority STRING,
            notes STRING,
            visit_status STRING DEFAULT 'pending',
            visit_date DATE,
            visit_notes STRING
        )
    """
    await asyncio.to_thread(_run_sql, create_query)

    # Insert the flag
    insert_query = f"""
        INSERT INTO {CATALOG}.{SCHEMA}.flagged_sites
        (site_id, flagged_by, flagged_at, priority, notes, visit_status)
        VALUES (:site_id, :flagged_by, current_timestamp(), :priority, :notes, 'pending')
    """
    await asyncio.to_thread(_run_sql, insert_query, {
        "site_id": site_id,
        "flagged_by": flagged_by,
        "priority": priority,
        "notes": notes,
    })

    return {"status": "flagged", "site_id": site_id, "priority": priority}


@app.get("/api/flagged-sites")
async def get_flagged_sites():
    """Return all flagged sites with their visit status."""
    query = f"""
        SELECT fs.site_id, fs.flagged_by, fs.flagged_at, fs.priority,
               fs.notes, fs.visit_status, fs.visit_date, fs.visit_notes,
               sl.predicted_annual_sales, sl.score_tier, sl.metro,
               sl.latitude, sl.longitude
        FROM {CATALOG}.{SCHEMA}.flagged_sites fs
        LEFT JOIN {CATALOG}.{SCHEMA}.scored_locations sl ON fs.site_id = sl.site_id
        ORDER BY fs.flagged_at DESC
    """
    rows = await asyncio.to_thread(_run_sql, query)
    return {"flagged_sites": rows, "count": len(rows)}


@app.post("/api/flag-site/{site_id}/update")
async def update_flag(site_id: str, request: Request):
    """Update a flagged site after field visit (complete the feedback loop)."""
    body = await request.json()
    visit_status = body.get("visit_status", "visited")
    visit_notes = body.get("visit_notes", "")

    query = f"""
        UPDATE {CATALOG}.{SCHEMA}.flagged_sites
        SET visit_status = :visit_status,
            visit_date = current_date(),
            visit_notes = :visit_notes
        WHERE site_id = :site_id
    """
    await asyncio.to_thread(_run_sql, query, {
        "site_id": site_id,
        "visit_status": visit_status,
        "visit_notes": visit_notes,
    })

    return {"status": "updated", "site_id": site_id, "visit_status": visit_status}


# ---------------------------------------------------------------------------
# API: Config
# ---------------------------------------------------------------------------
@app.get("/api/config/maps-key")
async def get_maps_key():
    """Return Google Maps API key for frontend."""
    return {"key": GOOGLE_MAPS_API_KEY}


@app.get("/api/config/app-info")
async def get_app_info():
    """Return app configuration including demo mode status.

    When demo_mode is true, the frontend should display a prominent banner:
    'DEMO DATA — NOT VALIDATED FOR PRODUCTION USE'

    This ensures viewers don't mistake synthetic demo outputs for validated
    site recommendations. The banner should be visible but not block interaction.
    """
    return {
        "demo_mode": DEMO_MODE,
        "catalog": CATALOG,
        "metros": METROS,
        "demo_banner": {
            "show": DEMO_MODE,
            "message": "DEMO DATA \u2014 NOT VALIDATED FOR PRODUCTION USE",
            "detail": (
                "This application is running on synthetic demo data. "
                "Site scores, SHAP explanations, and cannibalization analysis "
                "reflect generated patterns, not real market conditions. "
                "To use with real data, set DEMO_MODE=false and provide "
                "your own bronze-layer tables."
            ),
            "style": "warning",  # frontend: render as yellow/amber banner
        },
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "catalog": CATALOG, "warehouse": _warehouse_id}


@app.get("/api/debug")
async def debug():
    """Debug endpoint: test SQL connectivity and return diagnostics."""
    result = {
        "catalog": CATALOG,
        "schema": SCHEMA,
        "warehouse": _warehouse_id,
        "host": _host,
        "server_hostname": _wc.config.host.replace("https://", "") if _wc else None,
    }

    # Step 1: Test token extraction
    try:
        headers = _wc.config.authenticate()
        auth_header = headers.get("Authorization", "")
        result["auth_ok"] = bool(auth_header)
        result["auth_type"] = auth_header.split(" ", 1)[0] if auth_header else "none"
        result["token_len"] = len(auth_header.split(" ", 1)[1]) if " " in auth_header else 0
    except Exception as e:
        result["auth_ok"] = False
        result["auth_error"] = f"{type(e).__name__}: {e}"

    # Step 2: Test SQL query
    try:
        rows = await asyncio.to_thread(
            _run_sql,
            f"SELECT COUNT(*) AS cnt FROM {CATALOG}.{SCHEMA}.scored_locations",
        )
        result["scored_locations_count"] = rows[0]["cnt"] if rows else 0
        result["sql_ok"] = True
    except Exception as e:
        result["sql_ok"] = False
        result["sql_error"] = f"{type(e).__name__}: {str(e)[:500]}"
    return result
