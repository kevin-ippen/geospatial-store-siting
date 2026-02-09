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
]

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
    """Return full feature detail for a single site."""
    query = f"""
        SELECT f.*, s.predicted_annual_sales, s.percentile_rank,
               s.score_tier, s.shap_top5
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
    """Estimate sales cannibalization impact on nearby existing stores."""
    # Get candidate location
    site_query = f"""
        SELECT latitude, longitude, metro
        FROM {CATALOG}.{SCHEMA}.scored_locations
        WHERE site_id = :site_id
    """
    site_rows = await asyncio.to_thread(_run_sql, site_query, {"site_id": site_id})
    if not site_rows:
        raise HTTPException(404, "Site not found")

    site = site_rows[0]

    # Find nearby existing stores (within ~3 miles using bounding box approximation)
    delta_lat = 3.0 / 69.0  # ~3 miles in latitude degrees
    delta_lon = 3.0 / (69.0 * math.cos(math.radians(site["latitude"])))

    store_query = f"""
        SELECT store_id, store_name, latitude, longitude, annual_sales, format
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

    # Calculate impact for each nearby store
    impacts = []
    total_impact = 0
    for store in stores:
        dist_mi = math.sqrt(
            ((store["latitude"] - site["latitude"]) * 69.0) ** 2
            + ((store["longitude"] - site["longitude"]) * 69.0 * math.cos(math.radians(site["latitude"]))) ** 2
        )
        if dist_mi > 3.0:
            continue
        impact_pct = max(0, 0.15 * (1 - dist_mi / 3.0))
        impacted_sales = store["annual_sales"] * impact_pct
        total_impact += impacted_sales
        impacts.append({
            "store_id": store["store_id"],
            "store_name": store["store_name"],
            "latitude": store["latitude"],
            "longitude": store["longitude"],
            "distance_mi": round(dist_mi, 2),
            "current_sales": store["annual_sales"],
            "impact_pct": round(impact_pct * 100, 1),
            "impacted_sales": round(impacted_sales),
        })

    impacts.sort(key=lambda x: x["distance_mi"])
    return {
        "site_id": site_id,
        "impacts": impacts,
        "total_impacted_sales": round(total_impact),
        "stores_affected": len(impacts),
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
# API: Config (maps key)
# ---------------------------------------------------------------------------
@app.get("/api/config/maps-key")
async def get_maps_key():
    """Return Google Maps API key for frontend."""
    return {"key": GOOGLE_MAPS_API_KEY}


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
