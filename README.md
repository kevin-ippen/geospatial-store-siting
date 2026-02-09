# Geospatial Site Selection Accelerator

End-to-end Databricks solution for optimal retail/QSR store location siting. Generates synthetic demo data, trains an XGBoost site scoring model, and deploys an interactive map application with 12 analytical features.

## What's Included

| Component | Description |
|-----------|-------------|
| **Demo Data Pipeline** | One-click synthetic data generation for a configurable metro |
| **ML Pipeline** | Feature engineering, model training (XGBoost + SHAP), endpoint deployment, batch scoring |
| **Interactive App** | FastAPI + Leaflet SPA with heatmaps, competitor overlays, what-if scoring, comparison mode, PDF export |
| **Schema Validator** | Validates your own data matches the expected schema |

## Quick Start

### 1. Configure

Edit `databricks.yml` — set your catalog and warehouse:

```yaml
variables:
  catalog:
    default: "my_catalog"       # your Unity Catalog name
  warehouse_id:
    default: "abc123def456"     # your SQL Warehouse ID
  app_owner:
    default: "user@company.com" # who can manage the app
```

### 2. Deploy

```bash
databricks bundle deploy --target dev
```

### 3. Generate Demo Data

```bash
databricks bundle run phase1_demo_data
```

This creates ~6 bronze tables with synthetic Chicago data (~3K hexagons, 280 competitors, 1K candidate sites).

### 4. Run ML Pipeline

```bash
databricks bundle run phase2_ml_pipeline
```

This builds features, trains an XGBoost model, deploys a serving endpoint, and scores all candidate locations.

### 5. Open the App

```bash
databricks apps list
# Navigate to the qsr-siting-app URL
```

## Using Your Own Data

Set `demo_mode: "false"` in `databricks.yml` and populate `{catalog}.bronze.*` tables matching the schemas in [docs/DATA_REFERENCE.md](docs/DATA_REFERENCE.md). Then run:

```bash
databricks bundle run phase2_ml_pipeline
```

The pipeline starts with a schema validator that checks your tables before processing.

## Bundle Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `catalog` | Yes | `qsr_siting` | Unity Catalog name |
| `warehouse_id` | Yes | — | SQL Warehouse ID |
| `app_owner` | Yes | — | Email for app permissions |
| `demo_mode` | No | `true` | Generate synthetic data |
| `demo_metro` | No | `Chicago` | Metro for demo data |
| `model_endpoint_name` | No | `qsr-site-scoring` | Model serving endpoint name |
| `google_maps_api_key` | No | — | Enables Street View in app |

## Project Structure

```
├── databricks.yml          # Bundle config with variables
├── app/                    # Databricks App (FastAPI + Leaflet)
│   ├── main.py             # Backend: 15 API endpoints
│   └── static/index.html   # Frontend: interactive map SPA
├── notebooks/
│   ├── _config.py          # Shared parameterized config
│   ├── 00_validate_schema  # Schema validator for BYOD
│   ├── 01_seed_demo_data   # Synthetic data generation
│   ├── 10-14_*             # ML pipeline (features → train → deploy → score)
│   └── 00_quality_checks   # Reusable data quality gate
├── resources/
│   ├── phase1_demo_data    # Demo data generation job
│   ├── phase2_ml_pipeline  # ML pipeline job
│   └── app                 # App resource definition
└── docs/
    └── DATA_REFERENCE.md   # Table schemas + query patterns
```

## App Features

1. **H3 Heatmap Overlays** — Population density, income, traffic, competition
2. **Competitor Map Layer** — Color-coded QSR brands with clustering
3. **POI Layer** — Retail, office, school, healthcare, entertainment
4. **Cannibalization Analysis** — Impact on nearby existing stores
5. **Site Comparison** — Side-by-side up to 4 candidate sites
6. **Confidence Intervals** — Sales range from similar existing stores
7. **Similar Sites** — Find existing stores with matching characteristics
8. **What-If Scoring** — Adjust features, get real-time re-predictions
9. **Daypart Demand** — Breakfast/lunch/dinner/late-night potential
10. **Street View** — Google Maps panorama at any site (requires API key)
11. **PDF Export** — One-click site report generation
12. **Rich Tooltips** — SHAP-powered hover previews
