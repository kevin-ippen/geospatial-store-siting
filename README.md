# Geospatial Store Siting Accelerator

A turnkey Databricks solution for data-driven retail and QSR site selection. Deploy a complete pipeline — from synthetic data generation through ML scoring to an interactive map application — using a single `databricks bundle deploy`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  databricks.yml                                             │
│  (catalog, warehouse, metro, model endpoint)                │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
    Phase 1: Demo Data           Phase 2: ML Pipeline
             │                            │
  ┌──────────▼──────────┐    ┌────────────▼────────────────┐
  │  01_seed_demo_data   │    │  00_validate_schema         │
  │  ─────────────────   │    │  10_feature_engineering      │
  │  H3 hexagons         │    │  11_train_model (XGBoost)    │
  │  Demographics        │    │  12_deploy_endpoint          │
  │  Traffic patterns    │    │  13_score_candidates         │
  │  Competitors         │    │  14_phase2_summary           │
  │  POIs & candidates   │    └────────────┬───────────────┘
  │  Existing stores     │                 │
  │  Daypart demand      │                 ▼
  └──────────┬──────────┘    ┌─────────────────────────────┐
             │               │  Model Serving Endpoint      │
             ▼               │  (real-time site scoring)    │
  ┌─────────────────────┐    └──────────────┬──────────────┘
  │  Unity Catalog       │                  │
  │  bronze.* → gold.*   │◄─────────────────┘
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────────────────────────┐
  │  Databricks App (FastAPI + Leaflet.js)  │
  │  ─────────────────────────────────────  │
  │  Interactive map · SHAP explanations    │
  │  What-if scoring · PDF export           │
  └─────────────────────────────────────────┘
```

## Getting Started

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- A SQL Warehouse (serverless recommended)
- Databricks CLI v0.230+ authenticated to your workspace

### 1. Clone and configure

```bash
git clone https://github.com/kevin-ippen/geospatial-store-siting.git
cd geospatial-store-siting
```

Edit `databricks.yml` with your workspace details:

```yaml
variables:
  catalog:       { default: "qsr_siting" }       # Unity Catalog name
  warehouse_id:  { default: "your_warehouse_id" } # SQL Warehouse ID
  app_owner:     { default: "you@company.com" }   # App permissions
```

### 2. Deploy

```bash
databricks bundle deploy --target dev
```

### 3. Generate demo data

```bash
databricks bundle run phase1_demo_data
```

Creates ~6 bronze tables with synthetic data for the configured metro (default: Chicago). Includes ~3K H3 hexagons, 280 competitors, 1K candidate sites, existing stores, POIs, and daypart demand curves.

### 4. Train and deploy the model

```bash
databricks bundle run phase2_ml_pipeline
```

Runs schema validation, feature engineering (ring aggregations, Huff gravity model, competitive intensity), XGBoost training with SHAP, model serving endpoint deployment, and batch scoring of all candidates.

### 5. Open the app

```bash
databricks apps list  # find the app URL
```

---

## App Capabilities

| Feature | Description |
|---------|-------------|
| **H3 Heatmaps** | Population density, income, traffic, and competition overlays at resolution 8 |
| **Competitor Layer** | Color-coded QSR brands with marker clustering |
| **POI Layer** | Retail anchors, offices, schools, healthcare, entertainment |
| **Cannibalization Analysis** | Projected impact on nearby existing stores |
| **Site Comparison** | Side-by-side analysis of up to 4 candidate locations |
| **Confidence Intervals** | Sales range derived from similar existing store performance |
| **Similar Sites** | Find existing stores with matching feature profiles |
| **What-If Scoring** | Adjust input features, get real-time re-predictions from the serving endpoint |
| **Daypart Demand** | Breakfast / lunch / dinner / late-night potential curves |
| **Street View** | Google Maps panorama at any site (requires API key) |
| **PDF Export** | One-click site report generation |
| **SHAP Tooltips** | Hover to see feature importance driving each score |

---

## Bring Your Own Data

To use real data instead of the synthetic demo:

1. Set `demo_mode: "false"` in `databricks.yml`
2. Populate `{catalog}.bronze.*` tables matching the schemas in [docs/DATA_REFERENCE.md](docs/DATA_REFERENCE.md)
3. Run the ML pipeline — the schema validator will flag any mismatches before processing

```bash
databricks bundle run phase2_ml_pipeline
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `catalog` | `qsr_siting` | Unity Catalog name for all tables |
| `warehouse_id` | — | SQL Warehouse ID (required) |
| `app_owner` | — | Email with CAN_MANAGE on the app (required) |
| `demo_mode` | `true` | Generate synthetic data when `true` |
| `demo_metro` | `Chicago` | Metro for demo data (Chicago, Dallas, Phoenix, Atlanta, Denver) |
| `model_endpoint_name` | `qsr-site-scoring` | Serving endpoint name |
| `google_maps_api_key` | — | Enables Street View in the app (optional) |

## Project Layout

```
geospatial-siting-accelerator/
├── databricks.yml              # Bundle config + variables
├── pyproject.toml              # Python project metadata
├── app/
│   ├── main.py                 # FastAPI backend (15+ endpoints)
│   ├── app.yaml                # Databricks App manifest
│   └── static/                 # Leaflet.js frontend SPA
├── notebooks/
│   ├── _config.py              # Shared parameterized config
│   ├── 00_validate_schema.py   # Schema validation for BYOD
│   ├── 00_quality_checks.py    # Reusable data quality gate
│   ├── 01_seed_demo_data.py    # Synthetic data generation
│   ├── 10_feature_engineering.py
│   ├── 11_train_model.py       # XGBoost + SHAP
│   ├── 12_deploy_endpoint.py   # Model serving deployment
│   ├── 13_score_candidates.py  # Batch scoring
│   └── 14_phase2_summary.py    # Pipeline summary
├── resources/
│   ├── phase1_demo_data.yml    # Demo data job definition
│   ├── phase2_ml_pipeline.yml  # ML pipeline job definition
│   └── app.yml                 # App resource definition
└── docs/
    └── DATA_REFERENCE.md       # Bronze/gold table schemas
```

## Tech Stack

- **Compute**: Databricks serverless SQL + Jobs
- **Storage**: Delta Lake on Unity Catalog (bronze → gold medallion)
- **Geospatial**: H3 hexagonal indexing (resolution 8, ~460m cells)
- **ML**: XGBoost + SHAP explanations, Huff gravity model for demand estimation
- **Serving**: Databricks Model Serving (real-time scoring endpoint)
- **App**: FastAPI backend + Leaflet.js single-page application
- **Deployment**: Databricks Asset Bundles (multi-environment)
