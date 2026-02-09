# Data Reference

All tables live in `{catalog}.bronze` (raw data) and `{catalog}.gold` (ML features + scores).

## Bronze Tables

### `demographics`
Census-style data at H3 resolution 8 (~460m hexagons). One row per hex per metro.

| Column | Type | Description |
|--------|------|-------------|
| `h3_index` | string | H3 cell index (resolution 8) |
| `resolution` | int | H3 resolution (always 8) |
| `metro` | string | Metro area name |
| `latitude` | double | Hex centroid latitude |
| `longitude` | double | Hex centroid longitude |
| `population` | int | Estimated population |
| `households` | int | Number of households |
| `median_income` | double | Median household income ($) |
| `median_age` | double | Median age |
| `pct_under_18` | double | % under 18 |
| `pct_18_to_34` | double | % 18-34 |
| `pct_35_to_54` | double | % 35-54 |
| `pct_over_55` | double | % over 55 |
| `pct_college_educated` | double | % with college degree |
| `pct_renter` | double | % renter-occupied |
| `unemployment_rate` | double | Unemployment rate |
| `updated_at` | timestamp | Last updated |

### `traffic`
Vehicle/pedestrian/transit data at H3 resolution 8. **No metro column** — join with `demographics` on `h3_index`.

| Column | Type | Description |
|--------|------|-------------|
| `h3_index` | string | H3 cell index |
| `resolution` | int | H3 resolution |
| `avg_daily_traffic` | int | Average daily vehicle count |
| `peak_hour_traffic` | int | Peak hour vehicles |
| `pedestrian_index` | double | Pedestrian activity score |
| `transit_score` | double | Transit accessibility |
| `commute_inflow` | int | Workers commuting in |
| `commute_outflow` | int | Workers commuting out |
| `is_employment_center` | string | Y/N employment hub flag |
| `updated_at` | timestamp | Last updated |

### `competitors`
QSR competitor locations.

| Column | Type | Description |
|--------|------|-------------|
| `competitor_id` | string | Unique ID |
| `brand` | string | Brand name |
| `category` | string | QSR category (QSR_Burger, QSR_Mexican, QSR_Chicken, Fast_Casual) |
| `latitude` / `longitude` | double | Location |
| `h3_res8` / `h3_res9` | string | Pre-computed H3 indexes |
| `metro` | string | Metro area |
| `estimated_annual_sales` | double | Estimated revenue ($) |
| `drive_thru` | boolean | Has drive-thru |
| `opened_date` | date | Date opened |

### `poi`
Points of interest. Categories: `Entertainment`, `Grocery`, `Healthcare`, `Office`, `Retail`, `School`.

| Column | Type | Description |
|--------|------|-------------|
| `poi_id` | string | Unique ID |
| `name` | string | POI name |
| `category` | string | Primary category |
| `subcategory` | string | Detailed type |
| `latitude` / `longitude` | double | Location |
| `h3_res8` | string | Pre-computed H3 index |
| `metro` | string | Metro area |
| `size_category` | string | small / medium / large / anchor |
| `foot_traffic_index` | double | Relative foot traffic |

### `existing_stores`
Current store locations with operational metrics.

| Column | Type | Description |
|--------|------|-------------|
| `store_id` | string | Unique ID |
| `store_name` | string | Display name |
| `latitude` / `longitude` | double | Location |
| `h3_res8` / `h3_res9` | string | Pre-computed H3 indexes |
| `metro` | string | Metro area |
| `opened_date` | date | Opening date |
| `format` | string | traditional / express / drive_thru_only |
| `square_feet` | int | Store size |
| `annual_sales` | double | Annual revenue ($) |
| `transactions_per_day` | int | Daily transactions |
| `avg_ticket` | double | Average order ($) |
| `drive_thru_pct` | double | % revenue from drive-thru |
| `delivery_pct` | double | % revenue from delivery |
| `location_quality_score` | double | Quality score (0-100) |

### `locations`
Candidate real-estate sites.

| Column | Type | Description |
|--------|------|-------------|
| `location_id` | string | Unique ID |
| `latitude` / `longitude` | double | Site location |
| `h3_res7` / `h3_res8` / `h3_res9` | string | Multi-resolution H3 |
| `address` | string | Street address |
| `city` / `state` / `zip_code` | string | Location details |
| `metro` | string | Metro area |
| `property_type` | string | standalone_pad, freestanding_drive_thru, strip_mall_endcap, urban_inline, strip_mall_inline |
| `square_feet` | int | Available sqft |
| `parking_spaces` | int | Parking count |
| `drive_thru_capable` | boolean | Drive-thru feasible |
| `rent_per_sqft` | double | Annual rent/sqft ($) |

### `daypart_demand`
Time-of-day demand scores at H3 resolution 8.

| Column | Type | Description |
|--------|------|-------------|
| `h3_index` | string | H3 cell index |
| `metro` | string | Metro area |
| `breakfast_score` | double | 0-100 morning demand |
| `lunch_score` | double | 0-100 midday demand |
| `dinner_score` | double | 0-100 evening demand |
| `late_night_score` | double | 0-100 late-night demand |
| `weekend_multiplier` | double | 0.8-1.5 weekend lift |

## Gold Tables

### `scored_locations`
Model predictions for candidate sites.

| Column | Type | Description |
|--------|------|-------------|
| `site_id` | string | Matches `locations.location_id` |
| `h3_res8` | string | H3 index |
| `metro` | string | Metro area |
| `latitude` / `longitude` | double | Site location |
| `predicted_annual_sales` | float | Model prediction ($) |
| `shap_base_value` | double | SHAP baseline |
| `shap_top5` | string | JSON: top 5 SHAP drivers |
| `percentile_rank` | double | 0-1 percentile |
| `score_tier` | string | A / B / C / D |

### `location_features`
35 engineered features for all sites (candidates + existing stores).

See [_config.py](../notebooks/_config.py) for the full feature list organized by category: Demographics, Traffic, Competition, POI, Property, and Derived.

### `model_feature_columns`
Ordered list of 35 model input features (including one-hot encoded categoricals).

## Key Joins

```sql
demographics.h3_index = traffic.h3_index         -- traffic has NO metro column
demographics.h3_index = poi.h3_res8
demographics.h3_index = competitors.h3_res8
locations.location_id  = scored_locations.site_id
location_features.site_id = existing_stores.store_id  -- where site_type='store'
location_features.site_id = scored_locations.site_id  -- where site_type='candidate'
```

## H3 Functions (Databricks SQL)

```sql
h3_longlatash3string(longitude, latitude, resolution)  -- point to H3
h3_kring(h3_index, k)                                  -- k-ring neighbors
```

Note: `h3_centeraslat()` / `h3_centeraslon()` are NOT available. Use `latitude`/`longitude` columns from `demographics` instead.
