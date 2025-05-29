# Time Series Forecasting Pipeline

This directory contains a complete pipeline for forecasting Taiwan-China escalation using both GDELT events and SBERT-tagged article clusters.

## Overview

The pipeline combines:
- **10 years of GDELT data** (2015-2024) from BigQuery for Taiwan (TW) and China (CH)
- **SBERT-tagged article clusters** with HDBSCAN clustering
- **Temporal dynamics** and feature engineering
- **Temporal Fusion Transformer (TFT)** for forecasting

## Pipeline Components

### 1. `query_gdelt_bigquery.py`
Fetches 10 years of GDELT data from Google BigQuery.

**Prerequisites:**
- Google Cloud account with BigQuery access
- Service account JSON key file
- `GOOGLE_APPLICATION_CREDENTIALS` environment variable set

**Usage:**
```bash
# Set up Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Run the query
python "data/Time Series/query_gdelt_bigquery.py"
```

**Output:**
- `data/gdelt_bigquery_taiwan_china.csv` - Raw GDELT events

**Supabase Setup:**
Create a table `gdelt_events` with columns:
- `event_date` (date)
- `country` (text) 
- `goldstein_score` (float)

Upload the CSV data to this table.

### 2. `aggregate_time_series.py`
Merges GDELT and article data into daily aggregations.

**Features created:**
- **GDELT features:** avg/std Goldstein scores, event counts
- **Article features:** avg/std Goldstein scores, article counts
- **Cluster features:** number of clusters, cluster sizes, noise percentage
- **Temporal features:** lags, rolling means, interaction terms

**Usage:**
```bash
python "data/Time Series/aggregate_time_series.py"
```

**Output:**
- `data/aggregated_timeseries.csv` - Merged daily features
- Uploads to Supabase `daily_features` table

### 3. `train_tft_model.py` 
Trains a Temporal Fusion Transformer model for escalation forecasting.

**Model architecture:**
- **Input:** 21-day lookback window (3 weeks)
- **Output:** 1-day ahead Goldstein score prediction
- **Features:** 22+ engineered features from GDELT and articles
- **Attention:** Multi-head attention for long-range dependencies

**Usage:**
```bash
python "data/Time Series/train_tft_model.py"
```

**Output:**
- `models/tft_merged_escalation_model.pt` - Trained TFT model
- Validation metrics (MAE, RMSE)
- Feature importance analysis

## Setup Instructions

### 1. Install Dependencies
```bash
pip install google-cloud-bigquery>=3.0.0
pip install pytorch-forecasting>=1.0.0
pip install pytorch-lightning>=2.0.0
```

### 2. Environment Variables
Create `.env` file with:
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### 3. Supabase Tables

**gdelt_events table:**
```sql
CREATE TABLE gdelt_events (
    id SERIAL PRIMARY KEY,
    event_date DATE NOT NULL,
    country TEXT NOT NULL,
    goldstein_score FLOAT NOT NULL
);
```

**daily_features table:**
```sql
CREATE TABLE daily_features (
    event_date DATE PRIMARY KEY,
    avg_goldstein FLOAT,
    std_goldstein FLOAT,
    num_gdelt_events INTEGER,
    article_avg_goldstein FLOAT,
    article_std_goldstein FLOAT,
    num_articles INTEGER,
    num_clusters INTEGER,
    max_cluster_size INTEGER,
    avg_cluster_size FLOAT,
    pct_noise FLOAT
);
```

## Complete Workflow

```bash
# 1. Fetch GDELT data from BigQuery
python "data/Time Series/query_gdelt_bigquery.py"

# 2. Upload GDELT data to Supabase gdelt_events table
# (Manual step - upload gdelt_bigquery_taiwan_china.csv)

# 3. Generate merged daily features  
python "data/Time Series/aggregate_time_series.py"

# 4. Train forecasting model
python "data/Time Series/train_tft_model.py"

# 5. Model is ready for inference!
```

## Model Features

The TFT model learns from:

**Temporal Features:**
- Day of week, week, month, quarter
- Relative time index

**GDELT Features:**
- Average/std Goldstein scores
- Event counts
- 1-day and 3-day lags
- 7-day rolling averages

**Article Features:**
- Average/std Goldstein scores from SBERT analysis
- Article counts
- Lagged features
- Rolling averages

**Cluster Features:**
- Number of distinct clusters
- Maximum/average cluster sizes
- Noise percentage (unclustered articles)

**Interaction Features:**
- GDELT vs Article Goldstein differences
- Cross-correlations
- Event-to-article ratios
- Events per cluster

## Expected Performance

The model should achieve:
- **MAE < 1.0** on daily Goldstein score prediction
- **RMSE < 1.5** with proper regularization
- **Feature importance** highlighting most predictive signals

## Troubleshooting

**BigQuery Access Issues:**
- Verify service account has BigQuery Data Viewer role
- Check `GOOGLE_APPLICATION_CREDENTIALS` path
- Ensure project has GDELT dataset access

**Memory Issues:**
- Reduce `batch_size` in train_tft_model.py
- Use CPU instead of GPU for small datasets
- Decrease `max_encoder_length` if needed

**Missing Data:**
- Check Supabase table schemas match exactly
- Verify date formats are consistent
- Ensure no NULL values in key columns

## Next Steps

1. **Hyperparameter tuning** with Optuna
2. **Ensemble models** (TFT + XGBoost + LSTM)
3. **Real-time inference** pipeline
4. **Alert system** for high escalation probabilities
5. **Model monitoring** and retraining automation 