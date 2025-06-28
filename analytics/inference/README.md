# Inference Module

The inference module provides multiple forecasting approaches for predicting Goldstein scores. This module is designed to be robust and flexible, offering different forecasting methods depending on available resources and requirements.

## Available Forecasting Methods

### 1. Simple Forecast (`simple_forecast.py`)
A lightweight trend-based forecasting approach that doesn't require complex ML models.

**Features:**
- Linear trend analysis
- 7-day forecast horizon
- Minimal dependencies
- Fast execution
- Automatic fallback when scaler file is missing

**Usage:**
```python
from inference.simple_forecast import simple_forecast
result = simple_forecast()
```

**Output:** CSV file with country, event_date, and goldstein_forecast columns.

### 2. XGBoost Forecast (`xgb_forecast.py`)
A machine learning-based approach using XGBoost or linear regression fallback.

**Features:**
- Uses pre-trained XGBoost model if available
- Feature engineering (time-based, lag features, rolling averages)
- Automatic fallback to linear regression
- Country encoding
- 7-day forecast horizon

**Usage:**
```python
from inference.xgb_forecast import xgb_forecast
result = xgb_forecast()
```

**Output:** CSV file with country, event_date, and goldstein_forecast columns.

### 3. TFT Forecast (`forecast.py`)
Advanced neural network-based forecasting using Temporal Fusion Transformer.

**Features:**
- PyTorch Lightning-based model
- Temporal attention mechanisms
- Complex feature interactions
- Requires pre-trained TFT model

**Note:** Currently requires a valid PyTorch Lightning checkpoint. If the checkpoint is corrupted or missing, use one of the other methods.

## Data Requirements

All forecasting methods expect data in the following format:
- `date`: Date column (YYYY-MM-DD format)
- `country`: Country identifier
- `avg_goldstein`: Target variable (Goldstein score)
- Additional features: `num_mentions`, `avg_tone`, `num_events`, etc.

## Configuration

Each forecasting method uses these configurable paths:
- `DATA_PATH`: Input data file (default: "data/gdelt_daily_2023.csv")
- `MODEL_PATH`: Pre-trained model file (varies by method)
- `SCALER_PATH`: Scaler parameters (default: "models/goldstein_scaler.csv")
- `OUTPUT_PATH`: Output forecast file (varies by method)

## Testing

All forecasting methods include comprehensive tests:

```bash
# Test simple forecast
python -m pytest tests/test_simple_forecast.py -v

# Test XGBoost forecast
python -m pytest tests/test_xgb_forecast.py -v

# Test TFT forecast (with mocking)
python -m pytest tests/test_forecast.py -v
```

## Recommendations

1. **For production use:** Start with `xgb_forecast.py` as it provides good performance with built-in fallbacks
2. **For quick testing:** Use `simple_forecast.py` for rapid prototyping
3. **For advanced scenarios:** Use `forecast.py` when you have a properly trained TFT model

## Output Format

All methods produce CSV files with these columns:
- `country`: Country identifier
- `event_date`: Forecast date
- `goldstein_forecast`: Predicted Goldstein score

The forecasts extend 7 days into the future from the last available data point. 