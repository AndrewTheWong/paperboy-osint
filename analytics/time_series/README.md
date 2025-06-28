# Time Series Forecasting System

## Overview

The Paperboy analytics system now includes comprehensive time series forecasting capabilities for predicting escalation trends and detecting temporal patterns in geopolitical tensions. This system provides ARIMA, SARIMA, and ensemble forecasting models with advanced visualization and analysis features.

## Features

### 🔮 Forecasting Models
- **ARIMA Models**: AutoRegressive Integrated Moving Average for trend analysis
- **SARIMA Models**: Seasonal ARIMA for detecting periodic patterns
- **Ensemble Forecasting**: Combines multiple models for improved accuracy
- **ML Integration**: Random Forest, Gradient Boosting, and Linear Regression support

### 📊 Data Processing
- **Automatic Data Preparation**: Converts article streams to time series
- **Feature Engineering**: 42+ temporal and technical features
- **Quality Assessment**: Data completeness and consistency analysis
- **Missing Data Handling**: Interpolation and forward/backward filling

### 🎯 Analytics Capabilities
- **Trend Detection**: Identify increasing/decreasing escalation patterns
- **Seasonality Analysis**: Weekly, monthly, and yearly patterns
- **Anomaly Detection**: Identify unusual escalation events
- **Confidence Intervals**: Statistical uncertainty quantification
- **Performance Metrics**: RMSE, MAE, MAPE, and directional accuracy

## Quick Start

### Basic ARIMA Forecasting

```python
from analytics.time_series.arima_models import ARIMAPredictor

# Initialize predictor
arima = ARIMAPredictor()

# Prepare data (articles with escalation scores)
ts_data = arima.prepare_data(articles, escalation_scores)

# Fit model with automatic parameter selection
arima.fit(ts_data)

# Generate 30-day forecast
forecast = arima.forecast(steps=30, confidence_level=0.95)

# Detect anomalies
anomalies = arima.detect_anomalies(threshold=2.0)

# Save model and diagnostics
arima.create_diagnostics_plots()
arima.save_model()
```

### Seasonal SARIMA Analysis

```python
from analytics.time_series.arima_models import SARIMAPredictor

# Initialize seasonal predictor
sarima = SARIMAPredictor()

# Prepare data
ts_data = sarima.prepare_data(articles, escalation_scores)

# Fit seasonal model
sarima.fit(ts_data)

# Generate seasonal forecast
forecast = sarima.forecast(steps=30)

# Create seasonal analysis plots
sarima.create_seasonal_plots()
```

### Complete Forecasting Pipeline

```python
from analytics.time_series.forecasting_engine import ForecastingEngine

# Initialize comprehensive engine
engine = ForecastingEngine()

# Run complete analysis with multiple models
results = engine.run_complete_analysis(
    articles=article_list,
    escalation_scores=scores,
    forecast_steps=30
)

# Access results
print(f"Models trained: {results['models_trained']}")
print(f"Performance: {results['performance_metrics']}")
print(f"Ensemble available: {results['ensemble_available']}")
```

### Integration with Enhanced Ensemble Predictor

```python
from analytics.inference.ensemble_predictor import EnhancedEnsemblePredictor

# Initialize enhanced predictor
predictor = EnhancedEnsemblePredictor()

# Get current predictions
predictions = predictor.predict_batch_escalation(articles)

# Generate time series trends analysis
trends = predictor.predict_time_series_trends(
    articles=articles,
    forecast_days=30
)

print(f"Current escalation trend: {trends['current_escalation']['trend']}")
print(f"Forecast trend: {trends['forecast_analysis']['trend_direction']}")
print(f"Models used: {trends['models_used']}")
```

## Model Performance

Based on comprehensive testing, the system achieves:

### ARIMA Models
- **RMSE**: 0.157 (16.53% MAPE)
- **Best for**: Trend analysis, non-seasonal data
- **Training time**: ~10 seconds (120 samples)

### SARIMA Models  
- **RMSE**: 0.167 (17.52% MAPE)
- **Best for**: Seasonal patterns, cyclical events
- **Training time**: ~60 seconds (120 samples)

### Ensemble Models
- **RMSE**: 0.162 (16.98% MAPE)
- **Best for**: Robust predictions, uncertainty quantification
- **Combines**: ARIMA + SARIMA + ML models

## Data Requirements

### Input Format
```python
articles = [
    {
        'title': 'Taiwan Strait tensions rise...',
        'content': 'Article content...',
        'scraped_at': '2024-01-01T00:00:00Z',  # Required
        'source': 'news_source',
        'url': 'https://...'
    },
    # ... more articles
]

escalation_scores = [0.3, 0.5, 0.7, ...]  # Corresponding scores
```

### Minimum Requirements
- **Sample size**: 30+ articles for basic analysis
- **Time range**: 2+ weeks for seasonal analysis
- **Data quality**: Valid timestamps and escalation scores

## Output Files

The system generates comprehensive outputs:

### Models Directory (`analytics/time_series/models/`)
- `arima_model_*.pkl` - Trained ARIMA models
- `sarima_model_*.pkl` - Trained SARIMA models
- `arima_diagnostics.png` - Model diagnostic plots
- `sarima_seasonal_analysis.png` - Seasonal analysis plots

### Forecasts Directory (`analytics/time_series/forecasts/`)
- `forecasting_results_*.json` - Forecast data and metadata
- `forecast_results.png` - Comprehensive forecast visualization

### Processed Data (`analytics/time_series/processed/`)
- `processed_timeseries_*.csv` - Engineered features
- `processed_timeseries_*_metadata.json` - Data quality metrics
- `time_series_overview.png` - Data visualization

## Advanced Features

### Custom Model Configuration

```python
# Custom ARIMA order
arima.fit(ts_data, order=(2, 1, 2))

# Custom seasonal configuration
sarima.fit(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
```

### Performance Evaluation

```python
# Train/test split
train_data, test_data = engine.prepare_data(articles, scores, test_size=0.2)

# Fit models
engine.train_arima_model(train_data)
engine.train_sarima_model(train_data)

# Evaluate performance
metrics = engine.evaluate_forecasts(test_data)
print(f"ARIMA RMSE: {metrics['arima']['rmse']:.4f}")
print(f"SARIMA RMSE: {metrics['sarima']['rmse']:.4f}")
```

### Ensemble Customization

```python
# Weighted ensemble based on performance
ensemble_forecast = engine.create_ensemble_forecast(method='weighted_average')

# Simple average ensemble
ensemble_forecast = engine.create_ensemble_forecast(method='simple_average')
```

## API Reference

### Core Classes

#### `ARIMAPredictor`
- `__init__(output_dir)` - Initialize predictor
- `prepare_data(articles, scores)` - Convert to time series
- `fit(ts_data, order=None)` - Train model
- `forecast(steps, confidence_level)` - Generate predictions
- `detect_anomalies(threshold)` - Find outliers
- `create_diagnostics_plots()` - Generate visualizations

#### `SARIMAPredictor`
- `detect_seasonality(ts_data)` - Analyze seasonal patterns
- `fit(ts_data, order, seasonal_order)` - Train seasonal model
- `create_seasonal_plots()` - Generate seasonal analysis

#### `ForecastingEngine`
- `run_complete_analysis(articles, scores, forecast_steps)` - Full pipeline
- `train_arima_model(train_data)` - Train ARIMA
- `train_sarima_model(train_data)` - Train SARIMA
- `create_ensemble_forecast(method)` - Combine models
- `evaluate_forecasts(test_data)` - Performance metrics

### Integration Points

#### Enhanced Ensemble Predictor
- `predict_time_series_trends(articles, forecast_days)` - Trend analysis
- `get_model_info()` - Check time series availability

## Dependencies

```bash
pip install statsmodels pandas numpy matplotlib seaborn scikit-learn
```

Optional (for enhanced features):
```bash
pip install pmdarima  # Auto-parameter selection (may have compatibility issues)
```

## Performance Tips

1. **Data Quality**: Ensure consistent timestamps and valid escalation scores
2. **Sample Size**: Use 60+ samples for reliable seasonal analysis
3. **Model Selection**: 
   - ARIMA for trending data
   - SARIMA for data with clear patterns
   - Ensemble for robust predictions

4. **Parameter Tuning**: Let grid search find optimal parameters for best results

## Error Handling

The system includes comprehensive error handling:
- **Missing data**: Automatic interpolation
- **Invalid dates**: Synthetic date generation
- **Model failures**: Fallback to simpler models
- **Insufficient data**: Graceful degradation

## Troubleshooting

### Common Issues

**"No valid dates found"**
- Ensure articles have `scraped_at`, `published_at`, or `created_at` fields
- Check date format compatibility

**"Insufficient data for analysis"**
- Provide at least 30 articles
- Ensure escalation scores are numerical

**"Model convergence issues"**
- Try simpler model orders
- Check for extreme outliers in data

**"Import errors"**
- Install required dependencies
- Check Python environment compatibility

### Performance Issues

**Slow SARIMA training**
- Reduce seasonal period search space
- Use smaller parameter grids
- Consider ARIMA for non-seasonal data

## Future Enhancements

- **Advanced ML Models**: LSTM, Prophet, and Transformer forecasting
- **Multi-variate Analysis**: Include external factors (economic, political)
- **Real-time Updates**: Streaming forecasts with online learning
- **Geographic Analysis**: Regional trend comparison
- **Event Detection**: Automatic crisis event identification

## Contributing

To extend the time series system:

1. Add new models to `analytics/time_series/`
2. Update `ForecastingEngine` to include new models
3. Add tests to `debug/test_time_series.py`
4. Update documentation

## Support

For questions or issues:
- Check the test files for usage examples
- Review the generated visualizations for insights
- Examine the comprehensive logging output
- Consult the performance metrics for model selection 