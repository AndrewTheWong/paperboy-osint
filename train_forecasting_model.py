#!/usr/bin/env python3
"""
XGBoost Time-Series Forecasting Model for StraitWatch Phase 1

Trains XGBoost model to predict next-day average escalation score using:
- GDELT and ACLED daily conflict event counts
- Lagged escalation scores (t-1, t-2, t-3)
- Day of week features
- 7-day and 14-day rolling mean of escalation

Output: Regression model predicting escalation_score[t+1]
Saves to: models/xgboost_forecast.pkl
"""
import os
import pandas as pd
import numpy as np
import json
import joblib
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forecasting_model')

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def load_timeseries_data() -> pd.DataFrame:
    """
    Load and combine GDELT, ACLED data for time series forecasting.
    
    Returns:
        DataFrame with daily aggregated conflict events and escalation scores
    """
    logger.info("Loading time series data from available sources...")
    
    # Initialize empty dataframe with date range (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    date_range = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame({'date': date_range})
    
    # Load GDELT data if available
    try:
        gdelt_df = pd.read_csv("data/gdelt_events.csv")
        if 'Day' in gdelt_df.columns:
            gdelt_df['date'] = pd.to_datetime(gdelt_df['Day'], format='%Y%m%d', errors='coerce')
        elif 'SQLDATE' in gdelt_df.columns:
            gdelt_df['date'] = pd.to_datetime(gdelt_df['SQLDATE'], format='%Y%m%d', errors='coerce')
        
        # Count events per day
        gdelt_daily = gdelt_df.groupby('date').size().reset_index(name='gdelt_events')
        df = df.merge(gdelt_daily, on='date', how='left')
        logger.info(f"Loaded GDELT data: {len(gdelt_daily)} daily records")
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Could not load GDELT data: {e}")
        df['gdelt_events'] = 10  # Default baseline
    
    # Load ACLED data if available
    try:
        acled_df = pd.read_csv("data/acled_events_clean.csv")
        if 'event_date' in acled_df.columns:
            acled_df['date'] = pd.to_datetime(acled_df['event_date'], errors='coerce')
        elif 'date' in acled_df.columns:
            acled_df['date'] = pd.to_datetime(acled_df['date'], errors='coerce')
        
        # Count events per day
        acled_daily = acled_df.groupby('date').size().reset_index(name='acled_events')
        df = df.merge(acled_daily, on='date', how='left')
        logger.info(f"Loaded ACLED data: {len(acled_daily)} daily records")
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Could not load ACLED data: {e}")
        df['acled_events'] = 2  # Default baseline
    
    # Fill missing values with 0
    df['gdelt_events'] = df['gdelt_events'].fillna(0)
    df['acled_events'] = df['acled_events'].fillna(0)
    
    # Create synthetic escalation score based on event counts
    # This will be replaced with real escalation scores from tagged articles in production
    df['escalation_score'] = (
        np.log1p(df['gdelt_events']) * 0.3 + 
        np.log1p(df['acled_events']) * 0.7 + 
        np.random.normal(0, 0.1, len(df))  # Add noise
    )
    
    # Normalize escalation score to [0, 1]
    df['escalation_score'] = (df['escalation_score'] - df['escalation_score'].min()) / \
                             (df['escalation_score'].max() - df['escalation_score'].min())
    
    # Clip to reasonable range
    df['escalation_score'] = np.clip(df['escalation_score'], 0.0, 1.0)
    
    logger.info(f"Created time series with {len(df)} daily records")
    logger.info(f"Escalation score range: {df['escalation_score'].min():.3f} - {df['escalation_score'].max():.3f}")
    
    return df


def create_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time series features for forecasting.
    
    Args:
        df: DataFrame with date, escalation_score, gdelt_events, acled_events
        
    Returns:
        DataFrame with lag features, rolling means, and time features
    """
    logger.info("Creating forecasting features...")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Lag features (previous 3 days)
    df['escalation_lag_1'] = df['escalation_score'].shift(1)
    df['escalation_lag_2'] = df['escalation_score'].shift(2)
    df['escalation_lag_3'] = df['escalation_score'].shift(3)
    
    # Rolling mean features
    df['escalation_7d_mean'] = df['escalation_score'].rolling(window=7, min_periods=1).mean()
    df['escalation_14d_mean'] = df['escalation_score'].rolling(window=14, min_periods=1).mean()
    
    # Event count lag features
    df['gdelt_lag_1'] = df['gdelt_events'].shift(1)
    df['acled_lag_1'] = df['acled_events'].shift(1)
    
    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Rolling event counts
    df['gdelt_7d_mean'] = df['gdelt_events'].rolling(window=7, min_periods=1).mean()
    df['acled_7d_mean'] = df['acled_events'].rolling(window=7, min_periods=1).mean()
    
    # Create target variable (next day escalation)
    df['target'] = df['escalation_score'].shift(-1)
    
    # Drop rows with missing target
    df = df.dropna(subset=['target'])
    
    logger.info(f"Created features dataset with {len(df)} samples")
    
    return df


def train_forecasting_model(df: pd.DataFrame) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    Train XGBoost forecasting model.
    
    Args:
        df: DataFrame with features and target
        
    Returns:
        Trained model and evaluation metrics
    """
    logger.info("Training XGBoost forecasting model...")
    
    # Define feature columns
    feature_cols = [
        'escalation_lag_1', 'escalation_lag_2', 'escalation_lag_3',
        'escalation_7d_mean', 'escalation_14d_mean',
        'gdelt_events', 'acled_events', 'gdelt_lag_1', 'acled_lag_1',
        'gdelt_7d_mean', 'acled_7d_mean',
        'day_of_week', 'month', 'day_of_year'
    ]
    
    # Prepare features and target
    X = df[feature_cols].fillna(0)  # Fill any remaining NaNs
    y = df['target'].values
    
    # Time series split (preserve temporal order)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Use last split for final training
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_count': len(feature_cols)
    }
    
    logger.info(f"Model Performance:")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  R²: {r2:.4f}")
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"Top 5 features: {top_features}")
    
    return model, metrics


def save_model_and_metadata(model: xgb.XGBRegressor, metrics: Dict[str, float]) -> str:
    """
    Save trained model and metadata.
    
    Args:
        model: Trained XGBoost model
        metrics: Evaluation metrics
        
    Returns:
        Path to saved model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/xgboost_forecast_{timestamp}.pkl"
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_type': 'xgboost_forecasting',
        'timestamp': timestamp,
        'metrics': metrics,
        'feature_count': metrics['feature_count'],
        'model_path': model_path
    }
    
    metadata_path = f"logs/forecast_model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update latest model reference
    with open("models/latest_forecast.json", 'w') as f:
        json.dump({'model_path': model_path, 'timestamp': timestamp}, f)
    
    logger.info(f"Saved model to: {model_path}")
    logger.info(f"Saved metadata to: {metadata_path}")
    
    return model_path


def predict_next_day_escalation(model_path: str = None) -> float:
    """
    Predict tomorrow's escalation score using the latest model.
    
    Args:
        model_path: Optional path to specific model, otherwise uses latest
        
    Returns:
        Predicted escalation score for tomorrow
    """
    # Load model
    if model_path is None:
        try:
            with open("models/latest_forecast.json", 'r') as f:
                latest_info = json.load(f)
                model_path = latest_info['model_path']
        except FileNotFoundError:
            logger.error("No latest forecast model found")
            return 0.5  # Default neutral prediction
    
    model = joblib.load(model_path)
    
    # Load recent data to create features
    df = load_timeseries_data()
    df_features = create_forecast_features(df)
    
    # Get most recent features
    latest_features = df_features.iloc[-1:][
        ['escalation_lag_1', 'escalation_lag_2', 'escalation_lag_3',
         'escalation_7d_mean', 'escalation_14d_mean',
         'gdelt_events', 'acled_events', 'gdelt_lag_1', 'acled_lag_1',
         'gdelt_7d_mean', 'acled_7d_mean',
         'day_of_week', 'month', 'day_of_year']
    ].fillna(0)
    
    # Make prediction
    prediction = model.predict(latest_features)[0]
    prediction = np.clip(prediction, 0.0, 1.0)  # Ensure valid range
    
    logger.info(f"Predicted escalation for tomorrow: {prediction:.3f}")
    
    return float(prediction)


def plot_forecast_results(df: pd.DataFrame, model: xgb.XGBRegressor):
    """
    Create visualization of forecasting results.
    
    Args:
        df: DataFrame with features and target
        model: Trained model
    """
    # Prepare data for plotting
    feature_cols = [
        'escalation_lag_1', 'escalation_lag_2', 'escalation_lag_3',
        'escalation_7d_mean', 'escalation_14d_mean',
        'gdelt_events', 'acled_events', 'gdelt_lag_1', 'acled_lag_1',
        'gdelt_7d_mean', 'acled_7d_mean',
        'day_of_week', 'month', 'day_of_year'
    ]
    
    X = df[feature_cols].fillna(0)
    y_true = df['target'].values
    y_pred = model.predict(X)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('XGBoost Forecasting Model Results', fontsize=16)
    
    # Time series plot
    axes[0, 0].plot(df['date'], df['escalation_score'], label='Actual', alpha=0.7)
    axes[0, 0].plot(df['date'], y_pred, label='Predicted', alpha=0.7)
    axes[0, 0].set_title('Time Series: Actual vs Predicted')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Escalation Score')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Scatter plot
    axes[0, 1].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 1].set_title('Actual vs Predicted')
    axes[0, 1].set_xlabel('Actual Escalation Score')
    axes[0, 1].set_ylabel('Predicted Escalation Score')
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
    
    axes[1, 0].barh(list(top_features.keys()), list(top_features.values()))
    axes[1, 0].set_title('Top 10 Feature Importance')
    axes[1, 0].set_xlabel('Importance')
    
    # Residuals
    residuals = y_true - y_pred
    axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"logs/forecast_model_plots_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plots to: {plot_path}")
    plt.close()


def main():
    """Main training pipeline."""
    logger.info("Starting XGBoost forecasting model training...")
    
    # Load and prepare data
    df = load_timeseries_data()
    df_features = create_forecast_features(df)
    
    # Train model
    model, metrics = train_forecasting_model(df_features)
    
    # Save model
    model_path = save_model_and_metadata(model, metrics)
    
    # Create visualizations
    plot_forecast_results(df_features, model)
    
    # Test prediction
    tomorrow_score = predict_next_day_escalation(model_path)
    
    logger.info("✅ Forecasting model training completed successfully!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Tomorrow's predicted escalation: {tomorrow_score:.3f}")
    
    return model, metrics


if __name__ == "__main__":
    main() 