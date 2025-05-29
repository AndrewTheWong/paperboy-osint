#!/usr/bin/env python3
"""
XGBoost Time-Series Forecasting Model for Escalation Prediction

This module trains an XGBoost model to predict next-day average escalation scores
using GDELT and ACLED daily conflict event counts plus historical escalation patterns.

Features:
- Lagged escalation scores (t-1, t-2, t-3)
- Day of week encoding
- 7-day and 14-day rolling averages
- GDELT and ACLED daily conflict counts
- Time-based features

Output:
- Regression model predicting escalation_score[t+1]
- Saved to models/xgboost_forecast.pkl
- Metrics: MAE, RMSE, R² Score
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

ACLED_API_KEY = os.getenv("ACLED_API_KEY")
ACLED_EMAIL = os.getenv("ACLED_EMAIL")

def fetch_gdelt_daily_counts(days_back: int = 365) -> pd.DataFrame:
    """
    Fetch daily GDELT conflict event counts for time series features.
    
    Args:
        days_back: Number of days of historical data to fetch
        
    Returns:
        DataFrame with columns: date, gdelt_conflict_count, gdelt_total_count
    """
    logger.info(f"Fetching GDELT daily counts for past {days_back} days...")
    
    # For demonstration, we'll simulate GDELT data since the full implementation
    # would require substantial API calls. In production, this would fetch
    # actual GDELT daily aggregates.
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate realistic daily conflict patterns
    np.random.seed(42)  # For reproducible results
    
    gdelt_data = []
    for date in date_range:
        # Simulate conflict counts with some seasonality and trends
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Weekend effect (fewer reports on weekends)
        weekend_factor = 0.7 if date.weekday() >= 5 else 1.0
        
        # Base conflict count with noise
        base_conflicts = int(seasonal_factor * weekend_factor * np.random.poisson(15))
        total_events = int(base_conflicts * np.random.uniform(3, 8))
        
        gdelt_data.append({
            'date': date.date(),
            'gdelt_conflict_count': base_conflicts,
            'gdelt_total_count': total_events
        })
    
    df = pd.DataFrame(gdelt_data)
    logger.info(f"Generated {len(df)} days of GDELT data")
    return df

def fetch_acled_daily_counts(days_back: int = 365) -> pd.DataFrame:
    """
    Fetch daily ACLED conflict event counts for time series features.
    
    Args:
        days_back: Number of days of historical data to fetch
        
    Returns:
        DataFrame with columns: date, acled_conflict_count, acled_fatalities
    """
    logger.info(f"Fetching ACLED daily counts for past {days_back} days...")
    
    if not ACLED_API_KEY or not ACLED_EMAIL:
        logger.warning("ACLED credentials not found, generating synthetic data")
        return generate_synthetic_acled_data(days_back)
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # ACLED API request
        base_url = "https://api.acleddata.com/acled/read"
        params = {
            "key": ACLED_API_KEY,
            "email": ACLED_EMAIL,
            "event_date": f"{start_date_str}|{end_date_str}",
            "format": "json",
            "limit": 50000  # Max limit
        }
        
        logger.info("Fetching data from ACLED API...")
        response = requests.get(base_url, params=params, timeout=120)
        
        if response.status_code != 200:
            logger.error(f"ACLED API error: {response.status_code}")
            return generate_synthetic_acled_data(days_back)
        
        data = response.json()
        if "data" not in data or not data["data"]:
            logger.warning("No ACLED data returned, generating synthetic data")
            return generate_synthetic_acled_data(days_back)
        
        # Process ACLED data
        events = data["data"]
        logger.info(f"Received {len(events)} ACLED events")
        
        # Convert to DataFrame and aggregate by date
        df = pd.DataFrame(events)
        df['event_date'] = pd.to_datetime(df['event_date'])
        df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)
        
        # Conflict types
        conflict_types = ["Battles", "Explosions/Remote violence", "Violence against civilians"]
        df['is_conflict'] = df['event_type'].isin(conflict_types)
        
        # Daily aggregation
        daily_acled = df.groupby(df['event_date'].dt.date).agg({
            'is_conflict': 'sum',
            'fatalities': 'sum',
            'event_type': 'count'
        }).reset_index()
        
        daily_acled.columns = ['date', 'acled_conflict_count', 'acled_fatalities', 'acled_total_count']
        
        # Fill missing dates with zeros
        date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        full_range_df = pd.DataFrame({'date': date_range.date})
        daily_acled = full_range_df.merge(daily_acled, on='date', how='left').fillna(0)
        
        logger.info(f"Processed {len(daily_acled)} days of ACLED data")
        return daily_acled
        
    except Exception as e:
        logger.error(f"Error fetching ACLED data: {e}")
        return generate_synthetic_acled_data(days_back)

def generate_synthetic_acled_data(days_back: int) -> pd.DataFrame:
    """Generate synthetic ACLED data for demonstration."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(123)  # Different seed from GDELT
    
    acled_data = []
    for date in date_range:
        # Simulate more volatile conflict patterns for ACLED
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # ACLED tends to have more weekend reporting than GDELT
        weekend_factor = 0.9 if date.weekday() >= 5 else 1.0
        
        conflicts = int(seasonal_factor * weekend_factor * np.random.poisson(8))
        fatalities = int(conflicts * np.random.exponential(2))
        
        acled_data.append({
            'date': date.date(),
            'acled_conflict_count': conflicts,
            'acled_fatalities': fatalities,
            'acled_total_count': int(conflicts * np.random.uniform(1.2, 2.5))
        })
    
    return pd.DataFrame(acled_data)

def generate_synthetic_escalation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic historical escalation scores based on conflict data.
    In production, this would come from your actual inference pipeline results.
    """
    logger.info("Generating synthetic escalation scores...")
    
    df = df.copy()
    np.random.seed(456)
    
    # Create escalation scores based on conflict indicators
    df['base_escalation'] = (
        0.3 * (df['gdelt_conflict_count'] / df['gdelt_conflict_count'].max()) +
        0.4 * (df['acled_conflict_count'] / df['acled_conflict_count'].max()) +
        0.3 * (df['acled_fatalities'] / (df['acled_fatalities'].max() + 1))
    )
    
    # Add noise and ensure [0, 1] range
    noise = np.random.normal(0, 0.1, len(df))
    df['escalation_score'] = np.clip(df['base_escalation'] + noise, 0, 1)
    
    return df

def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time series features for forecasting model.
    
    Args:
        df: DataFrame with date and escalation_score columns
        
    Returns:
        DataFrame with added features
    """
    logger.info("Creating time series features...")
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Lagged features
    df['escalation_lag1'] = df['escalation_score'].shift(1)
    df['escalation_lag2'] = df['escalation_score'].shift(2)
    df['escalation_lag3'] = df['escalation_score'].shift(3)
    
    # Rolling averages
    df['escalation_7d_avg'] = df['escalation_score'].rolling(window=7, min_periods=1).mean()
    df['escalation_14d_avg'] = df['escalation_score'].rolling(window=14, min_periods=1).mean()
    
    # Time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Conflict trend features
    df['gdelt_conflict_lag1'] = df['gdelt_conflict_count'].shift(1)
    df['acled_conflict_lag1'] = df['acled_conflict_count'].shift(1)
    
    # Rolling conflict averages
    df['gdelt_conflict_7d_avg'] = df['gdelt_conflict_count'].rolling(window=7, min_periods=1).mean()
    df['acled_conflict_7d_avg'] = df['acled_conflict_count'].rolling(window=7, min_periods=1).mean()
    
    # Change indicators
    df['escalation_change'] = df['escalation_score'] - df['escalation_lag1']
    df['gdelt_conflict_change'] = df['gdelt_conflict_count'] - df['gdelt_conflict_lag1']
    
    return df

def prepare_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare features and target for model training.
    
    Args:
        df: DataFrame with features
        
    Returns:
        X: Feature matrix
        y: Target vector (next day escalation score)
        feature_names: List of feature names
    """
    logger.info("Preparing training data...")
    
    # Feature columns
    feature_cols = [
        'escalation_lag1', 'escalation_lag2', 'escalation_lag3',
        'escalation_7d_avg', 'escalation_14d_avg',
        'day_of_week', 'day_of_month', 'month', 'is_weekend',
        'gdelt_conflict_count', 'gdelt_conflict_lag1', 'gdelt_conflict_7d_avg',
        'acled_conflict_count', 'acled_conflict_lag1', 'acled_conflict_7d_avg',
        'acled_fatalities', 'escalation_change', 'gdelt_conflict_change'
    ]
    
    # Target: next day escalation score
    df['target'] = df['escalation_score'].shift(-1)
    
    # Remove rows with missing values (due to lags and shifts)
    valid_data = df.dropna(subset=feature_cols + ['target'])
    
    X = valid_data[feature_cols].values
    y = valid_data['target'].values
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Features: {feature_cols}")
    
    return X, y, feature_cols

def train_xgboost_forecast_model(X: np.ndarray, y: np.ndarray, feature_names: list) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Train XGBoost time series forecasting model.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        
    Returns:
        Trained model and metrics dictionary
    """
    logger.info("Training XGBoost forecasting model...")
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Use the last split for final validation
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_r2': r2_score(y_val, y_val_pred)
    }
    
    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_))
    metrics['feature_importance'] = importance
    
    logger.info(f"Training metrics - MAE: {metrics['train_mae']:.4f}, RMSE: {metrics['train_rmse']:.4f}, R²: {metrics['train_r2']:.4f}")
    logger.info(f"Validation metrics - MAE: {metrics['val_mae']:.4f}, RMSE: {metrics['val_rmse']:.4f}, R²: {metrics['val_r2']:.4f}")
    
    return model, metrics

def save_model_and_metrics(model: xgb.XGBRegressor, metrics: Dict, feature_names: list):
    """Save trained model and metrics."""
    logger.info("Saving model and metrics...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = "models/xgboost_forecast.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'xgboost_forecast',
        'feature_names': feature_names,
        'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                   for k, v in metrics.items() if k != 'feature_importance'},
        'feature_importance': {k: float(v) for k, v in metrics['feature_importance'].items()}
    }
    
    metadata_path = "models/xgboost_forecast_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main training pipeline."""
    logger.info("Starting XGBoost forecasting model training...")
    
    try:
        # Fetch conflict data
        gdelt_df = fetch_gdelt_daily_counts(days_back=365)
        acled_df = fetch_acled_daily_counts(days_back=365)
        
        # Merge data sources
        df = gdelt_df.merge(acled_df, on='date', how='inner')
        logger.info(f"Combined dataset: {len(df)} days")
        
        # Generate escalation scores (in production, load from actual data)
        df = generate_synthetic_escalation_scores(df)
        
        # Create time series features
        df = create_time_series_features(df)
        
        # Prepare training data
        X, y, feature_names = prepare_training_data(df)
        
        if len(X) < 50:
            logger.error("Insufficient data for training (need at least 50 samples)")
            return
        
        # Train model
        model, metrics = train_xgboost_forecast_model(X, y, feature_names)
        
        # Save results
        save_model_and_metrics(model, metrics, feature_names)
        
        logger.info("✅ Forecasting model training completed successfully!")
        
        # Show top features
        logger.info("Top 5 most important features:")
        for feature, importance in sorted(metrics['feature_importance'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
            
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 