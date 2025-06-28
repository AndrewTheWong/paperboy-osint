#!/usr/bin/env python3
"""
Comprehensive Forecasting Engine

This module provides a unified interface for time series forecasting using
multiple models including ARIMA, SARIMA, and ensemble methods.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from pathlib import Path

# Import our time series models
from .arima_models import ARIMAPredictor, SARIMAPredictor
from .time_series_processor import TimeSeriesProcessor

# Additional models if available
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available for ensemble models")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingEngine:
    """
    Comprehensive forecasting engine for escalation prediction.
    
    Features:
    - Multiple model types (ARIMA, SARIMA, ML models)
    - Ensemble forecasting
    - Model comparison and selection
    - Confidence intervals
    - Performance evaluation
    """
    
    def __init__(self, output_dir: str = "analytics/time_series/forecasts"):
        """Initialize forecasting engine."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processor = TimeSeriesProcessor()
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        self.ensemble_weights = {}
        
        logger.info("ForecastingEngine initialized")
    
    def prepare_data(self, articles: List[Dict], escalation_scores: List[float], 
                    test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare data for time series modeling.
        
        Args:
            articles: List of article dictionaries
            escalation_scores: List of escalation scores
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (train_series, test_series)
        """
        try:
            # Process data
            processed_df = self.processor.process_pipeline(articles, escalation_scores)
            
            if processed_df.empty or 'escalation_score_mean' not in processed_df.columns:
                logger.error("Failed to prepare time series data")
                return pd.Series(), pd.Series()
            
            # Extract main time series
            ts = processed_df['escalation_score_mean'].dropna()
            
            # Split into train/test
            split_idx = int(len(ts) * (1 - test_size))
            train_ts = ts[:split_idx]
            test_ts = ts[split_idx:]
            
            logger.info(f"Data prepared: {len(train_ts)} train, {len(test_ts)} test samples")
            
            return train_ts, test_ts
            
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            return pd.Series(), pd.Series()
    
    def train_arima_model(self, train_ts: pd.Series, model_name: str = "arima") -> bool:
        """
        Train ARIMA model.
        
        Args:
            train_ts: Training time series
            model_name: Name for the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Training ARIMA model: {model_name}")
            
            arima_model = ARIMAPredictor(output_dir=str(self.output_dir / "models"))
            
            if arima_model.fit(train_ts):
                self.models[model_name] = arima_model
                logger.info(f"ARIMA model {model_name} trained successfully")
                return True
            else:
                logger.error(f"Failed to train ARIMA model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to train ARIMA model: {e}")
            return False
    
    def train_sarima_model(self, train_ts: pd.Series, model_name: str = "sarima") -> bool:
        """
        Train SARIMA model.
        
        Args:
            train_ts: Training time series
            model_name: Name for the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Training SARIMA model: {model_name}")
            
            sarima_model = SARIMAPredictor(output_dir=str(self.output_dir / "models"))
            
            if sarima_model.fit(train_ts):
                self.models[model_name] = sarima_model
                logger.info(f"SARIMA model {model_name} trained successfully")
                return True
            else:
                logger.error(f"Failed to train SARIMA model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to train SARIMA model: {e}")
            return False
    
    def train_ml_models(self, processed_df: pd.DataFrame, target_col: str = 'escalation_score_mean',
                       test_size: float = 0.2) -> bool:
        """
        Train machine learning models for time series forecasting.
        
        Args:
            processed_df: Processed DataFrame with features
            target_col: Target column name
            test_size: Test set size
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available, skipping ML models")
                return False
            
            logger.info("Training ML models for time series forecasting")
            
            # Prepare features and target
            feature_cols = [col for col in processed_df.columns 
                           if col not in [target_col] and not col.startswith(target_col)]
            
            # Remove columns with too many NaN values
            feature_cols = [col for col in feature_cols 
                           if processed_df[col].notna().sum() > len(processed_df) * 0.5]
            
            # Remove categorical columns that cause issues
            categorical_cols = []
            for col in feature_cols:
                if processed_df[col].dtype == 'object':
                    categorical_cols.append(col)
            
            # Remove categorical columns from features
            feature_cols = [col for col in feature_cols if col not in categorical_cols]
            
            if not feature_cols:
                logger.warning("No suitable numerical features found for ML models")
                return False
            
            logger.info(f"Using {len(feature_cols)} numerical features for ML models")
            
            # Prepare data
            X = processed_df[feature_cols].fillna(method='ffill').fillna(method='bfill')
            y = processed_df[target_col].fillna(method='ffill')
            
            # Remove rows with NaN values
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 10:
                logger.warning("Insufficient data for ML models")
                return False
            
            # Split data
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train models
            ml_models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            for name, model in ml_models.items():
                try:
                    logger.info(f"Training {name}...")
                    model.fit(X_train, y_train)
                    
                    # Make predictions for evaluation
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Store model and metadata
                    self.models[name] = {
                        'model': model,
                        'features': feature_cols,
                        'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
                        'type': 'ml'
                    }
                    
                    logger.info(f"{name} trained - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {name}: {e}")
                    continue
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train ML models: {e}")
            return False
    
    def generate_forecasts(self, forecast_steps: int = 30, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecasts from all trained models.
        
        Args:
            forecast_steps: Number of steps to forecast
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with all forecasts
        """
        try:
            logger.info(f"Generating {forecast_steps}-step forecasts from {len(self.models)} models")
            
            all_forecasts = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'forecast'):
                        # Time series models (ARIMA, SARIMA)
                        forecast_result = model.forecast(forecast_steps, confidence_level)
                        if forecast_result:
                            all_forecasts[model_name] = forecast_result
                            logger.info(f"Generated forecast from {model_name}")
                    
                    elif isinstance(model, dict) and model.get('type') == 'ml':
                        # ML models - need to create future features
                        logger.info(f"Generating ML forecast from {model_name}...")
                        # For ML models, we'd need to engineer future features
                        # This is more complex and would require additional feature engineering
                        logger.warning(f"ML model forecasting not fully implemented for {model_name}")
                
                except Exception as e:
                    logger.warning(f"Failed to generate forecast from {model_name}: {e}")
                    continue
            
            self.forecasts = all_forecasts
            logger.info(f"Generated forecasts from {len(all_forecasts)} models")
            
            return all_forecasts
            
        except Exception as e:
            logger.error(f"Failed to generate forecasts: {e}")
            return {}
    
    def create_ensemble_forecast(self, method: str = 'simple_average') -> Dict[str, Any]:
        """
        Create ensemble forecast from multiple models.
        
        Args:
            method: Ensemble method ('simple_average', 'weighted_average')
            
        Returns:
            Ensemble forecast dictionary
        """
        try:
            if not self.forecasts:
                logger.error("No forecasts available for ensemble")
                return {}
            
            logger.info(f"Creating ensemble forecast using {method}")
            
            # Collect all forecasts
            forecast_arrays = []
            forecast_dates = None
            model_names = []
            
            for model_name, forecast_data in self.forecasts.items():
                if 'forecast' in forecast_data and 'dates' in forecast_data:
                    forecast_arrays.append(forecast_data['forecast'])
                    model_names.append(model_name)
                    if forecast_dates is None:
                        forecast_dates = forecast_data['dates']
            
            if not forecast_arrays:
                logger.error("No valid forecasts found for ensemble")
                return {}
            
            # Create ensemble
            forecast_arrays = np.array(forecast_arrays)
            
            if method == 'simple_average':
                ensemble_forecast = np.mean(forecast_arrays, axis=0)
                ensemble_std = np.std(forecast_arrays, axis=0)
            
            elif method == 'weighted_average':
                # Use performance-based weights if available
                weights = []
                for model_name in model_names:
                    if model_name in self.performance_metrics:
                        # Use inverse MSE as weight (lower MSE = higher weight)
                        mse = self.performance_metrics[model_name].get('mse', 1.0)
                        weights.append(1.0 / (mse + 1e-8))
                    else:
                        weights.append(1.0)
                
                weights = np.array(weights) / np.sum(weights)  # Normalize
                ensemble_forecast = np.average(forecast_arrays, axis=0, weights=weights)
                ensemble_std = np.sqrt(np.average((forecast_arrays - ensemble_forecast)**2, axis=0, weights=weights))
            
            else:
                logger.warning(f"Unknown ensemble method: {method}, using simple average")
                ensemble_forecast = np.mean(forecast_arrays, axis=0)
                ensemble_std = np.std(forecast_arrays, axis=0)
            
            # Create confidence intervals
            confidence_level = 0.95
            z_score = 1.96  # For 95% confidence
            lower_ci = ensemble_forecast - z_score * ensemble_std
            upper_ci = ensemble_forecast + z_score * ensemble_std
            
            ensemble_result = {
                'dates': forecast_dates,
                'forecast': ensemble_forecast,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'std': ensemble_std,
                'confidence_level': confidence_level,
                'method': method,
                'contributing_models': model_names,
                'model_count': len(model_names)
            }
            
            # Store ensemble forecast
            self.forecasts['ensemble'] = ensemble_result
            
            logger.info(f"Ensemble forecast created from {len(model_names)} models")
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Failed to create ensemble forecast: {e}")
            return {}
    
    def evaluate_forecasts(self, test_ts: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate forecast accuracy against test data.
        
        Args:
            test_ts: Test time series
            
        Returns:
            Performance metrics for each model
        """
        try:
            logger.info("Evaluating forecast performance")
            
            performance = {}
            
            for model_name, forecast_data in self.forecasts.items():
                try:
                    if 'forecast' not in forecast_data:
                        continue
                    
                    forecast = forecast_data['forecast']
                    
                    # Align forecast with test data
                    min_len = min(len(forecast), len(test_ts))
                    if min_len == 0:
                        continue
                    
                    y_true = test_ts.values[:min_len]
                    y_pred = forecast[:min_len]
                    
                    # Calculate metrics
                    mse = np.mean((y_true - y_pred) ** 2)
                    mae = np.mean(np.abs(y_true - y_pred))
                    rmse = np.sqrt(mse)
                    
                    # Mean Absolute Percentage Error
                    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                    
                    # Directional accuracy
                    if len(y_true) > 1:
                        true_direction = np.sign(np.diff(y_true))
                        pred_direction = np.sign(np.diff(y_pred))
                        directional_accuracy = np.mean(true_direction == pred_direction) * 100
                    else:
                        directional_accuracy = 0
                    
                    performance[model_name] = {
                        'mse': float(mse),
                        'mae': float(mae),
                        'rmse': float(rmse),
                        'mape': float(mape),
                        'directional_accuracy': float(directional_accuracy),
                        'forecast_points': min_len
                    }
                    
                    logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
                
                except Exception as e:
                    logger.warning(f"Failed to evaluate {model_name}: {e}")
                    continue
            
            self.performance_metrics = performance
            return performance
            
        except Exception as e:
            logger.error(f"Failed to evaluate forecasts: {e}")
            return {}
    
    def create_forecast_visualization(self, train_ts: pd.Series, test_ts: pd.Series = None) -> bool:
        """
        Create comprehensive forecast visualization.
        
        Args:
            train_ts: Training time series
            test_ts: Test time series (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.forecasts:
                logger.error("No forecasts to visualize")
                return False
            
            logger.info("Creating forecast visualization")
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle('Time Series Forecasting Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Main forecast comparison
            ax = axes[0, 0]
            
            # Plot historical data
            ax.plot(train_ts.index, train_ts.values, 'b-', label='Training Data', alpha=0.7)
            if test_ts is not None and len(test_ts) > 0:
                ax.plot(test_ts.index, test_ts.values, 'g-', label='Test Data', alpha=0.7)
            
            # Plot forecasts
            colors = ['red', 'orange', 'purple', 'brown', 'pink']
            for i, (model_name, forecast_data) in enumerate(self.forecasts.items()):
                if 'forecast' in forecast_data and 'dates' in forecast_data:
                    color = colors[i % len(colors)]
                    ax.plot(forecast_data['dates'], forecast_data['forecast'], 
                           color=color, label=f'{model_name} Forecast', alpha=0.8)
                    
                    # Add confidence intervals if available
                    if 'lower_ci' in forecast_data and 'upper_ci' in forecast_data:
                        ax.fill_between(forecast_data['dates'], 
                                      forecast_data['lower_ci'], 
                                      forecast_data['upper_ci'],
                                      color=color, alpha=0.2)
            
            ax.set_title('Forecast Comparison')
            ax.set_ylabel('Escalation Score')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Performance comparison
            ax = axes[0, 1]
            if self.performance_metrics:
                models = list(self.performance_metrics.keys())
                rmse_values = [self.performance_metrics[m]['rmse'] for m in models]
                
                bars = ax.bar(models, rmse_values, alpha=0.7)
                ax.set_title('Model Performance (RMSE)')
                ax.set_ylabel('RMSE')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, rmse_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Plot 3: Residuals (for best model)
            ax = axes[1, 0]
            if self.performance_metrics and test_ts is not None:
                # Find best model by RMSE
                best_model = min(self.performance_metrics.keys(), 
                               key=lambda x: self.performance_metrics[x]['rmse'])
                
                if best_model in self.forecasts:
                    forecast_data = self.forecasts[best_model]
                    forecast = forecast_data['forecast']
                    
                    min_len = min(len(forecast), len(test_ts))
                    residuals = test_ts.values[:min_len] - forecast[:min_len]
                    
                    ax.plot(range(len(residuals)), residuals, 'r-', alpha=0.7)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax.set_title(f'Residuals - {best_model}')
                    ax.set_ylabel('Residuals')
                    ax.grid(True, alpha=0.3)
            
            # Plot 4: Forecast uncertainty
            ax = axes[1, 1]
            if 'ensemble' in self.forecasts:
                ensemble_data = self.forecasts['ensemble']
                if 'std' in ensemble_data:
                    ax.plot(ensemble_data['dates'], ensemble_data['std'], 'purple', alpha=0.7)
                    ax.set_title('Ensemble Forecast Uncertainty')
                    ax.set_ylabel('Standard Deviation')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.output_dir / 'forecast_results.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast visualization saved to {viz_path}")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create forecast visualization: {e}")
            return False
    
    def save_forecasts(self, filename: str = None) -> bool:
        """
        Save all forecasts and performance metrics.
        
        Args:
            filename: Optional filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"forecasting_results_{timestamp}.json"
            
            # Prepare data for JSON serialization
            results = {
                'forecasts': {},
                'performance_metrics': self.performance_metrics,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_count': len(self.models),
                    'forecast_count': len(self.forecasts)
                }
            }
            
            # Convert forecasts to serializable format
            for model_name, forecast_data in self.forecasts.items():
                serializable_forecast = {}
                for key, value in forecast_data.items():
                    if hasattr(value, 'tolist'):
                        # Numpy array
                        serializable_forecast[key] = value.tolist()
                    elif hasattr(value, 'isoformat'):
                        # Single datetime
                        serializable_forecast[key] = value.isoformat()
                    elif isinstance(value, pd.DatetimeIndex):
                        # Pandas DatetimeIndex
                        serializable_forecast[key] = [d.isoformat() for d in value]
                    elif hasattr(value, '__iter__') and not isinstance(value, str):
                        # Iterable (list, array, etc.)
                        try:
                            serializable_list = []
                            for item in value:
                                if hasattr(item, 'isoformat'):
                                    serializable_list.append(item.isoformat())
                                elif hasattr(item, 'item'):  # numpy scalar
                                    serializable_list.append(item.item())
                                else:
                                    serializable_list.append(item)
                            serializable_forecast[key] = serializable_list
                        except:
                            # Fallback to string conversion
                            serializable_forecast[key] = str(value)
                    else:
                        serializable_forecast[key] = value
                
                results['forecasts'][model_name] = serializable_forecast
            
            # Save results
            import json
            results_path = self.output_dir / filename
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Forecasting results saved to {results_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save forecasts: {e}")
            return False
    
    def run_complete_analysis(self, articles: List[Dict], escalation_scores: List[float],
                             forecast_steps: int = 30) -> Dict[str, Any]:
        """
        Run complete time series forecasting analysis.
        
        Args:
            articles: List of article dictionaries
            escalation_scores: List of escalation scores
            forecast_steps: Number of steps to forecast
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info("Starting complete time series forecasting analysis")
            
            # Prepare data
            train_ts, test_ts = self.prepare_data(articles, escalation_scores)
            if train_ts.empty:
                logger.error("Failed to prepare data")
                return {}
            
            # Train models
            models_trained = 0
            
            # Train ARIMA
            if self.train_arima_model(train_ts):
                models_trained += 1
            
            # Train SARIMA
            if self.train_sarima_model(train_ts):
                models_trained += 1
            
            # Train ML models if we have processed features
            processed_df = self.processor.features
            if processed_df is not None and not processed_df.empty:
                if self.train_ml_models(processed_df):
                    models_trained += 1
            
            if models_trained == 0:
                logger.error("No models were successfully trained")
                return {}
            
            logger.info(f"Successfully trained {models_trained} model types")
            
            # Generate forecasts
            forecasts = self.generate_forecasts(forecast_steps)
            if not forecasts:
                logger.error("Failed to generate forecasts")
                return {}
            
            # Create ensemble forecast
            ensemble_forecast = self.create_ensemble_forecast()
            
            # Evaluate performance
            performance = {}
            if not test_ts.empty:
                performance = self.evaluate_forecasts(test_ts)
            
            # Create visualization
            self.create_forecast_visualization(train_ts, test_ts)
            
            # Save results
            self.save_forecasts()
            
            # Prepare summary
            analysis_results = {
                'models_trained': list(self.models.keys()),
                'forecasts_generated': list(self.forecasts.keys()),
                'performance_metrics': performance,
                'ensemble_available': 'ensemble' in self.forecasts,
                'forecast_horizon': forecast_steps,
                'data_summary': {
                    'train_samples': len(train_ts),
                    'test_samples': len(test_ts),
                    'date_range': {
                        'start': str(train_ts.index[0]) if len(train_ts) > 0 else 'N/A',
                        'end': str(train_ts.index[-1]) if len(train_ts) > 0 else 'N/A'
                    }
                }
            }
            
            logger.info("Complete time series analysis finished successfully")
            logger.info(f"Models trained: {len(self.models)}")
            logger.info(f"Forecasts generated: {len(self.forecasts)}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            return {} 