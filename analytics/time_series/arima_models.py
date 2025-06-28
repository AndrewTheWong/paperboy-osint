#!/usr/bin/env python3
"""
ARIMA and SARIMA Models for Escalation Forecasting

This module implements time series forecasting models to predict:
- Escalation trends over time
- Seasonal patterns in geopolitical tensions
- Anomaly detection in escalation patterns
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

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Install with: pip install statsmodels")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAPredictor:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model for escalation forecasting.
    
    Features:
    - Grid search parameter selection (p, d, q)
    - Model diagnostics and validation
    - Confidence intervals for predictions
    - Anomaly detection
    """
    
    def __init__(self, output_dir: str = "analytics/time_series/models"):
        """Initialize ARIMA predictor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.fitted_model = None
        self.data = None
        self.order = None
        self.diagnostics = {}
        
        logger.info("ARIMAPredictor initialized")
    
    def prepare_data(self, articles: List[Dict], escalation_scores: List[float]) -> pd.Series:
        """
        Prepare time series data from articles and escalation scores.
        
        Args:
            articles: List of article dictionaries
            escalation_scores: List of escalation scores
            
        Returns:
            Time series data indexed by date
        """
        try:
            # Extract dates and scores
            data_points = []
            for article, score in zip(articles, escalation_scores):
                try:
                    # Try different date fields
                    date_str = article.get('scraped_at') or article.get('published_at') or article.get('created_at')
                    if date_str:
                        # Parse various date formats
                        if isinstance(date_str, str):
                            try:
                                date = pd.to_datetime(date_str)
                            except:
                                # Try alternative parsing
                                date = pd.to_datetime(date_str, errors='coerce')
                        else:
                            date = pd.to_datetime(date_str)
                        
                        if pd.notna(date):
                            data_points.append({'date': date, 'escalation_score': score})
                except Exception as e:
                    logger.warning(f"Failed to parse date for article: {e}")
                    continue
            
            if not data_points:
                # Create synthetic time series for demonstration
                logger.warning("No valid dates found, creating synthetic time series")
                dates = pd.date_range(start='2024-01-01', periods=len(escalation_scores), freq='D')
                data_points = [{'date': date, 'escalation_score': score} 
                             for date, score in zip(dates, escalation_scores)]
            
            # Create DataFrame and aggregate by date
            df = pd.DataFrame(data_points)
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Aggregate multiple articles per day (mean escalation)
            daily_escalation = df.groupby('date')['escalation_score'].agg(['mean', 'count', 'std']).reset_index()
            daily_escalation.columns = ['date', 'escalation_score', 'article_count', 'escalation_std']
            
            # Create time series
            ts = pd.Series(
                daily_escalation['escalation_score'].values,
                index=pd.to_datetime(daily_escalation['date']),
                name='escalation_score'
            )
            
            # Fill missing dates with interpolation
            ts = ts.asfreq('D').interpolate(method='linear')
            
            logger.info(f"Prepared time series: {len(ts)} days, range {ts.index.min()} to {ts.index.max()}")
            logger.info(f"Escalation range: {ts.min():.3f} to {ts.max():.3f}")
            
            self.data = ts
            return ts
            
        except Exception as e:
            logger.error(f"Failed to prepare time series data: {e}")
            # Return synthetic data as fallback
            dates = pd.date_range(start='2024-01-01', periods=len(escalation_scores), freq='D')
            ts = pd.Series(escalation_scores, index=dates, name='escalation_score')
            self.data = ts
            return ts
    
    def check_stationarity(self, ts: pd.Series) -> Dict[str, Any]:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            ts: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        try:
            if not STATSMODELS_AVAILABLE:
                return {"stationary": False, "p_value": 1.0, "critical_values": {}}
            
            result = adfuller(ts.dropna())
            
            stationarity_result = {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'stationary': result[1] < 0.05
            }
            
            logger.info(f"Stationarity test - ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
            logger.info(f"Series is {'stationary' if stationarity_result['stationary'] else 'non-stationary'}")
            
            return stationarity_result
            
        except Exception as e:
            logger.error(f"Failed to check stationarity: {e}")
            return {"stationary": False, "p_value": 1.0, "critical_values": {}}
    
    def find_optimal_order(self, ts: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order (p, d, q) using grid search.
        
        Args:
            ts: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order  
            max_q: Maximum MA order
            
        Returns:
            Optimal (p, d, q) order
        """
        try:
            if STATSMODELS_AVAILABLE and len(ts.dropna()) > 20:
                # Grid search for optimal parameters
                logger.info("Using grid search for parameter selection...")
                best_aic = float('inf')
                best_order = (1, 1, 1)
                
                for p in range(max_p + 1):
                    for d in range(max_d + 1):
                        for q in range(max_q + 1):
                            try:
                                model = ARIMA(ts.dropna(), order=(p, d, q))
                                fitted = model.fit()
                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                logger.info(f"Grid search selected order: {best_order} (AIC: {best_aic:.2f})")
                return best_order
            
            else:
                # Default order
                logger.warning("Using default ARIMA order (1,1,1)")
                return (1, 1, 1)
                
        except Exception as e:
            logger.error(f"Failed to find optimal order: {e}")
            return (1, 1, 1)
    
    def fit(self, ts: pd.Series, order: Optional[Tuple[int, int, int]] = None) -> bool:
        """
        Fit ARIMA model to time series data.
        
        Args:
            ts: Time series data
            order: ARIMA order (p, d, q). If None, will auto-select
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not STATSMODELS_AVAILABLE:
                logger.error("Statsmodels not available for ARIMA fitting")
                return False
            
            # Store data
            self.data = ts
            
            # Check stationarity
            stationarity = self.check_stationarity(ts)
            self.diagnostics['stationarity'] = stationarity
            
            # Find optimal order if not provided
            if order is None:
                order = self.find_optimal_order(ts)
            
            self.order = order
            
            # Fit ARIMA model
            logger.info(f"Fitting ARIMA{order} model...")
            self.model = ARIMA(ts.dropna(), order=order)
            self.fitted_model = self.model.fit()
            
            # Store model diagnostics
            self.diagnostics.update({
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'llf': self.fitted_model.llf,
                'params': self.fitted_model.params.to_dict()
            })
            
            logger.info(f"ARIMA model fitted successfully")
            logger.info(f"AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            return False
    
    def forecast(self, steps: int = 30, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecasts with confidence intervals.
        
        Args:
            steps: Number of periods to forecast
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with forecasts and confidence intervals
        """
        try:
            if self.fitted_model is None:
                logger.error("Model not fitted. Call fit() first.")
                return {}
            
            # Generate forecast
            forecast_result = self.fitted_model.forecast(steps=steps)
            forecast_ci = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=1-confidence_level)
            
            # Create future dates
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
            
            # Organize results
            forecast_data = {
                'dates': future_dates,
                'forecast': forecast_result,
                'lower_ci': forecast_ci.iloc[:, 0],
                'upper_ci': forecast_ci.iloc[:, 1],
                'confidence_level': confidence_level
            }
            
            logger.info(f"Generated {steps}-step forecast")
            logger.info(f"Forecast range: {forecast_result.min():.3f} to {forecast_result.max():.3f}")
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            return {}
    
    def detect_anomalies(self, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect anomalies in the time series using residuals.
        
        Args:
            threshold: Standard deviation threshold for anomaly detection
            
        Returns:
            Dictionary with anomaly information
        """
        try:
            if self.fitted_model is None:
                logger.error("Model not fitted. Call fit() first.")
                return {}
            
            # Calculate residuals
            residuals = self.fitted_model.resid
            residual_std = residuals.std()
            
            # Identify anomalies
            anomaly_mask = np.abs(residuals) > threshold * residual_std
            anomaly_dates = self.data.index[anomaly_mask]
            anomaly_values = self.data[anomaly_mask]
            anomaly_residuals = residuals[anomaly_mask]
            
            anomaly_data = {
                'dates': anomaly_dates,
                'values': anomaly_values,
                'residuals': anomaly_residuals,
                'threshold': threshold,
                'count': len(anomaly_dates)
            }
            
            logger.info(f"Detected {len(anomaly_dates)} anomalies (threshold: {threshold}σ)")
            
            return anomaly_data
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return {}
    
    def create_diagnostics_plots(self) -> bool:
        """Create diagnostic plots for model evaluation."""
        try:
            if self.fitted_model is None:
                logger.error("Model not fitted. Call fit() first.")
                return False
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'ARIMA{self.order} Model Diagnostics', fontsize=16, fontweight='bold')
            
            # Plot 1: Original time series
            axes[0, 0].plot(self.data.index, self.data.values, 'b-', alpha=0.7)
            axes[0, 0].set_title('Original Time Series')
            axes[0, 0].set_ylabel('Escalation Score')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Fitted values vs actual
            fitted_values = self.fitted_model.fittedvalues
            axes[0, 1].plot(self.data.index[1:], self.data.values[1:], 'b-', label='Actual', alpha=0.7)
            axes[0, 1].plot(fitted_values.index, fitted_values.values, 'r-', label='Fitted', alpha=0.7)
            axes[0, 1].set_title('Fitted vs Actual Values')
            axes[0, 1].set_ylabel('Escalation Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Residuals
            residuals = self.fitted_model.resid
            axes[0, 2].plot(residuals.index, residuals.values, 'g-', alpha=0.7)
            axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 2].set_title('Residuals')
            axes[0, 2].set_ylabel('Residuals')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: ACF of residuals
            if STATSMODELS_AVAILABLE:
                plot_acf(residuals.dropna(), ax=axes[1, 0], lags=20, title='ACF of Residuals')
            
            # Plot 5: PACF of residuals  
            if STATSMODELS_AVAILABLE:
                plot_pacf(residuals.dropna(), ax=axes[1, 1], lags=20, title='PACF of Residuals')
            
            # Plot 6: Q-Q plot of residuals
            from scipy import stats
            stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 2])
            axes[1, 2].set_title('Q-Q Plot of Residuals')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / 'arima_diagnostics.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diagnostics plot saved to {plot_path}")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create diagnostics plots: {e}")
            return False
    
    def save_model(self, filename: str = None) -> bool:
        """Save the fitted model."""
        try:
            if self.fitted_model is None:
                logger.error("No model to save. Call fit() first.")
                return False
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"arima_model_{timestamp}.pkl"
            
            model_path = self.output_dir / filename
            
            # Save model and metadata
            model_data = {
                'model': self.fitted_model,
                'order': self.order,
                'diagnostics': self.diagnostics,
                'data_info': {
                    'start_date': str(self.data.index[0]),
                    'end_date': str(self.data.index[-1]),
                    'n_observations': len(self.data)
                }
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"ARIMA model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False


class SARIMAPredictor:
    """
    SARIMA (Seasonal ARIMA) model for escalation forecasting with seasonal patterns.
    
    Features:
    - Seasonal pattern detection
    - Grid search seasonal parameter selection
    - Advanced diagnostics
    - Multiple seasonality handling
    """
    
    def __init__(self, output_dir: str = "analytics/time_series/models"):
        """Initialize SARIMA predictor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.fitted_model = None
        self.data = None
        self.order = None
        self.seasonal_order = None
        self.diagnostics = {}
        
        logger.info("SARIMAPredictor initialized")
    
    def prepare_data(self, articles: List[Dict], escalation_scores: List[float]) -> pd.Series:
        """
        Prepare time series data from articles and escalation scores.
        
        Args:
            articles: List of article dictionaries
            escalation_scores: List of escalation scores
            
        Returns:
            Time series data indexed by date
        """
        try:
            # Extract dates and scores
            data_points = []
            for article, score in zip(articles, escalation_scores):
                try:
                    # Try different date fields
                    date_str = article.get('scraped_at') or article.get('published_at') or article.get('created_at')
                    if date_str:
                        # Parse various date formats
                        if isinstance(date_str, str):
                            try:
                                date = pd.to_datetime(date_str)
                            except:
                                # Try alternative parsing
                                date = pd.to_datetime(date_str, errors='coerce')
                        else:
                            date = pd.to_datetime(date_str)
                        
                        if pd.notna(date):
                            data_points.append({'date': date, 'escalation_score': score})
                except Exception as e:
                    logger.warning(f"Failed to parse date for article: {e}")
                    continue
            
            if not data_points:
                # Create synthetic time series for demonstration
                logger.warning("No valid dates found, creating synthetic time series")
                dates = pd.date_range(start='2024-01-01', periods=len(escalation_scores), freq='D')
                data_points = [{'date': date, 'escalation_score': score} 
                             for date, score in zip(dates, escalation_scores)]
            
            # Create DataFrame and aggregate by date
            df = pd.DataFrame(data_points)
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Aggregate multiple articles per day (mean escalation)
            daily_escalation = df.groupby('date')['escalation_score'].agg(['mean', 'count', 'std']).reset_index()
            daily_escalation.columns = ['date', 'escalation_score', 'article_count', 'escalation_std']
            
            # Create time series
            ts = pd.Series(
                daily_escalation['escalation_score'].values,
                index=pd.to_datetime(daily_escalation['date']),
                name='escalation_score'
            )
            
            # Fill missing dates with interpolation
            ts = ts.asfreq('D').interpolate(method='linear')
            
            logger.info(f"Prepared time series: {len(ts)} days, range {ts.index.min()} to {ts.index.max()}")
            logger.info(f"Escalation range: {ts.min():.3f} to {ts.max():.3f}")
            
            self.data = ts
            return ts
            
        except Exception as e:
            logger.error(f"Failed to prepare time series data: {e}")
            # Return synthetic data as fallback
            dates = pd.date_range(start='2024-01-01', periods=len(escalation_scores), freq='D')
            ts = pd.Series(escalation_scores, index=dates, name='escalation_score')
            self.data = ts
            return ts
    
    def detect_seasonality(self, ts: pd.Series, periods: List[int] = [7, 30, 365]) -> Dict[str, Any]:
        """
        Detect seasonal patterns in the time series.
        
        Args:
            ts: Time series data
            periods: List of potential seasonal periods to check
            
        Returns:
            Dictionary with seasonality information
        """
        try:
            seasonality_results = {}
            
            for period in periods:
                if len(ts) >= 2 * period:
                    try:
                        if STATSMODELS_AVAILABLE:
                            # Perform seasonal decomposition
                            decomposition = seasonal_decompose(ts.dropna(), model='additive', period=period)
                            
                            # Calculate seasonal strength
                            seasonal_var = np.var(decomposition.seasonal.dropna())
                            residual_var = np.var(decomposition.resid.dropna())
                            
                            seasonal_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0
                            
                            seasonality_results[f'period_{period}'] = {
                                'seasonal_strength': seasonal_strength,
                                'seasonal_variance': seasonal_var,
                                'residual_variance': residual_var,
                                'is_seasonal': seasonal_strength > 0.3
                            }
                            
                            logger.info(f"Period {period}: Seasonal strength = {seasonal_strength:.3f}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze period {period}: {e}")
                        continue
            
            # Determine best seasonal period
            best_period = None
            best_strength = 0
            
            for period_key, result in seasonality_results.items():
                if result['seasonal_strength'] > best_strength:
                    best_strength = result['seasonal_strength']
                    best_period = int(period_key.split('_')[1])
            
            seasonality_results['best_period'] = best_period
            seasonality_results['best_strength'] = best_strength
            
            logger.info(f"Best seasonal period: {best_period} (strength: {best_strength:.3f})")
            
            return seasonality_results
            
        except Exception as e:
            logger.error(f"Failed to detect seasonality: {e}")
            return {}
    
    def find_optimal_seasonal_order(self, ts: pd.Series, seasonal_period: int = 7) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Find optimal SARIMA orders using grid search.
        
        Args:
            ts: Time series data
            seasonal_period: Seasonal period
            
        Returns:
            Optimal (p, d, q) and (P, D, Q, s) orders
        """
        try:
            if STATSMODELS_AVAILABLE and len(ts.dropna()) > 3 * seasonal_period:
                # Grid search for seasonal parameters
                logger.info(f"Using grid search for seasonal parameter selection (period={seasonal_period})...")
                
                best_aic = float('inf')
                best_order = (1, 1, 1)
                best_seasonal_order = (1, 1, 1, seasonal_period)
                
                # Reduced search space for efficiency
                for p in range(2):
                    for d in range(2):
                        for q in range(2):
                            for P in range(2):
                                for D in range(2):
                                    for Q in range(2):
                                        try:
                                            order = (p, d, q)
                                            seasonal_order = (P, D, Q, seasonal_period)
                                            
                                            model = SARIMAX(ts.dropna(), 
                                                           order=order, 
                                                           seasonal_order=seasonal_order)
                                            fitted = model.fit(disp=False)
                                            
                                            if fitted.aic < best_aic:
                                                best_aic = fitted.aic
                                                best_order = order
                                                best_seasonal_order = seasonal_order
                                        except:
                                            continue
                
                logger.info(f"Grid search selected orders: {best_order}, seasonal: {best_seasonal_order}")
                return best_order, best_seasonal_order
            
            else:
                # Default orders
                logger.warning(f"Using default SARIMA orders")
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, seasonal_period)
                return order, seasonal_order
                
        except Exception as e:
            logger.error(f"Failed to find optimal seasonal order: {e}")
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 7)
            return order, seasonal_order
    
    def fit(self, ts: pd.Series, order: Optional[Tuple[int, int, int]] = None, 
            seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Fit SARIMA model to time series data.
        
        Args:
            ts: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not STATSMODELS_AVAILABLE:
                logger.error("Statsmodels not available for SARIMA fitting")
                return False
            
            # Store data
            self.data = ts
            
            # Detect seasonality
            seasonality = self.detect_seasonality(ts)
            self.diagnostics['seasonality'] = seasonality
            
            # Determine seasonal period
            seasonal_period = seasonality.get('best_period', 7)
            
            # Find optimal orders if not provided
            if order is None or seasonal_order is None:
                order, seasonal_order = self.find_optimal_seasonal_order(ts, seasonal_period)
            
            self.order = order
            self.seasonal_order = seasonal_order
            
            # Fit SARIMA model
            logger.info(f"Fitting SARIMA{order}x{seasonal_order} model...")
            self.model = SARIMAX(ts.dropna(), order=order, seasonal_order=seasonal_order)
            self.fitted_model = self.model.fit(disp=False)
            
            # Store model diagnostics
            self.diagnostics.update({
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'llf': self.fitted_model.llf,
                'params': self.fitted_model.params.to_dict()
            })
            
            logger.info(f"SARIMA model fitted successfully")
            logger.info(f"AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to fit SARIMA model: {e}")
            return False
    
    def forecast(self, steps: int = 30, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Generate seasonal forecasts."""
        try:
            if self.fitted_model is None:
                logger.error("Model not fitted. Call fit() first.")
                return {}
            
            # Generate forecast
            forecast_result = self.fitted_model.forecast(steps=steps)
            forecast_ci = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=1-confidence_level)
            
            # Create future dates
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
            
            # Organize results
            forecast_data = {
                'dates': future_dates,
                'forecast': forecast_result,
                'lower_ci': forecast_ci.iloc[:, 0],
                'upper_ci': forecast_ci.iloc[:, 1],
                'confidence_level': confidence_level
            }
            
            logger.info(f"Generated {steps}-step seasonal forecast")
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Failed to generate seasonal forecast: {e}")
            return {}
    
    def create_seasonal_plots(self) -> bool:
        """Create seasonal analysis plots."""
        try:
            if self.fitted_model is None:
                logger.error("Model not fitted. Call fit() first.")
                return False
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'SARIMA{self.order}x{self.seasonal_order} Seasonal Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Original series with seasonal decomposition
            if STATSMODELS_AVAILABLE and len(self.data) >= 2 * self.seasonal_order[3]:
                decomposition = seasonal_decompose(self.data.dropna(), model='additive', period=self.seasonal_order[3])
                
                axes[0, 0].plot(self.data.index, self.data.values, 'b-', alpha=0.7, label='Original')
                axes[0, 0].plot(decomposition.trend.index, decomposition.trend.values, 'r-', alpha=0.7, label='Trend')
                axes[0, 0].set_title('Original Series with Trend')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Seasonal component
                axes[0, 1].plot(decomposition.seasonal.index, decomposition.seasonal.values, 'g-', alpha=0.7)
                axes[0, 1].set_title('Seasonal Component')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Fitted vs actual
            fitted_values = self.fitted_model.fittedvalues
            axes[1, 0].plot(self.data.index[1:], self.data.values[1:], 'b-', label='Actual', alpha=0.7)
            axes[1, 0].plot(fitted_values.index, fitted_values.values, 'r-', label='Fitted', alpha=0.7)
            axes[1, 0].set_title('Fitted vs Actual Values')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Residuals
            residuals = self.fitted_model.resid
            axes[1, 1].plot(residuals.index, residuals.values, 'g-', alpha=0.7)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Residuals')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / 'sarima_seasonal_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Seasonal analysis plot saved to {plot_path}")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create seasonal plots: {e}")
            return False
    
    def save_model(self, filename: str = None) -> bool:
        """Save the fitted SARIMA model."""
        try:
            if self.fitted_model is None:
                logger.error("No model to save. Call fit() first.")
                return False
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sarima_model_{timestamp}.pkl"
            
            model_path = self.output_dir / filename
            
            # Save model and metadata
            model_data = {
                'model': self.fitted_model,
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'diagnostics': self.diagnostics,
                'data_info': {
                    'start_date': str(self.data.index[0]),
                    'end_date': str(self.data.index[-1]),
                    'n_observations': len(self.data)
                }
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"SARIMA model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False 