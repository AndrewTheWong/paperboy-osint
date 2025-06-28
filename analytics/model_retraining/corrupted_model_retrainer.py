"""
Corrupted Model Retrainer for Intelligence Analysis System.
Retrains corrupted models and improves model robustness.
"""

import logging
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import optuna
from dataclasses import dataclass

@dataclass
class ModelRetrainingResult:
    """Result of model retraining process."""
    model_name: str
    success: bool
    original_corrupted: bool
    retrain_method: str
    performance_metrics: Dict[str, float]
    model_size_mb: float
    training_time_seconds: float
    error_message: Optional[str] = None

class CorruptedModelRetrainer:
    """
    System to retrain corrupted models and improve model robustness.
    Handles XGBoost, PCA, and Optuna study retraining.
    """
    
    def __init__(self, models_dir: str = "models/", data_dir: str = "data/"):
        self.logger = logging.getLogger(__name__)
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.training_data = None
        
        # Model configurations
        self.model_configs = {
            'xgb_goldstein_gpu.pkl': {
                'type': 'xgboost',
                'target': 'goldstein_scale',
                'features': 'auto',
                'params': {
                    'objective': 'reg:squarederror',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            },
            'xgb_working_fast.pkl': {
                'type': 'xgboost',
                'target': 'escalation_score',
                'features': 'reduced',
                'params': {
                    'objective': 'reg:squarederror',
                    'max_depth': 4,
                    'learning_rate': 0.2,
                    'n_estimators': 100,
                    'random_state': 42
                }
            },
            'article_pca_model.pkl': {
                'type': 'pca',
                'n_components': 10,
                'features': 'content_features'
            },
            'optuna_study.pkl': {
                'type': 'optuna',
                'n_trials': 100,
                'optimization_direction': 'minimize'
            }
        }
        
        self.feature_sets = {
            'auto': self._get_all_features,
            'reduced': self._get_reduced_features,
            'content_features': self._get_content_features
        }
        
        self._load_training_data()
    
    def _load_training_data(self):
        """Load training data from various sources."""
        training_files = [
            'processed_gdelt_data.csv',
            'processed_osint_articles.csv',
            'unified_training_data.csv'
        ]
        
        dataframes = []
        for filename in training_files:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    dataframes.append(df)
                    self.logger.info(f"Loaded {len(df)} rows from {filename}")
                except Exception as e:
                    self.logger.warning(f"Could not load {filename}: {e}")
        
        if dataframes:
            self.training_data = pd.concat(dataframes, ignore_index=True)
            self.logger.info(f"Combined training data: {len(self.training_data)} total rows")
        else:
            self.logger.warning("No training data found, will generate synthetic data")
            self.training_data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data for testing."""
        np.random.seed(42)
        
        data = {
            'goldstein_scale': np.random.normal(-2, 4, n_samples),
            'avg_tone': np.random.normal(0, 3, n_samples),
            'num_mentions': np.random.poisson(5, n_samples),
            'num_sources': np.random.poisson(3, n_samples),
            'escalation_score': np.random.beta(2, 5, n_samples),
            'sentiment_score': np.random.normal(0, 1, n_samples),
            'content_length': np.random.exponential(500, n_samples),
            'num_actors': np.random.poisson(2, n_samples),
            'num_locations': np.random.poisson(1, n_samples)
        }
        
        # Add some correlations
        data['escalation_score'] = np.clip(
            0.5 * (1 - data['goldstein_scale'] / 10) + 0.3 * np.random.random(n_samples),
            0, 1
        )
        
        return pd.DataFrame(data)
    
    def _get_all_features(self, data: pd.DataFrame) -> List[str]:
        """Get all available features."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_targets = ['goldstein_scale', 'escalation_score']
        return [col for col in numeric_columns if col not in exclude_targets]
    
    def _get_reduced_features(self, data: pd.DataFrame) -> List[str]:
        """Get reduced feature set for fast models."""
        return ['avg_tone', 'num_mentions', 'num_sources', 'sentiment_score']
    
    def _get_content_features(self, data: pd.DataFrame) -> List[str]:
        """Get content-based features for PCA."""
        content_features = ['content_length', 'sentiment_score', 'avg_tone', 'num_actors', 'num_locations']
        return [col for col in content_features if col in data.columns]
    
    def diagnose_models(self) -> Dict[str, Dict[str, Any]]:
        """Diagnose all models and identify corruption issues."""
        model_files = [
            'xgb_goldstein_gpu.pkl',
            'xgb_goldstein_gpu_tuned.pkl',
            'xgb_goldstein_regressor.pkl',
            'xgb_working_fast.pkl',
            'article_pca_model.pkl',
            'optuna_study.pkl',
            'tft_gdelt_enhanced_model_state.pt',
            'goldstein_predictor/'
        ]
        
        diagnosis = {}
        
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            
            if not os.path.exists(model_path):
                diagnosis[model_file] = {
                    'status': 'missing',
                    'error': 'File not found',
                    'size_mb': 0,
                    'corrupted': False
                }
                continue
            
            try:
                # Get file size
                if os.path.isfile(model_path):
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                else:
                    size_mb = sum(os.path.getsize(os.path.join(model_path, f)) 
                                for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))) / (1024 * 1024)
                
                # Try to load the model
                if model_file.endswith('.pkl'):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    diagnosis[model_file] = {
                        'status': 'working',
                        'error': None,
                        'size_mb': size_mb,
                        'corrupted': False,
                        'model_type': type(model).__name__
                    }
                elif model_file.endswith('.pt'):
                    # PyTorch model - basic existence check
                    diagnosis[model_file] = {
                        'status': 'working',
                        'error': None,
                        'size_mb': size_mb,
                        'corrupted': False,
                        'model_type': 'PyTorch'
                    }
                else:
                    # Directory-based model
                    diagnosis[model_file] = {
                        'status': 'working',
                        'error': None,
                        'size_mb': size_mb,
                        'corrupted': False,
                        'model_type': 'Directory'
                    }
                    
            except Exception as e:
                diagnosis[model_file] = {
                    'status': 'corrupted',
                    'error': str(e),
                    'size_mb': size_mb if 'size_mb' in locals() else 0,
                    'corrupted': True
                }
        
        # Summary statistics
        total_models = len(diagnosis)
        working_models = sum(1 for d in diagnosis.values() if d['status'] == 'working')
        corrupted_models = sum(1 for d in diagnosis.values() if d['status'] == 'corrupted')
        missing_models = sum(1 for d in diagnosis.values() if d['status'] == 'missing')
        
        self.logger.info(f"Model Diagnosis: {working_models}/{total_models} working, {corrupted_models} corrupted, {missing_models} missing")
        
        return {
            'models': diagnosis,
            'summary': {
                'total': total_models,
                'working': working_models,
                'corrupted': corrupted_models,
                'missing': missing_models,
                'diagnosis_time': datetime.now().isoformat()
            }
        }
    
    def retrain_model(self, model_name: str, force_retrain: bool = False) -> ModelRetrainingResult:
        """Retrain a specific model."""
        start_time = datetime.now()
        
        if model_name not in self.model_configs:
            return ModelRetrainingResult(
                model_name=model_name,
                success=False,
                original_corrupted=True,
                retrain_method='unknown',
                performance_metrics={},
                model_size_mb=0,
                training_time_seconds=0,
                error_message=f"Unknown model configuration for {model_name}"
            )
        
        config = self.model_configs[model_name]
        model_type = config['type']
        
        try:
            if model_type == 'xgboost':
                result = self._retrain_xgboost_model(model_name, config)
            elif model_type == 'pca':
                result = self._retrain_pca_model(model_name, config)
            elif model_type == 'optuna':
                result = self._retrain_optuna_study(model_name, config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            result.training_time_seconds = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            return ModelRetrainingResult(
                model_name=model_name,
                success=False,
                original_corrupted=True,
                retrain_method=model_type,
                performance_metrics={},
                model_size_mb=0,
                training_time_seconds=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )
    
    def _retrain_xgboost_model(self, model_name: str, config: Dict) -> ModelRetrainingResult:
        """Retrain XGBoost model."""
        target = config['target']
        feature_set = config['features']
        params = config['params']
        
        # Prepare data
        if target not in self.training_data.columns:
            raise ValueError(f"Target column '{target}' not found in training data")
        
        feature_columns = self.feature_sets[feature_set](self.training_data)
        feature_columns = [col for col in feature_columns if col in self.training_data.columns]
        
        if not feature_columns:
            raise ValueError(f"No valid features found for feature set '{feature_set}'")
        
        X = self.training_data[feature_columns].fillna(0)
        y = self.training_data[target].fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_columns)
        }
        
        # Save model
        model_path = os.path.join(self.models_dir, model_name)
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(model, model_path)
        
        # Get model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        return ModelRetrainingResult(
            model_name=model_name,
            success=True,
            original_corrupted=True,
            retrain_method='xgboost',
            performance_metrics=metrics,
            model_size_mb=model_size_mb,
            training_time_seconds=0  # Will be set by caller
        )
    
    def _retrain_pca_model(self, model_name: str, config: Dict) -> ModelRetrainingResult:
        """Retrain PCA model."""
        n_components = config['n_components']
        feature_set = config['features']
        
        # Prepare data
        feature_columns = self.feature_sets[feature_set](self.training_data)
        feature_columns = [col for col in feature_columns if col in self.training_data.columns]
        
        if not feature_columns:
            raise ValueError(f"No valid features found for PCA")
        
        X = self.training_data[feature_columns].fillna(0)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train PCA
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        # Calculate metrics
        explained_variance_ratio = pca.explained_variance_ratio_
        metrics = {
            'explained_variance_ratio_total': sum(explained_variance_ratio),
            'n_components': n_components,
            'original_features': len(feature_columns),
            'training_samples': len(X)
        }
        
        # Save model with scaler
        model_data = {
            'pca': pca,
            'scaler': scaler,
            'feature_columns': feature_columns
        }
        
        model_path = os.path.join(self.models_dir, model_name)
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(model_data, model_path)
        
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        return ModelRetrainingResult(
            model_name=model_name,
            success=True,
            original_corrupted=True,
            retrain_method='pca',
            performance_metrics=metrics,
            model_size_mb=model_size_mb,
            training_time_seconds=0
        )
    
    def _retrain_optuna_study(self, model_name: str, config: Dict) -> ModelRetrainingResult:
        """Retrain Optuna study."""
        n_trials = config['n_trials']
        direction = config['optimization_direction']
        
        def objective(trial):
            # Sample hyperparameters
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.05, 0.3)
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            
            # Prepare data
            feature_columns = self._get_reduced_features(self.training_data)
            X = self.training_data[feature_columns].fillna(0)
            y = self.training_data['escalation_score'].fillna(0)
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = xgb.XGBRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            
            return mse
        
        # Create and run study
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        
        # Calculate metrics
        metrics = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        }
        
        # Save study
        model_path = os.path.join(self.models_dir, model_name)
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(study, model_path)
        
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        return ModelRetrainingResult(
            model_name=model_name,
            success=True,
            original_corrupted=True,
            retrain_method='optuna',
            performance_metrics=metrics,
            model_size_mb=model_size_mb,
            training_time_seconds=0
        )
    
    def retrain_all_corrupted_models(self) -> Dict[str, ModelRetrainingResult]:
        """Retrain all corrupted models."""
        diagnosis = self.diagnose_models()
        corrupted_models = [
            name for name, info in diagnosis['models'].items() 
            if info['status'] == 'corrupted' and name in self.model_configs
        ]
        
        results = {}
        
        for model_name in corrupted_models:
            self.logger.info(f"Retraining corrupted model: {model_name}")
            result = self.retrain_model(model_name)
            results[model_name] = result
            
            if result.success:
                self.logger.info(f"Successfully retrained {model_name}")
            else:
                self.logger.error(f"Failed to retrain {model_name}: {result.error_message}")
        
        return results


def main():
    """Test the model retraining system."""
    logging.basicConfig(level=logging.INFO)
    
    retrainer = CorruptedModelRetrainer()
    
    # Diagnose models
    diagnosis = retrainer.diagnose_models()
    print("Model Diagnosis:")
    for model_name, info in diagnosis['models'].items():
        print(f"- {model_name}: {info['status']} ({info['size_mb']:.2f} MB)")
        if info['error']:
            print(f"  Error: {info['error']}")
    
    print(f"\nSummary: {diagnosis['summary']}")
    
    # Retrain corrupted models
    results = retrainer.retrain_all_corrupted_models()
    
    print("\nRetraining Results:")
    for model_name, result in results.items():
        print(f"- {model_name}: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Performance: {result.performance_metrics}")
        else:
            print(f"  Error: {result.error_message}")


if __name__ == "__main__":
    main() 