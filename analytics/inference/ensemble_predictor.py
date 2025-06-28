#!/usr/bin/env python3
"""
Enhanced Ensemble Predictor - Advanced escalation prediction using multiple model types.
Combines XGBoost, OSINT transformers, and rule-based approaches for robust prediction.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sentence_transformers import SentenceTransformer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Import OSINT model architecture
try:
    from models.OSINT_Tension.student_model_training import OSINTStudentTransformer
    from models.train_transformer_model import IntelligenceTransformer
    HAS_OSINT_MODELS = True
except ImportError:
    HAS_OSINT_MODELS = False

class OSINTTransformerPredictor:
    """Wrapper for OSINT transformer models."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.text_encoder = None
        self.embedding_dim = 384
        self.sequence_length = 30
        
        self._load_model()
        self._load_text_encoder()
    
    def _load_model(self):
        """Load the OSINT transformer model."""
        if not HAS_TORCH or not HAS_OSINT_MODELS:
            logger.warning("PyTorch or OSINT models not available")
            return
            
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get model config
            model_config = checkpoint.get('model_config', {
                'd_model': 256, 'nhead': 8, 'num_layers': 4, 
                'dim_feedforward': 512, 'dropout': 0.1
            })
            
            # Create model instance
            self.model = OSINTStudentTransformer(
                embedding_dim=self.embedding_dim,
                sequence_length=self.sequence_length,
                **model_config
            ).to(self.device)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"✅ Loaded OSINT transformer: {os.path.basename(self.model_path)}")
            
        except Exception as e:
            logger.warning(f"Failed to load OSINT model {self.model_path}: {e}")
            self.model = None
    
    def _load_text_encoder(self):
        """Load the text encoder for embedding articles."""
        if not HAS_SKLEARN:
            return
            
        try:
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded text encoder for OSINT transformer")
        except Exception as e:
            logger.warning(f"Failed to load text encoder: {e}")
    
    def predict(self, articles: List[Dict[str, Any]]) -> List[float]:
        """Predict escalation scores for articles."""
        if not self.model or not self.text_encoder:
            return [0.5] * len(articles)  # Neutral predictions
        
        try:
            # Create embeddings for articles
            texts = []
            for article in articles:
                title = article.get('title', '')
                content = article.get('content', '')
                combined_text = f"{title}. {content}"[:1000]  # Limit length
                texts.append(combined_text)
            
            # Generate embeddings
            embeddings = self.text_encoder.encode(texts)
            
            # Create sequence data (pad/truncate as needed)
            batch_size = len(articles)
            sequences = np.zeros((batch_size, self.sequence_length, self.embedding_dim))
            
            # For single articles, repeat embedding across sequence
            for i, embedding in enumerate(embeddings):
                sequences[i, :, :] = embedding  # Broadcast across sequence
            
            # Convert to tensor and predict
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(sequences_tensor)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]  # Get main prediction
                
                predictions = torch.sigmoid(predictions)  # Ensure 0-1 range
                return predictions.cpu().numpy().tolist()
        
        except Exception as e:
            logger.warning(f"OSINT transformer prediction failed: {e}")
            return [0.5] * len(articles)

class CompatibleFeatureExtractor:
    """Feature extractor that creates both 8-feature and extended feature sets."""
    
    def __init__(self):
        """Initialize feature extractor."""
        # Core 8 features that match the trained XGBoost model
        self.core_features = [
            'escalation_keywords', 'location_relevance', 'actor_mentions',
            'urgency_score', 'sentiment_score', 'military_indicators',
            'temporal_indicators', 'conflict_density'
        ]
        
        # Extended features for other models
        self.escalation_keywords = [
            'war', 'conflict', 'attack', 'missile', 'military', 'threat', 'invasion',
            'combat', 'strike', 'assault', 'aggression', 'hostility', 'tension',
            'escalate', 'crisis', 'emergency', 'alert', 'mobilize', 'deploy',
            'retaliate', 'counter-attack', 'defend', 'offensive', 'defensive'
        ]
        
        self.location_keywords = [
            'taiwan', 'strait', 'china', 'beijing', 'taipei', 'kinmen', 'matsu',
            'south china sea', 'east china sea', 'pacific', 'indo-pacific'
        ]
        
        self.actors = [
            'china', 'taiwan', 'usa', 'japan', 'south korea', 'philippines',
            'pla', 'military', 'navy', 'air force', 'army'
        ]
        
        # Initialize TF-IDF for extended features if available
        if HAS_SKLEARN:
            self.tfidf = TfidfVectorizer(max_features=50, stop_words='english', lowercase=True)
        else:
            self.tfidf = None
    
    def extract_core_features(self, articles: List[Dict[str, Any]]) -> np.ndarray:
        """Extract core 8 features compatible with trained XGBoost model."""
        features = []
        
        for article in articles:
            feature_vector = self._extract_core_article_features(article)
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_extended_features(self, articles: List[Dict[str, Any]]) -> np.ndarray:
        """Extract extended feature set including TF-IDF features."""
        # Start with core features
        core_features = self.extract_core_features(articles)
        
        # Add TF-IDF features if available
        if self.tfidf and HAS_SKLEARN:
            texts = [f"{article.get('title', '')} {article.get('content', '')}" for article in articles]
            try:
                tfidf_features = self.tfidf.fit_transform(texts).toarray()
                extended_features = np.hstack([core_features, tfidf_features])
                return extended_features
            except Exception as e:
                logger.warning(f"Failed to add TF-IDF features: {e}")
        
        return core_features
    
    def _extract_core_article_features(self, article: Dict[str, Any]) -> List[float]:
        """Extract core 8 features from a single article."""
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        text = f"{title} {content}"
        
        features = []
        
        # 1. Escalation keyword density (normalized)
        escalation_count = sum(1 for keyword in self.escalation_keywords if keyword in text)
        features.append(min(1.0, escalation_count / 5.0))  # Normalize to reasonable range
        
        # 2. Location relevance
        location_count = sum(1 for keyword in self.location_keywords if keyword in text)
        features.append(min(1.0, location_count / 3.0))
        
        # 3. Actor mentions
        actor_count = sum(1 for actor in self.actors if actor in text)
        features.append(min(1.0, actor_count / 3.0))
        
        # 4. Urgency indicators
        urgency_words = ['urgent', 'breaking', 'alert', 'immediate', 'emergency', 'critical']
        urgency_count = sum(1 for word in urgency_words if word in text)
        features.append(min(1.0, urgency_count / 3.0))
        
        # 5. Sentiment approximation (conflict vs cooperation)
        positive_words = ['peace', 'cooperation', 'dialogue', 'agreement', 'stability']
        negative_words = ['conflict', 'tension', 'crisis', 'threat', 'danger', 'risk']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count + negative_count > 0:
            sentiment_score = (negative_count - positive_count) / (positive_count + negative_count)
        else:
            sentiment_score = 0.0
        features.append(max(-1.0, min(1.0, sentiment_score)))
        
        # 6. Military action indicators
        military_actions = ['exercise', 'drill', 'maneuver', 'patrol', 'surveillance', 'reconnaissance']
        military_count = sum(1 for action in military_actions if action in text)
        features.append(min(1.0, military_count / 3.0))
        
        # 7. Temporal indicators (time sensitivity)
        temporal_words = ['today', 'now', 'current', 'ongoing', 'latest', 'recent']
        temporal_count = sum(1 for word in temporal_words if word in text)
        features.append(min(1.0, temporal_count / 3.0))
        
        # 8. Conflict density (conflicts per word)
        text_length = len(text.split())
        if text_length > 0:
            conflict_density = escalation_count / text_length * 100  # Conflicts per 100 words
        else:
            conflict_density = 0.0
        features.append(min(1.0, conflict_density))
        
        return features

class EnhancedEnsemblePredictor:
    """
    Enhanced ensemble predictor combining multiple model types for escalation prediction.
    """
    
    def __init__(self):
        """Initialize enhanced ensemble predictor."""
        self.models = {}
        self.osint_transformers = []
        self.feature_extractor = CompatibleFeatureExtractor()
        self.scaler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load all available models
        self._load_models()
        
        logger.info("EnhancedEnsemblePredictor initialized")
    
    def _load_models(self):
        """Load all available models."""
        self._load_xgboost_models()
        self._load_osint_transformers()
        self._load_gdelt_transformers()
        
        logger.info(f"✅ Loaded {len(self.models)} XGB models and {len(self.osint_transformers)} transformer models")
    
    def _load_xgboost_models(self):
        """Load XGBoost models."""
        model_dir = Path(__file__).parent
        
        xgb_models = [
            'xgb_goldstein_regressor.pkl',  # This one works with 8 features
            'xgb_goldstein_gpu_tuned.pkl',
            'xgb_working_fast.pkl'
        ]
        
        for model_name in xgb_models:
            model_path = model_dir / model_name
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    if hasattr(model, 'predict'):
                        self.models[model_name] = model
                        n_features = getattr(model, 'n_features_in_', 'unknown')
                        logger.info(f"✅ Loaded XGBoost model: {model_name} (expects {n_features} features)")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load XGBoost model {model_name}: {e}")
    
    def _load_osint_transformers(self):
        """Load OSINT transformer models."""
        if not HAS_TORCH:
            return
            
        osint_model_dir = Path("models/OSINT_Tension/trained_models")
        
        if osint_model_dir.exists():
            # Load the best performing models
            model_files = [
                "best_osint_student_model_20250621_131738.pt",
                "best_osint_student_model_20250621_131739.pt"
            ]
            
            for model_file in model_files:
                model_path = osint_model_dir / model_file
                if model_path.exists():
                    try:
                        predictor = OSINTTransformerPredictor(str(model_path), self.device)
                        if predictor.model is not None:
                            self.osint_transformers.append(predictor)
                    except Exception as e:
                        logger.warning(f"Failed to load OSINT transformer {model_file}: {e}")
    
    def _load_gdelt_transformers(self):
        """Load GDELT transformer models."""
        # TODO: Implement GDELT transformer loading
        pass
    
    def predict(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict escalation for articles using ensemble approach."""
        return self.predict_batch_escalation(articles)
    
    def predict_batch_escalation(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict escalation for a batch of articles."""
        if not articles:
            return []
        
        results = []
        
        try:
            # Extract features for different model types
            core_features = self.feature_extractor.extract_core_features(articles)
            extended_features = self.feature_extractor.extract_extended_features(articles)
            
            # Get predictions from each model type
            for i, article in enumerate(articles):
                result = self._predict_single_article(
                    article, 
                    core_features[i:i+1], 
                    extended_features[i:i+1]
                )
                results.append(result)
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Fallback predictions
            results = self._fallback_predictions(articles)
        
        return results
    
    def _predict_single_article(self, article: Dict[str, Any], core_features: np.ndarray, extended_features: np.ndarray) -> Dict[str, Any]:
        """Predict escalation for a single article using ensemble approach."""
        predictions = []
        confidences = []
        model_results = {}
        
        # 1. XGBoost predictions (using core 8 features)
        xgb_predictions = self._get_xgboost_predictions(core_features)
        if xgb_predictions:
            predictions.extend(xgb_predictions)
            confidences.extend([0.8] * len(xgb_predictions))
            model_results['xgboost'] = xgb_predictions
        
        # 2. OSINT transformer predictions
        osint_predictions = self._get_osint_transformer_predictions([article])
        if osint_predictions:
            predictions.extend(osint_predictions)
            confidences.extend([0.9] * len(osint_predictions))
            model_results['osint_transformers'] = osint_predictions
        
        # 3. Rule-based prediction as fallback
        rule_prediction = self._rule_based_predict(article, core_features)
        predictions.append(rule_prediction['escalation_score'])
        confidences.append(rule_prediction['confidence'])
        model_results['rule_based'] = rule_prediction['escalation_score']
        
        # Ensemble the predictions
        if predictions:
            # Weighted average based on confidence
            weighted_score = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
            avg_confidence = np.mean(confidences)
            prediction_method = "ensemble"
        else:
            weighted_score = 0.3  # Conservative default
            avg_confidence = 0.5
            prediction_method = "default"
        
        return {
            'article_id': article.get('id', 'unknown'),
            'escalation_score': float(np.clip(weighted_score, 0, 1)),
            'confidence': float(avg_confidence),
            'prediction_method': prediction_method,
            'model_predictions': model_results,
            'risk_level': self._categorize_risk(weighted_score),
            'features_used': {
                'core_features': core_features.tolist()[0],
                'num_extended_features': extended_features.shape[1]
            }
        }
    
    def _get_xgboost_predictions(self, features: np.ndarray) -> List[float]:
        """Get predictions from XGBoost models."""
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                prediction = model.predict(features)
                escalation_score = float(prediction[0]) if hasattr(prediction, '__getitem__') else float(prediction)
                
                # Normalize to 0-1 range
                escalation_score = max(0, min(1, escalation_score))
                predictions.append(escalation_score)
                
            except Exception as e:
                logger.warning(f"XGBoost model {model_name} prediction failed: {e}")
                continue
        
        return predictions
    
    def _get_osint_transformer_predictions(self, articles: List[Dict[str, Any]]) -> List[float]:
        """Get predictions from OSINT transformer models."""
        predictions = []
        
        for predictor in self.osint_transformers:
            try:
                preds = predictor.predict(articles)
                if preds:
                    predictions.extend(preds)
            except Exception as e:
                logger.warning(f"OSINT transformer prediction failed: {e}")
                continue
        
        return predictions
    
    def _rule_based_predict(self, article: Dict[str, Any], features: np.ndarray) -> Dict[str, Any]:
        """Rule-based escalation prediction."""
        # Extract key indicators from features
        escalation_keywords = features[0][0] if len(features[0]) > 0 else 0
        location_relevance = features[0][1] if len(features[0]) > 1 else 0
        sentiment_score = features[0][4] if len(features[0]) > 4 else 0
        military_indicators = features[0][5] if len(features[0]) > 5 else 0
        
        # Calculate rule-based score
        base_score = 0.2  # Conservative baseline
        
        # High keyword density increases score
        if escalation_keywords > 0.3:
            base_score += 0.3
        elif escalation_keywords > 0.1:
            base_score += 0.2
        
        # Location relevance increases score
        if location_relevance > 0.2:
            base_score += 0.2
        
        # Negative sentiment increases score
        if sentiment_score > 0.2:
            base_score += 0.2
        
        # Military indicators increase score
        if military_indicators > 0.2:
            base_score += 0.3
        
        escalation_score = min(1.0, base_score)
        confidence = 0.7 if escalation_score > 0.4 else 0.6
        
        return {
            'escalation_score': escalation_score,
            'confidence': confidence,
            'method': 'rule_based'
        }
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level based on escalation score."""
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _fallback_predictions(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback predictions when all models fail."""
        results = []
        for article in articles:
            results.append({
                'article_id': article.get('id', 'unknown'),
                'escalation_score': 0.3,  # Conservative default
                'confidence': 0.5,
                'prediction_method': 'fallback',
                'risk_level': 'LOW',
                'model_predictions': {},
                'features_used': {}
            })
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models and capabilities."""
        try:
            # Add time series forecasting availability
            try:
                from analytics.time_series.forecasting_engine import ForecastingEngine
                time_series_available = True
            except ImportError:
                time_series_available = False
            
            model_info = {
                'num_models': len(self.models) + len(self.osint_transformers),
                'xgb_models': len(self.models),
                'transformer_models': len(self.osint_transformers),
                'model_names': list(self.models.keys()) + [f"transformer_{i}" for i in range(len(self.osint_transformers))],
                'is_fitted': len(self.models) > 0 or len(self.osint_transformers) > 0,
                'has_scaler': self.scaler is not None,
                'feature_extractor_ready': hasattr(self, 'feature_extractor'),
                'time_series_available': time_series_available,
                'capabilities': [
                    'escalation_prediction',
                    'confidence_estimation', 
                    'ensemble_voting',
                    'feature_extraction'
                ]
            }
            
            if time_series_available:
                model_info['capabilities'].append('time_series_forecasting')
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def predict_time_series_trends(self, articles: List[Dict], forecast_days: int = 30) -> Dict[str, Any]:
        """
        Predict escalation trends using time series analysis.
        
        Args:
            articles: List of article dictionaries
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with time series predictions and trends
        """
        try:
            # Check if time series module is available
            try:
                from analytics.time_series.forecasting_engine import ForecastingEngine
            except ImportError:
                logger.warning("Time series module not available")
                return {'error': 'Time series forecasting not available'}
            
            logger.info(f"Generating time series forecast for {forecast_days} days...")
            
            # Get current escalation predictions
            current_predictions = self.predict_batch_escalation(articles)
            if not current_predictions:
                logger.error("Failed to get current predictions")
                return {'error': 'Failed to get current predictions'}
            
            escalation_scores = [p['escalation_score'] for p in current_predictions]
            
            # Initialize forecasting engine
            engine = ForecastingEngine()
            
            # Run time series analysis
            ts_results = engine.run_complete_analysis(articles, escalation_scores, forecast_days)
            
            if not ts_results:
                logger.error("Time series analysis failed")
                return {'error': 'Time series analysis failed'}
            
            # Extract key insights
            trends_summary = {
                'current_escalation': {
                    'mean': float(np.mean(escalation_scores)),
                    'max': float(np.max(escalation_scores)),
                    'min': float(np.min(escalation_scores)),
                    'trend': 'increasing' if escalation_scores[-1] > escalation_scores[0] else 'decreasing'
                },
                'forecast_horizon': forecast_days,
                'models_used': ts_results.get('models_trained', []),
                'forecast_performance': ts_results.get('performance_metrics', {}),
                'data_quality': {
                    'train_samples': ts_results.get('data_summary', {}).get('train_samples', 0),
                    'test_samples': ts_results.get('data_summary', {}).get('test_samples', 0),
                    'date_range': ts_results.get('data_summary', {}).get('date_range', {})
                },
                'ensemble_available': ts_results.get('ensemble_available', False)
            }
            
            # Add forecast trend analysis
            if engine.forecasts and 'ensemble' in engine.forecasts:
                ensemble_forecast = engine.forecasts['ensemble']['forecast']
                
                # Analyze forecast trends
                forecast_trend = 'stable'
                if len(ensemble_forecast) > 1:
                    if ensemble_forecast[-1] > ensemble_forecast[0] * 1.1:
                        forecast_trend = 'increasing'
                    elif ensemble_forecast[-1] < ensemble_forecast[0] * 0.9:
                        forecast_trend = 'decreasing'
                
                trends_summary['forecast_analysis'] = {
                    'trend_direction': forecast_trend,
                    'forecast_mean': float(np.mean(ensemble_forecast)),
                    'forecast_max': float(np.max(ensemble_forecast)),
                    'forecast_min': float(np.min(ensemble_forecast)),
                    'volatility': float(np.std(ensemble_forecast))
                }
            
            logger.info("Time series trend analysis completed successfully")
            return trends_summary
            
        except Exception as e:
            logger.error(f"Failed to predict time series trends: {e}")
            return {'error': str(e)}

def predict_batch_escalation(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Main function for batch escalation prediction.
    """
    predictor = EnhancedEnsemblePredictor()
    return predictor.predict_batch_escalation(articles)

def test_predictor():
    """Test the enhanced ensemble predictor."""
    # Test data
    test_articles = [
        {
            'id': 'test_1',
            'title': 'Military tensions escalate in Taiwan Strait',
            'content': 'Chinese military exercises near Taiwan have increased tensions in the region...'
        },
        {
            'id': 'test_2', 
            'title': 'Peaceful diplomatic meeting scheduled',
            'content': 'Representatives from both sides will meet to discuss cooperation and stability...'
        }
    ]
    
    predictor = EnhancedEnsemblePredictor()
    results = predictor.predict_batch_escalation(test_articles)
    
    print("🧪 Enhanced Ensemble Predictor Test Results:")
    print(f"📊 Model Info: {predictor.get_model_info()}")
    
    for result in results:
        print(f"\n📄 Article: {result['article_id']}")
        print(f"⚡ Escalation Score: {result['escalation_score']:.3f}")
        print(f"🎯 Confidence: {result['confidence']:.3f}")
        print(f"🔍 Method: {result['prediction_method']}")
        print(f"⚠️  Risk Level: {result['risk_level']}")

if __name__ == "__main__":
    test_predictor() 