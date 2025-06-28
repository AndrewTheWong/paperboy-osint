import joblib
import numpy as np
import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import json

logger = logging.getLogger(__name__)

class EnhancedEscalationPredictor:
    """Enhanced escalation predictor using latest OSINT models with increased sensitivity"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.text_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available models with fallbacks"""
        logger.info("🚀 Loading enhanced escalation prediction models...")
        
        # 1. Load text encoder
        try:
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded SentenceTransformer encoder")
        except Exception as e:
            logger.error(f"Failed to load text encoder: {e}")
        
        # 2. Load latest OSINT student model (most recent)
        self._load_osint_student_model()
        
        # 3. Load XGBoost models with enhanced features
        self._load_xgboost_models()
        
        # 4. Load transformer models
        self._load_transformer_models()
        
        logger.info(f"✅ Loaded {len(self.models)} prediction models")
    
    def _load_osint_student_model(self):
        """Load the latest OSINT student model"""
        osint_model_paths = [
            "models/OSINT_Tension/trained_models/best_osint_student_model_20250621_131743.pt",
            "models/OSINT_Tension/trained_models/best_osint_student_model_20250621_131742.pt",
            "models/OSINT_Tension/trained_models/best_osint_student_model_20250621_131741.pt"
        ]
        
        for model_path in osint_model_paths:
            if os.path.exists(model_path):
                try:
                    # Load the model checkpoint
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Create model architecture (assuming it's in the checkpoint)
                    model = self._create_osint_student_architecture()
                    if model:
                        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                        model.to(self.device)
                        model.eval()
                        
                        self.models['osint_student'] = model
                        logger.info(f"✅ Loaded OSINT student model: {os.path.basename(model_path)}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load OSINT model {model_path}: {e}")
                    continue
    
    def _create_osint_student_architecture(self):
        """Create OSINT student model architecture"""
        try:
            # Simple transformer-based model for OSINT articles
            class OSINTStudentModel(nn.Module):
                def __init__(self, input_dim=384, hidden_dim=128):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim // 2, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    return self.encoder(x)
            
            return OSINTStudentModel()
            
        except Exception as e:
            logger.warning(f"Failed to create OSINT model architecture: {e}")
            return None
    
    def _load_xgboost_models(self):
        """Load XGBoost models with enhanced sensitivity"""
        xgb_model_paths = [
            "models/enhanced_xgb_goldstein_v2.pkl",
            "models/xgb_goldstein_gpu_tuned.pkl",
            "models/xgb_goldstein_regressor.pkl"
        ]
        
        for model_path in xgb_model_paths:
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    model_name = os.path.basename(model_path).replace('.pkl', '')
                    self.models[f'xgb_{model_name}'] = model
                    logger.info(f"✅ Loaded XGBoost model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load XGBoost model {model_path}: {e}")
                    continue
    
    def _load_transformer_models(self):
        """Load transformer models"""
        transformer_paths = [
            "models/transformer/best_transformer_full.pt",
            "models/transformer/best_transformer.pt"
        ]
        
        for model_path in transformer_paths:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    # Would need proper architecture - for now skip
                    logger.info(f"Found transformer model: {os.path.basename(model_path)}")
                    break
                except Exception as e:
                    continue
    
    def predict_escalation_enhanced(self, text: str, title: str = "") -> Dict[str, float]:
        """Enhanced escalation prediction with multiple models and increased sensitivity"""
        
        if not text and not title:
            return {'escalation_score': 0.0, 'confidence': 0.0}
        
        combined_text = f"{title} {text}".strip()
        predictions = {}
        
        try:
            # 1. Get text embedding
            if self.text_encoder:
                embedding = self.text_encoder.encode([combined_text])[0]
            else:
                embedding = np.random.randn(384)  # Fallback
            
            # 2. OSINT Student Model Prediction (most important)
            if 'osint_student' in self.models:
                try:
                    with torch.no_grad():
                        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
                        osint_score = self.models['osint_student'](embedding_tensor).cpu().item()
                        predictions['osint_student'] = osint_score
                except Exception as e:
                    logger.warning(f"OSINT student prediction failed: {e}")
            
            # 3. Enhanced text-based scoring (more sensitive)
            text_score = self._enhanced_text_scoring(combined_text)
            predictions['enhanced_text'] = text_score
            
            # 4. XGBoost prediction (if available)
            xgb_score = self._predict_xgboost_enhanced(combined_text)
            if xgb_score > 0:
                predictions['xgboost'] = xgb_score
            
            # 5. Keyword-based escalation boost
            keyword_boost = self._calculate_keyword_escalation_boost(combined_text)
            predictions['keyword_boost'] = keyword_boost
            
            # 6. Ensemble prediction with enhanced sensitivity
            final_score = self._ensemble_predictions_enhanced(predictions)
            
            # 7. Calculate confidence
            confidence = self._calculate_prediction_confidence(predictions)
            
            return {
                'escalation_score': final_score,
                'confidence': confidence,
                'model_predictions': predictions,
                'models_used': list(predictions.keys())
            }
            
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {e}")
            return {'escalation_score': 0.0, 'confidence': 0.0}
    
    def _enhanced_text_scoring(self, text: str) -> float:
        """Enhanced text scoring with increased sensitivity to escalation indicators"""
        
        # High-impact escalation keywords (weighted more heavily)
        high_impact_keywords = {
            'military': 0.15, 'war': 0.2, 'conflict': 0.15, 'attack': 0.18, 'missile': 0.2,
            'nuclear': 0.25, 'weapons': 0.15, 'bombing': 0.2, 'invasion': 0.25, 'strike': 0.18,
            'escalation': 0.2, 'crisis': 0.12, 'threat': 0.15, 'hostile': 0.15, 'aggression': 0.15,
            'violence': 0.15, 'combat': 0.15, 'forces': 0.1, 'deployment': 0.12, 'mobilization': 0.18
        }
        
        # Medium-impact keywords
        medium_impact_keywords = {
            'tension': 0.08, 'dispute': 0.08, 'sanctions': 0.1, 'embargo': 0.12, 'blockade': 0.15,
            'confrontation': 0.12, 'standoff': 0.1, 'warning': 0.08, 'ultimatum': 0.15, 'retaliation': 0.15,
            'provocation': 0.1, 'incursion': 0.12, 'breach': 0.1, 'violation': 0.1, 'exercise': 0.08
        }
        
        # Geographic/political amplifiers
        geopolitical_keywords = {
            'iran': 0.05, 'israel': 0.05, 'taiwan': 0.05, 'china': 0.05, 'russia': 0.05, 'ukraine': 0.05,
            'north korea': 0.08, 'south korea': 0.03, 'syria': 0.05, 'lebanon': 0.05, 'gaza': 0.08,
            'middle east': 0.05, 'strait': 0.05, 'border': 0.03, 'territory': 0.03
        }
        
        text_lower = text.lower()
        score = 0.0
        
        # Score high-impact keywords
        for keyword, weight in high_impact_keywords.items():
            if keyword in text_lower:
                score += weight
        
        # Score medium-impact keywords
        for keyword, weight in medium_impact_keywords.items():
            if keyword in text_lower:
                score += weight
        
        # Score geopolitical keywords
        for keyword, weight in geopolitical_keywords.items():
            if keyword in text_lower:
                score += weight
        
        # Boost for keyword combinations
        if any(k in text_lower for k in ['military', 'forces', 'army']) and any(k in text_lower for k in ['exercise', 'drill', 'maneuver']):
            score += 0.1  # Military exercises
        
        if any(k in text_lower for k in ['nuclear', 'missile']) and any(k in text_lower for k in ['test', 'launch', 'program']):
            score += 0.15  # Nuclear/missile programs
        
        if any(k in text_lower for k in ['cyber', 'hack']) and any(k in text_lower for k in ['attack', 'breach', 'warfare']):
            score += 0.12  # Cyber warfare
        
        # Cap the score but make it more sensitive
        return min(score, 1.0)
    
    def _calculate_keyword_escalation_boost(self, text: str) -> float:
        """Calculate additional boost for specific escalation patterns"""
        text_lower = text.lower()
        boost = 0.0
        
        # Emergency/urgent language
        urgent_patterns = ['breaking', 'urgent', 'emergency', 'immediate', 'alert']
        if any(pattern in text_lower for pattern in urgent_patterns):
            boost += 0.1
        
        # Escalatory verbs
        escalatory_verbs = ['threatens', 'warns', 'prepares', 'mobilizes', 'deploys', 'launches', 'strikes']
        verb_count = sum(1 for verb in escalatory_verbs if verb in text_lower)
        boost += min(verb_count * 0.05, 0.2)
        
        # Numbers and scale indicators
        scale_indicators = ['hundreds', 'thousands', 'massive', 'large-scale', 'major']
        if any(indicator in text_lower for indicator in scale_indicators):
            boost += 0.08
        
        return boost
    
    def _predict_xgboost_enhanced(self, text: str) -> float:
        """Enhanced XGBoost prediction with better feature engineering"""
        
        # Find XGBoost model
        xgb_model = None
        for model_name, model in self.models.items():
            if 'xgb' in model_name:
                xgb_model = model
                break
        
        if not xgb_model:
            return 0.0
        
        try:
            # Create enhanced features
            features = self._create_enhanced_features(text)
            
            # Try prediction with different feature counts
            for n_features in [8, 58, len(features)]:
                try:
                    if n_features <= len(features):
                        feature_input = features[:n_features].reshape(1, -1)
                    else:
                        # Pad with zeros
                        padded_features = np.zeros(n_features)
                        padded_features[:len(features)] = features
                        feature_input = padded_features.reshape(1, -1)
                    
                    # Get prediction
                    if hasattr(xgb_model, 'predict_proba'):
                        prob = xgb_model.predict_proba(feature_input)[0][1]
                    else:
                        # Regression model - convert to probability
                        pred = xgb_model.predict(feature_input)[0]
                        prob = 1 / (1 + np.exp(-pred))  # Sigmoid
                    
                    return float(prob)
                    
                except Exception:
                    continue
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"XGBoost enhanced prediction failed: {e}")
            return 0.0
    
    def _create_enhanced_features(self, text: str) -> np.ndarray:
        """Create enhanced features for XGBoost model"""
        
        # Enhanced tag vocabulary
        tag_keywords = {
            "military_movement": ["military", "army", "troop", "force", "drill", "exercise", "deployment"],
            "conflict": ["conflict", "fight", "battle", "war", "clash", "combat", "violence", "attack"],
            "cyberattack": ["cyber", "hack", "breach", "malware", "ransomware", "digital"],
            "protest": ["protest", "demonstration", "riot", "unrest", "march", "rally"],
            "diplomatic_meeting": ["diplomatic", "meeting", "summit", "talks", "negotiation"],
            "nuclear": ["nuclear", "atomic", "warhead", "missile", "icbm", "uranium"],
            "ceasefire": ["ceasefire", "truce", "peace", "armistice", "cease-fire"]
        }
        
        text_lower = text.lower()
        features = []
        
        # Binary features for each tag
        for tag, keywords in tag_keywords.items():
            has_tag = int(any(keyword in text_lower for keyword in keywords))
            features.append(has_tag)
        
        # Additional enhanced features
        features.extend([
            len(text) / 1000.0,  # Text length (normalized)
            text_lower.count('!') / 10.0,  # Exclamation marks
            len([w for w in text_lower.split() if w.isupper()]) / len(text_lower.split()),  # Caps ratio
            0.5  # Default confidence
        ])
        
        # Pad to standard length if needed
        while len(features) < 58:
            features.append(0.0)
        
        return np.array(features[:58])  # Limit to max expected features
    
    def _ensemble_predictions_enhanced(self, predictions: Dict[str, float]) -> float:
        """Enhanced ensemble with increased sensitivity and proper weighting"""
        
        if not predictions:
            return 0.0
        
        # Enhanced weights (favor OSINT student model and boost high scores)
        weights = {
            'osint_student': 0.4,      # Highest weight - latest trained model
            'enhanced_text': 0.25,     # High weight - sensitive text analysis
            'xgboost': 0.2,           # Medium weight - traditional ML
            'keyword_boost': 0.15      # Bonus for escalation indicators
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, score in predictions.items():
            weight = weights.get(model_name, 0.1)  # Default weight for unknown models
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return np.mean(list(predictions.values()))
        
        base_score = weighted_sum / total_weight
        
        # Apply sensitivity boost for high-confidence predictions
        max_individual_score = max(predictions.values())
        if max_individual_score > 0.7:
            # Boost high-confidence predictions
            boost_factor = 1.2
            base_score = min(base_score * boost_factor, 1.0)
        
        # Apply minimum escalation threshold for detected keywords
        if predictions.get('enhanced_text', 0) > 0.3 or predictions.get('keyword_boost', 0) > 0.1:
            base_score = max(base_score, 0.2)  # Minimum escalation score
        
        return float(base_score)
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, float]) -> float:
        """Calculate confidence based on model agreement and score consistency"""
        
        if len(predictions) < 2:
            return 0.5
        
        scores = list(predictions.values())
        
        # Higher agreement = higher confidence
        std_dev = np.std(scores)
        mean_score = np.mean(scores)
        
        # Confidence is inversely related to standard deviation
        confidence = 1.0 - min(std_dev / (mean_score + 0.1), 1.0)
        
        # Boost confidence for high scores with good agreement
        if mean_score > 0.6 and std_dev < 0.2:
            confidence = min(confidence * 1.2, 1.0)
        
        return float(max(0.1, min(confidence, 1.0)))

# Global enhanced predictor instance
enhanced_predictor = EnhancedEscalationPredictor()

# Legacy compatibility functions
def load_xgb_model():
    """Legacy function to load XGBoost model - returns the enhanced predictor's XGBoost model"""
    try:
        # Return the first available XGBoost model from enhanced predictor
        for model_name, model in enhanced_predictor.models.items():
            if 'xgb' in model_name:
                return model
        
        # If no XGBoost model available, try to load one directly
        import joblib
        model_paths = [
            "models/enhanced_xgb_goldstein_v2.pkl",
            "models/xgb_goldstein_gpu_tuned.pkl", 
            "models/xgb_goldstein_regressor.pkl"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                return joblib.load(path)
        
        logger.warning("No XGBoost model available")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
        return None

def predict_article_scores(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhanced prediction pipeline for articles with increased escalation sensitivity
    """
    logger.info(f"🎯 Enhanced escalation prediction for {len(articles)} articles...")
    
    scored_articles = []
    
    for article in articles:
        title = article.get("title", "")
        text = article.get("text", "") or article.get("content", "") or article.get("summary", "")
        
        # Get enhanced prediction
        prediction_result = enhanced_predictor.predict_escalation_enhanced(text, title)
        
        # Add enhanced scores to article
        article_with_scores = article.copy()
        article_with_scores.update({
            'escalation_score': prediction_result['escalation_score'],
            'confidence_score': prediction_result['confidence'],
            'model_predictions': prediction_result.get('model_predictions', {}),
            'models_used': prediction_result.get('models_used', [])
        })
        
        scored_articles.append(article_with_scores)
    
    logger.info(f"✅ Enhanced scoring complete - Average score: {np.mean([a['escalation_score'] for a in scored_articles]):.3f}")
    return scored_articles

# Additional compatibility functions
def predict_with_xgboost(tags: list, confidence: float) -> float:
    """Legacy XGBoost prediction function"""
    fake_text = " ".join(tags)  # Convert tags to text
    result = enhanced_predictor.predict_escalation_enhanced(fake_text)
    return result['escalation_score']

def predict_from_text_features(text: str) -> float:
    """Legacy text features function"""
    result = enhanced_predictor.predict_escalation_enhanced(text)
    return result['escalation_score'] 