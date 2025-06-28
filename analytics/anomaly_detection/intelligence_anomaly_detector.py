"""
Anomaly Detection System for Intelligence Analysis.
Detects unusual patterns, escalation spikes, and temporal anomalies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

@dataclass
class AnomalyAlert:
    alert_id: str
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    affected_articles: List[str]
    anomaly_score: float
    confidence: float
    timestamp: str
    metadata: Dict[str, Any]

class IntelligenceAnomalyDetector:
    """Advanced anomaly detection for intelligence analysis."""
    
    def __init__(self, model_path: str = "models/anomaly_models/"):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        
        # Initialize models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
        self.scaler = StandardScaler()
        self.baseline_metrics = {}
        self.trained = False
        
        # Thresholds
        self.thresholds = {
            'escalation_spike': 0.7,
            'volume_spike': 2.0,
            'source_anomaly': 0.8,
            'temporal_anomaly': 0.75
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models."""
        try:
            import os
            if os.path.exists(f"{self.model_path}/isolation_forest.pkl"):
                self.isolation_forest = joblib.load(f"{self.model_path}/isolation_forest.pkl")
                self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")
                self.baseline_metrics = joblib.load(f"{self.model_path}/baseline_metrics.pkl")
                self.trained = True
                self.logger.info("Loaded pre-trained anomaly models")
        except Exception as e:
            self.logger.info(f"No pre-trained models found: {e}")
    
    def _save_models(self):
        """Save trained models."""
        try:
            import os
            os.makedirs(self.model_path, exist_ok=True)
            joblib.dump(self.isolation_forest, f"{self.model_path}/isolation_forest.pkl")
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
            joblib.dump(self.baseline_metrics, f"{self.model_path}/baseline_metrics.pkl")
            self.logger.info("Saved anomaly models")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def extract_features(self, articles: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for anomaly detection."""
        features = []
        
        for article in articles:
            article_features = [
                article.get('escalation_score', 0.0),
                article.get('goldstein_scale', 0.0),
                article.get('avg_tone', 0.0),
                article.get('num_mentions', 1),
                len(article.get('content', '')),
                self._extract_hour_of_day(article.get('date_published', '')),
                self._calculate_source_reputation(article.get('source_name', '')),
                abs(article.get('sentiment_score', 0.0)),
                len(article.get('locations', [])),
                len(article.get('actors', []))
            ]
            features.append(article_features)
        
        return np.array(features)
    
    def _extract_hour_of_day(self, date_str: str) -> float:
        """Extract normalized hour of day."""
        try:
            if date_str:
                dt = pd.to_datetime(date_str)
                return dt.hour / 24.0
        except:
            pass
        return 0.5
    
    def _calculate_source_reputation(self, source_name: str) -> float:
        """Calculate source reputation score."""
        trusted_sources = ['reuters', 'ap', 'bbc', 'cnn', 'xinhua', 'ministry', 'government']
        source_lower = source_name.lower()
        for trusted in trusted_sources:
            if trusted in source_lower:
                return 0.9
        return 0.5
    
    def train_baseline(self, historical_articles: List[Dict[str, Any]]):
        """Train baseline models on historical data."""
        if not historical_articles:
            self.logger.warning("No historical data for training")
            return
        
        self.logger.info(f"Training baseline on {len(historical_articles)} articles")
        
        features = self.extract_features(historical_articles)
        if len(features) == 0:
            return
        
        features_scaled = self.scaler.fit_transform(features)
        self.isolation_forest.fit(features_scaled)
        
        # Calculate baseline metrics
        escalation_scores = [a.get('escalation_score', 0.0) for a in historical_articles]
        volumes = [a.get('num_mentions', 1) for a in historical_articles]
        
        self.baseline_metrics = {
            'mean_escalation': np.mean(escalation_scores),
            'std_escalation': np.std(escalation_scores),
            'mean_volume': np.mean(volumes),
            'training_size': len(historical_articles),
            'training_date': datetime.now().isoformat()
        }
        
        self.trained = True
        self._save_models()
        self.logger.info("Baseline training completed")
    
    def detect_anomalies(self, current_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalies in current articles."""
        start_time = datetime.now()
        alerts = []
        
        if not current_articles:
            return {'alerts': [], 'summary': {}, 'processing_time': 0.0}
        
        # Detect different types of anomalies
        alerts.extend(self._detect_escalation_spikes(current_articles))
        alerts.extend(self._detect_volume_anomalies(current_articles))
        alerts.extend(self._detect_source_anomalies(current_articles))
        alerts.extend(self._detect_temporal_anomalies(current_articles))
        
        if self.trained:
            alerts.extend(self._detect_ml_anomalies(current_articles))
        
        # Generate summary
        summary = {
            'total_alerts': len(alerts),
            'articles_analyzed': len(current_articles),
            'highest_severity': max([a.severity for a in alerts], default='NONE'),
            'max_anomaly_score': max([a.anomaly_score for a in alerts], default=0.0),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'alerts': [alert.__dict__ for alert in alerts],
            'summary': summary,
            'baseline_metrics': self.baseline_metrics,
            'processing_time': processing_time
        }
    
    def _detect_escalation_spikes(self, articles: List[Dict[str, Any]]) -> List[AnomalyAlert]:
        """Detect escalation spikes."""
        alerts = []
        high_escalation = [a for a in articles if a.get('escalation_score', 0.0) > self.thresholds['escalation_spike']]
        
        if high_escalation:
            max_escalation = max(a.get('escalation_score', 0.0) for a in high_escalation)
            severity = 'CRITICAL' if max_escalation > 0.9 else 'HIGH' if max_escalation > 0.7 else 'MEDIUM'
            
            alerts.append(AnomalyAlert(
                alert_id=f"escalation_spike_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type="ESCALATION_SPIKE",
                severity=severity,
                description=f"Detected {len(high_escalation)} high escalation articles (max: {max_escalation:.3f})",
                affected_articles=[a.get('id', str(i)) for i, a in enumerate(high_escalation)],
                anomaly_score=max_escalation,
                confidence=0.9,
                timestamp=datetime.now().isoformat(),
                metadata={'max_escalation': max_escalation, 'affected_count': len(high_escalation)}
            ))
        
        return alerts
    
    def _detect_volume_anomalies(self, articles: List[Dict[str, Any]]) -> List[AnomalyAlert]:
        """Detect volume anomalies."""
        alerts = []
        
        if not self.baseline_metrics:
            return alerts
        
        current_volume = len(articles)
        expected_volume = self.baseline_metrics.get('mean_volume', current_volume)
        
        if current_volume > expected_volume * self.thresholds['volume_spike']:
            spike_ratio = current_volume / expected_volume
            severity = 'CRITICAL' if spike_ratio > 5 else 'HIGH' if spike_ratio > 3 else 'MEDIUM'
            
            alerts.append(AnomalyAlert(
                alert_id=f"volume_spike_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type="VOLUME_SPIKE",
                severity=severity,
                description=f"Volume spike: {current_volume} articles (expected: {expected_volume:.1f})",
                affected_articles=[a.get('id', str(i)) for i, a in enumerate(articles)],
                anomaly_score=spike_ratio,
                confidence=0.85,
                timestamp=datetime.now().isoformat(),
                metadata={'current_volume': current_volume, 'expected_volume': expected_volume}
            ))
        
        return alerts
    
    def _detect_source_anomalies(self, articles: List[Dict[str, Any]]) -> List[AnomalyAlert]:
        """Detect source anomalies."""
        alerts = []
        source_counts = {}
        
        for article in articles:
            source = article.get('source_name', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        total_articles = len(articles)
        for source, count in source_counts.items():
            ratio = count / total_articles
            
            if ratio > 0.6 and total_articles > 5:
                severity = 'HIGH' if ratio > 0.8 else 'MEDIUM'
                
                alerts.append(AnomalyAlert(
                    alert_id=f"source_dominance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type="SOURCE_DOMINANCE",
                    severity=severity,
                    description=f"Source dominance: {source} ({ratio:.1%} of articles)",
                    affected_articles=[a.get('id', str(i)) for i, a in enumerate(articles) if a.get('source_name') == source],
                    anomaly_score=ratio,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat(),
                    metadata={'dominant_source': source, 'dominance_ratio': ratio}
                ))
        
        return alerts
    
    def _detect_temporal_anomalies(self, articles: List[Dict[str, Any]]) -> List[AnomalyAlert]:
        """Detect temporal anomalies."""
        alerts = []
        hour_counts = {}
        
        for article in articles:
            try:
                date_str = article.get('date_published', '')
                if date_str:
                    dt = pd.to_datetime(date_str)
                    hour = dt.hour
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
            except:
                continue
        
        if not hour_counts:
            return alerts
        
        total_articles = sum(hour_counts.values())
        for hour, count in hour_counts.items():
            ratio = count / total_articles
            
            if ratio > 0.5 and total_articles > 10:
                severity = 'HIGH' if ratio > 0.8 else 'MEDIUM'
                
                alerts.append(AnomalyAlert(
                    alert_id=f"temporal_concentration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type="TEMPORAL_CONCENTRATION",
                    severity=severity,
                    description=f"Temporal concentration: {count} articles at hour {hour} ({ratio:.1%})",
                    affected_articles=[a.get('id', str(i)) for i, a in enumerate(articles)],
                    anomaly_score=ratio,
                    confidence=0.75,
                    timestamp=datetime.now().isoformat(),
                    metadata={'concentrated_hour': hour, 'concentration_ratio': ratio}
                ))
        
        return alerts
    
    def _detect_ml_anomalies(self, articles: List[Dict[str, Any]]) -> List[AnomalyAlert]:
        """Detect ML-based anomalies."""
        alerts = []
        
        try:
            features = self.extract_features(articles)
            if len(features) == 0:
                return alerts
            
            features_scaled = self.scaler.transform(features)
            anomaly_scores = self.isolation_forest.decision_function(features_scaled)
            anomaly_predictions = self.isolation_forest.predict(features_scaled)
            
            outlier_indices = np.where(anomaly_predictions == -1)[0]
            
            for idx in outlier_indices:
                article = articles[idx]
                score = abs(anomaly_scores[idx])
                severity = 'HIGH' if score > 0.5 else 'MEDIUM'
                
                alerts.append(AnomalyAlert(
                    alert_id=f"ml_anomaly_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type="ML_ANOMALY",
                    severity=severity,
                    description=f"ML detected anomaly: {article.get('title', 'Unknown')[:100]}",
                    affected_articles=[article.get('id', str(idx))],
                    anomaly_score=score,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat(),
                    metadata={'ml_score': anomaly_scores[idx]}
                ))
                
        except Exception as e:
            self.logger.error(f"Error in ML anomaly detection: {e}")
        
        return alerts


def main():
    """Test the anomaly detection system."""
    logging.basicConfig(level=logging.INFO)
    
    detector = IntelligenceAnomalyDetector()
    
    # Sample test data
    sample_articles = [
        {
            'id': '1',
            'title': 'Military Exercise in Taiwan Strait',
            'escalation_score': 0.85,
            'goldstein_scale': -8.0,
            'avg_tone': -5.2,
            'num_mentions': 15,
            'source_name': 'Reuters',
            'date_published': '2024-01-15T14:30:00Z'
        },
        {
            'id': '2',
            'title': 'Diplomatic Meeting',
            'escalation_score': 0.2,
            'goldstein_scale': 2.0,
            'avg_tone': 1.5,
            'num_mentions': 3,
            'source_name': 'AP News',
            'date_published': '2024-01-15T09:15:00Z'
        }
    ]
    
    analysis = detector.detect_anomalies(sample_articles)
    
    print("Anomaly Detection Results:")
    print(f"Total Alerts: {len(analysis['alerts'])}")
    for alert in analysis['alerts']:
        print(f"- {alert['alert_type']}: {alert['severity']} (Score: {alert['anomaly_score']:.3f})")
        print(f"  {alert['description']}")
    
    print(f"\nSummary: {analysis['summary']}")


if __name__ == "__main__":
    main() 