"""
Inference module for escalation prediction and time series forecasting.
Provides ensemble prediction capabilities combining multiple models.
"""

from .ensemble_predictor import predict_batch_escalation, EnhancedEnsemblePredictor
# Backward compatibility alias
EnsembleEscalationPredictor = EnhancedEnsemblePredictor

__all__ = [
    'predict_batch_escalation',
    'EnsembleEscalationPredictor',
    'EnhancedEnsemblePredictor'
] 