"""
Enhanced Sentiment Analyzer for OSINT Articles
Analyzes sentiment toward Taiwan and escalation levels
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc
from textblob import TextBlob

logger = logging.getLogger(__name__)

@dataclass
class SentimentAnalysis:
    """Structured sentiment analysis result"""
    sentiment_toward_Taiwan: str  # hostile, neutral, supportive
    escalation_level: str  # low, medium, high
    intent_signal: str  # coercive, defensive, deterrent, symbolic
    strategic_posture_change: bool
    info_warfare_detected: bool
    confidence_score: float = 0.8
    model_used: str = "custom-sentiment-analyzer"

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analysis with Taiwan-specific focus"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the sentiment analyzer"""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model for sentiment analysis: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found, downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Taiwan-specific sentiment indicators
        self.taiwan_sentiment_indicators = self._load_taiwan_sentiment_indicators()
        
        # Escalation indicators
        self.escalation_indicators = self._load_escalation_indicators()
        
        # Intent signal indicators
        self.intent_indicators = self._load_intent_indicators()
        
        # Information warfare indicators
        self.info_warfare_indicators = self._load_info_warfare_indicators()
    
    def _load_taiwan_sentiment_indicators(self) -> Dict[str, List[str]]:
        """Load Taiwan-specific sentiment indicators"""
        return {
            'hostile': [
                'threaten', 'warn', 'intimidate', 'coerce', 'pressure',
                'invade', 'attack', 'strike', 'target', 'punish',
                'sanction', 'blockade', 'isolate', 'undermine', 'destabilize',
                'separatist', 'independence', 'reunification', 'unification',
                'one china', 'sovereignty', 'territorial integrity',
                'military option', 'force', 'aggression', 'hostility'
            ],
            'neutral': [
                'observe', 'monitor', 'watch', 'assess', 'evaluate',
                'maintain', 'preserve', 'status quo', 'stability',
                'dialogue', 'communication', 'exchange', 'cooperation',
                'peaceful', 'diplomatic', 'negotiation', 'consultation'
            ],
            'supportive': [
                'support', 'back', 'aid', 'assist', 'help',
                'defend', 'protect', 'guarantee', 'commitment', 'alliance',
                'partnership', 'friendship', 'cooperation', 'solidarity',
                'democracy', 'freedom', 'human rights', 'self-determination'
            ]
        }
    
    def _load_escalation_indicators(self) -> Dict[str, List[str]]:
        """Load escalation level indicators"""
        return {
            'low': [
                'routine', 'normal', 'regular', 'standard', 'usual',
                'peaceful', 'diplomatic', 'dialogue', 'communication',
                'cooperation', 'exchange', 'visit', 'meeting'
            ],
            'medium': [
                'concerning', 'worrisome', 'troubling', 'significant',
                'notable', 'important', 'serious', 'escalating',
                'increasing', 'growing', 'rising', 'mounting'
            ],
            'high': [
                'critical', 'urgent', 'alarming', 'dangerous',
                'threatening', 'hostile', 'aggressive', 'provocative',
                'escalation', 'crisis', 'conflict', 'war',
                'military action', 'force', 'attack', 'strike'
            ]
        }
    
    def _load_intent_indicators(self) -> Dict[str, List[str]]:
        """Load intent signal indicators"""
        return {
            'coercive': [
                'threaten', 'intimidate', 'coerce', 'pressure', 'force',
                'compel', 'demand', 'ultimatum', 'warning', 'punishment',
                'sanction', 'blockade', 'isolation', 'retaliation'
            ],
            'defensive': [
                'defend', 'protect', 'guard', 'secure', 'safeguard',
                'prevent', 'deter', 'counter', 'respond', 'react',
                'self-defense', 'security', 'safety', 'protection'
            ],
            'deterrent': [
                'deter', 'prevent', 'discourage', 'dissuade', 'warn',
                'caution', 'admonish', 'signal', 'demonstrate', 'show',
                'capability', 'readiness', 'preparedness', 'strength'
            ],
            'symbolic': [
                'symbolic', 'gesture', 'signal', 'message', 'statement',
                'demonstration', 'show', 'display', 'ceremony', 'ritual',
                'commemoration', 'anniversary', 'celebration', 'mark'
            ]
        }
    
    def _load_info_warfare_indicators(self) -> List[str]:
        """Load information warfare indicators"""
        return [
            'propaganda', 'disinformation', 'misinformation', 'fake news',
            'manipulation', 'influence', 'narrative', 'messaging',
            'psychological', 'information warfare', 'media campaign',
            'social media', 'online', 'digital', 'cyber',
            'troll', 'bot', 'coordinated', 'campaign'
        ]
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment with Taiwan-specific focus"""
        if not text:
            return self._get_default_sentiment()
        
        try:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Analyze Taiwan sentiment
            taiwan_sentiment = self._analyze_taiwan_sentiment(doc)
            
            # Analyze escalation level
            escalation_level = self._analyze_escalation_level(doc)
            
            # Analyze intent signal
            intent_signal = self._analyze_intent_signal(doc)
            
            # Check for strategic posture change
            strategic_change = self._detect_strategic_posture_change(doc)
            
            # Check for information warfare
            info_warfare = self._detect_info_warfare(doc)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(doc, taiwan_sentiment, escalation_level)
            
            result = SentimentAnalysis(
                sentiment_toward_Taiwan=taiwan_sentiment,
                escalation_level=escalation_level,
                intent_signal=intent_signal,
                strategic_posture_change=strategic_change,
                info_warfare_detected=info_warfare,
                confidence_score=confidence
            )
            
            logger.info(f"Sentiment analysis completed: {taiwan_sentiment}, {escalation_level}, {intent_signal}")
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            result = self._get_default_sentiment()
        
        return result
    
    def _analyze_taiwan_sentiment(self, doc: Doc) -> str:
        """Analyze sentiment specifically toward Taiwan"""
        text_lower = doc.text.lower()
        
        # Count indicators for each sentiment
        sentiment_scores = {}
        
        for sentiment, indicators in self.taiwan_sentiment_indicators.items():
            score = 0
            for indicator in indicators:
                # Count occurrences with word boundaries
                pattern = r'\b' + re.escape(indicator) + r'\b'
                matches = re.findall(pattern, text_lower)
                score += len(matches)
            sentiment_scores[sentiment] = score
        
        # Determine dominant sentiment
        if sentiment_scores['hostile'] > sentiment_scores['supportive'] and sentiment_scores['hostile'] > sentiment_scores['neutral']:
            return 'hostile'
        elif sentiment_scores['supportive'] > sentiment_scores['hostile'] and sentiment_scores['supportive'] > sentiment_scores['neutral']:
            return 'supportive'
        else:
            return 'neutral'
    
    def _analyze_escalation_level(self, doc: Doc) -> str:
        """Analyze escalation level"""
        text_lower = doc.text.lower()
        
        # Count indicators for each level
        escalation_scores = {}
        
        for level, indicators in self.escalation_indicators.items():
            score = 0
            for indicator in indicators:
                pattern = r'\b' + re.escape(indicator) + r'\b'
                matches = re.findall(pattern, text_lower)
                score += len(matches)
            escalation_scores[level] = score
        
        # Determine escalation level
        if escalation_scores['high'] > 0:
            return 'high'
        elif escalation_scores['medium'] > 0:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_intent_signal(self, doc: Doc) -> str:
        """Analyze intent signal"""
        text_lower = doc.text.lower()
        
        # Count indicators for each intent
        intent_scores = {}
        
        for intent, indicators in self.intent_indicators.items():
            score = 0
            for indicator in indicators:
                pattern = r'\b' + re.escape(indicator) + r'\b'
                matches = re.findall(pattern, text_lower)
                score += len(matches)
            intent_scores[intent] = score
        
        # Determine dominant intent
        max_score = max(intent_scores.values())
        if max_score == 0:
            return 'symbolic'  # Default
        
        for intent, score in intent_scores.items():
            if score == max_score:
                return intent
        
        return 'symbolic'
    
    def _detect_strategic_posture_change(self, doc: Doc) -> bool:
        """Detect if there's a strategic posture change"""
        text_lower = doc.text.lower()
        
        change_indicators = [
            'change', 'shift', 'adjustment', 'modification', 'alteration',
            'new', 'different', 'unprecedented', 'first time', 'never before',
            'policy change', 'strategy change', 'approach change', 'position change'
        ]
        
        for indicator in change_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def _detect_info_warfare(self, doc: Doc) -> bool:
        """Detect information warfare indicators"""
        text_lower = doc.text.lower()
        
        for indicator in self.info_warfare_indicators:
            pattern = r'\b' + re.escape(indicator) + r'\b'
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _calculate_confidence(self, doc: Doc, taiwan_sentiment: str, escalation_level: str) -> float:
        """Calculate confidence score for the analysis"""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on text length
        if len(doc.text) > 500:
            confidence += 0.1
        elif len(doc.text) < 100:
            confidence -= 0.2
        
        # Adjust based on sentiment clarity
        text_lower = doc.text.lower()
        sentiment_indicators = self.taiwan_sentiment_indicators[taiwan_sentiment]
        indicator_count = sum(len(re.findall(r'\b' + re.escape(indicator) + r'\b', text_lower)) 
                             for indicator in sentiment_indicators)
        
        if indicator_count > 3:
            confidence += 0.1
        elif indicator_count == 0:
            confidence -= 0.2
        
        # Adjust based on escalation level
        if escalation_level == 'high':
            confidence += 0.1  # High escalation is usually more clear
        
        # Clamp to valid range
        return max(0.3, min(1.0, confidence))
    
    def _get_default_sentiment(self) -> SentimentAnalysis:
        """Get default sentiment analysis result"""
        return SentimentAnalysis(
            sentiment_toward_Taiwan='neutral',
            escalation_level='low',
            intent_signal='symbolic',
            strategic_posture_change=False,
            info_warfare_detected=False,
            confidence_score=0.5
        )
    
    def get_sentiment_summary(self, sentiment: SentimentAnalysis) -> str:
        """Get a human-readable summary of the sentiment analysis"""
        summary_parts = []
        
        summary_parts.append(f"Sentiment toward Taiwan: {sentiment.sentiment_toward_Taiwan}")
        summary_parts.append(f"Escalation level: {sentiment.escalation_level}")
        summary_parts.append(f"Intent signal: {sentiment.intent_signal}")
        
        if sentiment.strategic_posture_change:
            summary_parts.append("Strategic posture change detected")
        
        if sentiment.info_warfare_detected:
            summary_parts.append("Information warfare indicators detected")
        
        summary_parts.append(f"Confidence: {sentiment.confidence_score:.2f}")
        
        return "; ".join(summary_parts) 