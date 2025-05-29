#!/usr/bin/env python3
"""
Production-ready inference pipeline for predicting escalation likelihood from OSINT article text.

This module provides a clean, efficient interface for escalation prediction that can be used by:
- Daily clustering systems
- Digest generation
- Streamlit dashboards  
- Auto-alert systems
- Batch processing jobs

The pipeline uses:
- Tag-based feature extraction for compatibility with existing XGBoost models
- Sentence-BERT encoder: "all-MiniLM-L6-v2" for text embedding (future models)
- Pre-trained classifier from: "models/escalation_model.pkl" (fallback to latest available)

Usage:
    # Single prediction
    from inference_pipeline import predict_escalation
    score = predict_escalation("Taiwan military conducts exercises near strait")
    
    # Batch processing via CLI
    python inference_pipeline.py input.csv output.csv
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Optional, List
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model objects (loaded once on import)
encoder = None
model = None

# Tag vocabulary for existing models (7 tags + confidence = 8 features)
TAG_VOCAB = [
    "military movement", "conflict", "cyberattack", "protest",
    "diplomatic meeting", "nuclear", "ceasefire"
]

def extract_tags_from_text(text: str) -> List[str]:
    """
    Extract tags from text based on keyword matching.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of matched tags
    """
    text = text.lower()
    matched_tags = []
    
    # Keyword mappings for each tag
    tag_keywords = {
        "military movement": ["military", "army", "troop", "force", "drill", "exercise", "maneuver", "deployment"],
        "conflict": ["conflict", "fight", "battle", "war", "clash", "combat", "violence", "attack"],
        "cyberattack": ["cyber", "hack", "breach", "malware", "ransomware", "digital attack"],
        "protest": ["protest", "demonstration", "riot", "unrest", "march", "rally"],
        "diplomatic meeting": ["diplomatic", "meeting", "summit", "talks", "negotiation", "conference"],
        "nuclear": ["nuclear", "atomic", "warhead", "missile", "icbm", "deterrent"],
        "ceasefire": ["ceasefire", "truce", "peace", "armistice", "cease-fire"]
    }
    
    for tag, keywords in tag_keywords.items():
        for keyword in keywords:
            if keyword in text:
                matched_tags.append(tag)
                break  # Only add tag once
    
    return matched_tags

def encode_tags(text: str, confidence: float = 0.5) -> np.ndarray:
    """
    Convert text to tag-based features compatible with existing models.
    
    Args:
        text: Input text to analyze
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        numpy array of features [tag1, tag2, ..., tag7, confidence]
    """
    matched_tags = extract_tags_from_text(text)
    
    # Create binary vector for each tag in vocabulary
    tag_vector = [1 if tag in matched_tags else 0 for tag in TAG_VOCAB]
    
    # Add confidence score as final feature
    tag_vector.append(confidence)
    
    return np.array(tag_vector).reshape(1, -1)

def load_models():
    """Load the sentence transformer and classifier models."""
    global encoder, model
    
    try:
        # Load Sentence-BERT encoder for future use
        logger.info("Loading Sentence-BERT encoder: all-MiniLM-L6-v2")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Try to load escalation model, fall back to latest available
        model_paths = [
            "models/escalation_model.pkl",
            "models/xgboost_conflict_model_20250519.pkl", 
            "models/xgboost_conflict_model.pkl",
            "models/xgboost_conflict_model_20250520.pkl",
            "models/calibrated_xgboost_model_20250520.pkl",
            "models/calibrated_xgboost_model_20250526.pkl"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                logger.info(f"Loading classifier from: {model_path}")
                model = joblib.load(model_path)
                break
        else:
            raise FileNotFoundError("No suitable escalation model found in models/ directory")
            
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def predict_escalation(text: str) -> float:
    """
    Predict escalation likelihood from raw article text.
    
    Args:
        text (str): Raw article text to analyze
        
    Returns:
        float: Escalation probability between 0.0 and 1.0
               Returns 0.0 for empty/invalid input or on error
    """
    # Input validation
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return 0.0
    
    try:
        # Ensure models are loaded
        if model is None:
            load_models()
        
        # Extract tags and create features compatible with existing model
        features = encode_tags(text.strip())
        
        # Get prediction from classifier
        probabilities = model.predict_proba(features)
        
        # Return escalation probability (class 1)
        escalation_prob = float(probabilities[0][1])
        
        return min(max(escalation_prob, 0.0), 1.0)  # Clamp to [0, 1]
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return 0.0

def process_csv_batch(input_file: str, output_file: str) -> None:
    """
    Process a CSV file with article data and add escalation scores.
    
    Expected CSV columns: id, title, text
    Output CSV adds: escalation_score
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    try:
        # Load input data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        # Validate required columns
        required_cols = ['id', 'title', 'text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process each article
        logger.info(f"Processing {len(df)} articles...")
        escalation_scores = []
        
        for idx, row in df.iterrows():
            # Combine title and text for better context
            full_text = f"{row['title']} {row['text']}"
            score = predict_escalation(full_text)
            escalation_scores.append(score)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} articles")
        
        # Add scores to dataframe
        df['escalation_score'] = escalation_scores
        
        # Save results
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        # Log summary statistics
        high_risk = (df['escalation_score'] > 0.7).sum()
        medium_risk = ((df['escalation_score'] > 0.3) & (df['escalation_score'] <= 0.7)).sum()
        low_risk = (df['escalation_score'] <= 0.3).sum()
        
        logger.info(f"Summary: {high_risk} high-risk, {medium_risk} medium-risk, {low_risk} low-risk articles")
        
    except Exception as e:
        logger.error(f"Error processing CSV batch: {e}")
        raise

def main():
    """CLI entry point for batch processing."""
    if len(sys.argv) != 3:
        print("Usage: python inference_pipeline.py <input.csv> <output.csv>")
        print("Expected CSV columns: id, title, text")
        print("Output will add: escalation_score")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    try:
        process_csv_batch(input_file, output_file)
        print(f"✅ Successfully processed {input_file} -> {output_file}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

# Load models on import
try:
    load_models()
except Exception as e:
    logger.warning(f"Failed to load models on import: {e}")

if __name__ == "__main__":
    # Test with sample Taiwan Strait news article
    if len(sys.argv) == 1:
        print("🧪 Testing inference pipeline with sample article...")
        print()
        
        sample_text = """
        Taiwan's military conducted large-scale exercises near the Taiwan Strait today, 
        prompting China to respond with its own naval maneuvers in the region. 
        The exercises included live-fire drills and involved multiple aircraft and warships. 
        Military analysts warn that the escalating tensions could lead to a broader conflict 
        if diplomatic channels fail to de-escalate the situation.
        """
        
        score = predict_escalation(sample_text.strip())
        print(f"Sample text: {sample_text.strip()[:100]}...")
        print(f"Escalation score: {score:.4f}")
        print()
        
        if score > 0.7:
            print("🔴 HIGH RISK: Strong indicators of potential escalation")
        elif score > 0.3:
            print("🟡 MEDIUM RISK: Some escalatory elements detected")
        else:
            print("🟢 LOW RISK: Minimal escalation indicators")
        
        print()
        print("Usage for batch processing:")
        print("python inference_pipeline.py input.csv output.csv")
    else:
        main() 