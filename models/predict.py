import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from storage.db import get_unprocessed_osint_entries, insert_prediction

# Load XGBoost model
xgb_model = joblib.load("models/xgboost_conflict_model.pkl")

# Load Transformer model
transformer_model = AutoModelForSequenceClassification.from_pretrained("models/transformer_conflict_model")
transformer_tokenizer = AutoTokenizer.from_pretrained("models/transformer_conflict_model")

# PART 2: INPUT ENCODING
# Tag-based encoding for XGBoost (assumes consistent tag vocab)
def encode_tags(tags: list, confidence: float) -> np.ndarray:
    TAG_VOCAB = [
        "military movement", "conflict", "cyberattack", "protest",
        "diplomatic meeting", "nuclear", "ceasefire"
    ]
    vec = [1 if tag in tags else 0 for tag in TAG_VOCAB]
    vec.append(confidence)
    return np.array(vec).reshape(1, -1)

# Text encoding for transformer
def prepare_text_input(text: str):
    return transformer_tokenizer(text, truncation=True, padding=True, return_tensors="pt")

# PART 3: MODEL PREDICTIONS
# XGBoost prediction
def predict_with_xgboost(tags: list, confidence: float) -> float:
    features = encode_tags(tags, confidence)
    prob = float(xgb_model.predict_proba(features)[0][1])
    return prob  # escalation likelihood from 0.0 to 1.0

# Transformer prediction
def predict_with_transformer(text: str) -> float:
    inputs = prepare_text_input(text)
    with torch.no_grad():
        logits = transformer_model(**inputs).logits
        prob = torch.softmax(logits, dim=1)[0][1].item()  # class 1 = escalation
    return prob

# PART 4: PREDICTION PIPELINE
def run_prediction_pipeline():
    osint_items = get_unprocessed_osint_entries()

    for item in osint_items:
        tags = item.get("tags", [])
        confidence = item.get("confidence_score", 0.5)
        text = item.get("content", "")

        # Run both models
        xgb_score = predict_with_xgboost(tags, confidence)
        transformer_score = predict_with_transformer(text)

        # Average for ensemble
        final_score = round((xgb_score + transformer_score) / 2, 4)

        insert_prediction(
            osint_id=item["id"],
            event_type="conflict",  # static for now
            region="unspecified",
            score=final_score,
            model="ensemble"
        )

    print(f"✅ Scored and logged {len(osint_items)} OSINT entries.") 