from storage.db import get_unprocessed_osint_entries, insert_prediction, update_osint_tags
import random
from transformers import pipeline
from datetime import datetime

# Load sentiment analysis pipeline for basic prediction
sentiment = pipeline("sentiment-analysis")

# Mapping of tags to regions (simplified for demonstration)
REGION_MAPPING = {
    "military movement": ["Eastern Europe", "Middle East", "East Asia", "South Asia"],
    "diplomatic meeting": ["North America", "Europe", "Asia"],
    "conflict": ["Middle East", "Africa", "Eastern Europe", "South Asia"],
    "cyberattack": ["Global", "North America", "Europe", "East Asia"],
    "protest": ["Europe", "Middle East", "South America", "Southeast Asia"],
    "ceasefire": ["Middle East", "Africa", "Eastern Europe"],
    "nuclear": ["East Asia", "South Asia", "Middle East"]
}

def extract_region_from_text(text):
    """Extract region mentions from text using simple keyword matching."""
    regions = {
        "North America": ["United States", "Canada", "Mexico", "USA", "America"],
        "South America": ["Brazil", "Argentina", "Colombia", "Venezuela", "Chile"],
        "Europe": ["EU", "European Union", "Germany", "France", "UK", "Britain", "Italy", "Spain"],
        "Eastern Europe": ["Russia", "Ukraine", "Poland", "Belarus", "Romania"],
        "Middle East": ["Iran", "Iraq", "Syria", "Saudi Arabia", "Israel", "Turkey"],
        "Africa": ["Egypt", "Nigeria", "South Africa", "Kenya", "Ethiopia"],
        "South Asia": ["India", "Pakistan", "Bangladesh", "Afghanistan"],
        "East Asia": ["China", "Japan", "North Korea", "South Korea", "Taiwan"],
        "Southeast Asia": ["Vietnam", "Thailand", "Indonesia", "Philippines", "Malaysia"],
        "Oceania": ["Australia", "New Zealand"]
    }
    
    found_regions = []
    for region, keywords in regions.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            found_regions.append(region)
    
    return found_regions or ["Global"]  # Default to Global if no specific region found

def predict_from_osint(osint_entry):
    """Generate predictions based on OSINT data."""
    content = osint_entry.get("content", "")
    tags = osint_entry.get("tags", [])
    
    if not content or not tags:
        return []
    
    # Use sentiment to help determine likelihood
    try:
        sentiment_result = sentiment(content[:512])[0]  # Limit input size
        # Convert sentiment to a numeric score (0-1)
        sentiment_score = 0.5
        if sentiment_result["label"] == "POSITIVE":
            sentiment_score = 0.3  # Lower likelihood for positive sentiment
        elif sentiment_result["label"] == "NEGATIVE":
            sentiment_score = 0.7  # Higher likelihood for negative sentiment
    except Exception:
        sentiment_score = 0.5  # Default if sentiment analysis fails
    
    # Extract potential regions from text
    extracted_regions = extract_region_from_text(content)
    
    predictions = []
    for tag in tags:
        # Determine event type based on tag
        event_type = tag
        
        # Select region based on tag and extracted regions
        potential_regions = REGION_MAPPING.get(tag, ["Global"])
        # Prioritize regions mentioned in the text
        matching_regions = [r for r in extracted_regions if r in potential_regions]
        region = random.choice(matching_regions) if matching_regions else random.choice(potential_regions)
        
        # Calculate likelihood score (0.3-0.9 range)
        base_score = 0.3 + (random.random() * 0.3)  # Random base between 0.3 and 0.6
        tag_weight = 0.1 if tag in ["conflict", "military movement", "nuclear"] else 0.05
        likelihood_score = min(0.9, base_score + tag_weight + (sentiment_score * 0.2))
        
        predictions.append({
            "osint_id": osint_entry.get("id"),
            "event_type": event_type,
            "region": region,
            "likelihood_score": likelihood_score,
            "model_used": "hybrid_sentiment_rule"
        })
    
    return predictions

def run_prediction_pipeline(limit=20):
    """Run the prediction pipeline on untagged OSINT entries."""
    # Get entries that have tags but haven't been manually reviewed
    entries = get_unprocessed_osint_entries(limit)
    processed_count = 0
    prediction_count = 0
    
    for entry in entries:
        try:
            # Only process entries with content
            if not entry.get("content"):
                continue
                
            # Auto-tag if not tagged yet
            if not entry.get("tags"):
                from models.tagging import auto_tag_text
                tags, confidence = auto_tag_text(entry.get("content", ""))
                if tags:
                    update_osint_tags(entry.get("id"), tags, confidence)
                    entry["tags"] = tags
                    entry["confidence_score"] = confidence
                else:
                    continue  # Skip if no tags could be assigned
            
            # Generate predictions
            predictions = predict_from_osint(entry)
            for prediction in predictions:
                insert_prediction(
                    prediction["osint_id"],
                    prediction["event_type"],
                    prediction["region"],
                    prediction["likelihood_score"],
                    prediction["model_used"]
                )
                prediction_count += 1
                
            processed_count += 1
                
        except Exception as e:
            print(f"Error processing entry {entry.get('id')}: {str(e)}")
    
    print(f"Generated {prediction_count} predictions from {processed_count} OSINT entries.")
    return processed_count, prediction_count

if __name__ == "__main__":
    run_prediction_pipeline() 