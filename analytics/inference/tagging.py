from transformers import pipeline
import re
from storage.db import update_osint_tags, get_unprocessed_osint_entries

# Initialize NER pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def auto_tag_text(text):
    """Tag text using NER and rule-based approaches."""
    # Run named entity recognition
    try:
        entities = ner(text[:5000])  # Limit text length for performance
        found = [e['word'].lower() for e in entities]
    except Exception as e:
        print(f"Error running NER: {str(e)}")
        found = []

    # Define rules for different event types
    rules = {
        "military movement": ["troop", "deployed", "airstrike", "naval", "military", "forces", "army", "battalion"],
        "diplomatic meeting": ["summit", "meeting", "negotiation", "talks", "diplomat", "embassy", "consulate"],
        "conflict": ["clash", "border", "shelling", "explosion", "attack", "war", "battle", "fighting"],
        "cyberattack": ["cyber", "ddos", "hacking", "malware", "ransomware", "phishing"],
        "protest": ["protest", "riot", "unrest", "demonstration", "rally", "march"],
        "ceasefire": ["ceasefire", "truce", "peace", "armistice"],
        "nuclear": ["uranium", "nuclear", "enrichment", "reactor", "isotope", "plutonium"]
    }

    # Apply rule-based tagging
    tags = []
    for tag, keywords in rules.items():
        if any(kw in text.lower() for kw in keywords):
            tags.append(tag)

    # Check for entities to enhance confidence
    entities_found = len(found)
    tag_count = len(tags)
    
    # Calculate confidence score
    confidence = min(1.0, 0.6 + 0.05 * tag_count + 0.01 * entities_found)

    return tags, confidence

def tag_and_score_osint_item(text: str, source_url: str):
    """Tag and score a single OSINT item."""
    tags, confidence = auto_tag_text(text)
    return {
        "source_url": source_url,
        "content": text,
        "tags": tags,
        "confidence_score": confidence
    }

def process_untagged_entries(limit=10):
    """Process all untagged entries in the database."""
    entries = get_unprocessed_osint_entries(limit)
    processed_count = 0
    
    for entry in entries:
        try:
            tags, confidence = auto_tag_text(entry["content"])
            if tags:
                update_osint_tags(entry["id"], tags, confidence)
                processed_count += 1
        except Exception as e:
            print(f"Error processing entry {entry['id']}: {str(e)}")
    
    print(f"Auto-tagged {processed_count} out of {len(entries)} entries.")
    return processed_count

if __name__ == "__main__":
    process_untagged_entries() 