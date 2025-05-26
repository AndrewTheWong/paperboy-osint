import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict
from fetch_and_clean_data import tag_encoder, TAG_VOCAB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/peaceful_events.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_peaceful_gdelt_events(force=False):
    """
    Extract non-conflict events from GDELT data.
    
    Args:
        force: Whether to reprocess even if output file exists
    
    Returns:
        DataFrame with non-conflict GDELT events
    """
    output_file = "data/gdelt_peaceful_events.csv"
    
    # Check if output already exists
    if os.path.exists(output_file) and not force:
        logger.info(f"Peaceful GDELT events file already exists at {output_file}")
        return pd.read_csv(output_file)
    
    # Load existing GDELT data
    try:
        gdelt_df = pd.read_csv("data/gdelt_events.csv")
        logger.info(f"Loaded {len(gdelt_df)} GDELT events for peaceful event extraction")
    except FileNotFoundError:
        logger.error("GDELT events file not found")
        return pd.DataFrame()
    
    # Extract non-conflict events (EventCode not starting with "19")
    nonconflict_gdelt = gdelt_df[~gdelt_df["EventCode"].astype(str).str.startswith("19")].copy()
    
    # Process non-conflict events
    nonconflict_gdelt["label"] = 0  # Non-conflict label
    nonconflict_gdelt["confidence"] = ((nonconflict_gdelt["GoldsteinScale"] + 10) / 20).clip(0, 1)
    
    # Generate descriptive text
    nonconflict_gdelt["text"] = (
        "Interaction between " + nonconflict_gdelt["actor1"].fillna("UNKNOWN") +
        " and " + nonconflict_gdelt["actor2"].fillna("UNKNOWN") +
        " with EventCode " + nonconflict_gdelt["EventCode"].astype(str)
    )
    
    # Add source field
    nonconflict_gdelt["source"] = "gdelt_peaceful"
    
    # Keep only needed columns
    peaceful_df = nonconflict_gdelt[["text", "label", "confidence", "source", "date"]]
    
    # Save to CSV
    peaceful_df.to_csv(output_file, index=False)
    logger.info(f"Extracted and saved {len(peaceful_df)} peaceful GDELT events to {output_file}")
    
    return peaceful_df

def create_synthetic_peaceful_events():
    """
    Create synthetic peaceful events data.
    
    Returns:
        DataFrame with synthetic peaceful events
    """
    logger.info("Creating synthetic peaceful events dataset")
    
    # Define synthetic peaceful events
    synthetic_events = [
        {"text": "World Bank funds road project in Botswana", "label": 0, "confidence": 0.9},
        {"text": "UN hosts development summit on climate change", "label": 0, "confidence": 0.9},
        {"text": "Costa Rica launches new solar energy program", "label": 0, "confidence": 0.9},
        {"text": "Iceland celebrates 20 years of continuous peace", "label": 0, "confidence": 0.95},
        {"text": "Japan and South Korea sign trade agreement", "label": 0, "confidence": 0.8},
        {"text": "UNESCO designates new World Heritage site in Vietnam", "label": 0, "confidence": 0.9},
        {"text": "Norway increases funding for global education initiatives", "label": 0, "confidence": 0.85},
        {"text": "New Zealand and Australia cooperate on marine conservation", "label": 0, "confidence": 0.9},
        {"text": "Singapore hosts international technology conference", "label": 0, "confidence": 0.8},
        {"text": "Canada and EU finalize terms of free trade agreement", "label": 0, "confidence": 0.85},
        {"text": "African Union announces infrastructure development plan", "label": 0, "confidence": 0.75},
        {"text": "Indonesia and Malaysia conduct joint cultural festival", "label": 0, "confidence": 0.8},
        {"text": "Switzerland mediates diplomatic talks between rival nations", "label": 0, "confidence": 0.7},
        {"text": "Brazil inaugurates new scientific research facility", "label": 0, "confidence": 0.85},
        {"text": "Denmark achieves 50% renewable energy milestone", "label": 0, "confidence": 0.9},
        {"text": "Ghana launches nationwide vaccination campaign", "label": 0, "confidence": 0.85},
        {"text": "Argentina and Chile resolve border dispute through peaceful negotiation", "label": 0, "confidence": 0.75},
        {"text": "Finland invests in early childhood education reform", "label": 0, "confidence": 0.9},
        {"text": "Luxembourg hosts international finance summit", "label": 0, "confidence": 0.8},
        {"text": "Thailand opens new cultural exchange center", "label": 0, "confidence": 0.85},
        {"text": "South Africa and Botswana sign water management treaty", "label": 0, "confidence": 0.8},
        {"text": "Portugal unveils offshore wind farm initiative", "label": 0, "confidence": 0.85},
        {"text": "Estonia launches e-residency program for entrepreneurs", "label": 0, "confidence": 0.8},
        {"text": "Uruguay legalizes cannabis for medical research", "label": 0, "confidence": 0.75},
        {"text": "Ireland announces increased foreign aid budget", "label": 0, "confidence": 0.85},
    ]
    
    # Create DataFrame
    synthetic_df = pd.DataFrame(synthetic_events)
    
    # Add source field
    synthetic_df["source"] = "synthetic_peaceful"
    
    logger.info(f"Created {len(synthetic_df)} synthetic peaceful events")
    return synthetic_df

def create_peaceful_dataset():
    """
    Create a combined peaceful events dataset.
    
    Returns:
        DataFrame with all peaceful events, encoded with tags
    """
    logger.info("Creating combined peaceful events dataset")
    
    # Get peaceful GDELT events
    gdelt_peaceful = extract_peaceful_gdelt_events()
    
    # Get synthetic peaceful events
    synthetic_peaceful = create_synthetic_peaceful_events()
    
    # Combine datasets
    all_peaceful = pd.concat([gdelt_peaceful, synthetic_peaceful], ignore_index=True)
    
    # Encode tags
    logger.info("Encoding tags for peaceful events...")
    all_peaceful["tags"] = all_peaceful["text"].apply(tag_encoder)
    
    # Calculate tag coverage metrics
    all_peaceful["tag_count"] = all_peaceful["tags"].apply(sum)
    all_peaceful["no_tags"] = all_peaceful["tag_count"] == 0
    
    # Log tag coverage statistics
    total = len(all_peaceful)
    no_tagged = all_peaceful["no_tags"].sum()
    logger.info(f"[STATS] Total peaceful events: {total}")
    logger.info(f"[WARNING] Peaceful events with 0 matched tags: {no_tagged} ({100 * no_tagged / total:.1f}%)")
    
    # Calculate and log most common tags
    if len(all_peaceful) > 0:
        tag_matrix = np.array(all_peaceful["tags"].tolist())
        tag_totals = tag_matrix.sum(axis=0)
        
        logger.info("Most common tags in peaceful events:")
        for tag, count in sorted(zip(TAG_VOCAB, tag_totals), key=lambda x: -x[1])[:5]:
            if count > 0:
                logger.info(f"[TAG] {tag}: {int(count)} matches")
    
    # Save to CSV
    output_file = "data/all_peaceful_events.csv"
    all_peaceful.to_csv(output_file, index=False)
    logger.info(f"Saved {len(all_peaceful)} peaceful events to {output_file}")
    
    return all_peaceful

def combine_with_conflict_data(peaceful_df, force=False):
    """
    Combine peaceful events with conflict events.
    
    Args:
        peaceful_df: DataFrame with peaceful events
        force: Whether to regenerate even if output file exists
    
    Returns:
        Combined DataFrame with both peaceful and conflict events
    """
    output_file = "data/balanced_conflict_data.csv"
    
    # Check if output already exists
    if os.path.exists(output_file) and not force:
        logger.info(f"Balanced dataset already exists at {output_file}")
        combined_df = pd.read_csv(output_file)
        
        # Check if the tags column needs fixing
        if "tags" in combined_df.columns:
            if combined_df["tags"].isna().any():
                logger.warning("Found NaN values in tags column, regenerating dataset")
                force = True
            elif isinstance(combined_df["tags"].iloc[0], str):
                try:
                    combined_df["tags"] = combined_df["tags"].apply(eval)
                except (ValueError, SyntaxError):
                    logger.warning("Failed to parse tags column, regenerating dataset")
                    force = True
        
        if not force:
            return combined_df
    
    # Load conflict data
    try:
        conflict_df = pd.read_csv("data/all_conflict_data.csv")
        logger.info(f"Loaded {len(conflict_df)} conflict events")
        
        # Ensure conflict data has valid tags
        if "tags" not in conflict_df.columns or conflict_df["tags"].isna().any():
            logger.warning("Conflict data missing tags column or contains NaN values")
            from fetch_and_clean_data import tag_encoder
            logger.info("Re-encoding tags for conflict events...")
            conflict_df["tags"] = conflict_df["text"].apply(tag_encoder)
        elif isinstance(conflict_df["tags"].iloc[0], str):
            try:
                conflict_df["tags"] = conflict_df["tags"].apply(eval)
            except:
                logger.warning("Failed to parse tags from string, re-encoding...")
                from fetch_and_clean_data import tag_encoder
                conflict_df["tags"] = conflict_df["text"].apply(tag_encoder)
    except FileNotFoundError:
        logger.error("Conflict data file not found")
        return peaceful_df
    
    # Ensure peaceful data has valid tags
    if peaceful_df["tags"].isna().any():
        logger.warning("Peaceful data contains NaN values in tags column")
        from fetch_and_clean_data import tag_encoder
        logger.info("Re-encoding tags for peaceful events...")
        peaceful_df["tags"] = peaceful_df["text"].apply(tag_encoder)
    
    # Combine datasets
    combined_df = pd.concat([conflict_df, peaceful_df], ignore_index=True)
    
    # Calculate tag statistics for verification
    combined_df["tag_count"] = combined_df["tags"].apply(lambda x: sum(x) if isinstance(x, list) else 0)
    combined_df["no_tags"] = combined_df["tag_count"] == 0
    
    # Log tag statistics
    no_tags_count = combined_df["no_tags"].sum()
    logger.info(f"Tag statistics: {no_tags_count}/{len(combined_df)} events have no tags ({no_tags_count/len(combined_df)*100:.1f}%)")
    
    # Save to CSV with safe handling of list columns
    # Convert tags to strings for safe CSV storage
    combined_df_for_csv = combined_df.copy()
    combined_df_for_csv["tags"] = combined_df_for_csv["tags"].apply(str)
    combined_df_for_csv.to_csv(output_file, index=False)
    logger.info(f"Saved {len(combined_df)} balanced events (with peaceful data) to {output_file}")
    
    # Log class balance statistics
    conflict_count = combined_df["label"].sum()
    peaceful_count = len(combined_df) - conflict_count
    logger.info(f"Balanced dataset statistics:")
    logger.info(f"  Total events: {len(combined_df)}")
    logger.info(f"  Conflict events: {conflict_count} ({conflict_count/len(combined_df)*100:.1f}%)")
    logger.info(f"  Peaceful events: {peaceful_count} ({peaceful_count/len(combined_df)*100:.1f}%)")
    
    return combined_df

def prepare_balanced_model_data(force=False):
    """
    Prepare balanced dataset for model training.
    
    Args:
        force: Whether to force regeneration of data
    
    Returns:
        X features, y labels, and original texts
    """
    # Create peaceful dataset
    peaceful_df = create_peaceful_dataset()
    
    # Combine with conflict data
    combined_df = combine_with_conflict_data(peaceful_df, force)
    
    # Validate that tags are properly formatted as lists 
    if "tags" in combined_df.columns:
        # Check if any tags are missing or invalid
        invalid_tags = False
        if combined_df["tags"].isna().any():
            logger.warning("Combined dataset has NaN tags, re-encoding...")
            invalid_tags = True
        elif not isinstance(combined_df["tags"].iloc[0], list):
            logger.warning(f"Tags are not in list format: {type(combined_df['tags'].iloc[0])}")
            invalid_tags = True
            
        # Fix invalid tags if needed
        if invalid_tags:
            from fetch_and_clean_data import tag_encoder
            logger.info("Re-encoding tags for all events...")
            combined_df["tags"] = combined_df["text"].apply(tag_encoder)
    
    # Ensure confidence column is valid (no NaNs)
    if "confidence" not in combined_df.columns:
        logger.warning("Confidence column missing, adding default values")
        combined_df["confidence"] = 0.5
    elif combined_df["confidence"].isna().any():
        logger.warning("Found NaN values in confidence column, filling with default value")
        combined_df["confidence"] = combined_df["confidence"].fillna(0.5)
    
    # Create feature matrix directly to avoid prepare_model_data errors
    logger.info("Building feature matrix...")
    X = np.hstack([
        np.array(combined_df["tags"].tolist()), 
        combined_df["confidence"].values.reshape(-1, 1)
    ])
    
    # Check for NaN values
    if np.isnan(X).any():
        logger.warning("Feature matrix contains NaN values, replacing with zeros")
        X = np.nan_to_num(X, nan=0.0)
    
    # Extract labels
    y = combined_df["label"].values
    
    # Extract texts
    texts = combined_df["text"].tolist()
    
    # Save model-ready data
    logger.info("Saving balanced model-ready data...")
    os.makedirs("data/model_ready", exist_ok=True)
    np.save("data/model_ready/X_features_balanced.npy", X)
    np.save("data/model_ready/y_labels_balanced.npy", y)
    
    # Save texts as pickle for reference
    import pickle
    with open("data/model_ready/texts_balanced.pkl", "wb") as f:
        pickle.dump(texts, f)
    
    logger.info(f"Saved balanced model-ready data: X.shape={X.shape}, y.shape={y.shape}")
    
    return X, y, texts

if __name__ == "__main__":
    prepare_balanced_model_data() 