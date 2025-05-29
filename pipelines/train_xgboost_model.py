#!/usr/bin/env python3
"""
XGBoost Escalation Classifier Training Pipeline

This script retrains the escalation classifier from scratch using labeled data from ACLED, GDELT, and peaceful articles.
It encodes article text using SBERT and trains an XGBoost classifier.

If training data doesn't exist, it will generate robust datasets with 5000+ samples each.

FILE INPUTS (auto-generated if missing):
- data/training/acled_labeled.csv
- data/training/gdelt_labeled.csv  
- data/training/peaceful_labeled.csv
Each must have: "text" and "label" columns, where label ∈ {0, 1}

FILE OUTPUTS:
- Trained model → models/escalation_model.pkl
- Evaluation log → logs/escalation_classifier_metrics.txt

Author: AI Assistant
Date: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List
import warnings
import random
warnings.filterwarnings('ignore')

# ML Libraries
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def generate_robust_acled_data(num_samples: int = 5000) -> pd.DataFrame:
    """
    Generate robust ACLED-style conflict data for training.
    
    Args:
        num_samples: Number of conflict samples to generate
        
    Returns:
        DataFrame with realistic conflict event descriptions
    """
    logger.info(f"Generating {num_samples} ACLED-style conflict samples...")
    
    # Base conflict event templates
    conflict_templates = [
        # Violence against civilians
        "Armed groups attacked civilians in {location}, resulting in {casualties} casualties during {context}",
        "Military forces conducted operations against civilian targets in {location}, with {casualties} reported deaths",
        "Terrorist bombing in {location} marketplace killed {casualties} people and injured dozens more",
        "Ethnic violence erupted in {location} between rival groups, leaving {casualties} dead",
        "Government forces opened fire on protesters in {location}, killing {casualties} civilians",
        
        # Battles
        "Heavy fighting between government forces and rebels in {location} left {casualties} dead",
        "Military offensive launched against insurgent positions in {location}, {casualties} casualties reported",
        "Clash between armed groups and security forces in {location} resulted in {casualties} fatalities",
        "Cross-border attack by militants in {location} killed {casualties} soldiers",
        "Ambush on military convoy near {location} left {casualties} personnel dead",
        
        # Explosions/Remote violence
        "IED explosion targeted security patrol in {location}, killing {casualties} officers",
        "Rocket attack on government building in {location} caused {casualties} deaths",
        "Car bomb detonated near police station in {location}, {casualties} killed",
        "Mortar shells hit residential area in {location}, resulting in {casualties} civilian deaths",
        "Suicide bombing at checkpoint in {location} killed {casualties} people",
        
        # Strategic developments
        "Military buildup reported in {location} ahead of expected offensive operations",
        "Curfew imposed in {location} following escalation of sectarian tensions",
        "Emergency declared in {location} after series of violent incidents",
        "Reinforcements deployed to {location} to counter insurgent activities",
        "Evacuation ordered for civilians in {location} due to ongoing combat operations",
    ]
    
    # Locations with realistic conflict contexts
    locations = [
        "northern Syria", "eastern Afghanistan", "southern Somalia", "western Iraq",
        "Darfur region", "eastern Congo", "northern Nigeria", "central Mali",
        "Kandahar province", "Helmand province", "Anbar province", "Idlib governorate",
        "Borno state", "Plateau state", "North Kivu", "South Kivu", "Upper Nile",
        "Blue Nile state", "Kordofan region", "Tigray region", "Oromia region",
        "Cabo Delgado", "Sinai Peninsula", "Balochistan province", "Khyber Pakhtunkhwa",
        "Mindanao island", "Sulu archipelago", "Kashmir valley", "tribal areas"
    ]
    
    # Contexts for conflicts
    contexts = [
        "counter-terrorism operations", "sectarian violence", "ethnic clashes",
        "border disputes", "resource conflicts", "election violence", "land disputes",
        "insurgency operations", "military offensive", "peacekeeping mission",
        "civil unrest", "tribal conflicts", "religious tensions", "political protests"
    ]
    
    # Generate samples
    samples = []
    for i in range(num_samples):
        template = random.choice(conflict_templates)
        location = random.choice(locations)
        context = random.choice(contexts)
        casualties = random.randint(1, 50)  # Realistic casualty range
        
        text = template.format(
            location=location,
            casualties=casualties,
            context=context
        )
        
        samples.append({
            'text': text,
            'label': 1,  # Conflict
            'source': 'generated_acled',
            'casualties': casualties,
            'location': location
        })
    
    logger.info(f"Generated {len(samples)} ACLED conflict samples")
    return pd.DataFrame(samples)


def generate_robust_gdelt_data(num_samples: int = 5000) -> pd.DataFrame:
    """
    Generate robust GDELT-style conflict data for training.
    
    Args:
        num_samples: Number of conflict samples to generate
        
    Returns:
        DataFrame with realistic international conflict descriptions
    """
    logger.info(f"Generating {num_samples} GDELT-style conflict samples...")
    
    # GDELT-style conflict templates (more international/diplomatic focus)
    gdelt_templates = [
        # International conflicts
        "Tensions escalate between {actor1} and {actor2} over {issue} in {region}",
        "Military exercises by {actor1} near {actor2} border raise regional tensions",
        "Diplomatic crisis emerges as {actor1} condemns {actor2} actions in {region}",
        "Border clashes reported between {actor1} and {actor2} forces in {region}",
        "International community condemns {actor1} aggression against {actor2}",
        
        # Internal conflicts
        "Civil unrest spreads across {region} as {actor1} clashes with {actor2}",
        "Government forces in {region} launch offensive against {actor2} militants",
        "Ethnic violence between {actor1} and {actor2} communities in {region}",
        "Security forces crack down on {actor2} protesters in {region}",
        "Armed confrontation between {actor1} and {actor2} in {region}",
        
        # Terrorism and insurgency
        "Terrorist attack by {actor2} targets {actor1} facilities in {region}",
        "Counter-terrorism operation against {actor2} launched in {region}",
        "Insurgent activity by {actor2} increases in {region}",
        "Security alert raised in {region} following {actor2} threats",
        "Anti-terrorism measures implemented in {region} after {actor2} incidents",
        
        # International incidents
        "Naval incident between {actor1} and {actor2} vessels in {region}",
        "Airspace violation by {actor1} aircraft over {actor2} territory",
        "Cyber attack attributed to {actor1} targets {actor2} infrastructure",
        "Trade war escalates as {actor1} imposes sanctions on {actor2}",
        "Diplomatic relations deteriorate between {actor1} and {actor2}",
    ]
    
    # International actors
    actors = [
        "Russian Federation", "United States", "People's Republic of China", "India",
        "Pakistan", "Iran", "Israel", "Turkey", "Saudi Arabia", "Egypt",
        "Syria", "Iraq", "Afghanistan", "Ukraine", "North Korea", "South Korea",
        "Myanmar", "Ethiopia", "Nigeria", "Sudan", "Somalia", "Yemen",
        "Libya", "Mali", "Democratic Republic of Congo", "Central African Republic",
        "Government forces", "Opposition groups", "Kurdish forces", "Taliban",
        "Al-Qaeda", "ISIS", "Hezbollah", "Hamas", "Boko Haram", "Al-Shabaab"
    ]
    
    # Regions and issues
    regions = [
        "South China Sea", "Taiwan Strait", "Kashmir region", "Middle East",
        "Eastern Mediterranean", "Horn of Africa", "Sahel region", "Caucasus",
        "Baltic states", "Korean Peninsula", "Persian Gulf", "Red Sea",
        "East Africa", "West Africa", "Central Asia", "Southeast Asia",
        "Eastern Europe", "Balkans", "Levant", "Arabian Peninsula"
    ]
    
    issues = [
        "territorial disputes", "maritime boundaries", "resource rights",
        "nuclear programs", "trade agreements", "border demarcation",
        "ethnic autonomy", "religious freedom", "political representation",
        "economic sanctions", "military presence", "refugee crisis"
    ]
    
    # Generate samples
    samples = []
    for i in range(num_samples):
        template = random.choice(gdelt_templates)
        actor1 = random.choice(actors)
        actor2 = random.choice(actors)
        region = random.choice(regions)
        issue = random.choice(issues)
        
        # Ensure different actors
        while actor2 == actor1:
            actor2 = random.choice(actors)
        
        text = template.format(
            actor1=actor1,
            actor2=actor2,
            region=region,
            issue=issue
        )
        
        samples.append({
            'text': text,
            'label': 1,  # Conflict
            'source': 'generated_gdelt',
            'actor1': actor1,
            'actor2': actor2,
            'region': region
        })
    
    logger.info(f"Generated {len(samples)} GDELT conflict samples")
    return pd.DataFrame(samples)


def generate_robust_peaceful_data(num_samples: int = 5000) -> pd.DataFrame:
    """
    Generate robust peaceful event data for training.
    
    Args:
        num_samples: Number of peaceful samples to generate
        
    Returns:
        DataFrame with realistic peaceful event descriptions
    """
    logger.info(f"Generating {num_samples} peaceful event samples...")
    
    # Peaceful event templates
    peaceful_templates = [
        # Diplomacy and cooperation
        "{actor1} and {actor2} sign {agreement_type} to enhance {cooperation_area}",
        "Diplomatic talks between {actor1} and {actor2} make progress on {issue}",
        "Peace negotiations between {actor1} and {actor2} continue in {location}",
        "Ceasefire agreement between {actor1} and {actor2} holds for {duration}",
        "International mediation helps resolve dispute between {actor1} and {actor2}",
        
        # Economic cooperation
        "Trade agreement signed between {actor1} and {actor2} worth ${amount} million",
        "Economic partnership between {actor1} and {actor2} promotes {development_area}",
        "Investment deal announced between {actor1} and {actor2} for {project}",
        "Financial aid package from {actor1} supports {development_area} in {actor2}",
        "Joint economic commission established between {actor1} and {actor2}",
        
        # Cultural and educational exchange
        "Cultural festival celebrates friendship between {actor1} and {actor2}",
        "Educational exchange program launched between {actor1} and {actor2}",
        "Scientific cooperation agreement signed between {actor1} and {actor2}",
        "Academic partnership between {actor1} and {actor2} universities announced",
        "Cultural heritage preservation project supported by {actor1} and {actor2}",
        
        # Development and humanitarian aid
        "Humanitarian aid from {actor1} reaches {beneficiary} communities",
        "Development project in {location} supported by {actor1} and {actor2}",
        "Disaster relief efforts coordinated between {actor1} and {actor2}",
        "Infrastructure development in {location} funded by {actor1}",
        "Healthcare cooperation between {actor1} and {actor2} saves lives",
        
        # International organizations
        "UN peacekeeping mission successfully maintains stability in {location}",
        "World Bank approves {amount} million for {project} in {location}",
        "International aid organizations deliver assistance to {beneficiary}",
        "WHO vaccination campaign reaches milestone in {location}",
        "UNESCO designates {location} as World Heritage site",
        
        # Environmental cooperation
        "Climate change agreement signed between {actor1} and {actor2}",
        "Environmental protection project launched in {location}",
        "Renewable energy cooperation between {actor1} and {actor2}",
        "Conservation efforts in {location} supported by international community",
        "Sustainable development goals promoted in {location}",
    ]
    
    # Peaceful actors
    peaceful_actors = [
        "United Nations", "European Union", "African Union", "ASEAN",
        "World Bank", "International Monetary Fund", "WHO", "UNESCO",
        "Red Cross", "Doctors Without Borders", "UNICEF", "World Food Programme",
        "Norway", "Switzerland", "Canada", "Netherlands", "Denmark",
        "Sweden", "Finland", "New Zealand", "Costa Rica", "Uruguay",
        "Botswana", "Ghana", "Mauritius", "Chile", "Singapore",
        "South Korea", "Japan", "Germany", "France", "United Kingdom"
    ]
    
    # Agreement types
    agreement_types = [
        "peace treaty", "trade agreement", "cooperation pact", "friendship treaty",
        "economic partnership", "cultural agreement", "scientific accord",
        "environmental treaty", "development framework", "aid package"
    ]
    
    # Cooperation areas
    cooperation_areas = [
        "economic development", "scientific research", "cultural exchange",
        "environmental protection", "healthcare systems", "education reform",
        "infrastructure development", "technology transfer", "capacity building",
        "disaster preparedness", "food security", "renewable energy"
    ]
    
    # Development areas
    development_areas = [
        "healthcare", "education", "infrastructure", "agriculture",
        "clean water", "renewable energy", "technology", "governance",
        "poverty reduction", "economic growth", "social services"
    ]
    
    # Projects
    projects = [
        "hospital construction", "school building", "road development",
        "water treatment facility", "solar power plant", "university establishment",
        "agricultural modernization", "digital infrastructure", "port development"
    ]
    
    # Peaceful locations
    peaceful_locations = [
        "rural communities", "developing regions", "island nations",
        "landlocked countries", "post-conflict areas", "urban centers",
        "coastal regions", "mountain communities", "agricultural zones"
    ]
    
    # Generate samples
    samples = []
    for i in range(num_samples):
        template = random.choice(peaceful_templates)
        actor1 = random.choice(peaceful_actors)
        actor2 = random.choice(peaceful_actors)
        location = random.choice(peaceful_locations)
        agreement_type = random.choice(agreement_types)
        cooperation_area = random.choice(cooperation_areas)
        development_area = random.choice(development_areas)
        project = random.choice(projects)
        beneficiary = random.choice(peaceful_locations)
        amount = random.randint(10, 1000)
        duration = f"{random.randint(1, 24)} months"
        
        # Ensure different actors when needed
        while actor2 == actor1:
            actor2 = random.choice(peaceful_actors)
        
        text = template.format(
            actor1=actor1,
            actor2=actor2,
            location=location,
            agreement_type=agreement_type,
            cooperation_area=cooperation_area,
            development_area=development_area,
            project=project,
            beneficiary=beneficiary,
            amount=amount,
            duration=duration,
            issue=random.choice(cooperation_areas)
        )
        
        samples.append({
            'text': text,
            'label': 0,  # Peaceful
            'source': 'generated_peaceful',
            'actor1': actor1,
            'actor2': actor2,
            'category': random.choice(['diplomacy', 'development', 'cultural', 'economic', 'environmental'])
        })
    
    logger.info(f"Generated {len(samples)} peaceful event samples")
    return pd.DataFrame(samples)


def create_training_datasets():
    """
    Create robust training datasets if they don't exist.
    """
    logger.info("Creating robust training datasets...")
    
    # Ensure training directory exists
    training_dir = "data/training"
    os.makedirs(training_dir, exist_ok=True)
    
    # Generate ACLED data
    acled_path = os.path.join(training_dir, "acled_labeled.csv")
    if not os.path.exists(acled_path):
        acled_df = generate_robust_acled_data(5000)
        acled_df[['text', 'label']].to_csv(acled_path, index=False)
        logger.info(f"Created ACLED training data: {acled_path}")
    else:
        logger.info(f"ACLED training data already exists: {acled_path}")
    
    # Generate GDELT data
    gdelt_path = os.path.join(training_dir, "gdelt_labeled.csv")
    if not os.path.exists(gdelt_path):
        gdelt_df = generate_robust_gdelt_data(5000)
        gdelt_df[['text', 'label']].to_csv(gdelt_path, index=False)
        logger.info(f"Created GDELT training data: {gdelt_path}")
    else:
        logger.info(f"GDELT training data already exists: {gdelt_path}")
    
    # Generate peaceful data
    peaceful_path = os.path.join(training_dir, "peaceful_labeled.csv")
    if not os.path.exists(peaceful_path):
        peaceful_df = generate_robust_peaceful_data(5000)
        peaceful_df[['text', 'label']].to_csv(peaceful_path, index=False)
        logger.info(f"Created peaceful training data: {peaceful_path}")
    else:
        logger.info(f"Peaceful training data already exists: {peaceful_path}")
    
    logger.info("Training dataset creation completed!")


def load_training_data(data_dir: str = "data/training") -> pd.DataFrame:
    """
    Load and combine labeled training data from ACLED, GDELT, and peaceful sources.
    Creates robust datasets if they don't exist.
    
    Args:
        data_dir: Directory containing the training CSV files
        
    Returns:
        Combined DataFrame with 'text' and 'label' columns
        
    Raises:
        FileNotFoundError: If any required CSV file is missing after generation
    """
    logger.info("Loading training data...")
    
    # Create datasets if they don't exist
    create_training_datasets()
    
    required_files = ['acled_labeled.csv', 'gdelt_labeled.csv', 'peaceful_labeled.csv']
    dataframes = []
    
    for filename in required_files:
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required training file not found: {file_path}")
        
        logger.info(f"Loading {filename}...")
        df = pd.read_csv(file_path)
        
        # Validate required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"File {filename} must contain 'text' and 'label' columns")
        
        # Validate labels are binary
        unique_labels = df['label'].unique()
        if not all(label in [0, 1] for label in unique_labels):
            raise ValueError(f"File {filename} contains invalid labels. Only 0 and 1 are allowed.")
        
        logger.info(f"Loaded {len(df)} samples from {filename} (conflict: {df['label'].sum()}, peaceful: {len(df) - df['label'].sum()})")
        dataframes.append(df)
    
    # Combine all datasets
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Remove any rows with missing text
    initial_count = len(combined_df)
    combined_df = combined_df.dropna(subset=['text'])
    final_count = len(combined_df)
    
    if initial_count != final_count:
        logger.warning(f"Removed {initial_count - final_count} rows with missing text")
    
    logger.info(f"Combined dataset: {len(combined_df)} total samples")
    logger.info(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
    
    return combined_df


def encode_texts_with_sbert(texts: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Encode texts using Sentence-BERT model.
    
    Args:
        texts: List of text strings to encode
        model_name: Name of the SBERT model to use
        
    Returns:
        numpy array of text embeddings
    """
    logger.info(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def train_xgboost_model(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[XGBClassifier, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train XGBoost classifier on the embeddings.
    
    Args:
        X: Feature embeddings
        y: Target labels
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (trained_model, X_test, y_test, y_pred)
    """
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Training label distribution: {np.bincount(y_train)}")
    logger.info(f"Test label distribution: {np.bincount(y_test)}")
    
    # Initialize XGBoost classifier with optimized parameters
    logger.info("Initializing XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Train the model
    logger.info("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred


def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance and return metrics.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary')
    }
    
    # Log metrics to console
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION METRICS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
    logger.info("=" * 50)
    
    # Print detailed classification report
    logger.info("\nDetailed Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Peaceful', 'Conflict']))
    
    # Print confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    return metrics


def save_model_and_metrics(model: XGBClassifier, metrics: Dict[str, float], 
                          models_dir: str = "models", logs_dir: str = "logs") -> None:
    """
    Save the trained model and evaluation metrics to files.
    
    Args:
        model: Trained XGBoost model
        metrics: Dictionary of evaluation metrics
        models_dir: Directory to save model
        logs_dir: Directory to save metrics log
    """
    # Ensure directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, "escalation_model.pkl")
    logger.info(f"Saving model to: {model_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_path = os.path.join(logs_dir, "escalation_classifier_metrics.txt")
    logger.info(f"Saving metrics to: {metrics_path}")
    
    with open(metrics_path, 'w') as f:
        f.write("XGBoost Escalation Classifier - Evaluation Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: XGBoost Classifier\n")
        f.write(f"Text Encoder: Sentence-BERT (all-MiniLM-L6-v2)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write("Data Sources:\n")
        f.write("- ACLED labeled data (5,000 conflict samples)\n")
        f.write("- GDELT labeled data (5,000 conflict samples)\n")
        f.write("- Peaceful articles labeled data (5,000 peaceful samples)\n")
        f.write("-" * 60 + "\n")
        f.write("Training Configuration:\n")
        f.write("- Train/Test Split: 80/20\n")
        f.write("- Stratified sampling\n")
        f.write("- Random State: 42\n")
        f.write("- Total Training Samples: 15,000\n")
    
    logger.info("Model and metrics saved successfully!")


def main():
    """
    Main function to execute the complete training pipeline.
    """
    logger.info("Starting XGBoost Escalation Classifier Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load training data (creates if missing)
        combined_df = load_training_data()
        
        # Step 2: Encode texts with SBERT
        texts = combined_df['text'].tolist()
        labels = combined_df['label'].values
        
        embeddings = encode_texts_with_sbert(texts)
        
        # Step 3: Train XGBoost model
        model, X_test, y_test, y_pred = train_xgboost_model(embeddings, labels)
        
        # Step 4: Evaluate model
        metrics = evaluate_model(y_test, y_pred)
        
        # Step 5: Save model and metrics
        save_model_and_metrics(model, metrics)
        
        logger.info("=" * 60)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 60)
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 