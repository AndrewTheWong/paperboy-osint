import os
import pandas as pd
import numpy as np
import json
import datetime
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
import torch
import nltk
import random
from nltk.corpus import wordnet

# Download required NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/transformer_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Generate timestamp for versioning
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d")

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def load_and_prepare_data():
    """
    Load combined data (GDELT, ACLED, UCDP) and prepare for transformer training.
    
    Returns:
        DataFrame with text, label, and date columns
    """
    logger.info("Loading data for transformer training...")
    
    # First try to load the combined dataset
    try:
        combined_df = pd.read_csv("data/all_conflict_data.csv")
        logger.info(f"Loaded combined dataset with {len(combined_df)} events")
        
        # Convert date column to datetime if needed
        if 'date' in combined_df.columns and not pd.api.types.is_datetime64_any_dtype(combined_df['date']):
            combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
            
        return combined_df
    except FileNotFoundError:
        logger.warning("Combined dataset not found. Trying individual data sources...")
    
    # Load GDELT data
    try:
        gdelt_df = pd.read_csv("data/gdelt_events.csv")
        logger.info(f"Loaded {len(gdelt_df)} GDELT events")
    except FileNotFoundError:
        logger.error("GDELT data file not found. Please run fetch_and_clean_data.py first.")
        return None
    
    # Load ACLED data if available
    try:
        with open("data/acled_events.json", "r") as f:
            acled_data = json.load(f)
        acled_df = pd.DataFrame(acled_data)
        logger.info(f"Loaded {len(acled_df)} ACLED events")
        
        # Convert date column to datetime if needed
        if 'date' in acled_df.columns and not pd.api.types.is_datetime64_any_dtype(acled_df['date']):
            acled_df['date'] = pd.to_datetime(acled_df['date'], errors='coerce')
        
        has_acled_data = len(acled_df) > 0
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("ACLED data file not found or empty. Proceeding with GDELT data only.")
        has_acled_data = False
        acled_df = pd.DataFrame()
    
    # Load UCDP CSV data if available
    try:
        ucdp_csv_df = pd.read_csv("data/ucdp/csv_processed.csv")
        logger.info(f"Loaded {len(ucdp_csv_df)} UCDP CSV events")
        has_ucdp_csv_data = len(ucdp_csv_df) > 0
    except FileNotFoundError:
        logger.warning("UCDP CSV data file not found. Proceeding without it.")
        has_ucdp_csv_data = False
        ucdp_csv_df = pd.DataFrame()
        
    # Load UCDP API data if available
    try:
        ucdp_api_df = pd.read_csv("data/ucdp_api_clean.csv")
        logger.info(f"Loaded {len(ucdp_api_df)} UCDP API events")
        has_ucdp_api_data = len(ucdp_api_df) > 0
    except FileNotFoundError:
        logger.warning("UCDP API data file not found. Proceeding without it.")
        has_ucdp_api_data = False
        ucdp_api_df = pd.DataFrame()
    
    # Prepare DataFrames to concatenate
    dfs_to_merge = []
    counts = {}
    
    if len(gdelt_df) > 0:
        dfs_to_merge.append(gdelt_df[['text', 'label', 'date']])
        counts["GDELT"] = len(gdelt_df)
    
    if has_acled_data and "text" in acled_df.columns and "label" in acled_df.columns:
        dfs_to_merge.append(acled_df[['text', 'label', 'date']])
        counts["ACLED"] = len(acled_df)
    
    if has_ucdp_csv_data and "text" in ucdp_csv_df.columns and "label" in ucdp_csv_df.columns:
        dfs_to_merge.append(ucdp_csv_df[['text', 'label', 'date']])
        counts["UCDP CSV"] = len(ucdp_csv_df)
    
    if has_ucdp_api_data and "text" in ucdp_api_df.columns and "label" in ucdp_api_df.columns:
        dfs_to_merge.append(ucdp_api_df[['text', 'label', 'date']])
        counts["UCDP API"] = len(ucdp_api_df)
    
    # Combine data sources
    if len(dfs_to_merge) > 0:
        combined_df = pd.concat(dfs_to_merge, ignore_index=True)
        
        # Log data sources
        logger.info(f"Combined data from {', '.join(counts.keys())}")
        for source, count in counts.items():
            logger.info(f" - {source}: {count} events")
        
        logger.info(f"Total combined dataset: {len(combined_df)} samples")
    else:
        # Fall back to GDELT only
        combined_df = gdelt_df[['text', 'label', 'date']]
        logger.info(f"Using GDELT data only: {len(combined_df)} samples")
    
    # Clean the data
    combined_df = combined_df.dropna(subset=['text'])
    logger.info(f"After cleaning: {len(combined_df)} samples")
    
    return combined_df

def compute_metrics(eval_pred: EvalPrediction):
    """
    Compute evaluation metrics for the Hugging Face Trainer.
    
    Args:
        eval_pred: EvalPrediction object containing predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    pred_probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    
    # Calculate metrics
    accuracy = (predictions == labels).mean()
    auc = roc_auc_score(labels, pred_probs)
    
    # Generate classification report
    report = classification_report(labels, predictions, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "roc_auc": auc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig(f"logs/transformer_confusion_matrix_{TIMESTAMP}.png")
    plt.close()

def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    
    # Save the plot
    plt.savefig(f"logs/transformer_roc_curve_{TIMESTAMP}.png")
    plt.close()

def plot_probability_distribution(y_true, y_prob, title="Probability Distribution"):
    """Plot and save probability distribution."""
    plt.figure(figsize=(10, 6))
    
    # Plot probability distribution for positive and negative classes
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.5, label='Non-conflict events')
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.5, label='Conflict events')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    
    # Save the plot
    plt.savefig(f"logs/transformer_probability_distribution_{TIMESTAMP}.png")
    plt.close()

def get_misclassifications(texts, y_true, y_pred, y_prob, n=10):
    """
    Get top misclassifications (false positives and false negatives).
    
    Args:
        texts: List of text examples
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        n: Number of examples to return
        
    Returns:
        Dictionary with false positives and false negatives
    """
    # Get indices of false positives (predicted 1, actually 0)
    fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
    # Sort by prediction confidence (highest first)
    fp_indices = fp_indices[np.argsort(-y_prob[fp_indices])]
    
    # Get indices of false negatives (predicted 0, actually 1)
    fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]
    # Sort by prediction confidence (lowest first)
    fn_indices = fn_indices[np.argsort(y_prob[fn_indices])]
    
    # Limit to n examples
    fp_indices = fp_indices[:n]
    fn_indices = fn_indices[:n]
    
    # Create misclassification examples
    false_positives = []
    false_negatives = []
    
    for idx in fp_indices:
        false_positives.append({
            'text': texts[idx],
            'probability': float(y_prob[idx])
        })
    
    for idx in fn_indices:
        false_negatives.append({
            'text': texts[idx],
            'probability': float(y_prob[idx])
        })
    
    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def augment_text(text, augmentation_strength=0.1):
    """
    Apply text augmentation by randomly replacing words with synonyms
    
    Args:
        text: Text string to augment
        augmentation_strength: Proportion of words to potentially replace [0-1]
        
    Returns:
        Augmented text
    """
    words = text.split()
    num_to_replace = max(1, int(len(words) * augmentation_strength))
    indices = random.sample(range(len(words)), min(num_to_replace, len(words)))
    
    for idx in indices:
        word = words[idx]
        # Skip very short words
        if len(word) <= 3:
            continue
            
        # Get synonyms for the word
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name().replace('_', ' '))
        
        # Replace word with a synonym if any are found
        if synonyms:
            words[idx] = random.choice(synonyms)
    
    return ' '.join(words)

def tokenize_function(examples, tokenizer, max_length=128):
    """
    Tokenize text data for transformer model.
    
    Args:
        examples: Dictionary of examples
        tokenizer: Transformers tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

def train_transformer_model(model_name="distilbert-base-uncased", num_epochs=3, batch_size=16, 
                           run_evaluation=False, time_split=False, learning_rate=5e-5,
                           data_augmentation=False, hyperparameter_tuning=False):
    """
    Train and save a transformer model for conflict prediction.
    
    Args:
        model_name: Name of the pre-trained model to use
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        run_evaluation: Whether to run detailed evaluation and create plots
        time_split: Whether to split train/test by time instead of randomly
        learning_rate: Learning rate for training
        data_augmentation: Whether to augment training data
        hyperparameter_tuning: Whether to use a validation set for tuning
        
    Returns:
        Dictionary with model metadata
    """
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        return None
    
    # Data augmentation
    if data_augmentation:
        # Find conflict examples (for balanced augmentation)
        conflict_examples = df[df["label"] == 1]
        
        # Create augmented examples for conflict class
        augmented_texts = []
        augmented_labels = []
        
        for _, row in conflict_examples.iterrows():
            # Create 1 additional augmented example for each conflict
            augmented_text = augment_text(row["text"])
            augmented_texts.append(augmented_text)
            augmented_labels.append(1)
        
        # Add augmented examples
        if augmented_texts:
            aug_df = pd.DataFrame({
                "text": augmented_texts,
                "label": augmented_labels,
                "date": conflict_examples["date"].iloc[0]  # Use the same date for all
            })
            df = pd.concat([df, aug_df], ignore_index=True)
            logger.info(f"Added {len(augmented_texts)} augmented examples. New dataset size: {len(df)}")
    
    # Split into train, validation (if tuning), and test sets
    if time_split and 'date' in df.columns:
        # Sort by date
        df = df.sort_values('date')
        
        if hyperparameter_tuning:
            # Use oldest 70% for training, next 10% for validation, newest 20% for testing
            train_size = int(0.7 * len(df))
            val_size = int(0.1 * len(df))
            
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:train_size+val_size]
            test_df = df.iloc[train_size+val_size:]
            
            logger.info(f"Split data by time: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
        else:
            # Use oldest 80% for training, newest 20% for testing
            train_size = int(0.8 * len(df))
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            val_df = None
            
            logger.info(f"Split data by time: {len(train_df)} training samples, {len(test_df)} test samples")
    else:
        # Random split
        if hyperparameter_tuning:
            # 70/10/20 split
            train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
            val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42, stratify=temp_df["label"])
            
            logger.info(f"Split data randomly: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
        else:
            # 80/20 split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
            val_df = None
            
            logger.info(f"Split data randomly: {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    if hyperparameter_tuning and val_df is not None:
        val_dataset = Dataset.from_pandas(val_df)
    else:
        val_dataset = None
    
    # Tokenize datasets
    tokenize = lambda examples: tokenize_function(examples, tokenizer)
    
    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)
    
    if val_dataset:
        tokenized_val = val_dataset.map(tokenize, batched=True)
    
    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    # Define output directory
    output_dir = f"models/transformer_conflict_model_{TIMESTAMP}"
    
    # Create evaluation callbacks
    callbacks = []
    if hyperparameter_tuning:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"logs/transformer_training_{TIMESTAMP}",
        logging_steps=100,
        load_best_model_at_end=hyperparameter_tuning,
        evaluation_strategy="epoch" if hyperparameter_tuning else "no",
        save_strategy="epoch" if hyperparameter_tuning else "no",
        report_to="none",
        save_total_limit=2
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val if hyperparameter_tuning else None,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    # Check for previous checkpoint to resume training
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
    
    # Train model
    logger.info(f"Training transformer model ({model_name}) for {num_epochs} epochs...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")
    
    # Evaluate model on test set
    logger.info("Evaluating model on test set...")
    test_results = trainer.predict(tokenized_test)
    
    # Extract predictions and metrics
    predictions = np.argmax(test_results.predictions, axis=-1)
    pred_probs = torch.nn.functional.softmax(torch.tensor(test_results.predictions), dim=-1).numpy()[:, 1]
    metrics = test_results.metrics
    
    logger.info(f"Test metrics: {metrics}")
    
    # Create detailed evaluation if requested
    if run_evaluation:
        logger.info("Running detailed evaluation...")
        
        # Plot confusion matrix
        plot_confusion_matrix(test_dataset["label"], predictions, "Transformer Confusion Matrix")
        
        # Plot ROC curve
        plot_roc_curve(test_dataset["label"], pred_probs, "Transformer ROC Curve")
        
        # Plot probability distribution
        plot_probability_distribution(test_dataset["label"], pred_probs, "Transformer Probability Distribution")
        
        # Get misclassifications
        misclassifications = get_misclassifications(
            test_dataset["text"], test_dataset["label"], predictions, pred_probs
        )
        
        # Log misclassifications
        logger.info("Top False Positives (predicted conflict, actually non-conflict):")
        for i, fp in enumerate(misclassifications['false_positives']):
            logger.info(f"{i+1}. \"{fp['text']}\" (Prob: {fp['probability']:.4f})")
        
        logger.info("Top False Negatives (predicted non-conflict, actually conflict):")
        for i, fn in enumerate(misclassifications['false_negatives']):
            logger.info(f"{i+1}. \"{fn['text']}\" (Prob: {fn['probability']:.4f})")
    else:
        misclassifications = None
    
    # Save metadata
    metadata = {
        "model_type": "transformer",
        "model_name": model_name,
        "timestamp": TIMESTAMP,
        "dataset_size": len(df),
        "training_size": len(train_df),
        "validation_size": len(val_df) if val_df is not None else 0,
        "test_size": len(test_df),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "data_augmentation": data_augmentation,
        "hyperparameter_tuning": hyperparameter_tuning,
        "metrics": {
            "accuracy": float(metrics["test_accuracy"]),
            "roc_auc": float(metrics["test_roc_auc"]),
            "precision": float(metrics["test_precision"]),
            "recall": float(metrics["test_recall"]),
            "f1": float(metrics["test_f1"])
        },
        "data_sources": list(df["source"].unique()) if "source" in df.columns else ["unknown"],
        "misclassifications": misclassifications
    }
    
    metadata_file = f"models/metadata_transformer_{TIMESTAMP}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_file}")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Train transformer model for conflict prediction")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", 
                        help="Model name to use (default: distilbert-base-uncased)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--evaluation", action="store_true", help="Run detailed evaluation and create plots")
    parser.add_argument("--time-split", action="store_true", help="Split train/test by time instead of randomly")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--augment", action="store_true", help="Perform data augmentation")
    parser.add_argument("--tuning", action="store_true", help="Use validation set for hyperparameter tuning")
    args = parser.parse_args()
    
    train_transformer_model(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        run_evaluation=args.evaluation,
        time_split=args.time_split,
        learning_rate=args.learning_rate,
        data_augmentation=args.augment,
        hyperparameter_tuning=args.tuning
    )

if __name__ == "__main__":
    main() 