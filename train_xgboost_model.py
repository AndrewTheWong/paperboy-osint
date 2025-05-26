import os
import pandas as pd
import numpy as np
import json
import joblib
import datetime
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix, 
                            roc_curve, precision_recall_curve, auc, accuracy_score, 
                            f1_score, precision_score, recall_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from imblearn.over_sampling import RandomOverSampler
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/xgboost_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Generate timestamp for versioning
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d")

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs(f"logs/xgb_eval_{TIMESTAMP}", exist_ok=True)

# Import tag vocabulary from fetch_and_clean_data
try:
    from fetch_and_clean_data import TAG_VOCAB
    logger.info(f"Imported {len(TAG_VOCAB)} tags from fetch_and_clean_data")
except ImportError:
    # Fallback if import fails
    logger.warning("Could not import TAG_VOCAB, using default tags")
    TAG_VOCAB = [
        "military movement", "armed clash", "ceasefire", "civil war",
        "diplomatic meeting", "protest", "coup", "airstrike",
        "terror attack", "cross-border raid", "shelling", "civilian deaths",
        "ethnic violence", "insurgency", "mass killing"
    ]

def load_model_ready_data(use_balanced_data=True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load model-ready data from saved numpy arrays and pickle.
    
    Args:
        use_balanced_data: Whether to preferentially load balanced dataset with peaceful events
        
    Returns:
        X features, y labels, and original texts
    """
    if use_balanced_data:
        try:
            # First try to load the balanced data (with peaceful events)
            X = np.load("data/model_ready/X_features_balanced.npy")
            y = np.load("data/model_ready/y_labels_balanced.npy")
            
            with open("data/model_ready/texts_balanced.pkl", "rb") as f:
                texts = pickle.load(f)
                
            logger.info(f"Loaded balanced model-ready data: X.shape={X.shape}, y.shape={y.shape}, {len(texts)} texts")
            
            # Report class balance
            positive = np.sum(y)
            total = len(y)
            logger.info(f"Class balance: {positive}/{total} positive examples ({positive/total*100:.1f}%)")
            
            return X, y, texts
        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Could not load balanced model-ready data: {e}")
            logger.info("Falling back to standard model-ready data")
    
    try:
        # Try to load the standard model-ready data
        X = np.load("data/model_ready/X_features.npy")
        y = np.load("data/model_ready/y_labels.npy")
        
        with open("data/model_ready/texts.pkl", "rb") as f:
            texts = pickle.load(f)
            
        logger.info(f"Loaded standard model-ready data: X.shape={X.shape}, y.shape={y.shape}, {len(texts)} texts")
        return X, y, texts
    except (FileNotFoundError, IOError) as e:
        logger.warning(f"Could not load model-ready data: {e}")
        logger.info("Falling back to loading and processing from CSV")
        return load_datasets(use_balanced_data)

def load_datasets(use_balanced_data=True):
    """
    Load all datasets and process them for model training.
    
    Args:
        use_balanced_data: Whether to preferentially load balanced dataset with peaceful events
        
    Returns:
        Combined DataFrame with the dataset
    """
    logger.info("Loading data from all available sources")
    
    # Try to load balanced dataset if requested
    if use_balanced_data:
        try:
            balanced_df = pd.read_csv("data/balanced_conflict_data.csv")
            logger.info(f"Loaded balanced dataset: {len(balanced_df)} events")
            
            # Extract features and labels
            if "tags" in balanced_df.columns:
                # Convert string representation of list to actual list
                if isinstance(balanced_df["tags"].iloc[0], str):
                    balanced_df["tags"] = balanced_df["tags"].apply(eval)
                
                # Create feature matrix
                X = np.hstack([
                    np.array(balanced_df["tags"].tolist()),
                    balanced_df["confidence"].values.reshape(-1, 1)
                ])
                
                # Extract labels
                y = balanced_df["label"].values
                
                # Extract texts
                texts = balanced_df["text"].tolist()
                
                # Report class balance
                positive = np.sum(y)
                total = len(y)
                logger.info(f"Class balance: {positive}/{total} positive examples ({positive/total*100:.1f}%)")
                
                return X, y, texts
            else:
                logger.warning("Balanced dataset does not contain 'tags' column")
        except FileNotFoundError:
            logger.warning("Balanced dataset file not found")
    
    # Load GDELT data
    try:
        gdelt_df = pd.read_csv("data/gdelt_events.csv")
        logger.info(f"Loaded GDELT dataset: {len(gdelt_df)} events")
    except FileNotFoundError:
        logger.warning("GDELT data file not found")
        gdelt_df = pd.DataFrame()
    
    # Load ACLED data
    try:
        acled_file = "data/acled_events_clean.csv"
        if not os.path.exists(acled_file):
            # Try to load from JSON if CSV doesn't exist
            with open("data/acled_events.json", "r") as f:
                acled_data = json.load(f)
            acled_df = pd.DataFrame(acled_data) if acled_data else pd.DataFrame()
        else:
            acled_df = pd.read_csv(acled_file)
        logger.info(f"Loaded ACLED dataset: {len(acled_df)} events")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load ACLED data: {e}")
        acled_df = pd.DataFrame()
    
    # Load UCDP data
    try:
        ucdp_df = pd.read_csv("data/ucdp_api_clean.csv")
        logger.info(f"Loaded UCDP API dataset: {len(ucdp_df)} events")
    except FileNotFoundError:
        logger.warning("UCDP API data file not found")
        # Try UCDP CSV data
        try:
            ucdp_df = pd.read_csv("data/ucdp_csv_clean.csv")
            logger.info(f"Loaded UCDP CSV dataset: {len(ucdp_df)} events")
        except FileNotFoundError:
            logger.warning("UCDP CSV data file not found")
            ucdp_df = pd.DataFrame()
    
    # Try loading the combined file if available
    try:
        combined_df = pd.read_csv("data/all_conflict_data.csv")
        logger.info(f"Loaded combined dataset: {len(combined_df)} events")
        
        # Extract features and labels
        if "tags" in combined_df.columns:
            # Convert string representation of list to actual list
            if isinstance(combined_df["tags"].iloc[0], str):
                combined_df["tags"] = combined_df["tags"].apply(eval)
            
            # Create feature matrix
            X = np.hstack([
                np.array(combined_df["tags"].tolist()),
                combined_df["confidence"].values.reshape(-1, 1)
            ])
            
            # Extract labels
            y = combined_df["label"].values
            
            # Extract texts
            texts = combined_df["text"].tolist()
            
            return X, y, texts
    except FileNotFoundError:
        logger.warning("Combined data file not found, generating features manually")
    
    # Fallback: generate features manually
    from fetch_and_clean_data import prepare_model_data, create_unified_dataset, preprocess_gdelt, preprocess_acled, preprocess_ucdp
    
    # Preprocess each dataset
    logger.info("Preprocessing datasets manually...")
    gdelt_processed = preprocess_gdelt(gdelt_df)
    acled_processed = preprocess_acled(acled_df)
    ucdp_processed = preprocess_ucdp(ucdp_df)
    
    # Create unified dataset
    combined_df = create_unified_dataset(gdelt_processed, acled_processed, ucdp_processed)
    
    # Try to add peaceful events if using balanced data
    if use_balanced_data:
        try:
            from fetch_peaceful_events import create_peaceful_dataset, combine_with_conflict_data
            logger.info("Attempting to add peaceful events to dataset...")
            peaceful_df = create_peaceful_dataset()
            combined_df = combine_with_conflict_data(peaceful_df, force=True)
        except ImportError:
            logger.warning("Could not import peaceful events module, using only conflict data")
    
    # Prepare model data
    X, y, texts = prepare_model_data(combined_df)
    
    return X, y, texts

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-Conflict', 'Conflict'],
               yticklabels=['Non-Conflict', 'Conflict'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig(f"logs/xgb_eval_{TIMESTAMP}/confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    
    # Save the plot
    plt.savefig(f"logs/xgb_eval_{TIMESTAMP}/roc_curve.png")
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve"):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    
    # Save the plot
    plt.savefig(f"logs/xgb_eval_{TIMESTAMP}/precision_recall_curve.png")
    plt.close()

def plot_calibration_curve(y_true, y_prob, title="Calibration Curve"):
    """Plot and save calibration curve."""
    plt.figure(figsize=(8, 6))
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(title)
    plt.legend(loc='lower right')
    
    # Save the plot
    plt.savefig(f"logs/xgb_eval_{TIMESTAMP}/calibration_curve.png")
    plt.close()

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Plot and save feature importance."""
    # Get feature importance
    importance = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feat_imp.head(20))
    plt.title(title)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"logs/xgb_eval_{TIMESTAMP}/feature_importance.png")
    plt.close()
    
    return feat_imp

def get_misclassifications(y_true, y_pred, y_prob, text_data, n=10):
    """
    Get top misclassifications (false positives and false negatives).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        text_data: Original text data
        n: Number of examples to return
        
    Returns:
        Dictionary with false positives and false negatives
    """
    # Make sure text_data has the same length as y_true
    if len(text_data) != len(y_true):
        logger.warning(f"Text data length ({len(text_data)}) doesn't match labels length ({len(y_true)})")
        text_data = text_data[:len(y_true)] if len(text_data) > len(y_true) else text_data + [""] * (len(y_true) - len(text_data))
    
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
            'text': text_data[idx],
            'probability': float(y_prob[idx])
        })
    
    for idx in fn_indices:
        false_negatives.append({
            'text': text_data[idx],
            'probability': float(y_prob[idx])
        })
    
    # Log the misclassifications
    logger.info("Top False Positives (predicted conflict, actually non-conflict):")
    for i, fp in enumerate(false_positives):
        logger.info(f"{i+1}. \"{fp['text']}\" (Prob: {fp['probability']:.4f})")
    
    logger.info("Top False Negatives (predicted non-conflict, actually conflict):")
    for i, fn in enumerate(false_negatives):
        logger.info(f"{i+1}. \"{fn['text']}\" (Prob: {fn['probability']:.4f})")
    
    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def save_metrics(metrics, filename="metrics.json"):
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filename: Output filename
    """
    # Convert numpy values to Python types for JSON serialization
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
            serializable_metrics[k] = float(v)
        else:
            serializable_metrics[k] = v
    
    # Save to JSON
    output_path = f"logs/xgb_eval_{TIMESTAMP}/{filename}"
    with open(output_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {output_path}")

def train_xgboost_model(hyperparameter_tuning=True, do_cross_validation=True, use_calibration=True, compare_to_rf=False, use_balanced_data=True):
    """
    Train and save an XGBoost model for conflict prediction with improved techniques.
    
    Args:
        hyperparameter_tuning: Whether to perform hyperparameter tuning using GridSearchCV
        do_cross_validation: Whether to perform cross-validation
        use_calibration: Whether to calibrate predicted probabilities
        compare_to_rf: Whether to compare with Random Forest baseline
        use_balanced_data: Whether to use balanced dataset with peaceful events
        
    Returns:
        Dictionary with model metadata
    """
    logger.info("Loading and preparing data for model training...")
    
    # Load model-ready data
    X, y, text_data = load_model_ready_data(use_balanced_data=use_balanced_data)
    
    if X.size == 0 or y.size == 0:
        logger.error("No data available for training")
        return None
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Split data into train and test sets (stratified)
    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, y, text_data, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Apply oversampling to the training data
    logger.info("Applying random oversampling to balance classes...")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    logger.info(f"After oversampling: {len(X_train_resampled)} training samples")
    logger.info(f"Class distribution after oversampling: {np.bincount(y_train_resampled)}")
    
    # Define feature names (for interpretation)
    feature_names = TAG_VOCAB + ["confidence_score"]
    
    # Create and train model with hyperparameter tuning
    if hyperparameter_tuning:
        logger.info("Performing hyperparameter tuning with GridSearchCV...")
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0]
        }
        
        grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid,
            scoring="f1",
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train_resampled, y_train_resampled)
        logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        
        # Use the best estimator
        gbm = grid_search.best_estimator_
    else:
        # Train with default hyperparameters
        gbm = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        gbm.fit(X_train_resampled, y_train_resampled)
    
    # Compare to Random Forest baseline if requested
    if compare_to_rf:
        logger.info("Training Random Forest for comparison...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate on test set
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_roc_auc = roc_auc_score(y_test, rf_prob)
        rf_f1 = f1_score(y_test, rf_pred)
        
        logger.info(f"Random Forest - Test accuracy: {rf_accuracy:.4f}")
        logger.info(f"Random Forest - Test ROC AUC: {rf_roc_auc:.4f}")
        logger.info(f"Random Forest - Test F1: {rf_f1:.4f}")
    
    # Perform cross-validation if requested
    if do_cross_validation:
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(gbm, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc')
        logger.info(f"Cross-validation ROC AUC scores: {cv_scores}")
        logger.info(f"CV ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Calibrate probabilities if requested
    if use_calibration:
        logger.info("Calibrating predicted probabilities...")
        calibrated_model = CalibratedClassifierCV(gbm, method="sigmoid", cv=3)
        calibrated_model.fit(X_train_resampled, y_train_resampled)
        model_for_prediction = calibrated_model
        
        # Save calibrated model
        joblib.dump(calibrated_model, f"models/calibrated_xgboost_model_{TIMESTAMP}.pkl")
    else:
        model_for_prediction = gbm
    
    # Evaluate model on test set
    y_pred = model_for_prediction.predict(X_test)
    y_prob = model_for_prediction.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info(f"Test ROC AUC: {roc_auc:.4f}")
    logger.info(f"Test precision: {precision:.4f}")
    logger.info(f"Test recall: {recall:.4f}")
    logger.info(f"Test F1 score: {f1:.4f}")
    
    # Create detailed classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Get feature importances (from base model if calibrated)
    if use_calibration:
        try:
            base_model = calibrated_model.base_estimator
        except AttributeError:
            # For scikit-learn 1.0+
            try:
                logger.info("Trying to access base estimator through estimators_[0]")
                base_model = calibrated_model.estimators_[0]
            except (AttributeError, IndexError) as e:
                logger.warning(f"Could not access base estimator: {e}, using original GBM")
                base_model = gbm
    else:
        base_model = gbm
    
    # Create feature importance dictionary
    feature_importance = dict(zip(feature_names, base_model.feature_importances_))
    
    # Save feature importance to JSON
    with open(f"models/xgb_feature_importance_{TIMESTAMP}.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    
    # Create metrics dictionary for saving
    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": class_report,
        "timestamp": TIMESTAMP,
        "model_type": "xgboost" + ("_calibrated" if use_calibration else ""),
        "feature_importance": feature_importance,
        "training_samples": len(X_train_resampled),
        "test_samples": len(X_test),
        "class_distribution": {
            "training": {
                "before_oversampling": {
                    "conflict": int(y_train.sum()),
                    "non_conflict": int(len(y_train) - y_train.sum())
                },
                "after_oversampling": {
                    "conflict": int(y_train_resampled.sum()),
                    "non_conflict": int(len(y_train_resampled) - y_train_resampled.sum())
                }
            },
            "test": {
                "conflict": int(y_test.sum()),
                "non_conflict": int(len(y_test) - y_test.sum())
            }
        }
    }
    
    # Add cross-validation scores if available
    if do_cross_validation:
        metrics["cross_validation"] = {
            "mean_roc_auc": float(cv_scores.mean()),
            "std_roc_auc": float(cv_scores.std()),
            "scores": [float(score) for score in cv_scores]
        }
    
    # Add random forest comparison if available
    if compare_to_rf:
        metrics["random_forest_comparison"] = {
            "accuracy": float(rf_accuracy),
            "roc_auc": float(rf_roc_auc),
            "f1": float(rf_f1)
        }
    
    # Save metrics to JSON
    save_metrics(metrics)
    
    # Create visualization and detailed evaluation
    logger.info("Creating evaluation visualizations...")
    
    # Create visualizations
    plot_confusion_matrix(y_test, y_pred, "XGBoost Confusion Matrix")
    plot_roc_curve(y_test, y_prob, "XGBoost ROC Curve")
    plot_precision_recall_curve(y_test, y_prob, "XGBoost Precision-Recall Curve")
    plot_calibration_curve(y_test, y_prob, "XGBoost Calibration Curve")
    
    # Plot feature importance
    feature_importance_df = plot_feature_importance(base_model, feature_names, "XGBoost Feature Importance")
    
    # Get misclassifications
    misclassifications = get_misclassifications(y_test, y_pred, y_prob, text_test)
    
    # Save misclassifications
    with open(f"logs/xgb_eval_{TIMESTAMP}/misclassifications.json", "w") as f:
        json.dump(misclassifications, f, indent=2)
    
    # Save feature importance
    feature_importance_df.to_csv(f"logs/xgb_eval_{TIMESTAMP}/feature_importance.csv", index=False)
    
    # Save base model
    model_file = f"models/xgboost_conflict_model_{TIMESTAMP}.pkl"
    joblib.dump(gbm, model_file)
    logger.info(f"Base model saved to {model_file}")
    
    # Log feature importance
    logger.info("Feature importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: -x[1])[:15]:
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Save metadata
    metadata = {
        "model_type": "xgboost" + ("_calibrated" if use_calibration else ""),
        "timestamp": TIMESTAMP,
        "dataset_size": len(X),
        "training_size": len(X_train_resampled),
        "test_size": len(X_test),
        "metrics": {
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        },
        "feature_importance": feature_importance
    }
    
    metadata_file = f"models/metadata_xgboost_{TIMESTAMP}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_file}")
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model for conflict prediction")
    parser.add_argument("--tuning", action="store_true", help="Perform hyperparameter tuning", default=True)
    parser.add_argument("--cross-validation", action="store_true", help="Perform cross-validation", default=True)
    parser.add_argument("--calibration", action="store_true", help="Calibrate predicted probabilities", default=True)
    parser.add_argument("--compare", action="store_true", help="Compare with Random Forest baseline")
    parser.add_argument("--use-balanced-data", action="store_true", help="Use balanced dataset with peaceful events", default=True)
    args = parser.parse_args()
    
    train_xgboost_model(
        hyperparameter_tuning=args.tuning,
        do_cross_validation=args.cross_validation,
        use_calibration=args.calibration,
        compare_to_rf=args.compare,
        use_balanced_data=args.use_balanced_data
    )

if __name__ == "__main__":
    main() 