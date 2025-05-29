#!/usr/bin/env python3
"""
Baseline Model A - XGBoost Classifier for Escalation Detection

This script trains the initial classifier for articles to determine if they are escalatory or not escalatory.
It uses existing conflict data from GDELT, ACLED, and UCDP sources, plus peaceful events data.

Features:
1. Loads structured escalation event data from multiple sources
2. Harmonizes into a unified feature schema using text embeddings
3. Trains an XGBoost classifier on labeled escalation vs. non-escalation events
4. Saves the model and outputs comprehensive evaluation metrics
5. Includes automated testing and validation

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import pickle
from datetime import datetime
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ModelA:
    """
    Model A - XGBoost Escalation Classifier
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize Model A
        
        Args:
            model_name: Sentence transformer model name
            test_size: Test set proportion
            random_state: Random state for reproducibility
        """
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.encoder_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and combine all available data sources
        
        Returns:
            Combined DataFrame with features and labels
        """
        logger.info("Loading data from all sources...")
        
        dataframes = []
        
        # Load conflict data
        conflict_files = [
            "data/all_conflict_data.csv",
            "data/gdelt_events.csv",
            "data/acled_events_clean.csv"
        ]
        
        for file_path in conflict_files:
            if os.path.exists(file_path):
                logger.info(f"Loading {file_path}...")
                df = pd.read_csv(file_path)
                
                # Standardize column names
                if 'text' in df.columns and 'label' in df.columns:
                    df_subset = df[['text', 'label']].copy()
                    df_subset['source'] = os.path.basename(file_path).replace('.csv', '')
                    dataframes.append(df_subset)
                    logger.info(f"Loaded {len(df)} samples from {file_path}")
                else:
                    logger.warning(f"Skipping {file_path} - missing required columns")
        
        # Load peaceful events
        peaceful_file = "data/all_peaceful_events.csv"
        if os.path.exists(peaceful_file):
            logger.info(f"Loading {peaceful_file}...")
            df = pd.read_csv(peaceful_file)
            if 'text' in df.columns and 'label' in df.columns:
                df_subset = df[['text', 'label']].copy()
                df_subset['source'] = 'peaceful_events'
                dataframes.append(df_subset)
                logger.info(f"Loaded {len(df)} peaceful samples")
        
        if not dataframes:
            raise FileNotFoundError("No valid data files found. Please ensure data files exist in the data/ directory.")
        
        # Combine all data
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Clean and validate data
        combined_df = combined_df.dropna(subset=['text', 'label'])
        combined_df['label'] = combined_df['label'].astype(int)
        
        # Check if we have any data after cleaning
        if len(combined_df) == 0:
            raise FileNotFoundError("No valid data found after cleaning. All data files appear to be empty or invalid.")
        
        # Ensure binary labels
        unique_labels = combined_df['label'].unique()
        if not all(label in [0, 1] for label in unique_labels):
            logger.warning(f"Found non-binary labels: {unique_labels}. Converting to binary.")
            combined_df['label'] = (combined_df['label'] > 0).astype(int)
        
        logger.info(f"Combined dataset: {len(combined_df)} total samples")
        logger.info(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
        
        return combined_df
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features from text using sentence embeddings and additional features
        
        Args:
            df: Input DataFrame with text and labels
            
        Returns:
            Feature matrix X and labels y
        """
        logger.info("Creating features from text data...")
        
        # Load sentence transformer
        logger.info(f"Loading sentence transformer: {self.model_name}")
        self.encoder_model = SentenceTransformer(self.model_name)
        
        # Generate text embeddings
        texts = df['text'].tolist()
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.encoder_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Create additional features
        additional_features = []
        
        # Text length features
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        # Source encoding
        if 'source' in df.columns:
            le_source = LabelEncoder()
            df['source_encoded'] = le_source.fit_transform(df['source'].fillna('unknown'))
            self.label_encoders['source'] = le_source
            additional_features.extend(['text_length', 'word_count', 'source_encoded'])
        else:
            additional_features.extend(['text_length', 'word_count'])
        
        # Combine embeddings with additional features
        if additional_features:
            extra_features = df[additional_features].values
            extra_features = self.scaler.fit_transform(extra_features)
            X = np.hstack([embeddings, extra_features])
        else:
            X = embeddings
        
        y = df['label'].values
        
        logger.info(f"Created feature matrix: {X.shape}")
        logger.info(f"Feature breakdown: {embeddings.shape[1]} embeddings + {len(additional_features)} additional features")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train XGBoost model with optimized hyperparameters
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        logger.info("Training XGBoost Model A...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Training label distribution: {np.bincount(y_train)}")
        logger.info(f"Test label distribution: {np.bincount(y_test)}")
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        
        # Initialize XGBoost with optimized parameters
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train model
        logger.info("Fitting XGBoost model...")
        self.model.fit(X_train, y_train)
        
        # Cross-validation score (handle small datasets)
        logger.info("Performing cross-validation...")
        try:
            # Determine appropriate CV folds based on data size
            min_class_count = min(np.bincount(y_train))
            cv_folds = min(5, min_class_count)  # Reduce folds if not enough samples
            
            if cv_folds >= 2:
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
                logger.info(f"Cross-validation AUC ({cv_folds}-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            else:
                logger.warning("Not enough samples for cross-validation. Skipping CV.")
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}. Continuing without CV.")
        
        logger.info("Model training completed!")
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate model performance on test set
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating Model A performance...")
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='binary'),
            'recall': recall_score(self.y_test, y_pred, average='binary'),
            'f1_score': f1_score(self.y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(self.y_test, y_prob)
        }
        
        self.metrics = metrics
        
        # Log detailed results
        logger.info("=" * 60)
        logger.info("MODEL A EVALUATION METRICS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info("=" * 60)
        
        # Detailed classification report
        logger.info("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Non-Escalatory', 'Escalatory']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        logger.info(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        return metrics
    
    def save_model(self, model_dir: str = "models") -> None:
        """
        Save trained model and components
        
        Args:
            model_dir: Directory to save model files
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save main model
        model_path = os.path.join(model_dir, "xgb_model_a.pkl")
        logger.info(f"Saving XGBoost model to: {model_path}")
        joblib.dump(self.model, model_path)
        
        # Save complete model object (including encoders, scaler, etc.)
        # But avoid pickling mock objects during testing
        full_model_path = os.path.join(model_dir, "model_a_complete.pkl")
        logger.info(f"Saving complete Model A to: {full_model_path}")
        
        try:
            with open(full_model_path, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            logger.warning(f"Could not pickle complete model (likely due to mocks in testing): {e}")
            # Save essential components separately
            model_components = {
                'model_name': self.model_name,
                'test_size': self.test_size,
                'random_state': self.random_state,
                'label_encoders': self.label_encoders,
                'metrics': self.metrics,
                'feature_columns': self.feature_columns
            }
            with open(full_model_path, 'wb') as f:
                pickle.dump(model_components, f)
        
        # Save metrics
        metrics_path = os.path.join(model_dir, "model_a_metrics.txt")
        logger.info(f"Saving metrics to: {metrics_path}")
        
        with open(metrics_path, 'w') as f:
            f.write("Model A - XGBoost Escalation Classifier - Evaluation Metrics\n")
            f.write("=" * 70 + "\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: XGBoost Classifier\n")
            f.write(f"Text Encoder: Sentence-BERT ({self.model_name})\n")
            f.write("-" * 70 + "\n")
            if self.metrics:
                for metric, value in self.metrics.items():
                    f.write(f"{metric.title()}: {value:.4f}\n")
            f.write("-" * 70 + "\n")
            f.write("Data Sources:\n")
            f.write("- ACLED conflict events\n")
            f.write("- GDELT conflict events\n") 
            f.write("- UCDP conflict events\n")
            f.write("- Peaceful events dataset\n")
            f.write("-" * 70 + "\n")
            f.write("Model Configuration:\n")
            f.write(f"- Train/Test Split: {int((1-self.test_size)*100)}/{int(self.test_size*100)}\n")
            f.write("- Stratified sampling\n")
            f.write(f"- Random State: {self.random_state}\n")
            f.write("- Cross-validation: Adaptive (2-5 fold)\n")
        
        logger.info("✅ Model A saved successfully!")
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None or self.encoder_model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Create features for new texts
        embeddings = self.encoder_model.encode(texts, convert_to_numpy=True)
        
        # Add additional features
        df_temp = pd.DataFrame({'text': texts})
        df_temp['text_length'] = df_temp['text'].str.len()
        df_temp['word_count'] = df_temp['text'].str.split().str.len()
        
        additional_features = [df_temp['text_length'].values, df_temp['word_count'].values]
        
        # Add source encoding if available
        if 'source' in self.label_encoders:
            # Use unknown source for new predictions
            source_encoded = np.zeros(len(texts))  # Default to first encoded value
            additional_features.append(source_encoded)
        
        if additional_features:
            extra_features = np.column_stack(additional_features)
            extra_features = self.scaler.transform(extra_features)
            X = np.hstack([embeddings, extra_features])
        else:
            X = embeddings
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities


def run_tests():
    """
    Run basic tests on Model A
    """
    logger.info("Running Model A tests...")
    
    # Test data loading
    model_a = ModelA()
    
    try:
        df = model_a.load_data()
        assert len(df) > 0, "No data loaded"
        assert 'text' in df.columns, "Missing text column"
        assert 'label' in df.columns, "Missing label column"
        logger.info("✅ Data loading test passed")
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False
    
    # Test feature creation
    try:
        X, y = model_a.create_features(df.head(10))  # Test with small sample
        assert X.shape[0] == 10, "Feature matrix size mismatch"
        assert len(y) == 10, "Label vector size mismatch"
        logger.info("✅ Feature creation test passed")
    except Exception as e:
        logger.error(f"❌ Feature creation test failed: {e}")
        return False
    
    # Test prediction functionality
    try:
        # Create a balanced sample for testing
        if len(df) > 100:
            # Get balanced sample
            conflict_samples = df[df['label'] == 1].head(50)
            peaceful_samples = df[df['label'] == 0].head(50)
            
            if len(conflict_samples) > 0 and len(peaceful_samples) > 0:
                test_df = pd.concat([conflict_samples, peaceful_samples], ignore_index=True)
                X_small, y_small = model_a.create_features(test_df)
            else:
                # Fall back to available data
                X_small, y_small = model_a.create_features(df.head(min(100, len(df))))
        else:
            X_small, y_small = model_a.create_features(df)
        
        # Check if we have both classes
        unique_labels = np.unique(y_small)
        if len(unique_labels) < 2:
            logger.warning("Only one class available in test data. Skipping prediction test.")
            logger.info("🎉 All available Model A tests passed!")
            return True
        
        model_a.train_model(X_small, y_small)
        
        # Test prediction
        test_texts = ["Peaceful diplomatic meeting between countries", "Armed conflict in region"]
        predictions, probabilities = model_a.predict(test_texts)
        assert len(predictions) == 2, "Prediction count mismatch"
        assert len(probabilities) == 2, "Probability count mismatch"
        logger.info("✅ Prediction test passed")
    except Exception as e:
        logger.error(f"❌ Prediction test failed: {e}")
        return False
    
    logger.info("🎉 All Model A tests passed!")
    return True


def main():
    """
    Main training pipeline for Model A
    """
    logger.info("Starting Model A Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize Model A
        model_a = ModelA()
        
        # Load and prepare data
        df = model_a.load_data()
        X, y = model_a.create_features(df)
        
        # Train model
        model_a.train_model(X, y)
        
        # Evaluate model
        metrics = model_a.evaluate_model()
        
        # Save model
        model_a.save_model()
        
        logger.info("=" * 60)
        logger.info("🎉 Model A training completed successfully!")
        logger.info("=" * 60)
        
        return model_a, metrics
        
    except Exception as e:
        logger.error(f"❌ Model A training failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run tests first
    if run_tests():
        # Run main training
        main()
    else:
        logger.error("Tests failed. Please fix issues before training.")
        sys.exit(1) 