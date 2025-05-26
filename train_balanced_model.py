import os
import logging
import argparse
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/balanced_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare peaceful data and train an XGBoost model with balanced dataset")
    parser.add_argument("--force", action="store_true", help="Force regeneration of peaceful events data")
    parser.add_argument("--tuning", action="store_true", help="Perform hyperparameter tuning", default=True)
    parser.add_argument("--cross-validation", action="store_true", help="Perform cross-validation", default=True)
    parser.add_argument("--calibration", action="store_true", help="Calibrate predicted probabilities", default=True)
    parser.add_argument("--compare", action="store_true", help="Compare with Random Forest baseline")
    args = parser.parse_args()
    
    # Step 1: Prepare balanced dataset with peaceful events
    logger.info("Step 1: Preparing balanced dataset with peaceful events")
    try:
        from fetch_peaceful_events import prepare_balanced_model_data
        X, y, texts = prepare_balanced_model_data(force=args.force)
        logger.info(f"Successfully prepared balanced dataset: {len(X)} samples")
        
        # Log class balance
        positive = sum(y)
        negative = len(y) - positive
        logger.info(f"Class balance: {positive} conflict events ({positive/len(y)*100:.1f}%), {negative} peaceful events ({negative/len(y)*100:.1f}%)")
    except Exception as e:
        logger.error(f"Error preparing balanced dataset: {e}")
        logger.error("Cannot proceed without balanced dataset")
        sys.exit(1)
    
    # Step 2: Train XGBoost model with the balanced dataset
    logger.info("Step 2: Training XGBoost model with balanced dataset")
    try:
        from train_xgboost_model import train_xgboost_model
        
        metadata = train_xgboost_model(
            hyperparameter_tuning=args.tuning,
            do_cross_validation=args.cross_validation,
            use_calibration=False,
            compare_to_rf=args.compare,
            use_balanced_data=True
        )
        
        if metadata:
            logger.info(f"Model training completed successfully with accuracy: {metadata['metrics']['accuracy']:.4f}")
            logger.info(f"Model F1 score: {metadata['metrics']['f1']:.4f}")
            logger.info(f"Model ROC AUC: {metadata['metrics']['roc_auc']:.4f}")
        else:
            logger.error("Model training failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        sys.exit(1)
    
    logger.info("Balanced model training pipeline completed successfully")

if __name__ == "__main__":
    main() 