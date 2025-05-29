"""
Hyperparameter Tuning for XGBoost Goldstein Regressor
- Uses RandomizedSearchCV to optimize key model parameters
- Trains on 80% of GDELT data, validates on 20%
- Saves best model and hyperparameters
"""

import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
from datetime import datetime

# === CONFIG ===
DATA_PATH = "data/gdelt_goldstein_dataset.csv"
BEST_MODEL_PATH = "models/xgb_goldstein_tuned.pkl"
BEST_PARAMS_PATH = "models/best_hyperparams.json"
N_ITER = 50  # Number of random parameter combinations to try
CV_FOLDS = 3  # Cross-validation folds

print("🔧 Starting XGBoost Hyperparameter Tuning...")

# === 1. LOAD DATA ===
df = pd.read_csv(DATA_PATH)
print(f"📊 Loaded {len(df)} rows from GDELT dataset")

features = ['event_date', 'event_code', 'quad_class', 'avg_tone', 'mentions', 'actor_1', 'actor_2', 'country']
target = 'goldstein'

# === 2. TRAIN/VALIDATION SPLIT (80/20) ===
X = df[features]
y = df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📈 Training set: {len(X_train)} rows, Validation set: {len(X_val)} rows")

# === 3. HYPERPARAMETER SEARCH SPACE ===
param_dist = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0, 2.0]
}

print(f"🎯 Search space: {len(param_dist)} parameters")

# === 4. RANDOMIZED SEARCH ===
xgb_base = XGBRegressor(random_state=42, verbosity=0)
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=N_ITER,
    cv=CV_FOLDS,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print(f"🚀 Running RandomizedSearchCV with {N_ITER} iterations...")
start_time = datetime.now()
random_search.fit(X_train, y_train)
end_time = datetime.now()

print(f"⏱️ Tuning completed in {end_time - start_time}")

# === 5. BEST MODEL EVALUATION ===
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = -random_search.best_score_

print(f"\n🏆 Best CV MSE: {round(best_score, 4)}")
print("🔧 Best Parameters:")
for param, value in best_params.items():
    print(f"   {param}: {value}")

# === 6. VALIDATION SET EVALUATION ===
val_preds = best_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)

print(f"\n📊 Validation Results:")
print(f"   MSE: {round(val_mse, 4)}")
print(f"   R²: {round(val_r2, 4)}")

# === 7. SAVE BEST MODEL AND PARAMETERS ===
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, BEST_MODEL_PATH)
print(f"✅ Best model saved to {BEST_MODEL_PATH}")

# Save hyperparameters with metadata
hyperparams_data = {
    "best_params": best_params,
    "best_cv_mse": best_score,
    "validation_mse": val_mse,
    "validation_r2": val_r2,
    "training_time": str(end_time - start_time),
    "n_iterations": N_ITER,
    "cv_folds": CV_FOLDS,
    "training_samples": len(X_train),
    "validation_samples": len(X_val)
}

with open(BEST_PARAMS_PATH, 'w') as f:
    json.dump(hyperparams_data, f, indent=2)

print(f"📄 Hyperparameters saved to {BEST_PARAMS_PATH}")

# === 8. FEATURE IMPORTANCE ===
importances = best_model.feature_importances_
feature_importance = dict(zip(features, importances))
print(f"\n🌟 Feature Importance (Top 5):")
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features[:5]:
    print(f"   {feature}: {round(importance, 4)}")

print("🎉 Hyperparameter tuning complete!") 