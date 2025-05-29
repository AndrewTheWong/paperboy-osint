"""
Prediction Visualizer for Goldstein Score Model
- Loads trained model and test data
- Creates scatter plots of predicted vs actual scores
- Generates residual plots and performance metrics
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import os

# === CONFIG ===
DATA_PATH = "data/gdelt_goldstein_dataset.csv"
MODEL_PATH = "models/xgb_goldstein_regressor.pkl"  # Default model
TUNED_MODEL_PATH = "models/xgb_goldstein_tuned.pkl"  # Tuned model
OUTPUT_DIR = "models/visualizations"

print("📊 Loading Goldstein Score Prediction Visualizer...")

# === 1. LOAD DATA ===
df = pd.read_csv(DATA_PATH)
print(f"📈 Loaded {len(df)} rows from GDELT dataset")

features = ['event_date', 'event_code', 'quad_class', 'avg_tone', 'mentions', 'actor_1', 'actor_2', 'country']
target = 'goldstein'

# === 2. RECREATE TEST SPLIT ===
# Use time-based split to match training script
df = df.sort_values('event_date')
total_len = len(df)
train_end = int(0.7 * total_len)
val_end = int(0.85 * total_len)
df_test = df.iloc[val_end:]

X_test = df_test[features]
y_test = df_test[target]
print(f"🧪 Test set: {len(X_test)} rows")

# === 3. LOAD MODEL(S) ===
models_to_test = []

# Try to load tuned model first
if os.path.exists(TUNED_MODEL_PATH):
    tuned_model = joblib.load(TUNED_MODEL_PATH)
    models_to_test.append(("Tuned XGBoost", tuned_model))
    print("✅ Loaded tuned XGBoost model")

# Load default model
if os.path.exists(MODEL_PATH):
    default_model = joblib.load(MODEL_PATH)
    models_to_test.append(("Default XGBoost", default_model))
    print("✅ Loaded default XGBoost model")

if not models_to_test:
    print("❌ No trained models found! Please train a model first.")
    exit(1)

# === 4. CREATE OUTPUT DIRECTORY ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 5. GENERATE PREDICTIONS AND VISUALIZATIONS ===
# Set style - use a simpler style that works across different seaborn versions
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

fig_size = (15, 10)

for model_name, model in models_to_test:
    print(f"\n🎯 Analyzing {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"📊 Performance Metrics:")
    print(f"   MSE: {round(mse, 4)}")
    print(f"   RMSE: {round(rmse, 4)}")
    print(f"   R²: {round(r2, 4)}")
    print(f"   MAE: {round(mae, 4)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle(f'{model_name} - Goldstein Score Predictions', fontsize=16, fontweight='bold')
    
    # 1. Predicted vs Actual Scatter Plot
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=1)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Goldstein Score')
    axes[0, 0].set_ylabel('Predicted Goldstein Score')
    axes[0, 0].set_title('Predicted vs Actual')
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # 2. Residual Plot
    residuals = y_pred - y_test
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Goldstein Score')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    
    # 3. Distribution of Residuals
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].text(0.05, 0.95, f'RMSE = {rmse:.4f}', transform=axes[1, 0].transAxes,
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # 4. Error vs Actual Values
    abs_errors = np.abs(residuals)
    axes[1, 1].scatter(y_test, abs_errors, alpha=0.6, s=1)
    axes[1, 1].set_xlabel('Actual Goldstein Score')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Absolute Error vs Actual')
    axes[1, 1].text(0.05, 0.95, f'MAE = {mae:.4f}', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    model_filename = model_name.lower().replace(' ', '_')
    save_path = os.path.join(OUTPUT_DIR, f'{model_filename}_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to {save_path}")
    
    plt.close()  # Close figure to free memory

# === 6. COMPARISON PLOT (if multiple models) ===
if len(models_to_test) > 1:
    print("\n🔄 Creating model comparison...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, (model_name, model) in enumerate(models_to_test):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        ax.scatter(y_test, y_pred, alpha=0.5, s=1, color=colors[i], 
                  label=f'{model_name} (R² = {r2:.4f})')
    
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Goldstein Score')
    ax.set_ylabel('Predicted Goldstein Score')
    ax.set_title('Model Comparison: Predicted vs Actual Goldstein Scores')
    ax.legend()
    
    comparison_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"📊 Model comparison saved to {comparison_path}")
    
    plt.close()

print("🎉 Visualization complete!") 