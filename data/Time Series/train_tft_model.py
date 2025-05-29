"""
Script: train_tft_model.py
Purpose: Train a Temporal Fusion Transformer (TFT) model to predict next-day escalation using 
         merged daily features from both GDELT events and SBERT-tagged article clusters.
"""

import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, RMSE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torch
import os

# Set seeds for reproducibility
seed_everything(42)

# Load merged dataset (GDELT + SBERT articles)
df = pd.read_csv("data/aggregated_timeseries.csv", parse_dates=["event_date"])
df = df.sort_values("event_date")

print(f"📊 Raw data shape: {df.shape}")
print(f"📅 Date range: {df['event_date'].min()} to {df['event_date'].max()}")

# Add time index and group id (single series)
df["time_idx"] = (df["event_date"] - df["event_date"].min()).dt.days
df["group"] = "taiwan_china_escalation"  # single group for this time series

# Add temporal features
df["day_of_week"] = df["event_date"].dt.dayofweek
df["week"] = df["event_date"].dt.isocalendar().week.astype(int)
df["month"] = df["event_date"].dt.month
df["quarter"] = df["event_date"].dt.quarter

# Create lagged features for both GDELT and article Goldstein scores
df["gdelt_goldstein_lag1"] = df["avg_goldstein"].shift(1)
df["gdelt_goldstein_lag3"] = df["avg_goldstein"].shift(3)
df["gdelt_goldstein_roll7"] = df["avg_goldstein"].rolling(7).mean()

df["article_goldstein_lag1"] = df["article_avg_goldstein"].shift(1)
df["article_goldstein_lag3"] = df["article_avg_goldstein"].shift(3)
df["article_goldstein_roll7"] = df["article_avg_goldstein"].rolling(7).mean()

# Create interaction features
df["gdelt_article_goldstein_diff"] = df["avg_goldstein"] - df["article_avg_goldstein"]
df["gdelt_article_correlation"] = df["avg_goldstein"] * df["article_avg_goldstein"]

# Add activity ratios
df["gdelt_per_article"] = df["num_gdelt_events"] / (df["num_articles"] + 1)  # +1 to avoid division by zero
df["events_per_cluster"] = df["num_gdelt_events"] / (df["num_clusters"] + 1)

# Drop NaNs from rolling/lag features
df = df.dropna().reset_index(drop=True)

print(f"📊 Data shape after preprocessing: {df.shape}")
print(f"📈 GDELT Goldstein range: {df['avg_goldstein'].min():.3f} to {df['avg_goldstein'].max():.3f}")
print(f"📈 Article Goldstein range: {df['article_avg_goldstein'].min():.3f} to {df['article_avg_goldstein'].max():.3f}")

# Define forecasting parameters
max_encoder_length = 21  # 3 weeks lookback
max_prediction_length = 1  # forecast 1 day ahead

# Split data (80/20 train/val)
training_cutoff = df["time_idx"].max() - max_prediction_length * 5  # Keep last 5 days for validation

print(f"🔀 Training cutoff: {training_cutoff} (total time steps: {df['time_idx'].max()})")

# Define features for the model
time_varying_known_reals = ["time_idx", "day_of_week", "week", "month", "quarter"]

time_varying_unknown_reals = [
    # GDELT features
    "avg_goldstein", "std_goldstein", "num_gdelt_events",
    "gdelt_goldstein_lag1", "gdelt_goldstein_lag3", "gdelt_goldstein_roll7",
    
    # Article features 
    "article_avg_goldstein", "article_std_goldstein", "num_articles",
    "article_goldstein_lag1", "article_goldstein_lag3", "article_goldstein_roll7",
    
    # Cluster features
    "num_clusters", "max_cluster_size", "avg_cluster_size", "pct_noise",
    
    # Interaction features
    "gdelt_article_goldstein_diff", "gdelt_article_correlation",
    "gdelt_per_article", "events_per_cluster"
]

# Create training dataset
training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="avg_goldstein",  # Predict GDELT Goldstein score
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group"],
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    target_normalizer=GroupNormalizer(groups=["group"], transformation=None),
    add_relative_time_idx=True,
    add_target_scales=True,
    allow_missing_timesteps=True,
)

# Create validation dataset
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

print(f"🎯 Training samples: {len(training)}")
print(f"🎯 Validation samples: {len(validation)}")

# Create dataloaders
batch_size = 16  # Small batch size for limited data
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# Setup training callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, mode="min")
lr_monitor = LearningRateMonitor()

# Setup trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    gradient_clip_val=0.1,
    callbacks=[lr_monitor, early_stop],
    enable_model_summary=True,
    logger=False,  # Disable logging for cleaner output
)

# Create TFT model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=3e-3,
    hidden_size=32,  # Increased from 16 for more complex features
    attention_head_size=2,  # Increased attention heads
    dropout=0.1,
    hidden_continuous_size=16,
    loss=MAE(),  # Mean Absolute Error
    log_interval=5,
    reduce_on_plateau_patience=3,
)

print(f"🚀 Starting training...")
print(f"   Model size: {sum(p.numel() for p in tft.parameters())} parameters")
print(f"   Features: {len(time_varying_unknown_reals)} unknown + {len(time_varying_known_reals)} known")

# Train the model
trainer.fit(tft, train_dataloader, val_dataloader)

# Save model
os.makedirs("models", exist_ok=True)
model_path = "models/tft_merged_escalation_model.pt"
tft.save(model_path)
print(f"✅ TFT model saved to {model_path}")

# Make predictions on validation set
predictions, x = tft.predict(val_dataloader, return_x=True)

# Calculate and display metrics
mae = MAE()
rmse = RMSE()

val_mae = mae(predictions, x["decoder_target"])
val_rmse = rmse(predictions, x["decoder_target"])

print(f"\n📊 Validation Metrics:")
print(f"   MAE: {val_mae:.4f}")
print(f"   RMSE: {val_rmse:.4f}")

# Feature importance (if available)
try:
    interpretation = tft.interpret_output(val_dataloader, return_x=True)
    feature_importance = interpretation["attention"].cpu().numpy().mean(axis=(0, 1))
    
    print(f"\n🎯 Top Feature Importances:")
    feature_names = time_varying_unknown_reals
    importance_pairs = list(zip(feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(importance_pairs[:10]):
        print(f"   {i+1:2d}. {feature:30s}: {importance:.4f}")
        
except Exception as e:
    print(f"⚠️  Could not compute feature importance: {e}")

print("\n🎉 Training completed successfully!") 