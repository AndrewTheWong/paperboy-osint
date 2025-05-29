"""
Script: train_tft_gdelt.py
Purpose: Train a TFT model using GDELT-only daily aggregates to predict next-day escalation (Goldstein score).
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
import torch

# Set seeds for reproducibility
seed_everything(42)

# Load GDELT-only aggregated data
df = pd.read_csv("data/aggregated_timeseries.csv", parse_dates=["event_date"])
df = df.sort_values("event_date")

# Create time index and static group ID
df["time_idx"] = (df["event_date"] - df["event_date"].min()).dt.days
df["group"] = "gdelt_only"  # single group id

# Add temporal features
df["day_of_week"] = df["event_date"].dt.dayofweek
df["week"] = df["event_date"].dt.isocalendar().week.astype(int)
df["month"] = df["event_date"].dt.month

# Rolling features for lag modeling
df["goldstein_lag1"] = df["avg_goldstein"].shift(1)
df["goldstein_lag3"] = df["avg_goldstein"].shift(3)
df["goldstein_roll3"] = df["avg_goldstein"].rolling(3).mean()

# Drop NaNs from lags
df = df.dropna().reset_index(drop=True)

print(f"📊 Data shape after preprocessing: {df.shape}")
print(f"📅 Date range: {df['event_date'].min()} to {df['event_date'].max()}")
print(f"📈 Goldstein range: {df['avg_goldstein'].min():.3f} to {df['avg_goldstein'].max():.3f}")

# TFT config
max_encoder_length = 14
max_prediction_length = 1

# Split data
training_cutoff = df["time_idx"].max() - max_prediction_length
print(f"🔀 Training cutoff: {training_cutoff} (total time steps: {df['time_idx'].max()})")

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="avg_goldstein",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group"],
    time_varying_known_reals=["time_idx", "day_of_week", "week", "month"],
    time_varying_unknown_reals=[
        "avg_goldstein", "goldstein_lag1", "goldstein_lag3", "goldstein_roll3",
        "num_articles", "num_clusters", "max_cluster_size", "avg_cluster_size", "pct_noise"
    ],
    target_normalizer=GroupNormalizer(groups=["group"], transformation=None),
    add_relative_time_idx=True,
    add_target_scales=True,
)

# Validation set
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

print(f"🎯 Training samples: {len(training)}")
print(f"🎯 Validation samples: {len(validation)}")

# Dataloaders
batch_size = 16  # Reduced batch size for small dataset
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
lr_monitor = LearningRateMonitor()

# Trainer
trainer = pl.Trainer(
    max_epochs=15,  # Reduced epochs for small dataset
    accelerator="cpu",  # Force CPU for small dataset
    gradient_clip_val=0.1,
    callbacks=[early_stop, lr_monitor],
    enable_model_summary=True,
    logger=False,  # Disable logging to reduce output
)

# Model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=3e-3,  # Slightly higher learning rate
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=MAE(),
    log_interval=5,
    reduce_on_plateau_patience=2,
)

print(f"🚀 Starting training...")
print(f"   Model size: {sum(p.numel() for p in tft.parameters())} parameters")

# Train
trainer.fit(tft, train_dataloader, val_dataloader)

# Save model
os.makedirs("models", exist_ok=True)
tft.save("models/tft_gdelt_only.pt")
print("✅ TFT model (GDELT-only) saved to models/tft_gdelt_only.pt") 