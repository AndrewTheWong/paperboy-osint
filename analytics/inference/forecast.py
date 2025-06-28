import os
import pandas as pd
from datetime import timedelta
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# -------------------
# Configuration paths
# -------------------
DATA_PATH = "data/gdelt_daily_2023.csv"
MODEL_PATH = "models/goldstein_predictor-epoch=14-val_loss=0.5480.ckpt"
SCALER_PATH = "models/goldstein_scaler.csv"
OUTPUT_PATH = "models/goldstein_forecast.csv"

def run_forecast():
    """Run the forecasting pipeline"""
    
    # -------------------
    # Load data
    # -------------------
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["avg_goldstein"]).sort_values(["country", "date"])
    df["time_idx"] = df.groupby("country").cumcount()
    df["country"] = df["country"].astype(str)

    # Load scaler
    scaler_df = pd.read_csv(SCALER_PATH)
    mean, scale = scaler_df["mean"][0], scaler_df["scale"][0]
    df["goldstein_scaled"] = (df["avg_goldstein"] - mean) / scale

    # -------------------
    # Build dataset object
    # -------------------
    max_encoder_length = 30
    max_prediction_length = 7

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="goldstein_scaled",
        group_ids=["country"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            "goldstein_scaled", "num_mentions", "avg_tone", "num_events",
            "num_verbal_conflict", "num_material_conflict",
            "num_verbal_coop", "num_material_coop"
        ],
        static_categoricals=["country"],
    )

    # -------------------
    # Prepare forecast input (future time steps)
    # -------------------
    latest_df = df.groupby("country").tail(max_encoder_length)

    future_rows = []
    for country in latest_df["country"].unique():
        last_time = latest_df[latest_df["country"] == country]["time_idx"].max()
        for i in range(1, max_prediction_length + 1):
            future_rows.append({
                "country": country,
                "time_idx": last_time + i,
            })

    future_df = pd.DataFrame(future_rows)

    # Forward fill predictors
    for col in [
        "num_mentions", "avg_tone", "num_events",
        "num_verbal_conflict", "num_material_conflict",
        "num_verbal_coop", "num_material_coop", "goldstein_scaled"
    ]:
        future_df[col] = future_df["country"].map(latest_df.groupby("country")[col].last())

    # Combine encoder + decoder input
    forecast_df = pd.concat([latest_df, future_df], ignore_index=True)

    # -------------------
    # Load model and forecast
    # -------------------
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
    predict_dataset = TimeSeriesDataSet.from_dataset(dataset, forecast_df, predict=True, stop_randomization=True)
    predict_loader = predict_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    forecast_scaled = model.predict(predict_loader, mode="prediction")
    forecast_df = pd.concat(predict_dataset.decode_prediction(forecast_scaled, mode="prediction"))

    # -------------------
    # Denormalize and assign forecast dates
    # -------------------
    forecast_df["goldstein_forecast"] = forecast_df["prediction"] * scale + mean
    latest_dates = df.groupby("country")["date"].max()
    forecast_df["event_date"] = forecast_df.apply(
        lambda row: latest_dates[row["country"]] + timedelta(days=row["decoder_time_idx"] - max_encoder_length + 1),
        axis=1,
    )

    # -------------------
    # Save output
    # -------------------
    output = forecast_df[["country", "event_date", "goldstein_forecast"]]
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Forecast saved to {OUTPUT_PATH}")
    
    return output

if __name__ == "__main__":
    run_forecast() 