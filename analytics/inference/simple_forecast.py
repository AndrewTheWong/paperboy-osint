import os
import pandas as pd
import numpy as np
from datetime import timedelta

# -------------------
# Configuration paths
# -------------------
DATA_PATH = "data/gdelt_daily_2023.csv"
SCALER_PATH = "models/goldstein_scaler.csv"
OUTPUT_PATH = "models/simple_forecast.csv"

def simple_forecast():
    """Run a simple forecasting pipeline using basic trend analysis"""
    
    # -------------------
    # Load data
    # -------------------
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["avg_goldstein"]).sort_values(["country", "date"])
    
    # Load scaler if exists
    if os.path.exists(SCALER_PATH):
        scaler_df = pd.read_csv(SCALER_PATH)
        mean, scale = scaler_df["mean"][0], scaler_df["scale"][0]
    else:
        # Calculate scaler from data
        mean = df["avg_goldstein"].mean()
        scale = df["avg_goldstein"].std()
    
    # -------------------
    # Simple trend-based forecasting
    # -------------------
    forecast_length = 7  # 7 days forecast
    forecast_results = []
    
    for country in df["country"].unique():
        country_data = df[df["country"] == country].copy()
        
        if len(country_data) < 7:  # Need at least 7 days of data
            continue
            
        # Get last 30 days for trend calculation
        recent_data = country_data.tail(30)
        
        # Calculate simple trend (linear)
        x = np.arange(len(recent_data))
        y = recent_data["avg_goldstein"].values
        trend = np.polyfit(x, y, 1)[0]  # Linear trend slope
        
        # Get last date and value
        last_date = recent_data["date"].iloc[-1]
        last_value = recent_data["avg_goldstein"].iloc[-1]
        
        # Generate forecasts
        for day in range(1, forecast_length + 1):
            forecast_date = last_date + timedelta(days=day)
            
            # Simple trend + random noise
            forecast_value = last_value + (trend * day) + np.random.normal(0, 0.1)
            
            forecast_results.append({
                "country": country,
                "event_date": forecast_date,
                "goldstein_forecast": forecast_value
            })
    
    # -------------------
    # Save output
    # -------------------
    output_df = pd.DataFrame(forecast_results)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Simple forecast saved to {OUTPUT_PATH}")
    
    if len(output_df) > 0:
        print(f"📊 Generated forecasts for {len(output_df['country'].unique())} countries")
        print(f"📅 Forecast period: {forecast_length} days")
    else:
        print("⚠️ No forecasts generated - insufficient data for all countries")
    
    return output_df

if __name__ == "__main__":
    simple_forecast() 