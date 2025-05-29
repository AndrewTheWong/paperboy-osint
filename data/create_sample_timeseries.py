"""
Script: create_sample_timeseries.py
Purpose: Generate sample GDELT-style time series data for testing TFT model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate 90 days of sample data
start_date = datetime(2024, 1, 1)
end_date = start_date + timedelta(days=90)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Create realistic Taiwan Strait escalation patterns
n_days = len(dates)

# Base Goldstein scores with some escalation periods
base_goldstein = np.random.normal(-0.5, 1.5, n_days)  # Slightly negative mean

# Add some escalation events (higher negative scores = more conflict)
escalation_days = [20, 21, 22, 45, 46, 60, 61, 62, 75]
for day in escalation_days:
    if day < n_days:
        base_goldstein[day] += np.random.normal(-3, 0.5)  # Escalation events

# Add some de-escalation periods (positive scores)
deescalation_days = [25, 26, 50, 51, 65, 66, 80, 81]
for day in deescalation_days:
    if day < n_days:
        base_goldstein[day] += np.random.normal(2, 0.5)  # Positive diplomatic events

# Create correlated features
num_articles = np.maximum(1, np.random.poisson(15, n_days) + 
                         (np.abs(base_goldstein) * 2).astype(int))  # More articles during events

num_clusters = np.maximum(1, np.random.poisson(4, n_days) + 
                         (np.abs(base_goldstein) * 0.5).astype(int))  # More topics during events

max_cluster_size = np.maximum(1, np.random.poisson(8, n_days) + 
                             (np.abs(base_goldstein) * 1.5).astype(int))

avg_cluster_size = np.maximum(1, max_cluster_size * np.random.uniform(0.3, 0.7, n_days))

pct_noise = np.random.uniform(0.1, 0.4, n_days)  # 10-40% noise in clustering

# Add some temporal correlation to make it more realistic
for i in range(1, n_days):
    # Goldstein scores tend to persist
    base_goldstein[i] = 0.7 * base_goldstein[i] + 0.3 * base_goldstein[i-1]
    
    # Article counts also show persistence
    num_articles[i] = int(0.6 * num_articles[i] + 0.4 * num_articles[i-1])

# Create DataFrame
df = pd.DataFrame({
    'event_date': dates,
    'avg_goldstein': base_goldstein,
    'std_goldstein': np.abs(np.random.normal(1.2, 0.3, n_days)),  # Always positive
    'num_articles': num_articles,
    'num_clusters': num_clusters,
    'max_cluster_size': max_cluster_size,
    'avg_cluster_size': avg_cluster_size,
    'pct_noise': pct_noise
})

# Round to realistic precision
df['avg_goldstein'] = df['avg_goldstein'].round(3)
df['std_goldstein'] = df['std_goldstein'].round(3)
df['avg_cluster_size'] = df['avg_cluster_size'].round(2)
df['pct_noise'] = df['pct_noise'].round(3)

# Save the data
output_path = "data/aggregated_timeseries.csv"
df.to_csv(output_path, index=False)

print(f"✅ Sample time series data created: {output_path}")
print(f"📊 Data shape: {df.shape}")
print(f"📅 Date range: {df['event_date'].min()} to {df['event_date'].max()}")
print(f"📈 Goldstein range: {df['avg_goldstein'].min():.3f} to {df['avg_goldstein'].max():.3f}")
print(f"📰 Article count range: {df['num_articles'].min()} to {df['num_articles'].max()}")

# Display sample of the data
print("\n📋 Sample data:")
print(df.head(10).to_string(index=False)) 