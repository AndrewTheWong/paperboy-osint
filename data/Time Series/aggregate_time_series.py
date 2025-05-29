"""
Script: aggregate_time_series.py
Purpose: Generate daily time-series features for escalation modeling, using article metadata, HDBSCAN cluster labels, and GDELT events.
"""

import pandas as pd
import os
from datetime import datetime
from supabase import create_client, Client

# Init Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_article_data():
    response = supabase.table("osint_articles").select("*").execute()
    df = pd.DataFrame(response.data)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    return df

def fetch_gdelt_data():
    """Fetch GDELT events from Supabase"""
    response = supabase.table("gdelt_events").select("*").execute()
    df_gdelt = pd.DataFrame(response.data)
    df_gdelt["event_date"] = pd.to_datetime(df_gdelt["event_date"]).dt.date
    return df_gdelt

def aggregate_gdelt_daily(df_gdelt: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GDELT events by day"""
    df_gdelt_daily = df_gdelt.groupby("event_date").agg(
        avg_goldstein=("goldstein_score", "mean"),
        std_goldstein=("goldstein_score", "std"),
        num_gdelt_events=("goldstein_score", "count")
    ).reset_index()
    return df_gdelt_daily

def aggregate_article_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article data by day"""
    def pct_noise(x): return (x == -1).sum() / len(x) if len(x) > 0 else 0

    df_valid = df[~df["goldstein_score"].isnull() & ~df["embedding_cluster_id"].isnull()]

    grouped = df_valid.groupby("event_date").agg(
        article_avg_goldstein=("goldstein_score", "mean"),
        article_std_goldstein=("goldstein_score", "std"),
        num_articles=("article_id", "count"),
        num_clusters=("embedding_cluster_id", lambda x: x[x != -1].nunique()),
        max_cluster_size=("embedding_cluster_id", lambda x: x[x != -1].value_counts().max() if x[x != -1].any() else 0),
        avg_cluster_size=("embedding_cluster_id", lambda x: x[x != -1].value_counts().mean() if x[x != -1].any() else 0),
        pct_noise=("embedding_cluster_id", pct_noise)
    ).reset_index()

    grouped["event_date"] = pd.to_datetime(grouped["event_date"])
    return grouped

def merge_datasets(df_articles: pd.DataFrame, df_gdelt: pd.DataFrame) -> pd.DataFrame:
    """Merge article and GDELT aggregations"""
    # Ensure both have datetime event_date
    df_articles["event_date"] = pd.to_datetime(df_articles["event_date"])
    df_gdelt["event_date"] = pd.to_datetime(df_gdelt["event_date"])
    
    # Merge on event_date
    df_merged = pd.merge(df_articles, df_gdelt, on="event_date", how="outer")
    
    # Fill NaN values with 0 for counts and means
    df_merged = df_merged.fillna({
        'num_articles': 0,
        'num_gdelt_events': 0,
        'num_clusters': 0,
        'max_cluster_size': 0,
        'avg_cluster_size': 0,
        'pct_noise': 0
    })
    
    return df_merged

def save_and_upload(df: pd.DataFrame, path="data/aggregated_timeseries.csv"):
    os.makedirs("data", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved to {path}")
    
    # Convert datetime to string for Supabase upload
    df_upload = df.copy()
    df_upload["event_date"] = df_upload["event_date"].dt.strftime('%Y-%m-%d')
    
    supabase.table("daily_features").upsert(df_upload.to_dict("records")).execute()
    print("✅ Uploaded to Supabase")

if __name__ == "__main__":
    print("🔄 Fetching article data...")
    df_articles = fetch_article_data()
    df_article_agg = aggregate_article_time_series(df_articles)
    
    print("🔄 Fetching GDELT data...")
    df_gdelt = fetch_gdelt_data()
    df_gdelt_agg = aggregate_gdelt_daily(df_gdelt)
    
    print("🔄 Merging datasets...")
    df_merged = merge_datasets(df_article_agg, df_gdelt_agg)
    
    print(f"📊 Final dataset shape: {df_merged.shape}")
    print(f"📅 Date range: {df_merged['event_date'].min()} to {df_merged['event_date'].max()}")
    
    save_and_upload(df_merged) 