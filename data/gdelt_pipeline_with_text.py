"""
Full GDELT Regression Pipeline with Article Text and SBERT
- Downloads sampled GDELT events with SourceURL
- Scrapes article text with robust error handling
- Embeds text with SBERT (all-MiniLM-L6-v2)
- Trains XGBoost Regressor on structured + semantic features
- Enhanced filtering for dead links and empty content
"""

import os, requests, zipfile, io, time
import pandas as pd
from datetime import datetime, timedelta
from newspaper import Article
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# === Setup ===
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
output_file = "data/gdelt_goldstein_text.csv"
if os.path.exists(output_file):
    os.remove(output_file)

# === Parameters ===
start_date = datetime(2024, 12, 1)  # Recent data for better URLs
end_date = datetime(2024, 12, 10)   # Short range for demo
interval = 1  # Daily sampling for demo
max_rows = 1000  # Limit for demo/test
batch_size = 3   # Smaller batches for better progress tracking
min_text_length = 50  # Minimum text length to keep
max_text_length = 2000  # Maximum text length for efficiency

print("🚀 Starting Enhanced GDELT Pipeline with Article Text & SBERT...")
print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"🎯 Target: {max_rows} events with valid article text")

# === Initialize SBERT Model ===
print("🤖 Loading SBERT model (all-MiniLM-L6-v2)...")
model_name = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)
print(f"   Model loaded: {embedder.get_sentence_embedding_dimension()} dimensions")

# === Helper Functions ===
def is_valid_url(url):
    """Check if URL is potentially valid"""
    try:
        parsed = urlparse(url)
        return all([parsed.scheme, parsed.netloc]) and parsed.scheme in ['http', 'https']
    except:
        return False

def scrape_article_text(url, timeout=10):
    """Scrape article with robust error handling"""
    try:
        if not is_valid_url(url):
            return "", "invalid_url"
        
        # Try newspaper3k first
        article = Article(url)
        article.download()
        article.parse()
        
        text = article.text.strip()
        if len(text) < min_text_length:
            return "", "text_too_short"
        
        # Truncate if too long
        if len(text) > max_text_length:
            text = text[:max_text_length]
        
        return text, "success"
        
    except Exception as e:
        return "", f"error_{type(e).__name__}"

# === Fetch and Process Loop ===
date = start_date
total_rows = 0
total_processed = 0
batch = []
scraping_stats = {"success": 0, "failed": 0, "invalid_url": 0, "text_too_short": 0}

while date <= end_date and total_rows < max_rows:
    batch_events = []
    
    for _ in range(batch_size):
        if date > end_date or total_rows >= max_rows:
            break
            
        date_str = date.strftime("%Y%m%d")
        url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
        
        try:
            print(f"📥 Downloading {date_str}...")
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, sep='\t', header=None, encoding='latin1', low_memory=False)
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    
                    # Select and rename columns
                    df = df.rename(columns={
                        'col_0': 'globaleventid',
                        'col_1': 'event_date',
                        'col_7': 'actor_1',
                        'col_17': 'actor_2',
                        'col_26': 'event_code',
                        'col_30': 'quad_class',
                        'col_31': 'avg_tone',
                        'col_33': 'mentions',
                        'col_34': 'goldstein',
                        'col_57': 'sourceurl',
                        'col_51': 'country'
                    })
                    
                    # Keep only needed columns
                    df = df[['event_date', 'event_code', 'quad_class', 'avg_tone', 'mentions',
                             'actor_1', 'actor_2', 'country', 'goldstein', 'sourceurl']]
                    
                    # Clean and convert data types
                    df = df.dropna(subset=['goldstein', 'sourceurl'])
                    df['event_code'] = pd.to_numeric(df['event_code'], errors='coerce')
                    df['quad_class'] = pd.to_numeric(df['quad_class'], errors='coerce')
                    df['avg_tone'] = pd.to_numeric(df['avg_tone'], errors='coerce')
                    df['mentions'] = pd.to_numeric(df['mentions'], errors='coerce')
                    df['goldstein'] = pd.to_numeric(df['goldstein'], errors='coerce')
                    
                    # Convert categorical columns to codes
                    for col in ['actor_1', 'actor_2', 'country']:
                        df[col] = df[col].astype('category').cat.codes
                    
                    # Convert date
                    df['event_date'] = pd.to_datetime(df['event_date'], format='%Y%m%d', errors='coerce')
                    df = df.dropna(subset=['event_date'])
                    df['event_date'] = df['event_date'].map(lambda x: x.toordinal())
                    
                    # Final cleanup
                    df = df.dropna()
                    
                    # Sample if too many events
                    if len(df) > 200:
                        df = df.sample(n=200, random_state=42)
                    
                    batch_events.append(df)
                    print(f"✅ {date_str}: {len(df)} events loaded")
                    
        except Exception as e:
            print(f"❌ Failed to download {date_str}: {e}")
        
        date += timedelta(days=interval)
    
    # Process batch for article scraping
    if batch_events:
        combined_batch = pd.concat(batch_events, ignore_index=True)
        print(f"\n📰 Scraping articles for batch ({len(combined_batch)} events)...")
        
        texts = []
        scrape_results = []
        valid_indices = []
        
        for idx, url in enumerate(combined_batch['sourceurl']):
            if total_rows >= max_rows:
                break
                
            text, status = scrape_article_text(url)
            total_processed += 1
            
            if status == "success":
                texts.append(text)
                scrape_results.append(status)
                valid_indices.append(idx)
                scraping_stats["success"] += 1
                print(f"   ✅ [{total_processed:3d}] Valid article: {len(text)} chars")
            else:
                scraping_stats[status] = scraping_stats.get(status, 0) + 1
                print(f"   ❌ [{total_processed:3d}] Failed: {status}")
            
            # Progress update every 10 articles
            if total_processed % 10 == 0:
                success_rate = (scraping_stats["success"] / total_processed) * 100
                print(f"   📊 Progress: {scraping_stats['success']} valid / {total_processed} processed ({success_rate:.1f}%)")
        
        # Keep only rows with valid text
        if valid_indices:
            valid_df = combined_batch.iloc[valid_indices].copy()
            valid_df['text'] = texts
            
            # Generate SBERT embeddings
            print(f"🧠 Generating SBERT embeddings for {len(texts)} articles...")
            embeddings = embedder.encode(texts, show_progress_bar=True)
            
            # Add embeddings as columns
            for i in range(embeddings.shape[1]):
                valid_df[f'sbert_{i}'] = embeddings[:, i]
            
            # Remove text and URL columns (keep only features)
            valid_df = valid_df.drop(columns=['text', 'sourceurl'])
            
            batch.append(valid_df)
            total_rows += len(valid_df)
            
            print(f"✅ Batch complete: {len(valid_df)} valid events (Total: {total_rows})")
        
        # Save batch periodically
        if batch and (total_rows >= max_rows or len(batch) >= 3):
            combined = pd.concat(batch, ignore_index=True)
            mode = 'a' if os.path.exists(output_file) else 'w'
            header = not os.path.exists(output_file)
            combined.to_csv(output_file, mode=mode, header=header, index=False)
            batch.clear()
            
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"💾 Saved batch to {output_file} ({file_size_mb:.1f} MB)")
            
        print()  # Add spacing between batches

# === Final Statistics ===
print("🎯 Data Collection Complete!")
print(f"📊 Scraping Statistics:")
print(f"   Total processed: {total_processed}")
print(f"   Successful: {scraping_stats['success']} ({(scraping_stats['success']/total_processed)*100:.1f}%)")
print(f"   Failed URLs: {scraping_stats.get('failed', 0)}")
print(f"   Invalid URLs: {scraping_stats.get('invalid_url', 0)}")
print(f"   Text too short: {scraping_stats.get('text_too_short', 0)}")

if total_rows == 0:
    print("❌ No valid data collected. Exiting...")
    exit(1)

# === Train Model ===
print(f"\n🧠 Training XGBoost Regressor with SBERT embeddings...")
df = pd.read_csv(output_file)
print(f"📊 Final dataset: {len(df)} rows, {len(df.columns)} features")

# Separate structural and embedding features
struct_cols = ['event_date', 'event_code', 'quad_class', 'avg_tone', 'mentions', 'actor_1', 'actor_2', 'country']
embed_cols = [c for c in df.columns if c.startswith('sbert_')]
features = struct_cols + embed_cols

print(f"   Structural features: {len(struct_cols)}")
print(f"   SBERT features: {len(embed_cols)}")
print(f"   Total features: {len(features)}")

X = df[features]
y = df['goldstein']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Training set: {len(X_train)} rows")
print(f"   Test set: {len(X_test)} rows")

# Train model
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

print("🚀 Training model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Model Performance:")
print(f"   MSE: {mse:.4f}")
print(f"   R²: {r2:.4f}")
print(f"   RMSE: {np.sqrt(mse):.4f}")

# Feature importance analysis
feature_importance = dict(zip(features, model.feature_importances_))
struct_importance = sum(feature_importance[f] for f in struct_cols)
embed_importance = sum(feature_importance[f] for f in embed_cols)

print(f"\n🌟 Feature Importance Analysis:")
print(f"   Structural features: {struct_importance:.3f} ({struct_importance*100:.1f}%)")
print(f"   SBERT embeddings: {embed_importance:.3f} ({embed_importance*100:.1f}%)")

# Top individual features
print(f"\n🔝 Top 10 Features:")
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for i, (feature, importance) in enumerate(sorted_features[:10], 1):
    feature_type = "STRUCT" if feature in struct_cols else "SBERT"
    print(f"   {i:2d}. {feature:15} ({feature_type}): {importance:.4f}")

# Save model
model_path = "models/xgb_goldstein_sbert.pkl"
joblib.dump(model, model_path)
print(f"\n✅ Model saved to {model_path}")

# Save metadata
metadata = {
    "model_type": "XGBoost with SBERT embeddings",
    "sbert_model": model_name,
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "features": {
        "structural": len(struct_cols),
        "sbert_dimensions": len(embed_cols),
        "total": len(features)
    },
    "performance": {
        "mse": float(mse),
        "r2": float(r2),
        "rmse": float(np.sqrt(mse))
    },
    "feature_importance": {
        "structural_weight": float(struct_importance),
        "sbert_weight": float(embed_importance)
    },
    "scraping_stats": scraping_stats,
    "data_range": {
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d')
    }
}

metadata_path = "models/sbert_model_metadata.json"
import json
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"📄 Metadata saved to {metadata_path}")
print(f"\n🎉 Enhanced GDELT pipeline complete!")
print(f"🔗 Next steps: Run visualization to compare with baseline model") 