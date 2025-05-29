"""
Fast Parallel GDELT Article Scraper with Progress Updates
- Uses hybrid approach: trafilatura + BeautifulSoup fallback
- ThreadPoolExecutor for multi-threaded scraping
- tqdm progress bar for live tracking
- URL caching to skip already-scraped articles
- Direct SBERT embedding + model inference pipeline
- Article snippet previews during scraping
"""

import os, requests, zipfile, io, time, json, hashlib, re
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from trafilatura import fetch_url, extract
from bs4 import BeautifulSoup
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import joblib
import numpy as np
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
CACHE_FILE = "data/url_cache.json"
OUTPUT_FILE = "data/gdelt_goldstein_fast.csv"
MODEL_PATH = "models/xgb_goldstein_fast.pkl"
MAX_WORKERS = 8  # Parallel scraping threads (reduced for debugging)
BATCH_SIZE = 50  # URLs to scrape per batch (reduced for testing)
MIN_TEXT_LENGTH = 50
MAX_TEXT_LENGTH = 2000
SNIPPET_LENGTH = 100  # Characters to show in preview

# === Setup ===
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Clear output file
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

print("🚀 Fast Parallel GDELT Pipeline with SBERT & Caching")
print(f"⚡ Using {MAX_WORKERS} parallel workers")
print(f"📦 Batch size: {BATCH_SIZE} URLs")

# === Cache Management ===
def load_cache():
    """Load URL cache from disk"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    """Save URL cache to disk"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def cache_key(url):
    """Generate cache key from URL"""
    return hashlib.md5(url.encode()).hexdigest()[:16]

# === Fast Scraper Functions ===
def clean_text(text):
    """Clean extracted text"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\n+', '\n', text)
    
    return text

def scrape_with_beautifulsoup(html):
    """Fallback scraper using BeautifulSoup"""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'article', '[role="main"]', '.content', '.article-content', 
            '.post-content', '.entry-content', '.story-body', 'main'
        ]
        
        text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([elem.get_text() for elem in elements])
                break
        
        # If no specific content area found, get all paragraph text
        if not text:
            paragraphs = soup.find_all(['p', 'div'], string=True)
            text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text().strip()) > 20])
        
        return clean_text(text)
        
    except Exception:
        return ""

def fast_scrape_article(url):
    """Fast article scraping with hybrid approach"""
    try:
        # First try with requests directly
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
        response.raise_for_status()
        html = response.text
        
        if not html:
            return "", "no_html"
        
        # Try trafilatura first
        try:
            text = extract(html, 
                          include_comments=False, 
                          include_tables=False,
                          include_formatting=False,
                          favor_precision=True)
            
            if text and len(text.strip()) >= MIN_TEXT_LENGTH:
                text = clean_text(text)
                if len(text) > MAX_TEXT_LENGTH:
                    text = text[:MAX_TEXT_LENGTH]
                return text, "success_trafilatura"
        except:
            pass
        
        # Fallback to BeautifulSoup
        text = scrape_with_beautifulsoup(html)
        
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return "", "too_short"
        
        text = clean_text(text)
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
        
        return text, "success_bs4"
        
    except requests.RequestException as e:
        return "", f"request_error"
    except Exception as e:
        return "", f"error_{type(e).__name__}"

def scrape_articles_with_progress(urls, cache=None):
    """Parallel scraping with progress bar and caching"""
    if cache is None:
        cache = {}
    
    # Filter URLs not in cache
    urls_to_scrape = []
    cached_results = []
    cache_hits = 0
    
    for url in urls:
        key = cache_key(url)
        if key in cache:
            cached_results.append((url, cache[key]['text'], cache[key]['status']))
            cache_hits += 1
        else:
            urls_to_scrape.append(url)
    
    print(f"📋 Cache: {cache_hits} hits, {len(urls_to_scrape)} URLs to scrape")
    
    # Scrape new URLs in parallel
    scraped_results = []
    stats = {"success_trafilatura": 0, "success_bs4": 0, "request_error": 0, "too_short": 0, "no_html": 0, "other_error": 0}
    
    if urls_to_scrape:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_url = {executor.submit(fast_scrape_article, url): url for url in urls_to_scrape}
            
            # Process results with progress bar
            for future in tqdm(as_completed(future_to_url), 
                             total=len(urls_to_scrape), 
                             desc="📰 Scraping articles",
                             unit="articles"):
                url = future_to_url[future]
                try:
                    text, status = future.result()
                    scraped_results.append((url, text, status))
                    
                    # Update cache
                    cache[cache_key(url)] = {'text': text, 'status': status}
                    
                    # Update stats
                    if status.startswith("success"):
                        stats[status] = stats.get(status, 0) + 1
                        # Show snippet if successful
                        snippet = text[:SNIPPET_LENGTH].replace('\n', ' ').replace('\r', ' ')
                        method = "🔧" if "trafilatura" in status else "🍲"
                        print(f"   ✅ {method} {len(text):4d} chars: {snippet}...")
                    elif status in stats:
                        stats[status] += 1
                    else:
                        stats["other_error"] += 1
                    
                except Exception as e:
                    scraped_results.append((url, "", f"future_error_{type(e).__name__}"))
                    stats["other_error"] += 1
    
    # Show scraping stats
    if urls_to_scrape:
        total = len(urls_to_scrape)
        total_success = stats["success_trafilatura"] + stats["success_bs4"]
        print(f"📊 Scraping stats:")
        print(f"   ✅ Success: {total_success} (🔧{stats['success_trafilatura']} + 🍲{stats['success_bs4']})")
        print(f"   ❌ Errors: 🌐{stats['request_error']} 📏{stats['too_short']} 📄{stats['no_html']} ⚠️{stats['other_error']}")
        success_rate = (total_success / total) * 100
        print(f"   📈 Success rate: {success_rate:.1f}%")
    
    # Combine cached and scraped results
    all_results = cached_results + scraped_results
    
    # Sort by original URL order
    url_to_result = {url: (text, status) for url, text, status in all_results}
    ordered_results = [(url_to_result.get(url, ("", "missing"))[0], 
                       url_to_result.get(url, ("", "missing"))[1]) for url in urls]
    
    return ordered_results, cache

# === GDELT Data Fetching ===
def fetch_gdelt_batch(start_date, end_date, max_events=1000):
    """Fetch GDELT events for date range"""
    print(f"📥 Fetching GDELT events from {start_date} to {end_date}")
    
    all_events = []
    current_date = start_date
    
    while current_date <= end_date and len(all_events) < max_events:
        date_str = current_date.strftime("%Y%m%d")
        url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
        
        try:
            print(f"📅 Downloading {date_str}...", end=" ")
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, sep='\t', header=None, encoding='latin1', low_memory=False)
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    
                    # Select relevant columns
                    df = df.rename(columns={
                        'col_1': 'event_date', 'col_26': 'event_code', 'col_30': 'quad_class',
                        'col_31': 'avg_tone', 'col_33': 'mentions', 'col_34': 'goldstein',
                        'col_57': 'sourceurl', 'col_7': 'actor_1', 'col_17': 'actor_2', 
                        'col_51': 'country'
                    })
                    
                    # Keep needed columns
                    cols = ['event_date', 'event_code', 'quad_class', 'avg_tone', 'mentions',
                           'actor_1', 'actor_2', 'country', 'goldstein', 'sourceurl']
                    df = df[cols]
                    
                    # Clean data
                    df = df.dropna(subset=['goldstein', 'sourceurl'])
                    for col in ['event_code', 'quad_class', 'avg_tone', 'mentions', 'goldstein']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Filter valid URLs
                    df = df[df['sourceurl'].str.startswith(('http://', 'https://'))]
                    df = df.dropna()
                    
                    # Sample if too many
                    if len(df) > 300:
                        df = df.sample(n=300, random_state=42)
                    
                    all_events.append(df)
                    print(f"✅ {len(df)} events")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
        
        current_date += timedelta(days=1)
    
    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        print(f"📊 Total events collected: {len(combined)}")
        return combined
    else:
        return pd.DataFrame()

# === SBERT + Model Pipeline ===
def setup_models():
    """Initialize SBERT and load/create XGBoost model"""
    print("🤖 Setting up models...")
    
    # Load SBERT
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"   SBERT: {embedder.get_sentence_embedding_dimension()} dimensions")
    
    # Initialize PCA for dimensionality reduction
    pca = PCA(n_components=50, random_state=42)
    print(f"   PCA: Reducing to 50 dimensions")
    
    return embedder, pca

def process_texts_to_features(texts, embedder, pca, fit_pca=False):
    """Convert texts to SBERT embeddings + PCA features"""
    print(f"🧠 Processing {len(texts)} texts to embeddings...")
    
    # Generate SBERT embeddings
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)
    
    # Apply PCA
    if fit_pca:
        print("📉 Fitting PCA...")
        pca_features = pca.fit_transform(embeddings)
    else:
        pca_features = pca.transform(embeddings)
    
    print(f"✅ Generated {pca_features.shape[1]} PCA features")
    return pca_features

def create_feature_matrix(df, pca_features):
    """Combine structural and text features"""
    # Structural features
    struct_cols = ['event_code', 'quad_class', 'avg_tone', 'mentions']
    
    # Convert categorical to codes
    for col in ['actor_1', 'actor_2', 'country']:
        df[col] = df[col].astype('category').cat.codes
    struct_cols.extend(['actor_1', 'actor_2', 'country'])
    
    # Convert date
    df['event_date'] = pd.to_datetime(df['event_date'], format='%Y%m%d', errors='coerce')
    df['event_date'] = df['event_date'].map(lambda x: x.toordinal() if pd.notna(x) else 0)
    struct_cols.append('event_date')
    
    # Combine features
    X_struct = df[struct_cols].values
    X_combined = np.hstack([X_struct, pca_features])
    
    feature_names = struct_cols + [f'pca_{i}' for i in range(pca_features.shape[1])]
    
    return X_combined, feature_names

# === Main Pipeline ===
def main():
    # Parameters
    start_date = datetime(2024, 12, 1)
    end_date = datetime(2024, 12, 5)  # Short range for demo
    max_events = 200  # Reduced for testing
    
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Load cache
    cache = load_cache()
    print(f"💾 Loaded cache with {len(cache)} entries")
    
    # Setup models
    embedder, pca = setup_models()
    
    # Fetch GDELT data
    df = fetch_gdelt_batch(start_date, end_date, max_events)
    if df.empty:
        print("❌ No data collected. Exiting...")
        return
    
    # Debug: Show sample URLs
    print(f"\n🔍 Sample URLs:")
    sample_urls = df['sourceurl'].head(5).tolist()
    for i, url in enumerate(sample_urls, 1):
        print(f"   {i}. {url}")
    
    # Test one URL manually
    print(f"\n🧪 Testing first URL manually...")
    test_url = sample_urls[0]
    test_text, test_status = fast_scrape_article(test_url)
    print(f"   URL: {test_url}")
    print(f"   Status: {test_status}")
    print(f"   Text length: {len(test_text)}")
    if test_text:
        print(f"   Preview: {test_text[:200]}...")
    
    print(f"\n📰 Starting article scraping for {len(df)} URLs...")
    
    # Scrape articles in batches
    all_texts = []
    all_indices = []
    
    urls = df['sourceurl'].tolist()
    for i in range(0, len(urls), BATCH_SIZE):
        batch_urls = urls[i:i+BATCH_SIZE]
        batch_indices = list(range(i, min(i+BATCH_SIZE, len(urls))))
        
        print(f"\n🔄 Processing batch {i//BATCH_SIZE + 1}/{(len(urls)-1)//BATCH_SIZE + 1}")
        
        # Scrape batch
        results, cache = scrape_articles_with_progress(batch_urls, cache)
        
        # Filter successful results
        valid_count = 0
        for j, (text, status) in enumerate(results):
            if status.startswith("success") and text:
                all_texts.append(text)
                all_indices.append(batch_indices[j])
                valid_count += 1
        
        # Save cache periodically
        save_cache(cache)
        
        print(f"✅ Batch complete: {valid_count} valid articles")
        
        # Stop if we have enough valid articles or this is just a test
        if len(all_texts) >= 50:  # Stop early for testing
            print("🛑 Stopping early for testing (got enough articles)")
            break
    
    print(f"\n📊 Final stats: {len(all_texts)} valid articles from {len(df)} events")
    
    if len(all_texts) < 10:
        print("❌ Too few valid articles. Exiting...")
        return
    
    # Filter dataframe to valid articles only
    df_valid = df.iloc[all_indices].copy().reset_index(drop=True)
    
    # Process texts to features
    pca_features = process_texts_to_features(all_texts, embedder, pca, fit_pca=True)
    
    # Create feature matrix
    X, feature_names = create_feature_matrix(df_valid, pca_features)
    y = df_valid['goldstein'].values
    
    print(f"🔢 Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    print(f"🚀 Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Model Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²: {r2:.4f}")
    print(f"   RMSE: {np.sqrt(mse):.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    struct_importance = importance[:8].sum()  # First 8 are structural
    text_importance = importance[8:].sum()    # Rest are text features
    
    print(f"\n🌟 Feature Importance:")
    print(f"   Structural features: {struct_importance:.3f} ({struct_importance*100:.1f}%)")
    print(f"   Text features (PCA): {text_importance:.3f} ({text_importance*100:.1f}%)")
    
    # Top features
    top_indices = np.argsort(importance)[-10:][::-1]
    print(f"\n🔝 Top 10 Features:")
    for i, idx in enumerate(top_indices, 1):
        feature_type = "STRUCT" if idx < 8 else "TEXT"
        print(f"   {i:2d}. {feature_names[idx]:15} ({feature_type}): {importance[idx]:.4f}")
    
    # Save model and components
    model_data = {
        'xgb_model': model,
        'pca': pca,
        'feature_names': feature_names,
        'embedder_name': "all-MiniLM-L6-v2"
    }
    
    joblib.dump(model_data, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    
    # Save final cache
    save_cache(cache)
    print(f"💾 Cache saved with {len(cache)} entries")
    
    print(f"\n🎉 Fast pipeline complete!")
    print(f"⚡ Successfully processed {len(all_texts)} articles")
    print(f"🎯 Final model R² = {r2:.4f}")

if __name__ == "__main__":
    main() 