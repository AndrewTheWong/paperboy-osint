"""
Working Fast Parallel GDELT Article Scraper
- Uses recent GDELT data for better URL success rates
- Simplified BeautifulSoup approach with retries
- Parallel processing with progress tracking
- URL caching and article snippet previews
"""

import os, requests, zipfile, io, json, hashlib, re
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
CACHE_FILE = "data/working_url_cache.json"
OUTPUT_FILE = "data/gdelt_working_fast.csv"
MODEL_PATH = "models/xgb_working_fast.pkl"
MAX_WORKERS = 8
BATCH_SIZE = 50
MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH = 1500
SNIPPET_LENGTH = 120

# === Setup ===
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("🚀 Working Fast GDELT Scraper with Real URLs")
print(f"⚡ Using {MAX_WORKERS} workers, {BATCH_SIZE} batch size")

# === Cache Management ===
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def cache_key(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]

# === Working Scraper ===
def scrape_article_working(url):
    """Working article scraper with retries"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Try request with retries
        for attempt in range(2):
            try:
                response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
                if response.status_code == 200:
                    break
                elif attempt == 0:
                    continue
                else:
                    return "", f"http_{response.status_code}"
            except:
                if attempt == 0:
                    continue
                else:
                    return "", "connection_error"
        
        html = response.text
        if len(html) < 500:
            return "", "html_too_small"
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for tag in ['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'button']:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Extract text from multiple strategies
        text_parts = []
        
        # Strategy 1: Look for article content
        article_selectors = ['article', '[role="main"]', '.article-body', '.content', '.post-content']
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                for elem in elements:
                    text_parts.append(elem.get_text())
                break
        
        # Strategy 2: Get all paragraphs if no article found
        if not text_parts:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                p_text = p.get_text().strip()
                if len(p_text) > 30:  # Only substantial paragraphs
                    text_parts.append(p_text)
        
        # Strategy 3: Fallback to all text
        if not text_parts:
            text_parts = [soup.get_text()]
        
        # Clean and combine text
        combined_text = ' '.join(text_parts)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', combined_text.strip())
        text = re.sub(r'\n+', ' ', text)
        
        # Filter by length
        if len(text) < MIN_TEXT_LENGTH:
            return "", "too_short"
        
        # Truncate if too long
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
        
        return text, "success"
        
    except Exception as e:
        return "", f"error_{type(e).__name__}"

# === Parallel Processing ===
def scrape_urls_parallel(urls, cache):
    """Scrape URLs in parallel with progress"""
    
    # Check cache first
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
    
    print(f"📋 Cache: {cache_hits} hits, {len(urls_to_scrape)} new URLs")
    
    # Scrape new URLs
    scraped_results = []
    stats = {"success": 0, "too_short": 0, "connection_error": 0, "http_error": 0, "other": 0}
    
    if urls_to_scrape:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(scrape_article_working, url): url for url in urls_to_scrape}
            
            for future in tqdm(as_completed(future_to_url), 
                             total=len(urls_to_scrape), 
                             desc="📰 Scraping",
                             unit="articles"):
                url = future_to_url[future]
                try:
                    text, status = future.result()
                    scraped_results.append((url, text, status))
                    
                    # Update cache
                    cache[cache_key(url)] = {'text': text, 'status': status}
                    
                    # Update stats and show snippets
                    if status == "success":
                        stats["success"] += 1
                        snippet = text[:SNIPPET_LENGTH].replace('\n', ' ')
                        print(f"   ✅ {len(text):4d} chars: {snippet}...")
                    elif "http_" in status:
                        stats["http_error"] += 1
                    elif status in stats:
                        stats[status] += 1
                    else:
                        stats["other"] += 1
                        
                except Exception as e:
                    scraped_results.append((url, "", f"future_error"))
                    stats["other"] += 1
    
    # Show stats
    if urls_to_scrape:
        total = len(urls_to_scrape)
        print(f"📊 Results: ✅{stats['success']} 📏{stats['too_short']} 🌐{stats['connection_error']} 🚫{stats['http_error']} ❓{stats['other']}")
        success_rate = (stats['success'] / total) * 100 if total > 0 else 0
        print(f"   📈 Success: {success_rate:.1f}%")
    
    # Combine results
    all_results = cached_results + scraped_results
    url_to_result = {url: (text, status) for url, text, status in all_results}
    ordered_results = [(url_to_result.get(url, ("", "missing"))[0], 
                       url_to_result.get(url, ("", "missing"))[1]) for url in urls]
    
    return ordered_results, cache

# === GDELT Data Fetching ===
def fetch_recent_gdelt(days_back=3, max_events=500):
    """Fetch recent GDELT data for better URL success"""
    print(f"📥 Fetching recent GDELT data ({days_back} days back)")
    
    all_events = []
    current_date = datetime.now() - timedelta(days=days_back)
    end_date = datetime.now() - timedelta(days=1)
    
    while current_date <= end_date and len(all_events) < max_events:
        date_str = current_date.strftime("%Y%m%d")
        url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
        
        try:
            print(f"📅 {date_str}...", end=" ")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, sep='\t', header=None, encoding='latin1', low_memory=False)
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
                    
                    # Select columns
                    df = df.rename(columns={
                        'col_1': 'event_date', 'col_26': 'event_code', 'col_30': 'quad_class',
                        'col_31': 'avg_tone', 'col_33': 'mentions', 'col_34': 'goldstein',
                        'col_57': 'sourceurl', 'col_7': 'actor_1', 'col_17': 'actor_2', 
                        'col_51': 'country'
                    })
                    
                    cols = ['event_date', 'event_code', 'quad_class', 'avg_tone', 'mentions',
                           'actor_1', 'actor_2', 'country', 'goldstein', 'sourceurl']
                    df = df[cols]
                    
                    # Clean and filter
                    df = df.dropna(subset=['goldstein', 'sourceurl'])
                    for col in ['event_code', 'quad_class', 'avg_tone', 'mentions', 'goldstein']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Filter for valid URLs from major domains
                    df = df[df['sourceurl'].str.contains('http', na=False)]
                    df = df[df['sourceurl'].str.contains('|'.join([
                        'reuters.com', 'yahoo.com', 'cnn.com', 'bbc.com', 'guardian.com',
                        'nytimes.com', 'washingtonpost.com', 'apnews.com', 'bloomberg.com'
                    ]), na=False)]
                    
                    df = df.dropna()
                    
                    # Sample randomly
                    if len(df) > 200:
                        df = df.sample(n=200, random_state=42)
                    
                    all_events.append(df)
                    print(f"✅ {len(df)} events")
                    
        except Exception as e:
            print(f"❌ {e}")
        
        current_date += timedelta(days=1)
    
    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        print(f"📊 Total events: {len(combined)}")
        return combined
    else:
        return pd.DataFrame()

# === Main Pipeline ===
def main():
    # Load cache
    cache = load_cache()
    print(f"💾 Cache loaded: {len(cache)} entries")
    
    # Get recent GDELT data
    df = fetch_recent_gdelt(days_back=2, max_events=300)
    if df.empty:
        print("❌ No data collected")
        return
    
    # Show sample URLs
    print(f"\n🔍 Sample URLs:")
    for i, url in enumerate(df['sourceurl'].head(3), 1):
        print(f"   {i}. {url}")
    
    # Scrape articles
    urls = df['sourceurl'].tolist()
    results, cache = scrape_urls_parallel(urls, cache)
    
    # Filter successful results
    valid_texts = []
    valid_indices = []
    
    for i, (text, status) in enumerate(results):
        if status == "success" and text:
            valid_texts.append(text)
            valid_indices.append(i)
    
    print(f"\n📊 Final results: {len(valid_texts)} valid articles from {len(df)} events")
    
    if len(valid_texts) < 20:
        print("⚠️ Few articles extracted, but continuing...")
        if len(valid_texts) < 5:
            print("❌ Too few articles to train model")
            save_cache(cache)
            return
    
    # Process with SBERT
    print(f"\n🤖 Loading SBERT...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"🧠 Generating embeddings...")
    embeddings = embedder.encode(valid_texts, show_progress_bar=True)
    
    # PCA
    pca = PCA(n_components=min(30, len(valid_texts)//2), random_state=42)
    pca_features = pca.fit_transform(embeddings)
    
    # Prepare structural features
    df_valid = df.iloc[valid_indices].copy().reset_index(drop=True)
    
    for col in ['actor_1', 'actor_2', 'country']:
        df_valid[col] = df_valid[col].astype('category').cat.codes
    
    df_valid['event_date'] = pd.to_datetime(df_valid['event_date'], format='%Y%m%d', errors='coerce')
    df_valid['event_date'] = df_valid['event_date'].map(lambda x: x.toordinal() if pd.notna(x) else 0)
    
    struct_cols = ['event_date', 'event_code', 'quad_class', 'avg_tone', 'mentions', 'actor_1', 'actor_2', 'country']
    X_struct = df_valid[struct_cols].values
    
    # Combine features
    X = np.hstack([X_struct, pca_features])
    y = df_valid['goldstein'].values
    
    print(f"🔢 Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"   Structural: {X_struct.shape[1]}, Text (PCA): {pca_features.shape[1]}")
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
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
    struct_imp = importance[:len(struct_cols)].sum()
    text_imp = importance[len(struct_cols):].sum()
    
    print(f"\n🌟 Feature Importance:")
    print(f"   Structural: {struct_imp:.3f} ({struct_imp*100:.1f}%)")
    print(f"   Text (PCA): {text_imp:.3f} ({text_imp*100:.1f}%)")
    
    # Save model
    model_data = {
        'xgb_model': model,
        'pca': pca,
        'struct_cols': struct_cols,
        'embedder_name': "all-MiniLM-L6-v2"
    }
    
    joblib.dump(model_data, MODEL_PATH)
    save_cache(cache)
    
    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"💾 Cache saved with {len(cache)} entries")
    print(f"🎉 Pipeline complete! R² = {r2:.4f}")

if __name__ == "__main__":
    main() 