import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import json
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from dotenv import load_dotenv
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# PART 1: ENV AND FOLDER SETUP
load_dotenv()  # Load environment variables from .env
ACLED_API_KEY = os.getenv("ACLED_API_KEY")
ACLED_EMAIL = os.getenv("ACLED_EMAIL")

# Generate timestamp for versioning
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d")

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# PART 2: FETCH GDELT EVENTS
def fetch_gdelt(years_back=25, sample_limit=50000):
    """
    Fetch GDELT data going back multiple years, extract relevant fields, and save to CSV.
    
    Args:
        years_back: Number of years of historical data to fetch
        sample_limit: Target number of samples to collect
    """
    print(f"Fetching GDELT data going back {years_back} years (target: {sample_limit} samples)...")
    
    # Get the current date
    current_date = datetime.datetime.now()
    start_date = current_date - datetime.timedelta(days=365*years_back)
    
    # Format dates for GDELT query
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = current_date.strftime("%Y%m%d")
    
    all_gdelt_data = []
    total_samples = 0
    
    # For historical data, we'll use GDELT 2.0 Events and historical data sources
    print(f"Downloading historical GDELT data from {start_date_str} to {end_date_str}...")
    
    # Define multiple data source patterns to try
    data_sources = [
        # Standard monthly reduced GDELT 2.0 data
        lambda month: f"http://data.gdeltproject.org/gdeltv2/reduced/{month}.reduced.txt.zip",
        
        # GDELT 1.0 historical data (spanning 1979-2013)
        lambda month: f"http://data.gdeltproject.org/events/{month}.zip" if int(month[:4]) < 2014 else None,
        
        # GDELT 2.0 daily data for recent years
        lambda month: f"http://data.gdeltproject.org/gdeltv2/{month}0101.export.CSV.zip" if int(month[:4]) > 2014 else None
    ]
    
    # Query monthly data (to keep requests manageable)
    current_month = start_date
    
    while current_month < current_date and total_samples < sample_limit:
        month_str = current_month.strftime("%Y%m")
        month_success = False
        
        # Try different data source patterns
        for data_source_fn in data_sources:
            url = data_source_fn(month_str)
            if not url:
                continue
                
            try:
                print(f"Downloading GDELT data for {month_str} from {url}...")
                response = requests.get(url, timeout=180)
                
                if response.status_code == 200:
                    # Extract and process the CSV/TXT file
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                        for file_name in zip_file.namelist():
                            with zip_file.open(file_name) as data_file:
                                try:
                                    # Determine the format based on the file extension
                                    if file_name.endswith('.csv') or "export.CSV" in file_name:
                                        # GDELT 2.0 format
                                        cols = [
                                            "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
                                            "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
                                            "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
                                            "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
                                            "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
                                            "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
                                            "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
                                            "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
                                            "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
                                            "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
                                            "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
                                            "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName",
                                            "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
                                            "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type",
                                            "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code",
                                            "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
                                            "DATEADDED", "SOURCEURL"
                                        ]
                                        
                                        monthly_df = pd.read_csv(
                                            data_file, 
                                            sep="\t", 
                                            header=None,
                                            names=cols,
                                            usecols=["SQLDATE", "Actor1CountryCode", "Actor2CountryCode", "EventCode", "GoldsteinScale", "SOURCEURL"],
                                            encoding="latin-1",
                                            on_bad_lines='skip',
                                            low_memory=False
                                        )
                                    else:
                                        # GDELT 1.0 format or reduced format
                                        cols = [
                                            "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
                                            "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
                                            "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code",
                                            "Actor1Type2Code", "Actor1Type3Code", "Actor2Code", "Actor2Name",
                                            "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2Religion1Code",
                                            "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code",
                                            "Actor2Type3Code", "IsRootEvent", "EventCode", "EventBaseCode",
                                            "EventRootCode", "QuadClass", "GoldsteinScale", "NumMentions",
                                            "NumSources", "NumArticles", "AvgTone", "Actor1Geo_Type",
                                            "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_Lat",
                                            "Actor1Geo_Long", "Actor2Geo_Type", "Actor2Geo_FullName",
                                            "Actor2Geo_CountryCode", "Actor2Geo_Lat", "Actor2Geo_Long",
                                            "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
                                            "ActionGeo_Lat", "ActionGeo_Long", "DATEADDED", "SOURCEURL"
                                        ]
                                        
                                        monthly_df = pd.read_csv(
                                            data_file,
                                            sep="\t",
                                            header=None,
                                            names=cols,
                                            usecols=["Day", "Actor1CountryCode", "Actor2CountryCode", "EventCode", "GoldsteinScale", "SOURCEURL"],
                                            encoding="latin-1",
                                            on_bad_lines='skip',
                                            low_memory=False
                                        )
                                        monthly_df["SQLDATE"] = monthly_df["Day"]
                                    
                                    # Process the data
                                    monthly_df["label"] = monthly_df["EventCode"].astype(str).str.startswith("19").astype(int)
                                    
                                    # Extract text for each event
                                    monthly_df["text"] = "GDELT event between " + monthly_df["Actor1CountryCode"].fillna("Unknown") + " and " + \
                                                       monthly_df["Actor2CountryCode"].fillna("Unknown") + " with code " + \
                                                       monthly_df["EventCode"].astype(str)
                                    
                                    # Sample to avoid overloading with a single month
                                    # But ensure we get a balanced dataset with conflict events
                                    conflict_events = monthly_df[monthly_df["label"] == 1]
                                    non_conflict_events = monthly_df[monthly_df["label"] == 0]
                                    
                                    # Determine sample size - prioritize conflict events (which are rarer)
                                    max_conflicts = min(2000, len(conflict_events))
                                    max_non_conflicts = min(3000, len(non_conflict_events))
                                    
                                    # Sample from each group
                                    if len(conflict_events) > 0:
                                        conflict_sample = conflict_events.sample(max_conflicts) if max_conflicts > 0 else conflict_events
                                        non_conflict_sample = non_conflict_events.sample(max_non_conflicts) if max_non_conflicts > 0 else non_conflict_events
                                        monthly_sample = pd.concat([conflict_sample, non_conflict_sample])
                                    else:
                                        monthly_sample = non_conflict_events.sample(max_non_conflicts) if max_non_conflicts > 0 else non_conflict_events
                                    
                                    all_gdelt_data.append(monthly_sample)
                                    total_samples += len(monthly_sample)
                                    print(f"  Added {len(monthly_sample)} events from {month_str} (total: {total_samples})")
                                    
                                    month_success = True
                                    if total_samples >= sample_limit:
                                        break
                                        
                                except Exception as e:
                                    print(f"  Error processing file {file_name}: {e}")
                            
                            if total_samples >= sample_limit:
                                break
                    
                    if month_success:
                        break  # Successfully processed this month, move to next
                else:
                    print(f"  Could not download data for {month_str} from {url}: Status code {response.status_code}")
                    
            except Exception as e:
                print(f"  Error downloading data for {month_str} from {url}: {e}")
        
        # Move to next month
        if not month_success:
            print(f"  No data sources available for {month_str}, skipping to next month")
            
        current_month = current_month + datetime.timedelta(days=32)
        current_month = current_month.replace(day=1)
        
        # Break if we've reached our sample limit
        if total_samples >= sample_limit:
            print(f"Reached target sample limit of {sample_limit}")
            break
    
    # Combine all data
    if not all_gdelt_data:
        print("No GDELT data was successfully retrieved. Aborting.")
        return None
        
    gdelt_df = pd.concat(all_gdelt_data, ignore_index=True)
    
    # Balance the dataset one more time if needed
    conflict_events = gdelt_df[gdelt_df["label"] == 1]
    non_conflict_events = gdelt_df[gdelt_df["label"] == 0]
    print(f"Total conflict events: {len(conflict_events)}, non-conflict events: {len(non_conflict_events)}")
    
    # Ensure we have at least 40% conflict events for good training
    min_conflict_ratio = 0.4
    current_conflict_ratio = len(conflict_events) / len(gdelt_df) if len(gdelt_df) > 0 else 0
    
    if current_conflict_ratio < min_conflict_ratio and len(non_conflict_events) > 0:
        # Downsample non-conflict events
        target_non_conflict = int(len(conflict_events) * (1-min_conflict_ratio) / min_conflict_ratio)
        non_conflict_sample = non_conflict_events.sample(min(target_non_conflict, len(non_conflict_events)))
        gdelt_df = pd.concat([conflict_events, non_conflict_sample])
        print(f"Balanced dataset: {len(conflict_events)} conflict, {len(non_conflict_sample)} non-conflict")
    
    # Save to CSV
    gdelt_df.to_csv("data/gdelt_events.csv", index=False)
    print(f"✅ Saved {len(gdelt_df)} GDELT events to data/gdelt_events.csv")
    print(f"   - Conflict events: {len(gdelt_df[gdelt_df['label'] == 1])}")
    print(f"   - Non-conflict events: {len(gdelt_df[gdelt_df['label'] == 0])}")
    
    return gdelt_df

# PART 3: FETCH ACLED EVENTS
def fetch_acled(years_back=25, sample_limit=20000):
    """
    Fetch global ACLED conflict data via API, process, and save to JSON.
    
    Args:
        years_back: Number of years of historical data to fetch
        sample_limit: Target number of samples to collect
    """
    print(f"Fetching ACLED data going back {years_back} years (target: {sample_limit} samples)...")
    
    if not ACLED_API_KEY:
        print("Error: ACLED_API_KEY not found in .env file")
        return None
        
    if not ACLED_EMAIL:
        print("Error: ACLED_EMAIL not found in .env file")
        return None
    
    # Calculate start date
    current_date = datetime.datetime.now()
    start_date = current_date - datetime.timedelta(days=365*years_back)
    
    # Format dates for ACLED API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = current_date.strftime("%Y-%m-%d")
    
    # To get more data, we'll query by multiple regions and time chunks
    regions = [
        "Middle East", "Africa", "Asia", "Europe", "South America", 
        "North America", "Central America", "Oceania"
    ]
    
    # Break the time period into smaller chunks to avoid API limitations
    time_chunks = []
    chunk_size = 365 * 2  # 2 years per chunk
    current_chunk_start = start_date
    
    while current_chunk_start < current_date:
        chunk_end = min(current_chunk_start + datetime.timedelta(days=chunk_size), current_date)
        time_chunks.append((
            current_chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        ))
        current_chunk_start = chunk_end + datetime.timedelta(days=1)
    
    url = "https://api.acleddata.com/acled/read"
    all_events = []
    total_samples = 0
    
    # Query each region and time chunk
    for region in regions:
        for time_start, time_end in time_chunks:
            if total_samples >= sample_limit:
                break
                
            print(f"Fetching ACLED data for {region} from {time_start} to {time_end}...")
            page = 1
            limit = 1000  # API limit per request
            
            while True:
                if total_samples >= sample_limit:
                    break
                    
                params = {
                    "key": ACLED_API_KEY,
                    "email": ACLED_EMAIL,
                    "page": page,
                    "limit": limit,
                    "format": "json",
                    "event_date": f"{time_start}|{time_end}",  # Filter by date range
                    "region": region
                }
                
                try:
                    response = requests.get(url, params=params, timeout=60)
                    
                    if response.status_code != 200:
                        print(f"Error fetching ACLED data: {response.status_code}")
                        print(f"Response: {response.text}")
                        break
                    
                    data = response.json()
                    events = data.get("data", [])
                    
                    if not events:
                        print(f"  No more events for {region} in {time_start} to {time_end}")
                        break  # No more events to fetch
                    
                    all_events.extend(events)
                    total_samples += len(events)
                    print(f"  Fetched {len(events)} events from page {page} (total: {total_samples})")
                    
                    # Check if we've reached the last page
                    if len(events) < limit:
                        break
                    
                    page += 1
                    
                except Exception as e:
                    print(f"Error fetching ACLED data: {e}")
                    break
    
    # Process events
    processed_events = []
    conflict_count = 0
    non_conflict_count = 0
    
    # Define conflict event types
    conflict_types = ["Battles", "Explosions/Remote violence", "Violence against civilians"]
    
    for event in all_events:
        is_conflict = event.get("event_type") in conflict_types
        
        # Track counts
        if is_conflict:
            conflict_count += 1
        else:
            non_conflict_count += 1
            
        processed_event = {
            "event_date": event.get("event_date"),
            "event_type": event.get("event_type"),
            "notes": event.get("notes"),
            "fatalities": event.get("fatalities", 0),
            "actor1": event.get("actor1"),
            "actor2": event.get("actor2"),
            "region": event.get("region", "Unknown"),
            "country": event.get("country", "Unknown"),
            "label": 1 if is_conflict else 0,
            "text": f"ACLED event in {event.get('country', 'Unknown')}: {event.get('notes', '')}"
        }
        
        processed_events.append(processed_event)
    
    # Balance the dataset if needed
    if conflict_count > 0 and non_conflict_count > 0:
        conflict_events = [e for e in processed_events if e["label"] == 1]
        non_conflict_events = [e for e in processed_events if e["label"] == 0]
        
        # Ensure we have at least 40% conflict events for good training
        min_conflict_ratio = 0.4
        current_conflict_ratio = conflict_count / len(processed_events)
        
        if current_conflict_ratio < min_conflict_ratio:
            # Downsample non-conflict events
            target_non_conflict = int(conflict_count * (1-min_conflict_ratio) / min_conflict_ratio)
            if target_non_conflict < non_conflict_count:
                import random
                random.shuffle(non_conflict_events)
                non_conflict_events = non_conflict_events[:target_non_conflict]
                processed_events = conflict_events + non_conflict_events
                print(f"Balanced ACLED dataset: {len(conflict_events)} conflict, {len(non_conflict_events)} non-conflict")
    
    # Save to JSON
    with open("data/acled_events.json", "w") as f:
        json.dump(processed_events, f)
    
    print(f"✅ Saved {len(processed_events)} ACLED events to data/acled_events.json")
    print(f"   - Conflict events: {len([e for e in processed_events if e['label'] == 1])}")
    print(f"   - Non-conflict events: {len([e for e in processed_events if e['label'] == 0])}")
    
    # Also return as DataFrame for further processing
    acled_df = pd.DataFrame(processed_events)
    return acled_df

# PART 4: PREPROCESS FOR XGBOOST
def prepare_xgb_data(df) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare tag-based features for XGBoost training.
    """
    # Define tag vocabulary
    TAG_VOCAB = [
        "military movement", "conflict", "cyberattack", "protest",
        "diplomatic meeting", "nuclear", "ceasefire"
    ]
    
    # Create feature array
    X = np.zeros((len(df), len(TAG_VOCAB) + 1))  # +1 for confidence
    
    for i, text in enumerate(df["text"]):
        if pd.isna(text):
            continue
            
        text_lower = str(text).lower()
        
        # Encode binary tag presence
        for j, tag in enumerate(TAG_VOCAB):
            if tag in text_lower:
                X[i, j] = 1
        
        # Add confidence score (use GoldsteinScale if available, else default)
        if "GoldsteinScale" in df.columns:
            # Normalize to [0, 1] range (Goldstein scale is -10 to +10)
            X[i, -1] = (df.iloc[i]["GoldsteinScale"] + 10) / 20
        else:
            # Default confidence
            X[i, -1] = 0.5
    
    # Get labels
    y = df["label"].values
    
    return X, y

# Plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig(f"logs/{model_name}_confusion_matrix_{TIMESTAMP}.png")
    plt.close()

# PART 5: TRAIN XGBOOST
def train_xgboost():
    """
    Train and save an XGBoost model for conflict prediction.
    """
    print("Training XGBoost model...")
    
    # Load data
    gdelt_df = pd.read_csv("data/gdelt_events.csv")
    
    try:
        with open("data/acled_events.json", "r") as f:
            acled_data = json.load(f)
        acled_df = pd.DataFrame(acled_data)
        
        # Check if ACLED data is empty
        has_acled_data = len(acled_df) > 0
    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠️ ACLED data file not found or empty. Proceeding with GDELT data only.")
        has_acled_data = False
    
    # Prepare GDELT data for XGBoost
    gdelt_subset = gdelt_df[["text", "label", "GoldsteinScale"]]
    X_gdelt, y_gdelt = prepare_xgb_data(gdelt_subset)
    
    # If ACLED data exists, add it to the training set
    if has_acled_data and "text" in acled_df.columns and "label" in acled_df.columns:
        acled_subset = acled_df[["text", "label"]]
        X_acled, y_acled = prepare_xgb_data(acled_subset)
        
        # Combine datasets
        X = np.vstack([X_gdelt, X_acled])
        y = np.concatenate([y_gdelt, y_acled])
        print(f"Training on combined dataset: {len(X)} samples")
    else:
        # Use only GDELT data
        X = X_gdelt
        y = y_gdelt
        print(f"Training on GDELT data only: {len(X)} samples")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"XGBoost ROC AUC: {auc:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, "xgboost")
    
    # Save model with version
    model_path = f"models/xgboost_conflict_model_{TIMESTAMP}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ XGBoost model saved to {model_path}")
    
    # Also save as the latest version
    joblib.dump(model, "models/xgboost_conflict_model.pkl")
    
    # Save metadata
    metadata = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "model": "xgboost",
        "version": TIMESTAMP,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "roc_auc": float(auc),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1_score": float(report["weighted avg"]["f1-score"])
    }
    
    metadata_path = f"models/metadata_xgboost_{TIMESTAMP}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Update latest.json
    with open("models/latest.json", "w") as f:
        json.dump({"xgboost": TIMESTAMP, "last_updated": datetime.datetime.now().strftime("%Y-%m-%d")}, f, indent=2)
    
    return metadata

# PART 6: TRAIN TRANSFORMER MODEL
def train_transformer():
    """
    Train and save a transformer model for conflict prediction.
    """
    print("Training Transformer model...")
    
    # Load data
    gdelt_df = pd.read_csv("data/gdelt_events.csv")
    
    # Initialize empty acled_df
    acled_df = pd.DataFrame()
    
    try:
        with open("data/acled_events.json", "r") as f:
            acled_data = json.load(f)
        acled_df = pd.DataFrame(acled_data)
        
        # Check if ACLED data is empty or missing required columns
        has_acled_data = len(acled_df) > 0 and "text" in acled_df.columns and "label" in acled_df.columns
    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠️ ACLED data file not found or empty. Proceeding with GDELT data only.")
        has_acled_data = False
    
    # Combine text and labels, using only GDELT if ACLED is unavailable
    if has_acled_data:
        combined_data = {
            "text": list(gdelt_df["text"]) + list(acled_df["text"]),
            "label": list(gdelt_df["label"]) + list(acled_df["label"])
        }
        print(f"Training on combined dataset: {len(combined_data['text'])} samples")
    else:
        combined_data = {
            "text": list(gdelt_df["text"]),
            "label": list(gdelt_df["label"])
        }
        print(f"Training on GDELT data only: {len(combined_data['text'])} samples")
    
    # Create Dataset
    dataset = Dataset.from_dict(combined_data)
    
    # Split dataset
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Load tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # Tokenize the dataset
    tokenized_train = train_test["train"].map(tokenize_function, batched=True)
    tokenized_test = train_test["test"].map(tokenize_function, batched=True)
    
    # Load model for fine-tuning
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Create versioned model directory
    model_dir = f"models/transformer_conflict_model_{TIMESTAMP}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="logs",
    )
    
    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Transformer evaluation results: {eval_results}")
    
    # Make predictions on test set
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=1)
    
    # Calculate ROC AUC
    probs = predictions.predictions[:, 1]
    auc = roc_auc_score(train_test["test"]["label"], probs)
    print(f"Transformer ROC AUC: {auc:.4f}")
    
    # Generate classification report
    report = classification_report(train_test["test"]["label"], preds, output_dict=True)
    print(classification_report(train_test["test"]["label"], preds))
    
    # Plot confusion matrix
    plot_confusion_matrix(train_test["test"]["label"], preds, "transformer")
    
    # Save the model with version
    model_path = f"models/transformer_conflict_model_{TIMESTAMP}"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✅ Transformer model saved to {model_path}")
    
    # Also save as the latest version
    latest_model_path = "models/transformer_conflict_model"
    model.save_pretrained(latest_model_path)
    tokenizer.save_pretrained(latest_model_path)
    
    # Save metadata
    metadata = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "model": "transformer",
        "version": TIMESTAMP,
        "train_size": len(tokenized_train),
        "test_size": len(tokenized_test),
        "roc_auc": float(auc),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1_score": float(report["weighted avg"]["f1-score"]),
        "eval_loss": float(eval_results["eval_loss"])
    }
    
    metadata_path = f"models/metadata_transformer_{TIMESTAMP}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Update latest.json to include transformer version
    try:
        with open("models/latest.json", "r") as f:
            latest = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        latest = {}
    
    latest["transformer"] = TIMESTAMP
    latest["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d")
    
    with open("models/latest.json", "w") as f:
        json.dump(latest, f, indent=2)
    
    return metadata

# PART 7: ENTRY POINT
if __name__ == "__main__":
    # Step 1: Fetch data if not already present or refresh requested
    # You can change these to True to force data refresh
    refresh_gdelt = True  # Setting to True to get more data
    refresh_acled = True  # Setting to True to get more data
    
    if refresh_gdelt or not os.path.exists("data/gdelt_events.csv"):
        fetch_gdelt(years_back=25, sample_limit=30000)
    else:
        print("GDELT data already exists, skipping download")
    
    if refresh_acled or not os.path.exists("data/acled_events.json"):
        fetch_acled(years_back=25, sample_limit=20000)
    else:
        print("ACLED data already exists, skipping download")
    
    # Step 2: Train models
    xgboost_metadata = train_xgboost()
    transformer_metadata = train_transformer()
    
    print("✅ All models trained and saved successfully.")
    print(f"XGBoost ROC AUC: {xgboost_metadata['roc_auc']:.4f}")
    print(f"Transformer ROC AUC: {transformer_metadata['roc_auc']:.4f}")
    print(f"Models saved with version: {TIMESTAMP}")
    print(f"Metadata saved to models/metadata_xgboost_{TIMESTAMP}.json and models/metadata_transformer_{TIMESTAMP}.json") 