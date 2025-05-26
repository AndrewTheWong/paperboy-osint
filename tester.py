# test_acled_api.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()

ACLED_API_KEY = os.getenv("ACLED_API_KEY")
ACLED_EMAIL = os.getenv("ACLED_EMAIL")

if not ACLED_API_KEY or not ACLED_EMAIL:
    print("❌ Missing ACLED_API_KEY or ACLED_EMAIL in .env file.")
    exit()

BASE_URL = "https://api.acleddata.com/acled/read"

params = {
    "key": ACLED_API_KEY,
    "email": ACLED_EMAIL,
    "limit": 10,
    "format": "json",
    "event_date": "2023-01-01|2024-12-31"
}

print("📡 Requesting ACLED data...")
response = requests.get(BASE_URL, params=params)

if response.status_code != 200:
    print(f"❌ API Error {response.status_code}")
    print(response.text)
    exit()

data = response.json()

if "data" not in data or not data["data"]:
    print("⚠️ No data returned from ACLED.")
    exit()

print(f"✅ Received {len(data['data'])} events.")
print("📋 First record:")
for k, v in data["data"][0].items():
    print(f"  {k}: {v}")
