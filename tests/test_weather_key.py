
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENWEATHER_API_KEY")
print(f"🔑 Testing API Key: {api_key}")

if not api_key:
    print("❌ No API key found in .env")
    exit()

url = "https://api.openweathermap.org/data/2.5/weather"
params = {"q": "London,UK", "appid": api_key}

try:
    response = requests.get(url, params=params)
    print(f"📡 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ API Key is VALID!")
        print(response.json())
    else:
        print(f"❌ API Request Failed: {response.text}")

except Exception as e:
    print(f"❌ Connection Error: {e}")
