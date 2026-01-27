"""
Weather Intelligence Service - AgriVision v3.0
Real-time weather data and agricultural alerts.
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

ALERT_RULES = {
    "frost_warning": {
        "condition": lambda w: w.get("temp", 20) < 5,
        "severity": "critical",
        "message": "🥶 FROST WARNING: Protect crops with covers.",
        "affected_crops": ["Tomato", "Potato", "Banana"]
    },
    "heat_stress": {
        "condition": lambda w: w.get("temp", 20) > 40,
        "severity": "high",
        "message": "🔥 HEAT STRESS: Increase irrigation.",
        "affected_crops": ["Wheat", "Rice", "Vegetables"]
    },
    "fungal_risk": {
        "condition": lambda w: w.get("humidity", 50) > 85 and w.get("temp", 20) > 20,
        "severity": "high",
        "message": "🍄 FUNGAL RISK: Apply preventive fungicide.",
        "affected_crops": ["Tomato", "Potato", "Rice"]
    },
    "heavy_rain": {
        "condition": lambda w: w.get("rain", 0) > 50,
        "severity": "high",
        "message": "🌧️ HEAVY RAIN: Ensure drainage.",
        "affected_crops": ["Groundnut", "Onion"]
    },
    "optimal_spray": {
        "condition": lambda w: 15 < w.get("temp", 20) < 30 and w.get("wind_speed", 0) < 15,
        "severity": "info",
        "message": "✅ OPTIMAL: Good for spraying.",
        "affected_crops": ["All crops"]
    }
}


class WeatherService:
    def __init__(self):
        self.api_key = WEATHER_API_KEY
        self.cache = {}
    
    def get_current_weather(self, city="Delhi", country="IN"):
        if not self.api_key:
            return self._get_simulated_weather(city)
        
        try:
            response = requests.get(
                f"{WEATHER_BASE_URL}/weather",
                params={"q": f"{city},{country}", "appid": self.api_key, "units": "metric"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                weather = self._parse_weather(data, city, country)
                weather["alerts"] = self._generate_alerts(weather["current"])
                return weather
            else:
                print(f"❌ Weather API Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Weather Service Exception: {e}")
        
        print(f"⚠️ Falling back to simulated weather for {city}")
        return self._get_simulated_weather(city)
    
    def _parse_weather(self, data, city, country):
        return {
            "location": {"city": data.get("name", city), "country": country},
            "current": {
                "temp": data.get("main", {}).get("temp", 0),
                "humidity": data.get("main", {}).get("humidity", 0),
                "wind_speed": data.get("wind", {}).get("speed", 0) * 3.6,
                "rain": data.get("rain", {}).get("1h", 0),
                "description": data.get("weather", [{}])[0].get("description", "")
            },
            "timestamp": datetime.now().isoformat(),
            "source": "OpenWeatherMap API"
        }
    
    def _get_simulated_weather(self, city):
        import random
        month = datetime.now().month
        temp = random.uniform(25, 35) if month in [3,4,5] else random.uniform(20, 30)
        humidity = random.uniform(50, 85)
        
        current = {
            "temp": round(temp, 1),
            "humidity": round(humidity),
            "wind_speed": round(random.uniform(5, 20), 1),
            "rain": round(random.uniform(0, 10), 1) if humidity > 70 else 0,
            "description": "simulated"
        }
        
        return {
            "location": {"city": city, "country": "IN"},
            "current": current,
            "timestamp": datetime.now().isoformat(),
            "source": "Simulation",
            "alerts": self._generate_alerts(current)
        }
    
    def _generate_alerts(self, weather_data):
        alerts = []
        w = {
            "temp": weather_data.get("temp", 20),
            "humidity": weather_data.get("humidity", 50),
            "wind_speed": weather_data.get("wind_speed", 10),
            "rain": weather_data.get("rain", 0)
        }
        for rule_id, rule in ALERT_RULES.items():
            if rule["condition"](w):
                alerts.append({
                    "id": rule_id,
                    "severity": rule["severity"],
                    "message": rule["message"],
                    "affected_crops": rule["affected_crops"]
                })
        return alerts
    
    def get_forecast(self, city="Delhi", days=5):
        """Get 5-day forecast from API or simulation"""
        if not self.api_key:
            return self._get_simulated_forecast(city, days)
            
        try:
            response = requests.get(
                f"{WEATHER_BASE_URL}/forecast",
                params={"q": city, "appid": self.api_key, "units": "metric"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                forecast_list = []
                # API returns data every 3 hours. We pick one entry per day (e.g., noon)
                # or aggregate. Simple approach: Pick distinct days.
                seen_dates = set()
                
                for item in data['list']:
                    date_txt = item['dt_txt'].split(' ')[0]
                    if date_txt not in seen_dates:
                        seen_dates.add(date_txt)
                        forecast_list.append({
                            "date": date_txt,
                            "temp_high": round(item['main']['temp_max']),
                            "temp_low": round(item['main']['temp_min']),
                            "humidity": item['main']['humidity'],
                            "rain": item.get('rain', {}).get('3h', 0),
                            "description": item['weather'][0]['description']
                        })
                        if len(forecast_list) >= days:
                            break
                            
                return {"success": True, "location": city, "forecast": forecast_list}
            else:
                print(f"❌ Forecast API Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Forecast Error: {e}")
            
        return self._get_simulated_forecast(city, days)

    def _get_simulated_forecast(self, city, days):
        # Simulated forecast
        import random
        forecast_list = []
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            forecast_list.append({
                "date": date,
                "temp_high": round(random.uniform(28, 38), 1),
                "temp_low": round(random.uniform(18, 25), 1),
                "humidity": random.randint(50, 85),
                "rain": round(random.uniform(0, 15), 1),
                "description": random.choice(["clear", "cloudy", "light rain"])
            })
        return {"success": True, "location": city, "forecast": forecast_list}


weather_service = None

def get_weather_service():
    global weather_service
    if weather_service is None:
        weather_service = WeatherService()
    return weather_service


if __name__ == "__main__":
    print("🌦️ Weather Service Test")
    service = get_weather_service()
    weather = service.get_current_weather("Delhi")
    print(f"Temp: {weather['current']['temp']}°C")
    print(f"Alerts: {len(weather.get('alerts', []))}")
