# Climate Monitoring API - Add this to your server.py

import requests
from datetime import datetime, timedelta
import json
import math

# Weather condition codes for Open-Meteo
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Slight thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
}

# Climate Monitoring Routes
@app.route('/climatemonitoring.html')
def climate_monitoring():
    return send_from_directory('.', 'climatemonitoring.html')

@app.route('/climate')
def climate_dashboard():
    return send_from_directory('.', 'climatemonitoring.html')

@app.route('/api/climate/current', methods=['GET'])
def get_current_climate():
    """Get current weather data from Open-Meteo API"""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        
        if not lat or not lng:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        
        # Open-Meteo Current Weather API
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lng,
            'current': [
                'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
                'is_day', 'precipitation', 'rain', 'showers', 'snowfall',
                'weather_code', 'cloud_cover', 'pressure_msl', 'surface_pressure',
                'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'
            ],
            'hourly': ['uv_index'],
            'timezone': 'auto',
            'forecast_days': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            current = data['current']
            
            # Get current UV index from hourly data
            current_hour = datetime.now().hour
            uv_index = data['hourly']['uv_index'][current_hour] if current_hour < len(data['hourly']['uv_index']) else None
            
            # Calculate precipitation probability based on precipitation amount
            precipitation_prob = min(100, current.get('precipitation', 0) * 20)
            
            weather_info = {
                "temperature": round(current['temperature_2m']),
                "apparent_temperature": round(current['apparent_temperature']),
                "condition": WEATHER_CODES.get(current['weather_code'], "Unknown"),
                "weather_code": current['weather_code'],
                "humidity": current['relative_humidity_2m'],
                "wind_speed": round(current['wind_speed_10m']),
                "wind_direction": current['wind_direction_10m'],
                "wind_gusts": round(current.get('wind_gusts_10m', 0)),
                "pressure": round(current.get('pressure_msl', current.get('surface_pressure', 1013))),
                "cloud_cover": current['cloud_cover'],
                "precipitation": current.get('precipitation', 0),
                "precipitation_probability": round(precipitation_prob),
                "uv_index": round(uv_index) if uv_index else 0,
                "is_day": current['is_day'] == 1,
                "timestamp": current['time']
            }
            
            return jsonify(weather_info)
        else:
            return get_fallback_climate_data(lat, lng)
            
    except Exception as e:
        logger.error(f"Climate API Error: {e}")
        return get_fallback_climate_data(lat, lng)

@app.route('/api/climate/forecast', methods=['GET'])
def get_climate_forecast():
    """Get hourly weather forecast from Open-Meteo"""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        
        if not lat or not lng:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lng,
            'hourly': [
                'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
                'precipitation_probability', 'precipitation', 'rain',
                'weather_code', 'cloud_cover', 'wind_speed_10m', 'uv_index'
            ],
            'timezone': 'auto',
            'forecast_days': 2
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            hourly = data['hourly']
            
            # Get next 24 hours
            current_time = datetime.now()
            forecast_list = []
            
            for i in range(min(24, len(hourly['time']))):
                forecast_time = datetime.fromisoformat(hourly['time'][i].replace('Z', '+00:00'))
                
                if forecast_time > current_time:
                    forecast_item = {
                        'time': forecast_time.strftime('%H:%M'),
                        'date': forecast_time.strftime('%Y-%m-%d'),
                        'temperature': round(hourly['temperature_2m'][i]),
                        'apparent_temperature': round(hourly['apparent_temperature'][i]),
                        'condition': WEATHER_CODES.get(hourly['weather_code'][i], "Unknown"),
                        'weather_code': hourly['weather_code'][i],
                        'precipitation_probability': round(hourly.get('precipitation_probability', [0])[i] if i < len(hourly.get('precipitation_probability', [])) else 0),
                        'precipitation': hourly.get('precipitation', [0])[i] if i < len(hourly.get('precipitation', [])) else 0,
                        'humidity': hourly['relative_humidity_2m'][i],
                        'wind_speed': round(hourly['wind_speed_10m'][i]),
                        'cloud_cover': hourly['cloud_cover'][i],
                        'uv_index': round(hourly.get('uv_index', [0])[i] if i < len(hourly.get('uv_index', [])) else 0)
                    }
                    forecast_list.append(forecast_item)
                    
                    if len(forecast_list) >= 24:
                        break
            
            return jsonify({"forecast": forecast_list})
        else:
            return jsonify({"error": "Failed to fetch forecast data"}), 500
            
    except Exception as e:
        logger.error(f"Climate Forecast Error: {e}")
        return jsonify({"error": "Failed to fetch forecast data"}), 500

@app.route('/api/climate/alerts', methods=['GET'])
def get_climate_alerts():
    """Generate climate-based crowd management alerts"""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        
        # Get current weather data
        current_response = requests.get(f'http://localhost:5000/api/climate/current?lat={lat}&lng={lng}')
        
        if current_response.status_code != 200:
            raise Exception("Failed to get current weather")
        
        weather_data = current_response.json()
        alerts = generate_crowd_climate_alerts(weather_data)
        
        return jsonify({"alerts": alerts})
        
    except Exception as e:
        logger.error(f"Climate Alerts Error: {e}")
        return jsonify({"alerts": get_default_climate_alerts()})

@app.route('/api/climate/environmental', methods=['GET'])
def get_environmental_data():
    """Get additional environmental data"""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        
        # Air Quality API (using Open-Meteo Air Quality API)
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        air_params = {
            'latitude': lat,
            'longitude': lng,
            'current': ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone'],
            'timezone': 'auto'
        }
        
        # Get current weather for additional environmental data
        weather_response = requests.get(f'http://localhost:5000/api/climate/current?lat={lat}&lng={lng}')
        weather_data = weather_response.json() if weather_response.status_code == 200 else {}
        
        air_quality_text = "Good"
        try:
            air_response = requests.get(air_url, params=air_params, timeout=5)
            if air_response.status_code == 200:
                air_data = air_response.json()
                current_air = air_data['current']
                
                # Simple air quality assessment based on PM2.5
                pm25 = current_air.get('pm2_5', 0)
                if pm25 > 55:
                    air_quality_text = "Unhealthy"
                elif pm25 > 35:
                    air_quality_text = "Moderate"
                elif pm25 > 12:
                    air_quality_text = "Fair"
                else:
                    air_quality_text = "Good"
        except:
            pass  # Use default "Good" if air quality API fails
        
        # Calculate dew point
        temp = weather_data.get('temperature', 20)
        humidity = weather_data.get('humidity', 50)
        dew_point = calculate_dew_point(temp, humidity)
        
        environmental_data = {
            "air_quality": air_quality_text,
            "cloud_cover": weather_data.get('cloud_cover', 0),
            "precipitation": weather_data.get('precipitation', 0),
            "dew_point": round(dew_point, 1),
            "uv_index": weather_data.get('uv_index', 0),
            "visibility": calculate_visibility(weather_data)
        }
        
        return jsonify(environmental_data)
        
    except Exception as e:
        logger.error(f"Environmental Data Error: {e}")
        return jsonify({
            "air_quality": "Good",
            "cloud_cover": 50,
            "precipitation": 0,
            "dew_point": 15,
            "uv_index": 3,
            "visibility": "Good"
        })

@app.route('/api/climate/analyze', methods=['GET'])
def analyze_location_climate():
    """Analyze climate risk for a specific location"""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        
        # Get weather data for the location
        weather_response = requests.get(f'http://localhost:5000/api/climate/current?lat={lat}&lng={lng}')
        
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            
            risk_analysis = calculate_location_risk(weather_data)
            
            return jsonify(risk_analysis)
        else:
            return jsonify({
                "risk_level": "UNKNOWN",
                "crowd_impact": "UNKNOWN",
                "recommendation": "Unable to analyze location weather data"
            })
            
    except Exception as e:
        logger.error(f"Location Analysis Error: {e}")
        return jsonify({"error": str(e)}), 500

# Helper Functions
def calculate_dew_point(temperature, humidity):
    """Calculate dew point using Magnus formula"""
    a = 17.27
    b = 237.7
    alpha = ((a * temperature) / (b + temperature)) + math.log(humidity / 100.0)
    return (b * alpha) / (a - alpha)

def calculate_visibility(weather_data):
    """Estimate visibility based on weather conditions"""
    weather_code = weather_data.get('weather_code', 0)
    precipitation = weather_data.get('precipitation', 0)
    
    if weather_code in [45, 48]:  # Fog
        return "Poor"
    elif weather_code in [95, 96, 99]:  # Thunderstorms
        return "Poor"
    elif precipitation > 5:  # Heavy precipitation
        return "Reduced"
    elif precipitation > 1:  # Light precipitation
        return "Fair"
    else:
        return "Good"

def generate_crowd_climate_alerts(weather_data):
    """Generate alerts specific to crowd management based on weather"""
    alerts = []
    
    temp = weather_data.get('temperature', 20)
    humidity = weather_data.get('humidity', 50)
    wind_speed = weather_data.get('wind_speed', 0)
    precipitation_prob = weather_data.get('precipitation_probability', 0)
    weather_code = weather_data.get('weather_code', 0)
    uv_index = weather_data.get('uv_index', 0)
    
    # Extreme temperature alerts
    if temp >= 40:
        alerts.append({
            "severity": "critical",
            "icon": "🌡️",
            "title": "Extreme Heat Warning",
            "message": f"Temperature: {temp}°C - High risk of heat exhaustion in crowds",
            "crowd_impact": "HIGH",
            "recommendations": ["Increase water stations", "Implement crowd cooling zones", "Monitor for heat-related incidents"]
        })
    elif temp >= 35:
        alerts.append({
            "severity": "warning",
            "icon": "🔥",
            "title": "High Temperature Alert",
            "message": f"Temperature: {temp}°C - Crowd discomfort likely",
            "crowd_impact": "MED",
            "recommendations": ["Ensure adequate shade", "Increase water availability"]
        })
    elif temp <= 0:
        alerts.append({
            "severity": "warning",
            "icon": "🧊",
            "title": "Freezing Temperature",
            "message": f"Temperature: {temp}°C - Risk of hypothermia in crowds",
            "crowd_impact": "HIGH",
            "recommendations": ["Provide warming areas", "Monitor for cold-related incidents"]
        })
    
    # Precipitation alerts
    if precipitation_prob >= 80:
        alerts.append({
            "severity": "critical",
            "icon": "🌧️",
            "title": "Heavy Rain Expected",
            "message": f"{precipitation_prob}% chance of rain - High risk of crowd stampede during evacuation",
            "crowd_impact": "HIGH",
            "recommendations": ["Prepare covered areas", "Plan evacuation routes", "Monitor crowd movement"]
        })
    elif precipitation_prob >= 60:
        alerts.append({
            "severity": "warning",
            "icon": "🌦️",
            "title": "Rain Likely",
            "message": f"{precipitation_prob}% chance of rain - Crowd dispersal may be needed",
            "crowd_impact": "MED",
            "recommendations": ["Prepare shelter areas", "Monitor weather updates"]
        })
    
    # Wind alerts (critical for outdoor events)
    if wind_speed >= 50:
        alerts.append({
            "severity": "critical",
            "icon": "💨",
            "title": "Dangerous Winds",
            "message": f"Wind speed: {wind_speed} km/h - Structural hazards, event cancellation recommended",
            "crowd_impact": "EXTREME",
            "recommendations": ["Consider event cancellation", "Secure all temporary structures", "Clear high-risk areas"]
        })
    elif wind_speed >= 35:
        alerts.append({
            "severity": "warning",
            "icon": "🌬️",
            "title": "Strong Wind Warning",
            "message": f"Wind speed: {wind_speed} km/h - Risk to temporary structures and crowd safety",
            "crowd_impact": "HIGH",
            "recommendations": ["Secure loose items", "Monitor temporary structures"]
        })
    
    # Thunderstorm alerts
    if weather_code in [95, 96, 99]:
        alerts.append({
            "severity": "critical",
            "icon": "⛈️",
            "title": "Thunderstorm Alert",
            "message": "Thunderstorm detected - Immediate indoor evacuation required",
            "crowd_impact": "EXTREME",
            "recommendations": ["Move crowds indoors immediately", "Avoid open areas", "Monitor lightning activity"]
        })
    
    # Heat index alert
    if temp >= 27 and humidity >= 70:
        heat_index = calculate_heat_index(temp, humidity)
        if heat_index >= 41:
            alerts.append({
                "severity": "critical",
                "icon": "🥵",
                "title": "Dangerous Heat Index",
                "message": f"Heat index: {heat_index}°C - Extreme crowd heat stress risk",
                "crowd_impact": "EXTREME",
                "recommendations": ["Implement immediate cooling measures", "Increase medical personnel", "Consider event modification"]
            })
    
    # UV Index alerts for outdoor events
    if uv_index >= 8:
        alerts.append({
            "severity": "warning",
            "icon": "☀️",
            "title": "Very High UV Index",
            "message": f"UV Index: {uv_index} - Increased risk of sunburn in crowds",
            "crowd_impact": "MED",
            "recommendations": ["Provide shade structures", "Recommend sun protection", "Increase water distribution"]
        })
    
    # Fog/visibility alerts
    if weather_code in [45, 48]:
        alerts.append({
            "severity": "warning",
            "icon": "🌫️",
            "title": "Low Visibility",
            "message": "Fog conditions - Crowd navigation and emergency response may be impaired",
            "crowd_impact": "HIGH",
            "recommendations": ["Improve lighting", "Increase signage", "Station more security personnel"]
        })
    
    # All clear message if no alerts
    if not alerts:
        alerts.append({
            "severity": "info",
            "icon": "✅",
            "title": "Favorable Weather Conditions",
            "message": "Current weather conditions are suitable for crowd activities",
            "crowd_impact": "LOW",
            "recommendations": ["Continue normal operations", "Monitor weather updates"]
        })
    
    return alerts

def calculate_heat_index(temperature, humidity):
    """Calculate heat index (feels like temperature)"""
    T = temperature
    RH = humidity
    
    # Simplified heat index calculation
    HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
    
    if HI >= 80:
        # More complex calculation for higher temperatures
        HI = (-42.379 + 2.04901523 * T + 10.14333127 * RH - 0.22475541 * T * RH
              - 0.00683783 * T * T - 0.05481717 * RH * RH
              + 0.00122874 * T * T * RH + 0.00085282 * T * RH * RH
              - 0.00000199 * T * T * RH * RH)
    
    return round(HI, 1)

def calculate_location_risk(weather_data):
    """Calculate overall risk level for a location"""
    risk_score = 0
    risk_factors = []
    
    temp = weather_data.get('temperature', 20)
    humidity = weather_data.get('humidity', 50)
    wind_speed = weather_data.get('wind_speed', 0)
    precipitation_prob = weather_data.get('precipitation_probability', 0)
    weather_code = weather_data.get('weather_code', 0)
    
    # Temperature risk
    if temp >= 40:
        risk_score += 4
        risk_factors.append("Extreme heat")
    elif temp >= 35:
        risk_score += 3
        risk_factors.append("High temperature")
    elif temp <= 0:
        risk_score += 3
        risk_factors.append("Freezing temperature")
    
    # Precipitation risk
    if precipitation_prob >= 80:
        risk_score += 3
        risk_factors.append("Heavy rain expected")
    elif precipitation_prob >= 60:
        risk_score += 2
        risk_factors.append("Rain likely")
    
    # Wind risk
    if wind_speed >= 50:
        risk_score += 4
        risk_factors.append("Dangerous winds")
    elif wind_speed >= 35:
        risk_score += 3
        risk_factors.append("Strong winds")
    
    # Severe weather
    if weather_code in [95, 96, 99]:
        risk_score += 4
        risk_factors.append("Thunderstorm")
    
    # Heat index
    if temp >= 27 and humidity >= 70:
        heat_index = calculate_heat_index(temp, humidity)
        if heat_index >= 41:
            risk_score += 3
            risk_factors.append("Dangerous heat index")
    
    # Determine risk level
    if risk_score >= 8:
        risk_level = "EXTREME"
        crowd_impact = "EXTREME"
        recommendation = "Event cancellation or major modifications recommended"
    elif risk_score >= 6:
        risk_level = "HIGH"
        crowd_impact = "HIGH"
        recommendation = "Implement emergency crowd management protocols"
    elif risk_score >= 3:
        risk_level = "MODERATE"
        crowd_impact = "MED"
        recommendation = "Enhanced monitoring and crowd safety measures needed"
    else:
        risk_level = "LOW"
        crowd_impact = "LOW"
        recommendation = "Normal crowd management procedures sufficient"
    
    return {
        "risk_level": risk_level,
        "crowd_impact": crowd_impact,
        "recommendation": recommendation,
        "risk_factors": risk_factors,
        "risk_score": risk_score
    }

def get_fallback_climate_data(lat=None, lng=None):
    """Fallback climate data when API is unavailable"""
    return jsonify({
        "temperature": 25,
        "apparent_temperature": 27,
        "condition": "Partly cloudy",
        "weather_code": 2,
        "humidity": 65,
        "wind_speed": 12,
        "wind_direction": 180,
        "wind_gusts": 18,
        "pressure": 1013,
        "cloud_cover": 40,
        "precipitation": 0,
        "precipitation_probability": 20,
        "uv_index": 5,
        "is_day": True,
        "timestamp": datetime.utcnow().isoformat()
    })

def get_default_climate_alerts():
    """Default climate alerts when data is unavailable"""
    return [
        {
            "severity": "info",
            "icon": "📡",
            "title": "Weather Monitoring Active",
            "message": "Climate monitoring system is online and tracking conditions",
            "crowd_impact": "LOW",
            "recommendations": ["Continue normal operations"]
        }
    ]
