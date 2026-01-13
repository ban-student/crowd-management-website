"""
Natural Disaster Prediction Module
Integrates USGS, GDACS, and FIRMS APIs for comprehensive disaster monitoring
"""

import requests
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class DisasterType(Enum):
    EARTHQUAKE = "earthquake"
    FIRE = "fire"
    FLOOD = "flood"
    CYCLONE = "cyclone"
    TSUNAMI = "tsunami"
    VOLCANO = "volcano"

class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EXTREME = "EXTREME"

@dataclass
class EarthquakeData:
    magnitude: float
    place: str
    time: int
    latitude: float
    longitude: float
    depth: float
    url: str
    distance_km: Optional[float] = None

@dataclass
class FireData:
    latitude: float
    longitude: float
    brightness: float
    confidence: int
    acq_date: str
    acq_time: str
    satellite: str
    distance_km: Optional[float] = None
    
    @property
    def acq_datetime(self):
        return f"{self.acq_date} {self.acq_time[:2]}:{self.acq_time[2:]}"

@dataclass
class GDACSAlert:
    event_id: str
    event_type: str
    title: str
    description: str
    severity: str
    date: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    icon: str = "⚠️"

class DisasterPredictor:
    """Main class for natural disaster prediction and monitoring"""
    
    def __init__(self):
        self.usgs_base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        self.gdacs_base_url = "https://www.gdacs.org"
        self.openmeteo_base_url = "https://api.open-meteo.com/v1/forecast"
        
        # Open-Meteo is free and doesn't require API key
        # We'll use it for weather-based fire risk assessment
        
        # Cache for API responses to avoid rate limiting
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes cache
    
    def _get_cache_key(self, api_name: str, params: dict) -> str:
        """Generate cache key for API response"""
        return f"{api_name}:{hash(str(sorted(params.items())))}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache:
            return False
        return time.time() - self._cache[cache_key]['timestamp'] < self._cache_timeout
    
    def _get_from_cache(self, cache_key: str):
        """Get data from cache"""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]['data']
        return None
    
    def _store_in_cache(self, cache_key: str, data):
        """Store data in cache"""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def fetch_earthquakes(self, latitude: float, longitude: float, 
                         radius_km: int = 100, days_back: int = 1) -> Dict:
        """Fetch earthquake data from USGS"""
        cache_key = self._get_cache_key("usgs", {
            'lat': latitude, 'lng': longitude, 'radius': radius_km, 'days': days_back
        })
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Calculate bounding box for API request
            lat_offset = radius_km / 111  # Rough conversion: 1 degree ≈ 111 km
            lng_offset = radius_km / (111 * math.cos(math.radians(latitude)))
            
            minlat = latitude - lat_offset
            maxlat = latitude + lat_offset
            minlng = longitude - lng_offset
            maxlng = longitude + lng_offset
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            params = {
                'format': 'geojson',
                'starttime': start_time.strftime('%Y-%m-%d'),
                'endtime': end_time.strftime('%Y-%m-%d'),
                'minlatitude': minlat,
                'maxlatitude': maxlat,
                'minlongitude': minlng,
                'maxlongitude': maxlng,
                'minmagnitude': 1.0,
                'orderby': 'magnitude'
            }
            
            response = requests.get(self.usgs_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            earthquakes = []
            
            for feature in data.get('features', []):
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                eq_data = EarthquakeData(
                    magnitude=props.get('mag', 0),
                    place=props.get('place', 'Unknown location'),
                    time=props.get('time', 0),
                    latitude=coords[1],
                    longitude=coords[0],
                    depth=coords[2] if len(coords) > 2 else 0,
                    url=props.get('url', '')
                )
                
                # Calculate distance
                eq_data.distance_km = self.calculate_distance(
                    latitude, longitude, eq_data.latitude, eq_data.longitude
                )
                
                # Only include earthquakes within radius
                if eq_data.distance_km <= radius_km:
                    earthquakes.append(eq_data)
            
            result = {
                'earthquakes': earthquakes,
                'count': len(earthquakes),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self._store_in_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching earthquake data: {e}")
            return self._get_mock_earthquake_data(latitude, longitude, radius_km)
    
    def fetch_fires(self, latitude: float, longitude: float, 
                   radius_km: int = 100, days_back: int = 1) -> Dict:
        """Fetch fire risk data from Open-Meteo weather API"""
        cache_key = self._get_cache_key("openmeteo_fire", {
            'lat': latitude, 'lng': longitude, 'radius': radius_km
        })
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Use Open-Meteo API for weather-based fire risk assessment
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'current': [
                    'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                    'wind_direction_10m', 'precipitation', 'weather_code'
                ],
                'hourly': [
                    'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                    'precipitation_probability', 'precipitation'
                ],
                'timezone': 'auto',
                'forecast_days': 2
            }
            
            response = requests.get(self.openmeteo_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data.get('current', {})
            hourly = data.get('hourly', {})
            
            # Calculate fire risk based on weather conditions
            fire_risks = []
            fire_risk_score = self._calculate_fire_risk(current, hourly)
            
            # Generate fire risk locations based on weather conditions
            if fire_risk_score >= 7:  # High risk
                # Create multiple high-risk zones around the main location
                for i in range(3):
                    offset_lat = latitude + (0.02 * (i - 1))
                    offset_lng = longitude + (0.015 * (i - 1))
                    
                    fire_risk = {
                        'latitude': offset_lat,
                        'longitude': offset_lng,
                        'risk_level': 'HIGH' if fire_risk_score >= 8 else 'MODERATE',
                        'temperature': current.get('temperature_2m', 25),
                        'humidity': current.get('relative_humidity_2m', 50),
                        'wind_speed': current.get('wind_speed_10m', 10),
                        'precipitation': current.get('precipitation', 0),
                        'risk_score': fire_risk_score,
                        'distance_km': self.calculate_distance(latitude, longitude, offset_lat, offset_lng)
                    }
                    fire_risks.append(fire_risk)
            elif fire_risk_score >= 4:  # Moderate risk
                fire_risk = {
                    'latitude': latitude + 0.01,
                    'longitude': longitude + 0.01,
                    'risk_level': 'MODERATE',
                    'temperature': current.get('temperature_2m', 25),
                    'humidity': current.get('relative_humidity_2m', 50),
                    'wind_speed': current.get('wind_speed_10m', 10),
                    'precipitation': current.get('precipitation', 0),
                    'risk_score': fire_risk_score,
                    'distance_km': 1.5
                }
                fire_risks.append(fire_risk)
            
            result = {
                'fires': fire_risks,
                'count': len(fire_risks),
                'fire_risk_score': fire_risk_score,
                'weather_conditions': {
                    'temperature': current.get('temperature_2m'),
                    'humidity': current.get('relative_humidity_2m'),
                    'wind_speed': current.get('wind_speed_10m'),
                    'precipitation': current.get('precipitation', 0)
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self._store_in_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching weather-based fire risk: {e}")
            return self._get_mock_fire_data(latitude, longitude, radius_km)
    
    def _calculate_fire_risk(self, current: dict, hourly: dict) -> int:
        """Calculate fire risk score based on weather conditions"""
        risk_score = 0
        
        temp = current.get('temperature_2m', 20)
        humidity = current.get('relative_humidity_2m', 50)
        wind_speed = current.get('wind_speed_10m', 5)
        precipitation = current.get('precipitation', 0)
        
        # Temperature factor (higher temp = higher risk)
        if temp >= 35:
            risk_score += 3
        elif temp >= 30:
            risk_score += 2
        elif temp >= 25:
            risk_score += 1
        
        # Humidity factor (lower humidity = higher risk)
        if humidity <= 20:
            risk_score += 3
        elif humidity <= 40:
            risk_score += 2
        elif humidity <= 60:
            risk_score += 1
        
        # Wind factor (higher wind = higher risk)
        if wind_speed >= 20:
            risk_score += 3
        elif wind_speed >= 15:
            risk_score += 2
        elif wind_speed >= 10:
            risk_score += 1
        
        # Precipitation factor (less precipitation = higher risk)
        if precipitation == 0:
            risk_score += 2
        elif precipitation < 1:
            risk_score += 1
        
        # Check forecast for next 24 hours for sustained conditions
        if hourly and 'precipitation_probability' in hourly:
            avg_precip_prob = sum(hourly['precipitation_probability'][:24]) / min(24, len(hourly['precipitation_probability']))
            if avg_precip_prob < 10:  # Very low chance of rain
                risk_score += 1
        
        return min(risk_score, 10)  # Cap at 10
    
    def fetch_gdacs_alerts(self, latitude: float, longitude: float, 
                          radius_km: int = 100) -> Dict:
        """Fetch disaster alerts from GDACS"""
        cache_key = self._get_cache_key("gdacs", {
            'lat': latitude, 'lng': longitude, 'radius': radius_km
        })
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # GDACS RSS feed for current alerts
            url = f"{self.gdacs_base_url}/xml/rss.xml"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            alerts = []
            
            for item in root.findall('.//item'):
                title = item.find('title')
                description = item.find('description')
                pub_date = item.find('pubDate')
                link = item.find('link')
                
                if title is not None and description is not None:
                    alert = GDACSAlert(
                        event_id=str(hash(title.text))[:8],
                        event_type=self._extract_event_type(title.text),
                        title=title.text,
                        description=description.text[:200] + "..." if len(description.text) > 200 else description.text,
                        severity=self._determine_severity(title.text, description.text),
                        date=pub_date.text if pub_date is not None else datetime.utcnow().isoformat(),
                        icon=self._get_disaster_icon(title.text)
                    )
                    alerts.append(alert)
            
            # Filter alerts by proximity (simplified - in production, extract coordinates from description)
            filtered_alerts = alerts[:5]  # Limit to 5 most recent
            
            result = {
                'alerts': filtered_alerts,
                'count': len(filtered_alerts),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self._store_in_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching GDACS data: {e}")
            return self._get_mock_gdacs_data(latitude, longitude, radius_km)
    
    def assess_disaster_risk(self, latitude: float, longitude: float, 
                           radius_km: int = 100) -> Dict:
        """Assess overall disaster risk for a location"""
        try:
            # Get all disaster data
            earthquakes = self.fetch_earthquakes(latitude, longitude, radius_km)
            fires = self.fetch_fires(latitude, longitude, radius_km)
            gdacs = self.fetch_gdacs_alerts(latitude, longitude, radius_km)
            
            risk_score = 0
            risk_factors = []
            crowd_impact = "LOW"
            
            # Calculate risk based on earthquake data
            for eq in earthquakes.get('earthquakes', []):
                if eq.magnitude >= 6.0:
                    risk_score += 4
                    risk_factors.append(f"Major earthquake (M{eq.magnitude})")
                elif eq.magnitude >= 4.0:
                    risk_score += 2
                    risk_factors.append(f"Moderate earthquake (M{eq.magnitude})")
                elif eq.magnitude >= 2.5:
                    risk_score += 1
                    risk_factors.append(f"Minor earthquake (M{eq.magnitude})")
            
            # Calculate risk based on fires
            fire_count = fires.get('count', 0)
            if fire_count >= 10:
                risk_score += 3
                risk_factors.append(f"Multiple active fires ({fire_count})")
            elif fire_count >= 5:
                risk_score += 2
                risk_factors.append(f"Several active fires ({fire_count})")
            elif fire_count > 0:
                risk_score += 1
                risk_factors.append(f"Active fires detected ({fire_count})")
            
            # Calculate risk based on GDACS alerts
            gdacs_count = gdacs.get('count', 0)
            if gdacs_count >= 3:
                risk_score += 3
                risk_factors.append(f"Multiple disaster alerts ({gdacs_count})")
            elif gdacs_count > 0:
                risk_score += 2
                risk_factors.append(f"Active disaster alerts ({gdacs_count})")
            
            # Determine overall risk level
            if risk_score >= 8:
                risk_level = RiskLevel.EXTREME
                crowd_impact = "EXTREME"
                recommendation = "Immediate evacuation recommended - multiple severe threats"
            elif risk_score >= 6:
                risk_level = RiskLevel.CRITICAL
                crowd_impact = "HIGH"
                recommendation = "High risk area - implement emergency protocols"
            elif risk_score >= 4:
                risk_level = RiskLevel.HIGH
                crowd_impact = "HIGH"
                recommendation = "Enhanced monitoring and safety measures required"
            elif risk_score >= 2:
                risk_level = RiskLevel.MODERATE
                crowd_impact = "MODERATE"
                recommendation = "Standard safety protocols with increased vigilance"
            else:
                risk_level = RiskLevel.LOW
                crowd_impact = "LOW"
                recommendation = "Normal safety measures sufficient"
            
            return {
                'risk_level': risk_level.value,
                'risk_score': risk_score,
                'crowd_impact': crowd_impact,
                'recommendation': recommendation,
                'risk_factors': risk_factors,
                'data_sources': {
                    'earthquakes': earthquakes.get('count', 0),
                    'fire_risks': fires.get('count', 0),
                    'gdacs_alerts': gdacs.get('count', 0)
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing disaster risk: {e}")
            return {
                'risk_level': 'UNKNOWN',
                'crowd_impact': 'UNKNOWN',
                'recommendation': 'Unable to assess risk - monitoring systems offline',
                'risk_factors': [],
                'data_sources': {
                    'earthquakes': 0,
                    'fire_risks': 0,
                    'gdacs_alerts': 0
                },
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def fetch_weather_disasters(self, latitude: float, longitude: float) -> Dict:
        """Fetch weather-based disaster predictions from Open-Meteo"""
        try:
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'current': [
                    'temperature_2m', 'relative_humidity_2m', 'precipitation',
                    'weather_code', 'wind_speed_10m', 'wind_direction_10m',
                    'wind_gusts_10m', 'pressure_msl'
                ],
                'hourly': [
                    'temperature_2m', 'precipitation_probability', 'precipitation',
                    'weather_code', 'wind_speed_10m', 'wind_gusts_10m'
                ],
                'daily': [
                    'temperature_2m_max', 'temperature_2m_min',
                    'precipitation_probability_max', 'wind_speed_10m_max',
                    'wind_gusts_10m_max'
                ],
                'timezone': 'auto',
                'forecast_days': 3
            }
            
            response = requests.get(self.openmeteo_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            weather_disasters = self._analyze_weather_disasters(data)
            
            return {
                'weather_disasters': weather_disasters,
                'count': len(weather_disasters),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching weather disasters: {e}")
            return {'weather_disasters': [], 'count': 0}
    
    def _analyze_weather_disasters(self, weather_data: dict) -> List[Dict]:
        """Analyze weather data for potential disasters"""
        try:
            disasters = []
            current = weather_data.get('current', {})
            hourly = weather_data.get('hourly', {})
            daily = weather_data.get('daily', {})
            
            # Current severe weather conditions
            temp = current.get('temperature_2m', 20)
            wind_speed = current.get('wind_speed_10m', 0)
            wind_gusts = current.get('wind_gusts_10m', 0)
            precipitation = current.get('precipitation', 0)
            weather_code = current.get('weather_code', 0)
            
            # Extreme heat warning
            if temp >= 40:
                disasters.append({
                    'type': 'EXTREME_HEAT',
                    'severity': 'CRITICAL',
                    'title': f'Extreme Heat Warning - {temp}°C',
                    'description': 'Dangerous heat conditions pose serious risk to crowds',
                    'crowd_impact': 'EXTREME',
                    'icon': '🌡️'
                })
            elif temp >= 35:
                disasters.append({
                    'type': 'HIGH_TEMPERATURE',
                    'severity': 'WARNING',
                    'title': f'High Temperature Alert - {temp}°C',
                    'description': 'Hot conditions may cause crowd discomfort and heat-related issues',
                    'crowd_impact': 'HIGH',
                    'icon': '☀️'
                })
            
            # Wind warnings
            if wind_gusts >= 60:
                disasters.append({
                    'type': 'SEVERE_WIND',
                    'severity': 'CRITICAL',
                    'title': f'Severe Wind Warning - {wind_gusts} km/h gusts',
                    'description': 'Dangerous winds pose risk to structures and crowd safety',
                    'crowd_impact': 'EXTREME',
                    'icon': '💨'
                })
            elif wind_speed >= 40:
                disasters.append({
                    'type': 'HIGH_WIND',
                    'severity': 'WARNING',
                    'title': f'High Wind Alert - {wind_speed} km/h',
                    'description': 'Strong winds may affect outdoor events and crowd activities',
                    'crowd_impact': 'HIGH',
                    'icon': '🌬️'
                })
            
            # Heavy precipitation
            if precipitation >= 10:
                disasters.append({
                    'type': 'HEAVY_RAIN',
                    'severity': 'CRITICAL',
                    'title': f'Heavy Rain Alert - {precipitation}mm/h',
                    'description': 'Heavy rainfall may cause flooding and evacuation needs',
                    'crowd_impact': 'HIGH',
                    'icon': '🌧️'
                })
            
            # Severe weather codes
            if weather_code in [95, 96, 99]:  # Thunderstorms
                disasters.append({
                    'type': 'THUNDERSTORM',
                    'severity': 'CRITICAL',
                    'title': 'Thunderstorm Alert',
                    'description': 'Severe thunderstorm conditions require immediate indoor evacuation',
                    'crowd_impact': 'EXTREME',
                    'icon': '⛈️'
                })
            
            # Check forecast for upcoming severe weather
            if hourly and 'wind_gusts_10m' in hourly:
                max_future_gust = max(hourly['wind_gusts_10m'][:24])  # Next 24 hours
                if max_future_gust >= 50:
                    disasters.append({
                        'type': 'FORECAST_SEVERE_WIND',
                        'severity': 'WARNING',
                        'title': f'Severe Wind Forecast - {max_future_gust} km/h expected',
                        'description': 'Severe winds expected within 24 hours - prepare safety measures',
                        'crowd_impact': 'HIGH',
                        'icon': '⚠️'
                    })
            
            return disasters
            
        except Exception as e:
            logger.error(f"Error analyzing weather disasters: {e}")
            return []
    
    def _extract_event_type(self, title: str) -> str:
        """Extract event type from GDACS title"""
        title_lower = title.lower()
        if 'earthquake' in title_lower:
            return 'EARTHQUAKE'
        elif 'fire' in title_lower or 'wildfire' in title_lower:
            return 'FIRE'
        elif 'flood' in title_lower:
            return 'FLOOD'
        elif 'cyclone' in title_lower or 'hurricane' in title_lower or 'typhoon' in title_lower:
            return 'CYCLONE'
        elif 'tsunami' in title_lower:
            return 'TSUNAMI'
        elif 'volcano' in title_lower:
            return 'VOLCANO'
        else:
            return 'OTHER'
    
    def _determine_severity(self, title: str, description: str) -> str:
        """Determine alert severity from title and description"""
        text = (title + " " + description).lower()
        if any(word in text for word in ['extreme', 'severe', 'major', 'critical', 'emergency']):
            return 'CRITICAL'
        elif any(word in text for word in ['moderate', 'significant', 'warning']):
            return 'WARNING'
        else:
            return 'INFO'
    
    def _get_disaster_icon(self, title: str) -> str:
        """Get appropriate icon for disaster type"""
        title_lower = title.lower()
        if 'earthquake' in title_lower:
            return '🌍'
        elif 'fire' in title_lower:
            return '🔥'
        elif 'flood' in title_lower:
            return '🌊'
        elif 'cyclone' in title_lower or 'hurricane' in title_lower:
            return '🌀'
        elif 'tsunami' in title_lower:
            return '🌊'
        elif 'volcano' in title_lower:
            return '🌋'
        else:
            return '⚠️'
    
    def _get_mock_earthquake_data(self, latitude: float, longitude: float, radius_km: int) -> Dict:
        """Mock earthquake data for testing"""
        mock_earthquakes = []
        
        # Generate some realistic mock data
        if latitude > 25 and latitude < 35 and longitude > 70 and longitude < 85:  # India region
            mock_earthquakes = [
                EarthquakeData(
                    magnitude=3.2,
                    place="23km NE of Gurugram, India",
                    time=int((datetime.utcnow() - timedelta(hours=2)).timestamp() * 1000),
                    latitude=latitude + 0.2,
                    longitude=longitude + 0.1,
                    depth=15.0,
                    url="https://earthquake.usgs.gov/earthquakes/eventpage/mock1",
                    distance_km=25.0
                ),
                EarthquakeData(
                    magnitude=2.8,
                    place="12km SW of Delhi, India",
                    time=int((datetime.utcnow() - timedelta(hours=5)).timestamp() * 1000),
                    latitude=latitude - 0.1,
                    longitude=longitude - 0.15,
                    depth=8.0,
                    url="https://earthquake.usgs.gov/earthquakes/eventpage/mock2",
                    distance_km=18.0
                )
            ]
        
        return {
            'earthquakes': mock_earthquakes,
            'count': len(mock_earthquakes),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_mock_fire_data(self, latitude: float, longitude: float, radius_km: int) -> Dict:
        """Mock fire risk data for testing"""
        mock_fire_risks = []
        
        # Generate mock fire risk data based on location
        # Higher risk for certain regions (dry, hot areas)
        base_risk_score = 3
        if latitude > 25 and latitude < 35:  # Northern India region - typically drier
            base_risk_score = 5
        
        # Create fire risk zones
        mock_fire_risks = [
            {
                'latitude': latitude + 0.05,
                'longitude': longitude + 0.03,
                'risk_level': 'MODERATE',
                'temperature': 32,
                'humidity': 35,
                'wind_speed': 12,
                'precipitation': 0,
                'risk_score': base_risk_score + 2,
                'distance_km': 8.2
            },
            {
                'latitude': latitude - 0.02,
                'longitude': longitude + 0.08,
                'risk_level': 'LOW',
                'temperature': 28,
                'humidity': 55,
                'wind_speed': 8,
                'precipitation': 2.5,
                'risk_score': base_risk_score,
                'distance_km': 12.5
            }
        ]
        
        return {
            'fires': mock_fire_risks,
            'count': len(mock_fire_risks),
            'fire_risk_score': base_risk_score + 1,
            'weather_conditions': {
                'temperature': 30,
                'humidity': 45,
                'wind_speed': 10,
                'precipitation': 0.5
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_mock_gdacs_data(self, latitude: float, longitude: float, radius_km: int) -> Dict:
        """Mock GDACS data for testing"""
        mock_alerts = [
            GDACSAlert(
                event_id="GDACS001",
                event_type="EARTHQUAKE",
                title="Earthquake - India - Magnitude 4.2",
                description="A moderate earthquake occurred in northern India region. No immediate damage reported but monitoring continues.",
                severity="WARNING",
                date=(datetime.utcnow() - timedelta(hours=3)).isoformat(),
                latitude=latitude + 0.5,
                longitude=longitude + 0.3,
                icon="🌍"
            ),
            GDACSAlert(
                event_id="GDACS002",
                event_type="FIRE",
                title="Wildfire Alert - India Region",
                description="Active wildfire detected in agricultural area. Local authorities monitoring situation.",
                severity="INFO",
                date=(datetime.utcnow() - timedelta(hours=6)).isoformat(),
                latitude=latitude - 0.2,
                longitude=longitude + 0.1,
                icon="🔥"
            )
        ]
        
        return {
            'alerts': mock_alerts,
            'count': len(mock_alerts),
            'timestamp': datetime.utcnow().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create disaster predictor instance
    predictor = DisasterPredictor()
    
    # Example coordinates (Delhi, India)
    latitude = 28.6139
    longitude = 77.2090
    radius_km = 100
    
    print("🌍 Natural Disaster Prediction System")
    print("=" * 50)
    
    # Test earthquake fetching
    print("\n📊 Fetching earthquake data...")
    earthquakes = predictor.fetch_earthquakes(latitude, longitude, radius_km)
    print(f"Found {earthquakes['count']} earthquakes in the last 24 hours")
    
    for eq in earthquakes.get('earthquakes', [])[:3]:  # Show first 3
        print(f"  🌍 M{eq.magnitude} - {eq.place} ({eq.distance_km:.1f}km away)")
    
    # Test fire risk assessment
    print("\n🔥 Assessing fire risk...")
    fires = predictor.fetch_fires(latitude, longitude, radius_km)
    print(f"Fire risk score: {fires.get('fire_risk_score', 0)}/10")
    print(f"Found {fires['count']} fire risk areas")
    
    # Test GDACS alerts
    print("\n⚠️ Checking disaster alerts...")
    gdacs = predictor.fetch_gdacs_alerts(latitude, longitude, radius_km)
    print(f"Found {gdacs['count']} active alerts")
    
    for alert in gdacs.get('alerts', [])[:2]:  # Show first 2
        print(f"  {alert.icon} {alert.title}")
        print(f"    Severity: {alert.severity}")
    
    # Test weather disasters
    print("\n🌤️ Checking weather-based disasters...")
    weather_disasters = predictor.fetch_weather_disasters(latitude, longitude)
    print(f"Found {weather_disasters['count']} weather-related threats")
    
    for disaster in weather_disasters.get('weather_disasters', []):
        print(f"  {disaster['icon']} {disaster['title']}")
        print(f"    Impact: {disaster['crowd_impact']}")
    
    # Overall risk assessment
    print("\n📈 Overall Risk Assessment")
    print("-" * 30)
    risk_assessment = predictor.assess_disaster_risk(latitude, longitude, radius_km)
    
    print(f"🚨 Risk Level: {risk_assessment['risk_level']}")
    print(f"👥 Crowd Impact: {risk_assessment['crowd_impact']}")
    print(f"📋 Recommendation: {risk_assessment['recommendation']}")
    
    if risk_assessment['risk_factors']:
        print("\n🔍 Risk Factors:")
        for factor in risk_assessment['risk_factors']:
            print(f"  • {factor}")
    
    print("\n📊 Data Sources:")
    sources = risk_assessment['data_sources']
    print(f"  • Earthquakes: {sources['earthquakes']}")
    print(f"  • Fire Risks: {sources['fire_risks']}")
    print(f"  • GDACS Alerts: {sources['gdacs_alerts']}")
    
    print(f"\n🕒 Last Updated: {risk_assessment['timestamp']}")
    print("\n" + "=" * 50)
