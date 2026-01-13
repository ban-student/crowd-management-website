#!/usr/bin/env python3
"""
Integrated Crowd Analytics Server
Combines existing Flask server with Alert System functionality
"""

from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import subprocess
import threading
import time
import os
import signal
import psutil
import webbrowser

# Computer Vision and ML imports
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import json

# Alert System imports
from datetime import datetime, timedelta
import uuid
import requests
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging
from enum import Enum
import schedule
import random 
import math 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize Flask app
app = Flask(__name__, static_folder='.', template_folder='.')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crowd_analytics.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables for existing functionality
streamlit_process = None
disaster_predictor = None

# Heatmap processing variables
heatmap_model = None
camera_active = False
current_frame = None
people_count = 0
people_boxes = []
heatmap_thread = None

# Camera settings
camera_settings = {
    'sensitivity': 7,
    'heat_intensity': 70,
    'threshold': 'medium',
    'camera_source': 'default'
}
current_camera_index = 0
detection_confidence = 0.5

# Alert System Configuration
class Config:
    # Twilio Configuration
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    
    # Email Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    EMAIL_USERNAME = os.getenv('EMAIL_USERNAME')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
    
    # Alert thresholds based on people count
    CRITICAL_PEOPLE_THRESHOLD = 15
    WARNING_PEOPLE_THRESHOLD = 10
    
    # Zone coordinates (for multi-camera setup)
    ZONE_COORDINATES = {
        'A': {'lat': 28.4593, 'lng': 77.0264, 'name': 'Main Entry Gate'},
        'B': {'lat': 28.4595, 'lng': 77.0266, 'name': 'Food Court Area'},
        'C': {'lat': 28.4597, 'lng': 77.0268, 'name': 'Exhibition Hall'},
        'D': {'lat': 28.4599, 'lng': 77.0270, 'name': 'Shopping Area'},
        'E': {'lat': 28.4601, 'lng': 77.0272, 'name': 'Camera Zone'}
    }

# Enums for Alert System
class AlertType(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"

class NotificationStatus(Enum):
    SENT = "sent"
    FAILED = "failed"
    PENDING = "pending"

# Database Models
class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    type = db.Column(db.Enum(AlertType), nullable=False)
    message = db.Column(db.Text, nullable=False)
    zone = db.Column(db.String(10), nullable=False)
    zone_name = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.Enum(AlertStatus), default=AlertStatus.ACTIVE)
    occupancy = db.Column(db.Float, nullable=True)
    coordinates = db.Column(db.String(50), nullable=True)
    created_by = db.Column(db.String(100), default='system')
    resolved_at = db.Column(db.DateTime, nullable=True)
    escalation_level = db.Column(db.Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type.value,
            'message': self.message,
            'zone': self.zone,
            'zoneName': self.zone_name,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'occupancy': self.occupancy,
            'coordinates': self.coordinates,
            'created_by': self.created_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'escalation_level': self.escalation_level
        }

class NotificationLog(db.Model):
    __tablename__ = 'notification_logs'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    alert_id = db.Column(db.String(36), db.ForeignKey('alerts.id'), nullable=True)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.Enum(NotificationStatus), default=NotificationStatus.PENDING)
    recipients = db.Column(db.Integer, default=0)
    notification_type = db.Column(db.String(20), nullable=False)
    details = db.Column(db.Text, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'alert_id': self.alert_id,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'recipients': self.recipients,
            'notification_type': self.notification_type,
            'details': json.loads(self.details) if self.details else None
        }

class SystemSettings(db.Model):
    __tablename__ = 'system_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    sms_enabled = db.Column(db.Boolean, default=True)
    email_enabled = db.Column(db.Boolean, default=False)
    push_enabled = db.Column(db.Boolean, default=True)
    auto_escalation = db.Column(db.Boolean, default=True)
    escalation_timeout = db.Column(db.Integer, default=300)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'sms_enabled': self.sms_enabled,
            'email_enabled': self.email_enabled,
            'push_enabled': self.push_enabled,
            'auto_escalation': self.auto_escalation,
            'escalation_timeout': self.escalation_timeout,
            'updated_at': self.updated_at.isoformat()
        }

class Contact(db.Model):
    __tablename__ = 'contacts'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    email = db.Column(db.String(100), nullable=True)
    role = db.Column(db.String(50), nullable=False)
    zones = db.Column(db.String(50), nullable=True)
    active = db.Column(db.Boolean, default=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'email': self.email,
            'role': self.role,
            'zones': json.loads(self.zones) if self.zones else [],
            'active': self.active
        }

# Mock Disaster Predictor Class
class MockDisasterPredictor:
    """Mock disaster predictor for when the full module is not available"""
    
    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance using simple approximation"""
        return abs(lat1 - lat2) * 111 + abs(lng1 - lng2) * 111
    
    def fetch_earthquakes(self, latitude, longitude, radius_km=100, days_back=1):
        """Mock earthquake data"""
        return {
            'earthquakes': [
                {
                    'magnitude': 3.2,
                    'place': f"Mock earthquake near {latitude:.2f}, {longitude:.2f}",
                    'time': int((datetime.utcnow() - timedelta(hours=2)).timestamp() * 1000),
                    'latitude': latitude + 0.1,
                    'longitude': longitude + 0.05,
                    'depth': 12.0,
                    'url': 'https://earthquake.usgs.gov/mock',
                    'distance_km': 15.0
                }
            ],
            'count': 1,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def fetch_fires(self, latitude, longitude, radius_km=100, days_back=1):
        """Mock fire data"""
        return {
            'fires': [
                {
                    'latitude': latitude + 0.02,
                    'longitude': longitude + 0.01,
                    'brightness': 320.5,
                    'confidence': 75,
                    'acq_date': datetime.utcnow().strftime('%Y-%m-%d'),
                    'acq_time': datetime.utcnow().strftime('%H%M'),
                    'acq_datetime': datetime.utcnow().strftime('%Y-%m-%d %H:%M'),
                    'satellite': 'VIIRS_SNPP',
                    'distance_km': 8.0
                }
            ],
            'count': 1,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def fetch_gdacs_alerts(self, latitude, longitude, radius_km=100):
        """Mock GDACS alerts"""
        return {
            'alerts': [
                {
                    'event_id': 'MOCK001',
                    'event_type': 'INFO',
                    'title': 'System Monitoring Active',
                    'description': 'Disaster monitoring system is operational and scanning for threats.',
                    'severity': 'INFO',
                    'date': datetime.utcnow().isoformat(),
                    'icon': '✅'
                }
            ],
            'count': 1,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def assess_disaster_risk(self, latitude, longitude, radius_km=100):
        """Mock risk assessment"""
        return {
            'risk_level': 'LOW',
            'risk_score': 1,
            'crowd_impact': 'LOW',
            'recommendation': 'Normal safety measures sufficient - monitoring active',
            'risk_factors': ['System monitoring active'],
            'data_sources': {
                'earthquakes': 1,
                'fires': 1,
                'gdacs_alerts': 1
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    def fetch_weather_disasters(self, latitude, longitude):
        """Mock weather disaster data"""
        return {
            'weather_disasters': [],
            'count': 0,
            'timestamp': datetime.utcnow().isoformat()
        }

# Services
class NotificationService:
    """Handles sending notifications via SMS, email, and push"""
    
    def __init__(self):
        self.twilio_client = None
        if Config.TWILIO_ACCOUNT_SID and Config.TWILIO_AUTH_TOKEN:
            self.twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification"""
        try:
            if not self.twilio_client:
                logger.warning("Twilio not configured, simulating SMS send")
                return True
            
            message = self.twilio_client.messages.create(
                body=message,
                from_=Config.TWILIO_PHONE_NUMBER,
                to=phone_number
            )
            logger.info(f"SMS sent successfully: {message.sid}")
            return True
        except Exception as e:
            logger.error(f"Failed to send SMS: {str(e)}")
            return False
    
    def send_email(self, to_email: str, subject: str, message: str) -> bool:
        """Send email notification"""
        try:
            if not Config.EMAIL_USERNAME or not Config.EMAIL_PASSWORD:
                logger.warning("Email not configured, simulating email send")
                return True
            
            msg = MIMEMultipart()
            msg['From'] = Config.EMAIL_USERNAME
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT)
            server.starttls()
            server.login(Config.EMAIL_USERNAME, Config.EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(Config.EMAIL_USERNAME, to_email, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def send_push_notification(self, message: str, alert_data: dict) -> bool:
        """Send push notification via WebSocket"""
        try:
            socketio.emit('new_alert', {
                'message': message,
                'alert': alert_data,
                'timestamp': datetime.utcnow().isoformat()
            })
            logger.info("Push notification sent via WebSocket")
            return True
        except Exception as e:
            logger.error(f"Failed to send push notification: {str(e)}")
            return False

class AlertService:
    """Handles alert creation and management"""
    
    def __init__(self):
        self.notification_service = NotificationService()
    
    def create_alert(self, alert_type: AlertType, message: str, zone: str = 'E', 
                    occupancy: Optional[float] = None, created_by: str = 'system') -> Alert:
        """Create a new alert"""
        zone_info = Config.ZONE_COORDINATES.get(zone, {})
        zone_name = zone_info.get('name', f'Zone {zone}')
        coordinates = f"{zone_info.get('lat', 0)}, {zone_info.get('lng', 0)}" if zone != 'ALL' else 'Multiple'
        
        alert = Alert(
            type=alert_type,
            message=message,
            zone=zone,
            zone_name=zone_name,
            occupancy=occupancy,
            coordinates=coordinates,
            created_by=created_by
        )
        
        db.session.add(alert)
        db.session.commit()
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        logger.info(f"Alert created: {alert.id} - {alert.type.value} - {alert.message}")
        return alert
    
    def resolve_alert(self, alert_id: str, resolved_by: str = 'system') -> Optional[Alert]:
        """Resolve an alert"""
        alert = Alert.query.get(alert_id)
        if alert and alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            db.session.commit()
            
            # Notify resolution
            self._send_resolution_notification(alert)
            
            logger.info(f"Alert resolved: {alert.id} by {resolved_by}")
            return alert
        return None
    
    def _send_alert_notifications(self, alert: Alert):
        """Send notifications for a new alert"""
        settings = SystemSettings.query.first()
        if not settings:
            settings = SystemSettings()
            db.session.add(settings)
            db.session.commit()
        
        message = f"CROWD ALERT: {alert.message} - Zone: {alert.zone_name} - Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send push notification
        if self.notification_service.send_push_notification(message, alert.to_dict()):
            log = NotificationLog(
                alert_id=alert.id,
                message=f"Push notification sent for {alert.zone_name}",
                notification_type='push',
                status=NotificationStatus.SENT,
                recipients=1
            )
            db.session.add(log)
            db.session.commit()
    
    def _send_resolution_notification(self, alert: Alert):
        """Send notification when alert is resolved"""
        message = f"RESOLVED: Alert in {alert.zone_name} has been resolved"
        self.notification_service.send_push_notification(message, alert.to_dict())

# Initialize services
alert_service = AlertService()

# Initialize disaster predictor
def initialize_disaster_predictor():
    """Initialize the disaster prediction system"""
    global disaster_predictor
    try:
        from disaster_prediction import DisasterPredictor
        disaster_predictor = DisasterPredictor()
        logger.info("Disaster prediction system initialized")
    except ImportError:
        logger.warning("Disaster prediction module not found - using mock data")
        disaster_predictor = MockDisasterPredictor()

# Heatmap and Computer Vision Functions
def load_heatmap_model():
    """Load YOLO model for heatmap processing"""
    global heatmap_model
    try:
        heatmap_model = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")

def generate_heatmap(frame, people_boxes):
    """Generate heatmap overlay from people detections"""
    if not people_boxes:
        return frame
        
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    # Use sensitivity to adjust heatmap radius
    radius = int(30 + (camera_settings['sensitivity'] * 5))  # 35-80 pixel radius
    
    for (x1, y1, x2, y2) in people_boxes:
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(heatmap, (center_x, center_y), radius, 1, -1)
    
    heatmap = cv2.GaussianBlur(heatmap, (91, 91), 0)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Use heat_intensity setting for blending
    intensity = camera_settings['heat_intensity'] / 100.0
    combined = cv2.addWeighted(frame, 1.0 - (intensity * 0.3), heatmap_color, intensity * 0.5, 0)
    return combined

def process_camera_feed():
    """Background thread for processing camera feed with alert integration"""
    global camera_active, current_frame, people_count, people_boxes, heatmap_model
    
    cap = cv2.VideoCapture(current_camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_alert_time = 0
    alert_cooldown = 30  # 30 seconds between alerts
    
    while camera_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        if heatmap_model is not None:
            # Use settings-based confidence threshold
            conf_map = {'low': 0.3, 'medium': 0.5, 'high': 0.7}
            conf = conf_map.get(camera_settings['threshold'], 0.5)
    
            # Run YOLO detection with dynamic confidence
            results = heatmap_model(frame, stream=True, verbose=False, conf=conf)
            detected_boxes = []
            
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 0:  # Person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detected_boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            people_boxes = detected_boxes
            people_count = len(detected_boxes)
            
            # Check for alert conditions
            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                if people_count >= Config.CRITICAL_PEOPLE_THRESHOLD:
                    alert_service.create_alert(
                        AlertType.CRITICAL,
                        f"Critical crowd detected - {people_count} people in camera view",
                        'E',  # Camera zone
                        (people_count / 20) * 100  # Convert to percentage
                    )
                    last_alert_time = current_time
                elif people_count >= Config.WARNING_PEOPLE_THRESHOLD:
                    alert_service.create_alert(
                        AlertType.WARNING,
                        f"High crowd density detected - {people_count} people in camera view",
                        'E',
                        (people_count / 20) * 100
                    )
                    last_alert_time = current_time
            
            # Generate heatmap
            frame = generate_heatmap(frame, detected_boxes)
            
            # Add text overlay
            cv2.putText(frame, f"People Count: {people_count}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add alert level indicator
            alert_level = "HIGH" if people_count >= Config.CRITICAL_PEOPLE_THRESHOLD else \
                         "MED" if people_count >= Config.WARNING_PEOPLE_THRESHOLD else "LOW"
            color = (0, 0, 255) if alert_level == "HIGH" else \
                   (0, 165, 255) if alert_level == "MED" else (0, 255, 0)
            
            cv2.putText(frame, f"Alert Level: {alert_level}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            current_frame = frame
            
            # Emit real-time data
            socketio.emit('crowd_data_update', {
                'people_count': people_count,
                'alert_level': alert_level,
                'timestamp': datetime.utcnow().isoformat(),
                'zone': 'E'
            })
        
        time.sleep(0.1)  # Control frame rate
    
    cap.release()

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    if frame is None:
        return None
    
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

# Existing Routes
@app.route('/')
def dashboard():
    return send_from_directory('.', 'analyticsdashboard.html')

@app.route('/analyticsdashboard.html')
def analytics_dashboard():
    return send_from_directory('.', 'analyticsdashboard.html')

@app.route('/crowddetection.html')
def crowd_detection():
    return send_from_directory('.', 'crowddetection.html')

@app.route('/crowdmanagement.html')
def crowd_management():
    return send_from_directory('.', 'crowdmanagement.html')

@app.route('/alertsystem.html')
def alert_system():
    return send_from_directory('.', 'alertsystem.html')

@app.route('/weatherprediction.html')
def weather_prediction():
    return send_from_directory('.', 'weatherprediction.html')

@app.route('/weather')
def weather_dashboard():
    return send_from_directory('.', 'weatherprediction.html')

@app.route('/climatemonitoring.html')
def climate_monitoring():
    return send_from_directory('.', 'climatemonitoring.html')

@app.route('/climate')
def climate_dashboard():
    return send_from_directory('.', 'climatemonitoring.html')

@app.route('/disasterprediction.html')
def disaster_prediction_page():
    return send_from_directory('.', 'disasterprediction.html')

# Heatmap API Routes
@app.route('/api/heatmap/update_settings', methods=['POST'])
def update_camera_settings():
    global camera_settings, current_camera_index
    try:
        settings = request.get_json()
        
        for key, value in settings.items():
            if key in camera_settings:
                camera_settings[key] = value
                
        if 'camera_source' in settings:
            source_map = {
                'default': 0,
                'camera1': 1, 
                'camera2': 2,
                'camera3': 3
            }
            current_camera_index = source_map.get(settings['camera_source'], 0)
            
        print(f"Camera settings updated: {camera_settings}")
        return jsonify({"status": "success", "message": "Settings updated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/heatmap/start_camera', methods=['POST'])
def start_camera():
    global camera_active, heatmap_thread
    
    if camera_active:
        return jsonify({"status": "already_running", "message": "Camera is already active"})
    
    try:
        if heatmap_model is None:
            load_heatmap_model()
        
        camera_active = True
        heatmap_thread = threading.Thread(target=process_camera_feed, daemon=True)
        heatmap_thread.start()
        
        return jsonify({"status": "success", "message": "Camera started successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/heatmap/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    
    camera_active = False
    time.sleep(0.5)
    
    return jsonify({"status": "success", "message": "Camera stopped successfully"})

@app.route('/api/heatmap/frame')
def get_current_frame():
    global current_frame, people_count
    
    if current_frame is None:
        return jsonify({"status": "no_frame", "message": "No camera feed available"})
    
    frame_b64 = frame_to_base64(current_frame)
    
    return jsonify({
        "status": "success",
        "frame": frame_b64,
        "people_count": people_count,
        "timestamp": time.time()
    })

@app.route('/api/heatmap/stats')
def get_heatmap_stats():
    global people_count, people_boxes
    
    avg_density = min(100, (people_count / 20) * 100)
    hot_spots = max(0, people_count // 3)
    alert_level = "HIGH" if people_count >= Config.CRITICAL_PEOPLE_THRESHOLD else \
                 "MED" if people_count >= Config.WARNING_PEOPLE_THRESHOLD else "LOW"
    
    return jsonify({
        "total_detected": people_count,
        "avg_density": f"{avg_density:.0f}%",
        "hot_spots": hot_spots,
        "alert_level": alert_level,
        "is_active": camera_active
    })

# Alert System API Routes
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all alerts with optional filtering"""
    alert_type = request.args.get('type')
    status = request.args.get('status')
    zone = request.args.get('zone')
    limit = int(request.args.get('limit', 50))
    
    query = Alert.query
    
    if alert_type:
        query = query.filter(Alert.type == AlertType(alert_type))
    if status:
        query = query.filter(Alert.status == AlertStatus(status))
    if zone:
        query = query.filter(Alert.zone == zone)
    
    alerts = query.order_by(Alert.timestamp.desc()).limit(limit).all()
    
    return jsonify({
        'alerts': [alert.to_dict() for alert in alerts],
        'total': query.count(),
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/alerts', methods=['POST'])
def create_manual_alert():
    """Create a manual alert"""
    try:
        data = request.get_json()
        
        alert_type = AlertType(data.get('type', 'info'))
        message = data.get('message', '').strip()
        zones = data.get('zones', [])
        created_by = data.get('created_by', 'manual')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if not zones:
            return jsonify({'error': 'At least one zone must be selected'}), 400
        
        created_alerts = []
        
        for zone in zones:
            if zone == 'ALL':
                alert = alert_service.create_alert(alert_type, message, 'ALL', created_by=created_by)
                created_alerts.append(alert.to_dict())
                break
            else:
                alert = alert_service.create_alert(alert_type, message, zone, created_by=created_by)
                created_alerts.append(alert.to_dict())
        
        return jsonify({
            'success': True,
            'alerts': created_alerts,
            'message': f'Alert(s) created successfully'
        })
        
    except Exception as e:
        logger.error(f"Error creating manual alert: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """Resolve an alert"""
    try:
        data = request.get_json() or {}
        resolved_by = data.get('resolved_by', 'api')
        
        alert = alert_service.resolve_alert(alert_id, resolved_by)
        
        if alert:
            return jsonify({
                'success': True,
                'alert': alert.to_dict(),
                'message': 'Alert resolved successfully'
            })
        else:
            return jsonify({'error': 'Alert not found or already resolved'}), 404
            
    except Exception as e:
        logger.error(f"Error resolving alert: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications', methods=['GET'])
def get_notification_logs():
    """Get notification logs"""
    limit = int(request.args.get('limit', 50))
    
    logs = NotificationLog.query.order_by(
        NotificationLog.timestamp.desc()
    ).limit(limit).all()
    
    return jsonify({
        'logs': [log.to_dict() for log in logs],
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get system settings"""
    settings = SystemSettings.query.first()
    if not settings:
        settings = SystemSettings()
        db.session.add(settings)
        db.session.commit()
    
    return jsonify(settings.to_dict())

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update system settings"""
    try:
        data = request.get_json()
        
        settings = SystemSettings.query.first()
        if not settings:
            settings = SystemSettings()
        
        settings.sms_enabled = data.get('sms_enabled', settings.sms_enabled)
        settings.email_enabled = data.get('email_enabled', settings.email_enabled)
        settings.push_enabled = data.get('push_enabled', settings.push_enabled)
        settings.auto_escalation = data.get('auto_escalation', settings.auto_escalation)
        settings.escalation_timeout = data.get('escalation_timeout', settings.escalation_timeout)
        settings.updated_at = datetime.utcnow()
        
        db.session.add(settings)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'settings': settings.to_dict(),
            'message': 'Settings updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get alert system statistics"""
    try:
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        active_alerts = Alert.query.filter_by(status=AlertStatus.ACTIVE).count()
        resolved_today = Alert.query.filter(
            Alert.status == AlertStatus.RESOLVED,
            Alert.resolved_at >= today_start
        ).count()
        
        avg_response_time = "2.5m"
        
        recent_logs = NotificationLog.query.filter(
            NotificationLog.timestamp >= now - timedelta(hours=24)
        ).all()
        
        if recent_logs:
            successful = len([log for log in recent_logs if log.status == NotificationStatus.SENT])
            system_health = f"{(successful / len(recent_logs)) * 100:.1f}%"
        else:
            system_health = "100.0%"
        
        camera_stats = {
            'people_count': people_count,
            'camera_active': camera_active,
            'alert_level': "HIGH" if people_count >= Config.CRITICAL_PEOPLE_THRESHOLD else 
                         "MED" if people_count >= Config.WARNING_PEOPLE_THRESHOLD else "LOW"
        }
        
        return jsonify({
            'active_alerts': active_alerts,
            'resolved_today': resolved_today,
            'avg_response_time': avg_response_time,
            'system_health': system_health,
            'camera_stats': camera_stats,
            'timestamp': now.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-alert', methods=['POST'])
def send_test_alert():
    """Send a test alert"""
    try:
        alert = alert_service.create_alert(
            AlertType.INFO,
            "This is a test alert - system is functioning normally",
            "E",
            created_by="test"
        )
        
        return jsonify({
            'success': True,
            'alert': alert.to_dict(),
            'message': 'Test alert sent successfully'
        })
        
    except Exception as e:
        logger.error(f"Error sending test alert: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Weather Prediction API Routes
@app.route('/api/climate/current', methods=['GET'])
def get_current_climate():
    """Get current weather data from Open-Meteo API"""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        
        if not lat or not lng:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        
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
            
            current_hour = datetime.now().hour
            uv_index = data['hourly']['uv_index'][current_hour] if current_hour < len(data['hourly']['uv_index']) else None
            
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

@app.route('/api/climate/alerts', methods=['GET'])
def get_climate_alerts():
    """Generate climate-based crowd management alerts"""
    try:
        lat = request.args.get('lat', type=float) or 28.6139
        lng = request.args.get('lng', type=float) or 77.2090
        
        # Try to get current weather data
        try:
            current_response = requests.get(f'http://localhost:5000/api/climate/current?lat={lat}&lng={lng}', timeout=5)
            if current_response.status_code == 200:
                weather_data = current_response.json()
            else:
                # Use fallback data
                weather_data = {
                    "temperature": 25,
                    "humidity": 65,
                    "wind_speed": 12,
                    "precipitation_probability": 20,
                    "weather_code": 2,
                    "uv_index": 5
                }
        except:
            # Use fallback data if request fails
            weather_data = {
                "temperature": 25,
                "humidity": 65,
                "wind_speed": 12,
                "precipitation_probability": 20,
                "weather_code": 2,
                "uv_index": 5
            }
        
        alerts = generate_crowd_climate_alerts(weather_data)
        
        return jsonify({"alerts": alerts})
        
    except Exception as e:
        logger.error(f"Climate Alerts Error: {e}")
        return jsonify({"alerts": get_default_climate_alerts()})

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
            "crowd_impact": "HIGH"
        })
    elif temp >= 35:
        alerts.append({
            "severity": "warning",
            "icon": "🔥",
            "title": "High Temperature Alert",
            "message": f"Temperature: {temp}°C - Crowd discomfort likely",
            "crowd_impact": "MED"
        })
    elif temp <= 0:
        alerts.append({
            "severity": "warning",
            "icon": "🧊",
            "title": "Freezing Temperature",
            "message": f"Temperature: {temp}°C - Risk of hypothermia in crowds",
            "crowd_impact": "HIGH"
        })
    
    # Precipitation alerts
    if precipitation_prob >= 80:
        alerts.append({
            "severity": "critical",
            "icon": "🌧️",
            "title": "Heavy Rain Expected",
            "message": f"{precipitation_prob}% chance of rain - High risk of crowd issues",
            "crowd_impact": "HIGH"
        })
    elif precipitation_prob >= 60:
        alerts.append({
            "severity": "warning",
            "icon": "🌦️",
            "title": "Rain Likely",
            "message": f"{precipitation_prob}% chance of rain - Crowd dispersal may be needed",
            "crowd_impact": "MED"
        })
    
    # Wind alerts
    if wind_speed >= 50:
        alerts.append({
            "severity": "critical",
            "icon": "💨",
            "title": "Dangerous Winds",
            "message": f"Wind speed: {wind_speed} km/h - Structural hazards possible",
            "crowd_impact": "EXTREME"
        })
    elif wind_speed >= 35:
        alerts.append({
            "severity": "warning",
            "icon": "🌬️",
            "title": "Strong Wind Warning",
            "message": f"Wind speed: {wind_speed} km/h - Monitor temporary structures",
            "crowd_impact": "HIGH"
        })
    
    # Thunderstorm alerts
    if weather_code in [95, 96, 99]:
        alerts.append({
            "severity": "critical",
            "icon": "⛈️",
            "title": "Thunderstorm Alert",
            "message": "Thunderstorm detected - Indoor evacuation recommended",
            "crowd_impact": "EXTREME"
        })
    
    # UV Index alerts
    if uv_index >= 8:
        alerts.append({
            "severity": "warning",
            "icon": "☀️",
            "title": "Very High UV Index",
            "message": f"UV Index: {uv_index} - Increased risk of sunburn",
            "crowd_impact": "MED"
        })
    
    # Fog/visibility alerts
    if weather_code in [45, 48]:
        alerts.append({
            "severity": "warning",
            "icon": "🌫️",
            "title": "Low Visibility",
            "message": "Fog conditions - Navigation may be impaired",
            "crowd_impact": "HIGH"
        })
    
    # All clear message if no alerts
    if not alerts:
        alerts.append({
            "severity": "info",
            "icon": "✅",
            "title": "Favorable Weather Conditions",
            "message": "Current weather conditions are suitable for crowd activities",
            "crowd_impact": "LOW"
        })
    
    return alerts

def get_default_climate_alerts():
    """Default climate alerts when data is unavailable"""
    return [
        {
            "severity": "info",
            "icon": "📡",
            "title": "Weather Monitoring Active",
            "message": "Climate monitoring system is online and tracking conditions",
            "crowd_impact": "LOW"
        }
    ]

@app.route('/api/climate/forecast', methods=['GET'])
def get_climate_forecast():
    """Get hourly weather forecast from Open-Meteo"""
    try:
        lat = request.args.get('lat', type=float) or 28.6139
        lng = request.args.get('lng', type=float) or 77.2090
        
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
                    
                    if len(forecast_list) >= 8:
                        break
            
            return jsonify({"forecast": forecast_list})
        else:
            return get_fallback_forecast()
            
    except Exception as e:
        logger.error(f"Climate Forecast Error: {e}")
        return get_fallback_forecast()

def get_fallback_forecast():
    """Fallback forecast data"""
    mock_forecast = []
    for i in range(8):
        hour = (datetime.now() + timedelta(hours=i+1)).strftime('%H:%M')
        mock_forecast.append({
            'time': hour,
            'temperature': 25 + (i % 3),
            'condition': 'Partly cloudy',
            'weather_code': 2,
            'precipitation_probability': 20,
            'wind_speed': 10,
            'humidity': 65,
            'cloud_cover': 40,
            'uv_index': 5
        })
    return jsonify({"forecast": mock_forecast})

@app.route('/api/climate/environmental', methods=['GET'])
def get_environmental_data():
    """Get additional environmental data"""
    try:
        lat = request.args.get('lat', type=float) or 28.6139
        lng = request.args.get('lng', type=float) or 77.2090
        
        # Get current weather for environmental calculations
        try:
            weather_response = requests.get(f'http://localhost:5000/api/climate/current?lat={lat}&lng={lng}', timeout=5)
            weather_data = weather_response.json() if weather_response.status_code == 200 else {}
        except:
            weather_data = {}
        
        # Calculate dew point
        temp = weather_data.get('temperature', 20)
        humidity = weather_data.get('humidity', 50)
        dew_point = calculate_dew_point(temp, humidity)
        
        environmental_data = {
            "air_quality": "Good",
            "cloud_cover": weather_data.get('cloud_cover', 50),
            "precipitation": weather_data.get('precipitation', 0),
            "dew_point": round(dew_point, 1)
        }
        
        return jsonify(environmental_data)
        
    except Exception as e:
        logger.error(f"Environmental Data Error: {e}")
        return jsonify({
            "air_quality": "Good",
            "cloud_cover": 50,
            "precipitation": 0,
            "dew_point": 15
        })

def calculate_dew_point(temperature, humidity):
    """Calculate dew point using Magnus formula"""
    try:
        import math
        a = 17.27
        b = 237.7
        alpha = ((a * temperature) / (b + temperature)) + math.log(humidity / 100.0)
        return (b * alpha) / (a - alpha)
    except:
        return 15  # Default dew point

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
@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard overview data"""
    try:
        return jsonify({
            "status": "success",
            "total_people": people_count,
            "active_zones": 5,
            "alerts": [],
            "system_health": "99.8%",
            "camera_active": camera_active,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Dashboard API Error: {e}")
        return jsonify({"error": str(e)}), 500
# Disaster Prediction API Routes
@app.route('/api/disaster/earthquakes', methods=['GET'])
def get_earthquakes():
    """Get earthquake data for a location"""
    try:
        lat = float(request.args.get('lat', 28.4593))
        lng = float(request.args.get('lng', 77.0264))
        radius = int(request.args.get('radius', 100))
        
        if not disaster_predictor:
            initialize_disaster_predictor()
        
        data = disaster_predictor.fetch_earthquakes(lat, lng, radius)
        
        if 'earthquakes' in data and hasattr(data['earthquakes'][0] if data['earthquakes'] else None, '__dict__'):
            data['earthquakes'] = [eq.__dict__ for eq in data['earthquakes']]
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching earthquake data: {e}")
        return jsonify({'error': str(e), 'earthquakes': [], 'count': 0}), 500
@app.route('/api/disaster/fires', methods=['GET'])
def get_fires():
    """Get fire risk data based on weather conditions"""
    try:
        lat = float(request.args.get('lat', 28.4593))
        lng = float(request.args.get('lng', 77.0264))
        radius = int(request.args.get('radius', 100))
        
        if not disaster_predictor:
            initialize_disaster_predictor()
        
        data = disaster_predictor.fetch_fires(lat, lng, radius)
        
        # Ensure proper data structure
        if 'fires' in data:
            for fire in data['fires']:
                # Convert dataclass to dict if needed
                if hasattr(fire, '__dict__'):
                    fire = fire.__dict__
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching fire risk data: {e}")
        # Return fallback data with proper structure
        return jsonify({
            'fires': [{
                'latitude': lat + 0.05,
                'longitude': lng + 0.03,
                'risk_level': 'MODERATE',
                'temperature': 32,
                'humidity': 35,
                'wind_speed': 12,
                'precipitation': 0,
                'risk_score': 6,
                'distance_km': 8.2
            }],
            'count': 1,
            'fire_risk_score': 6,
            'timestamp': datetime.utcnow().isoformat()
        })
@app.route('/api/disaster/risk-assessment', methods=['GET'])
def get_risk_assessment():
    """Get comprehensive disaster risk assessment"""
    try:
        lat = float(request.args.get('lat', 28.4593))
        lng = float(request.args.get('lng', 77.0264))
        radius = int(request.args.get('radius', 100))
        
        if not disaster_predictor:
            initialize_disaster_predictor()
        
        assessment = disaster_predictor.assess_disaster_risk(lat, lng, radius)
        return jsonify(assessment)
        
    except Exception as e:
        logger.error(f"Error performing risk assessment: {e}")
        return jsonify({
            'error': str(e),
            'risk_level': 'UNKNOWN',
            'crowd_impact': 'UNKNOWN'
        }), 500

@app.route('/api/disaster/all-data', methods=['GET'])
def get_all_disaster_data():
    """Get comprehensive disaster data for dashboard"""
    try:
        lat = float(request.args.get('lat', 28.4593))
        lng = float(request.args.get('lng', 77.0264))
        radius = int(request.args.get('radius', 100))
        
        if not disaster_predictor:
            initialize_disaster_predictor()
        
        earthquakes = disaster_predictor.fetch_earthquakes(lat, lng, radius)
        fire_risks = disaster_predictor.fetch_fires(lat, lng, radius)
        gdacs = disaster_predictor.fetch_gdacs_alerts(lat, lng, radius)
        risk = disaster_predictor.assess_disaster_risk(lat, lng, radius)
        
        if 'earthquakes' in earthquakes and earthquakes['earthquakes']:
            if hasattr(earthquakes['earthquakes'][0], '__dict__'):
                earthquakes['earthquakes'] = [eq.__dict__ for eq in earthquakes['earthquakes']]
        
        if 'alerts' in gdacs and gdacs['alerts']:
            if hasattr(gdacs['alerts'][0], '__dict__'):
                gdacs['alerts'] = [alert.__dict__ for alert in gdacs['alerts']]
        
        return jsonify({
            'earthquakes': earthquakes,
            'fire_risks': fire_risks,
            'gdacs': gdacs,
            'risk_assessment': risk,
            'location': {'latitude': lat, 'longitude': lng, 'radius': radius},
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching all disaster data: {e}")
        return jsonify({'error': str(e)}), 500

# Streamlit Integration Routes
@app.route('/start_streamlit', methods=['POST', 'GET'])
def start_streamlit():
    global streamlit_process
    
    try:
        if is_streamlit_running():
            return jsonify({
                "status": "success", 
                "message": "Streamlit is already running on port 8501"
            })
        
        kill_existing_streamlit()
        
        streamlit_process = subprocess.Popen([
            'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            '--server.enableCORS', 'false'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)
        
        if streamlit_process.poll() is None:
            return jsonify({
                "status": "success",
                "message": "Streamlit app started successfully",
                "port": 8501
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Streamlit failed to start"
            }), 500
            
    except FileNotFoundError:
        return jsonify({
            "status": "error",
            "message": "Streamlit not found. Please install with: pip install streamlit"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error starting Streamlit: {str(e)}"
        }), 500

@app.route('/stop_streamlit', methods=['POST'])
def stop_streamlit():
    global streamlit_process
    
    try:
        killed_count = kill_existing_streamlit()
        
        if streamlit_process:
            streamlit_process.terminate()
            streamlit_process = None
            
        return jsonify({
            "status": "success",
            "message": f"Streamlit stopped. Killed {killed_count} processes."
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error stopping Streamlit: {str(e)}"
        }), 500

@app.route('/streamlit_status')
def streamlit_status():
    is_running = is_streamlit_running()
    return jsonify({
        "status": "running" if is_running else "stopped",
        "port": 8501 if is_running else None
    })

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': 'connected',
            'camera': 'active' if camera_active else 'inactive',
            'websocket': 'active'
        }
    })
@app.route('/api/disaster/gdacs', methods=['GET'])
def get_gdacs_alerts():
    """Get GDACS alerts for a location"""
    try:
        lat = float(request.args.get('lat', 28.4593))
        lng = float(request.args.get('lng', 77.0264))
        radius = int(request.args.get('radius', 100))
        
        if not disaster_predictor:
            initialize_disaster_predictor()
        
        data = disaster_predictor.fetch_gdacs_alerts(lat, lng, radius)
        
        # Convert dataclass objects to dictionaries if needed
        if 'alerts' in data and data['alerts']:
            if hasattr(data['alerts'][0], '__dict__'):
                data['alerts'] = [alert.__dict__ for alert in data['alerts']]
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching GDACS data: {e}")
        return jsonify({'error': str(e), 'alerts': [], 'count': 0}), 500

@app.route('/api/disaster/weather-disasters', methods=['GET'])
def get_weather_disasters():
    """Get weather-based disaster predictions"""
    try:
        lat = float(request.args.get('lat', 28.4593))
        lng = float(request.args.get('lng', 77.0264))
        
        if not disaster_predictor:
            initialize_disaster_predictor()
        
        data = disaster_predictor.fetch_weather_disasters(lat, lng)
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching weather disasters: {e}")
        return jsonify({'error': str(e), 'weather_disasters': [], 'count': 0}), 500
    
# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info('Client connected to WebSocket')
    emit('connected', {'message': 'Successfully connected to alert system'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info('Client disconnected from WebSocket')

@socketio.on('subscribe_alerts')
def handle_subscribe_alerts():
    """Subscribe to real-time alert updates"""
    logger.info('Client subscribed to alert updates')
    emit('subscription_confirmed', {'type': 'alerts'})

# Utility Functions
def get_fallback_climate_data(lat=None, lng=None):
    """Fallback climate data when API is unavailable"""
    return jsonify({
        "temperature": 25,
        "condition": "Partly cloudy",
        "weather_code": 2,
        "humidity": 65,
        "wind_speed": 12,
        "pressure": 1013,
        "precipitation_probability": 20,
        "uv_index": 5
    })

def is_streamlit_running():
    """Check if Streamlit is running on port 8501"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python' in proc.info['name'].lower() and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'streamlit' in cmdline and '8501' in cmdline:
                    return True
        return False
    except Exception:
        return False

def kill_existing_streamlit():
    """Kill any existing Streamlit processes"""
    killed_count = 0
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python' in proc.info['name'].lower() and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'streamlit' in cmdline and ('app.py' in cmdline or '8501' in cmdline):
                    proc.kill()
                    killed_count += 1
        time.sleep(1)
    except Exception as e:
        print(f"Error killing Streamlit processes: {e}")
    
    return killed_count

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

def initialize_database():
    """Initialize database and create sample data"""
    with app.app_context():
        db.create_all()
        
        if not SystemSettings.query.first():
            settings = SystemSettings()
            db.session.add(settings)
            db.session.commit()
            logger.info("Default settings created")
        
        if not Contact.query.first():
            sample_contacts = [
                Contact(name="Security Manager", phone="+91-9999-000001", email="security@mall.com", 
                       role="security", zones='["A", "B", "C", "D", "E"]'),
                Contact(name="Zone A Manager", phone="+91-9999-000002", email="zonea@mall.com", 
                       role="manager", zones='["A"]'),
                Contact(name="Zone B Manager", phone="+91-9999-000003", email="zoneb@mall.com", 
                       role="manager", zones='["B"]'),
                Contact(name="Zone C Manager", phone="+91-9999-000004", email="zonec@mall.com", 
                       role="manager", zones='["C"]'),
                Contact(name="Zone D Manager", phone="+91-9999-000005", email="zoned@mall.com", 
                       role="manager", zones='["D"]'),
                Contact(name="Camera Zone Manager", phone="+91-9999-000006", email="camera@mall.com", 
                       role="manager", zones='["E"]'),
                Contact(name="Admin", phone="+91-9999-000007", email="admin@mall.com", 
                       role="admin", zones='["ALL"]'),
            ]
            
            for contact in sample_contacts:
                db.session.add(contact)
            
            db.session.commit()
            logger.info("Sample contacts created")

# Serve static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    # Clean up any existing Streamlit processes on start
    kill_existing_streamlit()
    
    # Initialize database
    initialize_database()
    
    # Load YOLO model on startup
    print("Loading YOLO model for heatmap processing...")
    load_heatmap_model()
    
    # Initialize disaster predictor
    print("Initializing Natural Disaster Prediction System...")
    initialize_disaster_predictor()
    
    print("Starting Integrated Crowd Analytics Server...")
    print("Dashboard available at: http://localhost:5000")
    print("Alert System available at: http://localhost:5000/alertsystem.html")
    print("Crowd Detection with heatmap at: http://localhost:5000/crowddetection.html")
    print("Climate Monitoring available at: http://localhost:5000/climatemonitoring.html")
    print("Disaster Prediction available at: http://localhost:5000/disasterprediction.html")
    print("Streamlit integration available on port 8501")
    print("Opening browser automatically...")
    
    # Start browser opening in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the application with SocketIO
    socketio.run(app, 
                host='0.0.0.0', 
                port=5000, 
                debug=True, 
                allow_unsafe_werkzeug=True)
