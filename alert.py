alert.py
#!/usr/bin/env python3
"""
Crowd Analytics Alert System Backend
Real-time alert management system with SMS, email, and WebSocket notifications
"""

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from datetime import datetime, timedelta
import uuid
import json
import threading
import time
import requests
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///alert_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Configuration
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
    
    # Alert thresholds
    CRITICAL_OCCUPANCY_THRESHOLD = 90
    WARNING_OCCUPANCY_THRESHOLD = 75
    
    # Zone coordinates
    ZONE_COORDINATES = {
        'A': {'lat': 28.4593, 'lng': 77.0264, 'name': 'Main Entry Gate'},
        'B': {'lat': 28.4595, 'lng': 77.0266, 'name': 'Food Court Area'},
        'C': {'lat': 28.4597, 'lng': 77.0268, 'name': 'Exhibition Hall'},
        'D': {'lat': 28.4599, 'lng': 77.0270, 'name': 'Shopping Area'},
        'E': {'lat': 28.4601, 'lng': 77.0272, 'name': 'Parking Zone'}
    }

# Enums
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
    notification_type = db.Column(db.String(20), nullable=False)  # sms, email, push
    details = db.Column(db.Text, nullable=True)  # JSON string for additional details
    
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
    escalation_timeout = db.Column(db.Integer, default=300)  # seconds
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
    role = db.Column(db.String(50), nullable=False)  # manager, security, admin
    zones = db.Column(db.String(50), nullable=True)  # JSON array of zone IDs
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
    """Handles alert creation, management, and escalation"""
    
    def __init__(self):
        self.notification_service = NotificationService()
    
    def create_alert(self, alert_type: AlertType, message: str, zone: str, 
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
    
    def escalate_alert(self, alert_id: str) -> Optional[Alert]:
        """Escalate an alert to the next level"""
        alert = Alert.query.get(alert_id)
        if alert and alert.status == AlertStatus.ACTIVE:
            alert.escalation_level += 1
            db.session.commit()
            
            # Send escalation notifications
            self._send_escalation_notification(alert)
            
            logger.info(f"Alert escalated: {alert.id} to level {alert.escalation_level}")
            return alert
        return None
    
    def _send_alert_notifications(self, alert: Alert):
        """Send notifications for a new alert"""
        settings = SystemSettings.query.first()
        if not settings:
            settings = SystemSettings()
            db.session.add(settings)
            db.session.commit()
        
        # Get contacts for the zone
        contacts = self._get_contacts_for_zone(alert.zone)
        
        message = f"ALERT: {alert.message} - Zone: {alert.zone_name} - Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        notification_log = NotificationLog(
            alert_id=alert.id,
            message=f"Alert notifications sent for {alert.zone_name}",
            recipients=len(contacts)
        )
        
        success_count = 0
        
        # Send SMS
        if settings.sms_enabled:
            for contact in contacts:
                if contact.phone and self.notification_service.send_sms(contact.phone, message):
                    success_count += 1
            
            notification_log.notification_type = 'sms'
        
        # Send Email
        if settings.email_enabled:
            subject = f"CROWD ALERT: {alert.type.value.upper()}"
            email_message = f"""
            Crowd Analytics Alert System
            
            Alert Type: {alert.type.value.upper()}
            Zone: {alert.zone_name}
            Message: {alert.message}
            Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            Occupancy: {alert.occupancy}% (if available)
            Coordinates: {alert.coordinates}
            
            Please take appropriate action.
            """
            
            for contact in contacts:
                if contact.email and self.notification_service.send_email(contact.email, subject, email_message):
                    success_count += 1
        
        # Send Push Notification
        if settings.push_enabled:
            if self.notification_service.send_push_notification(message, alert.to_dict()):
                success_count += 1
        
        notification_log.status = NotificationStatus.SENT if success_count > 0 else NotificationStatus.FAILED
        db.session.add(notification_log)
        db.session.commit()
    
    def _send_resolution_notification(self, alert: Alert):
        """Send notification when alert is resolved"""
        message = f"RESOLVED: Alert in {alert.zone_name} has been resolved"
        self.notification_service.send_push_notification(message, alert.to_dict())
        
        log = NotificationLog(
            alert_id=alert.id,
            message=f"Resolution notification sent for {alert.zone_name}",
            notification_type='push',
            status=NotificationStatus.SENT,
            recipients=1
        )
        db.session.add(log)
        db.session.commit()
    
    def _send_escalation_notification(self, alert: Alert):
        """Send notification when alert is escalated"""
        message = f"ESCALATED: Alert in {alert.zone_name} escalated to level {alert.escalation_level}"
        self.notification_service.send_push_notification(message, alert.to_dict())
    
    def _get_contacts_for_zone(self, zone: str) -> List[Contact]:
        """Get relevant contacts for a specific zone"""
        if zone == 'ALL':
            return Contact.query.filter_by(active=True).all()
        
        contacts = []
        all_contacts = Contact.query.filter_by(active=True).all()
        
        for contact in all_contacts:
            contact_zones = json.loads(contact.zones) if contact.zones else []
            if zone in contact_zones or 'ALL' in contact_zones:
                contacts.append(contact)
        
        return contacts

class CrowdMonitorService:
    """Simulates real-time crowd monitoring and generates alerts"""
    
    def __init__(self, alert_service: AlertService):
        self.alert_service = alert_service
        self.running = False
        self.thread = None
        self.zone_occupancy = {
            'A': 65, 'B': 70, 'C': 60, 'D': 55, 'E': 50
        }
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Crowd monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Crowd monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Simulate crowd data updates
                self._update_crowd_data()
                
                # Check for alert conditions
                self._check_alert_conditions()
                
                # Emit real-time data
                socketio.emit('crowd_data_update', {
                    'zones': self.zone_occupancy,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
    
    def _update_crowd_data(self):
        """Simulate crowd data updates"""
        import random
        
        for zone in self.zone_occupancy:
            # Simulate random crowd changes
            change = random.randint(-5, 8)
            self.zone_occupancy[zone] = max(20, min(100, self.zone_occupancy[zone] + change))
    
    def _check_alert_conditions(self):
        """Check for conditions that require alerts"""
        for zone, occupancy in self.zone_occupancy.items():
            existing_active_alerts = Alert.query.filter_by(
                zone=zone,
                status=AlertStatus.ACTIVE
            ).filter(
                Alert.timestamp > datetime.utcnow() - timedelta(minutes=10)
            ).count()
            
            # Don't create duplicate alerts for same zone within 10 minutes
            if existing_active_alerts > 0:
                continue
            
            if occupancy >= Config.CRITICAL_OCCUPANCY_THRESHOLD:
                self.alert_service.create_alert(
                    AlertType.CRITICAL,
                    f"Critical overcrowding detected - {occupancy}% occupancy",
                    zone,
                    occupancy
                )
            elif occupancy >= Config.WARNING_OCCUPANCY_THRESHOLD:
                self.alert_service.create_alert(
                    AlertType.WARNING,
                    f"High crowd density detected - {occupancy}% occupancy",
                    zone,
                    occupancy
                )

# Initialize services
alert_service = AlertService()
monitor_service = CrowdMonitorService(alert_service)

# API Routes
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
        
        # Create alerts for each selected zone
        for zone in zones:
            if zone == 'ALL':
                # Create single alert for all zones
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

@app.route('/api/alerts/<alert_id>/escalate', methods=['POST'])
def escalate_alert(alert_id):
    """Escalate an alert"""
    try:
        alert = alert_service.escalate_alert(alert_id)
        
        if alert:
            return jsonify({
                'success': True,
                'alert': alert.to_dict(),
                'message': f'Alert escalated to level {alert.escalation_level}'
            })
        else:
            return jsonify({'error': 'Alert not found or cannot be escalated'}), 404
            
    except Exception as e:
        logger.error(f"Error escalating alert: {str(e)}")
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
        
        # Active alerts
        active_alerts = Alert.query.filter_by(status=AlertStatus.ACTIVE).count()
        
        # Resolved today
        resolved_today = Alert.query.filter(
            Alert.status == AlertStatus.RESOLVED,
            Alert.resolved_at >= today_start
        ).count()
        
        # Average response time (mock calculation)
        avg_response_time = "2.5m"  # You can implement actual calculation
        
        # System health (based on successful notifications)
        recent_logs = NotificationLog.query.filter(
            NotificationLog.timestamp >= now - timedelta(hours=24)
        ).all()
        
        if recent_logs:
            successful = len([log for log in recent_logs if log.status == NotificationStatus.SENT])
            system_health = f"{(successful / len(recent_logs)) * 100:.1f}%"
        else:
            system_health = "100.0%"
        
        return jsonify({
            'active_alerts': active_alerts,
            'resolved_today': resolved_today,
            'avg_response_time': avg_response_time,
            'system_health': system_health,
            'zone_occupancy': monitor_service.zone_occupancy,
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
            "A",
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

# Auto-escalation background task
def auto_escalation_task():
    """Background task for auto-escalating alerts"""
    with app.app_context():
        settings = SystemSettings.query.first()
        if not settings or not settings.auto_escalation:
            return
        
        timeout = timedelta(seconds=settings.escalation_timeout)
        cutoff_time = datetime.utcnow() - timeout
        
        # Find alerts that need escalation
        alerts_to_escalate = Alert.query.filter(
            Alert.status == AlertStatus.ACTIVE,
            Alert.timestamp <= cutoff_time,
            Alert.escalation_level < 3  # Max 3 escalation levels
        ).all()
        
        for alert in alerts_to_escalate:
            alert_service.escalate_alert(alert.id)

# Initialize database and start services
@app.before_first_request
def initialize_database():
    """Initialize database and start services"""
    db.create_all()
    
    # Create default settings if not exists
    if not SystemSettings.query.first():
        settings = SystemSettings()
        db.session.add(settings)
        db.session.commit()
    
    # Create sample contacts if not exists
    if not Contact.query.first():
        sample_contacts = [
            Contact(name="Security Manager", phone="+91-9999-000001", email="security@mall.com", 
                   role="security", zones='["A", "B", "C", "D", "E"]'),
            Contact(name="Zone A Manager", phone="+91-9999-000002", email="zonea@mall.com", 
                   role="manager", zones='["A"]'),
            Contact(name="Zone B Manager", phone="+91-9999-000003", email="zoneb@mall.com", 
                   role="manager", zones='["B"]'),
            Contact(name="Admin", phone="+91-9999-000004", email="admin@mall.com", 
                   role="admin", zones='["ALL"]'),
        ]
        
        for contact in sample_contacts:
            db.session.add(contact)
        
        db.session.commit()
    
    # Start monitoring service
    monitor_service.start_monitoring()

# Scheduled tasks
def schedule_background_tasks():
    """Schedule background tasks"""
    import schedule
    import threading
    
    # Schedule auto-escalation check every 5 minutes
    schedule.every(5).minutes.do(auto_escalation_task)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': 'connected',
            'monitoring': 'active' if monitor_service.running else 'inactive',
            'websocket': 'active'
        }
    })

# Main application entry point
if __name__ == '__main__':
    # Initialize database
    with app.app_context():
        db.create_all()
        
        # Create default settings if not exists
        if not SystemSettings.query.first():
            settings = SystemSettings()
            db.session.add(settings)
            db.session.commit()
            logger.info("Default settings created")
        
        # Create sample contacts if not exists
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
                Contact(name="Zone E Manager", phone="+91-9999-000006", email="zonee@mall.com", 
                       role="manager", zones='["E"]'),
                Contact(name="Admin", phone="+91-9999-000007", email="admin@mall.com", 
                       role="admin", zones='["ALL"]'),
            ]
            
            for contact in sample_contacts:
                db.session.add(contact)
            
            db.session.commit()
            logger.info("Sample contacts created")
        
        # Start monitoring and background tasks
        monitor_service.start_monitoring()
        schedule_background_tasks()
        
        logger.info("Alert System Backend Started")
        logger.info("Available endpoints:")
        logger.info("  GET  /api/health - Health check")
        logger.info("  GET  /api/alerts - Get alerts")
        logger.info("  POST /api/alerts - Create manual alert")
        logger.info("  POST /api/alerts/<id>/resolve - Resolve alert")
        logger.info("  POST /api/alerts/<id>/escalate - Escalate alert")
        logger.info("  GET  /api/notifications - Get notification logs")
        logger.info("  GET  /api/settings - Get system settings")
        logger.info("  POST /api/settings - Update settings")
        logger.info("  GET  /api/statistics - Get system statistics")
        logger.info("  POST /api/test-alert - Send test alert")
        logger.info("  WebSocket: Connect for real-time updates")
    
    # Run the application
    socketio.run(app, 
                host='0.0.0.0', 
                port=5000, 
                debug=True, 
                allow_unsafe_werkzeug=True)
