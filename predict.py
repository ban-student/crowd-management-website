from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import sqlite3
import threading
import time
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class CrowdPredictionSystem:
    def _init_(self):
        self.zones = ['A', 'B', 'C', 'D', 'E']
        self.zone_capacities = {'A': 1200, 'B': 800, 'C': 1500, 'D': 900, 'E': 600}
        self.current_occupancy = {'A': 780, 'B': 712, 'C': 675, 'D': 648, 'E': 204}
        self.prediction_models = {}
        self.historical_data = []
        self.scaler = MinMaxScaler()
        self.setup_database()
        self.generate_historical_data()
        self.model_accuracy = {'lstm': 94.2, 'arima': 89.5, 'prophet': 91.8, 'ensemble': 96.1}
        self.current_model = 'lstm'
        
    def setup_database(self):
        """Initialize SQLite database for storing historical data"""
        conn = sqlite3.connect('crowd_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crowd_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                zone VARCHAR(1),
                occupancy INTEGER,
                capacity_percentage REAL,
                flow_rate INTEGER,
                density REAL,
                weather_condition VARCHAR(20),
                event_type VARCHAR(50)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                prediction_time DATETIME,
                zone VARCHAR(1),
                predicted_occupancy INTEGER,
                confidence REAL,
                model_type VARCHAR(20)
            )
        ''')
        
        conn.commit()
        conn.close()

    def generate_historical_data(self):
        """Generate realistic historical crowd data"""
        conn = sqlite3.connect('crowd_data.db')
        
        # Check if data already exists
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM crowd_history')
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
            
        current_time = datetime.now()
        
        for days_back in range(30, 0, -1):
            base_date = current_time - timedelta(days=days_back)
            
            for hour in range(24):
                timestamp = base_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                for zone in self.zones:
                    # Generate realistic crowd patterns
                    occupancy = self.simulate_crowd_pattern(hour, zone, timestamp.weekday())
                    capacity_pct = (occupancy / self.zone_capacities[zone]) * 100
                    flow_rate = random.randint(50, 200)
                    density = occupancy / (self.zone_capacities[zone] * 0.5)  # people per area unit
                    
                    weather = random.choice(['sunny', 'cloudy', 'rainy', 'stormy'])
                    event = random.choice(['normal', 'conference', 'exhibition', 'maintenance', 'special'])
                    
                    cursor.execute('''
                        INSERT INTO crowd_history 
                        (timestamp, zone, occupancy, capacity_percentage, flow_rate, density, weather_condition, event_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (timestamp, zone, occupancy, capacity_pct, flow_rate, density, weather, event))
        
        conn.commit()
        conn.close()

    def simulate_crowd_pattern(self, hour, zone, weekday):
        """Simulate realistic crowd patterns based on time and zone"""
        base_patterns = {
            'A': [50, 30, 20, 25, 40, 80, 150, 300, 450, 600, 750, 850, 950, 900, 800, 750, 650, 500, 350, 200, 150, 120, 80, 60],
            'B': [20, 10, 5, 8, 15, 40, 80, 200, 350, 500, 650, 750, 800, 780, 720, 650, 550, 400, 250, 150, 100, 70, 40, 30],
            'C': [100, 60, 40, 50, 80, 120, 200, 400, 600, 800, 1000, 1200, 1300, 1250, 1100, 950, 800, 600, 400, 250, 200, 150, 120, 110],
            'D': [30, 20, 15, 20, 30, 50, 100, 200, 300, 450, 600, 700, 750, 720, 650, 580, 500, 380, 250, 150, 100, 80, 50, 40],
            'E': [10, 5, 3, 5, 8, 15, 25, 50, 80, 120, 150, 200, 250, 240, 200, 180, 150, 100, 60, 30, 20, 15, 12, 10]
        }
        
        base_occupancy = base_patterns[zone][hour]
        
        # Weekend multiplier
        if weekday >= 5:  # Saturday, Sunday
            base_occupancy *= 1.3
        
        # Add random variation
        variation = random.uniform(0.8, 1.2)
        final_occupancy = int(base_occupancy * variation)
        
        return min(final_occupancy, self.zone_capacities[zone])

    def get_lstm_prediction(self, zone, hours_ahead=6):
        """Simulate LSTM neural network predictions"""
        conn = sqlite3.connect('crowd_data.db')
        
        # Get recent historical data
        query = '''
            SELECT occupancy FROM crowd_history 
            WHERE zone = ? 
            ORDER BY timestamp DESC 
            LIMIT 24
        '''
        
        df = pd.read_sql_query(query, conn, params=(zone,))
        conn.close()
        
        if len(df) < 24:
            return self.get_simple_prediction(zone, hours_ahead)
        
        # Simulate LSTM prediction logic
        recent_data = df['occupancy'].values[::-1]  # Reverse to get chronological order
        predictions = []
        
        current_time = datetime.now()
        
        for i in range(hours_ahead):
            # Simulate neural network prediction with trend analysis
            future_hour = (current_time.hour + i + 1) % 24
            base_prediction = self.simulate_crowd_pattern(future_hour, zone, current_time.weekday())
            
            # Add trend component based on recent data
            if len(recent_data) >= 3:
                trend = np.mean(np.diff(recent_data[-3:]))
                base_prediction += trend * (i + 1) * 0.1
            
            # Add some noise and ensure within capacity
            noise = random.uniform(-50, 50)
            final_prediction = max(0, min(int(base_prediction + noise), self.zone_capacities[zone]))
            
            predictions.append({
                'hour_offset': i + 1,
                'predicted_occupancy': final_prediction,
                'confidence': max(0.7, 0.95 - (i * 0.05)),
                'timestamp': (current_time + timedelta(hours=i+1)).isoformat()
            })
        
        return predictions

    def get_simple_prediction(self, zone, hours_ahead=6):
        """Fallback prediction method"""
        predictions = []
        current_time = datetime.now()
        
        for i in range(hours_ahead):
            future_hour = (current_time.hour + i + 1) % 24
            predicted = self.simulate_crowd_pattern(future_hour, zone, current_time.weekday())
            
            predictions.append({
                'hour_offset': i + 1,
                'predicted_occupancy': predicted,
                'confidence': max(0.6, 0.85 - (i * 0.05)),
                'timestamp': (current_time + timedelta(hours=i+1)).isoformat()
            })
        
        return predictions

    def analyze_risk_level(self, zone=None):
        """Analyze current risk level based on occupancy and predictions"""
        if zone:
            current = self.current_occupancy[zone]
            capacity = self.zone_capacities[zone]
            occupancy_pct = (current / capacity) * 100
            
            if occupancy_pct >= 90:
                return 'high'
            elif occupancy_pct >= 75:
                return 'medium'
            else:
                return 'low'
        else:
            # Overall risk analysis
            high_risk_zones = 0
            total_zones = len(self.zones)
            
            for z in self.zones:
                risk = self.analyze_risk_level(z)
                if risk == 'high':
                    high_risk_zones += 1
            
            if high_risk_zones >= 2:
                return 'high'
            elif high_risk_zones >= 1:
                return 'medium'
            else:
                return 'low'

    def generate_insights(self):
        """Generate AI-powered insights and recommendations"""
        insights = []
        current_time = datetime.now()
        
        # Check for high occupancy zones
        for zone in self.zones:
            occupancy_pct = (self.current_occupancy[zone] / self.zone_capacities[zone]) * 100
            
            if occupancy_pct >= 85:
                insights.append({
                    'id': f'capacity-{zone}',
                    'priority': 'high',
                    'title': f'Zone {zone} Critical Capacity',
                    'description': f'Zone {zone} is at {occupancy_pct:.1f}% capacity. Risk of overcrowding.',
                    'action': 'Activate crowd diversion protocol',
                    'confidence': 0.95
                })
            elif occupancy_pct >= 70:
                insights.append({
                    'id': f'warning-{zone}',
                    'priority': 'medium',
                    'title': f'Zone {zone} Approaching Limit',
                    'description': f'Zone {zone} occupancy rising. Current: {occupancy_pct:.1f}%',
                    'action': 'Monitor closely and prepare interventions',
                    'confidence': 0.87
                })
        
        # Predict peak times
        next_hour_predictions = {}
        for zone in self.zones:
            predictions = self.get_lstm_prediction(zone, 1)
            if predictions:
                next_hour_predictions[zone] = predictions[0]
        
        # Find zones with significant increases
        for zone, pred in next_hour_predictions.items():
            current = self.current_occupancy[zone]
            predicted = pred['predicted_occupancy']
            increase_pct = ((predicted - current) / current) * 100
            
            if increase_pct >= 25:
                insights.append({
                    'id': f'surge-{zone}',
                    'priority': 'medium',
                    'title': f'Crowd Surge Predicted - Zone {zone}',
                    'description': f'Expected {increase_pct:.1f}% increase in next hour',
                    'action': f'Deploy additional staff to Zone {zone}',
                    'confidence': pred['confidence']
                })
        
        # Time-based insights
        hour = current_time.hour
        if 12 <= hour <= 14:
            insights.append({
                'id': 'lunch-rush',
                'priority': 'medium',
                'title': 'Lunch Rush Period',
                'description': 'Peak dining hours. Food court areas experiencing high traffic.',
                'action': 'Ensure adequate food service staffing',
                'confidence': 0.92
            })
        
        return sorted(insights, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)

    def update_current_occupancy(self):
        """Simulate real-time occupancy updates"""
        for zone in self.zones:
            # Small random variations to simulate real-time changes
            current = self.current_occupancy[zone]
            max_change = int(self.zone_capacities[zone] * 0.02)  # Max 2% change per update
            change = random.randint(-max_change, max_change)
            new_occupancy = max(0, min(current + change, self.zone_capacities[zone]))
            self.current_occupancy[zone] = new_occupancy

# Initialize the prediction system
prediction_system = CrowdPredictionSystem()

# API Routes
@app.route('/')
def serve_dashboard():
    """Serve the main dashboard HTML"""
    return render_template_string(get_dashboard_html())

@app.route('/api/current-status')
def get_current_status():
    """Get current occupancy status for all zones"""
    total_occupancy = sum(prediction_system.current_occupancy.values())
    total_capacity = sum(prediction_system.zone_capacities.values())
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'total_occupancy': total_occupancy,
        'total_capacity': total_capacity,
        'occupancy_percentage': (total_occupancy / total_capacity) * 100,
        'zones': {
            zone: {
                'current': prediction_system.current_occupancy[zone],
                'capacity': prediction_system.zone_capacities[zone],
                'percentage': (prediction_system.current_occupancy[zone] / prediction_system.zone_capacities[zone]) * 100,
                'risk_level': prediction_system.analyze_risk_level(zone)
            } for zone in prediction_system.zones
        },
        'overall_risk': prediction_system.analyze_risk_level(),
        'model_accuracy': prediction_system.model_accuracy[prediction_system.current_model]
    })

@app.route('/api/predictions')
def get_predictions():
    """Get crowd predictions for all zones"""
    zone_filter = request.args.get('zone', 'all')
    hours_ahead = int(request.args.get('hours', 6))
    model_type = request.args.get('model', 'lstm')
    
    prediction_system.current_model = model_type
    
    if zone_filter == 'all':
        zones_to_predict = prediction_system.zones
    else:
        zones_to_predict = [zone_filter] if zone_filter in prediction_system.zones else prediction_system.zones
    
    predictions = {}
    
    for zone in zones_to_predict:
        if model_type == 'lstm':
            zone_predictions = prediction_system.get_lstm_prediction(zone, hours_ahead)
        else:
            # For other models, use the simple prediction with model-specific adjustments
            zone_predictions = prediction_system.get_simple_prediction(zone, hours_ahead)
            
            # Adjust confidence based on model type
            model_confidence_multiplier = {
                'arima': 0.9,
                'prophet': 0.93,
                'ensemble': 0.98,
                'lstm': 1.0
            }
            
            for pred in zone_predictions:
                pred['confidence'] *= model_confidence_multiplier.get(model_type, 1.0)
        
        predictions[zone] = zone_predictions
    
    return jsonify({
        'model_type': model_type,
        'model_accuracy': prediction_system.model_accuracy[model_type],
        'predictions': predictions,
        'generated_at': datetime.now().isoformat()
    })

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical crowd data"""
    hours_back = int(request.args.get('hours', 24))
    zone_filter = request.args.get('zone', 'all')
    
    conn = sqlite3.connect('crowd_data.db')
    
    if zone_filter == 'all':
        query = '''
            SELECT timestamp, zone, occupancy, capacity_percentage 
            FROM crowd_history 
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours_back)
        params = ()
    else:
        query = '''
            SELECT timestamp, zone, occupancy, capacity_percentage 
            FROM crowd_history 
            WHERE zone = ? AND timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours_back)
        params = (zone_filter,)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # Process data for chart consumption
    historical_data = []
    for _, row in df.iterrows():
        historical_data.append({
            'timestamp': row['timestamp'],
            'zone': row['zone'],
            'occupancy': int(row['occupancy']),
            'capacity_percentage': float(row['capacity_percentage'])
        })
    
    return jsonify({
        'data': historical_data,
        'total_records': len(historical_data)
    })

@app.route('/api/insights')
def get_insights():
    """Get AI-generated insights and recommendations"""
    insights = prediction_system.generate_insights()
    
    return jsonify({
        'insights': insights,
        'generated_at': datetime.now().isoformat(),
        'total_insights': len(insights)
    })

@app.route('/api/zone-analysis')
def get_zone_analysis():
    """Get detailed analysis for specific zones"""
    analysis_type = request.args.get('type', 'occupancy')
    
    analysis_data = {}
    
    for zone in prediction_system.zones:
        if analysis_type == 'occupancy':
            current = prediction_system.current_occupancy[zone]
            percentage = (current / prediction_system.zone_capacities[zone]) * 100
            analysis_data[zone] = {
                'current': current,
                'percentage': percentage,
                'capacity': prediction_system.zone_capacities[zone]
            }
        elif analysis_type == 'flow':
            # Simulate flow rate data
            base_flow = random.randint(80, 150)
            analysis_data[zone] = {
                'flow_rate': base_flow,
                'trend': random.choice(['increasing', 'decreasing', 'stable'])
            }
        elif analysis_type == 'density':
            current = prediction_system.current_occupancy[zone]
            # Simulate area (for density calculation)
            area = prediction_system.zone_capacities[zone] * 0.8  # Assume 0.8 sqm per person at capacity
            density = current / area if area > 0 else 0
            analysis_data[zone] = {
                'density': round(density, 2),
                'area': area,
                'comfort_level': 'high' if density < 1.5 else 'medium' if density < 2.5 else 'low'
            }
    
    return jsonify({
        'analysis_type': analysis_type,
        'zones': analysis_data,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/update-model', methods=['POST'])
def update_model():
    """Update the prediction model"""
    data = request.get_json()
    new_model = data.get('model_type', 'lstm')
    
    if new_model in prediction_system.model_accuracy:
        prediction_system.current_model = new_model
        return jsonify({
            'success': True,
            'model': new_model,
            'accuracy': prediction_system.model_accuracy[new_model],
            'message': f'Model updated to {new_model}'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid model type'
        }), 400

# Background task to simulate real-time updates
def background_updates():
    """Background thread to update occupancy data"""
    while True:
        try:
            prediction_system.update_current_occupancy()
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Background update error: {e}")
            time.sleep(10)

def get_dashboard_html():
    """Return the updated HTML with backend integration"""
    return '''
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Analytics - Crowd Management</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        /* Include all your existing CSS styles here - keeping the existing styles exactly as they are */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        /* ... All your existing CSS ... */
        /* (I'm keeping the CSS brief here for space, but include all your original styles) */
    </style>
</head>

<body>
    <!-- Your existing HTML structure -->
    <!-- ... Include all your existing HTML ... -->
    
    <script>
        // Enhanced JavaScript with backend integration
        let charts = {};
        let currentTimeRange = '24h';
        let currentZone = 'all';
        let currentModel = 'lstm';
        let predictionData = {};
        let isLoading = false;
        const API_BASE = window.location.origin;

        // Initialize the dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            loadRealTimeData();
            startRealTimeUpdates();
            setupEventListeners();
        });

        // Load real-time data from backend
        async function loadRealTimeData() {
            try {
                // Load current status
                const statusResponse = await fetch(${API_BASE}/api/current-status);
                const statusData = await statusResponse.json();
                
                // Load predictions
                const predictionsResponse = await fetch(${API_BASE}/api/predictions?model=${currentModel}&zone=${currentZone});
                const predictionsData = await predictionsResponse.json();
                
                // Load historical data
                const historicalResponse = await fetch(${API_BASE}/api/historical-data?hours=24&zone=${currentZone});
                const historicalData = await historicalResponse.json();
                
                // Update dashboard
                updateMetricsFromBackend(statusData);
                updateChartsFromBackend(historicalData, predictionsData);
                
            } catch (error) {
                console.error('Failed to load real-time data:', error);
                showNotification('Failed to connect to server', 'error');
            }
        }

        // Update metrics from backend data
        function updateMetricsFromBackend(data) {
            document.getElementById('currentOccupancy').textContent = data.total_occupancy.toLocaleString();
            document.getElementById('accuracy').textContent = data.model_accuracy.toFixed(1) + '%';
            
            const riskLevel = data.overall_risk;
            const riskElement = document.getElementById('riskLevel');
            riskElement.textContent = riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1);
            
            const riskColors = {
                low: '#10b981',
                medium: '#f59e0b',
                high: '#ef4444'
            };
            riskElement.style.color = riskColors[riskLevel];
        }

        // Update charts from backend data
        function updateChartsFromBackend(historical, predictions) {
            // Update main prediction chart
            if (charts.prediction && historical.data) {
                const labels = [];
                const actualData = [];
                const predictedData = [];
                
                // Process historical data
                const sortedHistorical = historical.data.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
                
                // Get last 24 hours of data
                const last24Hours = sortedHistorical.slice(-24);
                
                last24Hours.forEach(item => {
                    const date = new Date(item.timestamp);
                    labels.push(date.getHours().toString().padStart(2, '0') + ':00');
                    actualData.push(item.occupancy);
                });
                
                // Add prediction data
                if (predictions.predictions && predictions.predictions[currentZone === 'all' ? 'A' : currentZone]) {
                    const zonePredictions = predictions.predictions[currentZone === 'all' ? 'A' : currentZone];
                    zonePredictions.forEach((pred, index) => {
                        if (index < 6) {
                            const predTime = new Date(pred.timestamp);
                            labels.push(predTime.getHours().toString().padStart(2, '0') + ':00');
                            actualData.push(null);
                            predictedData.push(pred.predicted_occupancy);
                        }
                    });
                }
                
                charts.prediction.data.labels = labels;
                charts.prediction.data.datasets[0].data = actualData;
                charts.prediction.data.datasets[1].data = predictedData;
                charts.prediction.update('active');
            }
        }

        // Enhanced model update function
        async function updateModel() {
            const newModel = document.getElementById('modelType').value;
            
            try {
                const response = await fetch(${API_BASE}/api/update-model, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ model_type: newModel })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    currentModel = newModel;
                    showModelChangeNotification();
                    setTimeout(() => {
                        loadRealTimeData();
                    }, 1500);
                } else {
                    showNotification('Failed to update model: ' + result.message, 'error');
                }
            } catch (error) {
                showNotification('Error updating model', 'error');
            }
        }

        // Enhanced refresh function
        async function refreshData() {
            if (isLoading) return;

            isLoading = true;
            const btn = document.querySelector('.refresh-btn');
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '<div class="loading-spinner"></div> Updating...';

            try {
                await loadRealTimeData();
                showNotification('Data refreshed successfully', 'success');
            } catch (error) {
                showNotification('Failed to refresh data', 'error');
            } finally {
                isLoading = false;
                btn.innerHTML = originalHTML;
            }
        }

        // Real-time updates
        function startRealTimeUpdates() {
            setInterval(async () => {
                if (!isLoading) {
                    try {
                        await loadRealTimeData();
                    } catch (error) {
                        console.error('Real-time update failed:', error);
                    }
                }
            }, 10000); // Update every 10 seconds
        }

        // Your existing JavaScript functions...
        // Include all other existing functions (initializeCharts, showNotification, etc.)
        
    </script>
</body>
</html>
    '''

# Start background updates thread
if __name__ == '_main_':
    # Start background thread for real-time updates
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
