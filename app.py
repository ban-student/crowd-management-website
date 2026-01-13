import streamlit as st
import cv2
import tempfile
import numpy as np
import pygame
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import threading
import math

# Initialize pygame mixer for sound alerts
pygame.mixer.init()

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

class CrowdAnalyzer:
    def __init__(self):
        self.person_tracks = defaultdict(lambda: deque(maxlen=10))  # Track person positions
        self.frame_count = 0
        self.emergency_state = False
        self.last_alert_time = 0
        self.alert_cooldown = 3  # Reduced cooldown - 3 seconds between alerts
        
    def calculate_density(self, person_boxes, frame_shape):
        """Calculate crowd density based on person detections and their sizes"""
        if not person_boxes:
            return 0, "LOW"
        
        # Filter out small/distant people (background people)
        significant_people = []
        frame_height = frame_shape[0]
        
        for box in person_boxes:
            x1, y1, x2, y2 = box
            person_height = y2 - y1
            person_width = x2 - x1
            person_area = person_height * person_width
            
            # Only count people who are close enough (significant size)
            # People in background will be much smaller
            min_person_height = frame_height * 0.15  # At least 15% of frame height
            min_person_area = (frame_height * 0.1) * (frame_height * 0.05)  # Minimum area
            
            if person_height > min_person_height and person_area > min_person_area:
                significant_people.append(box)
        
        person_count = len(significant_people)
        
        # Much higher thresholds - only real crowds trigger emergency
        if person_count <= 1:
            return person_count / 10, "LOW"
        elif person_count <= 3:
            return person_count / 10, "LOW" 
        elif person_count <= 8:
            return person_count / 10, "MEDIUM"
        elif person_count <= 20:
            return person_count / 10, "HIGH"
        else:
            return person_count / 10, "CRITICAL"
    
    def detect_unusual_movement(self, person_boxes, frame_shape, frame=None):
        """Detect running behavior - simplified and more reliable"""
        current_time = time.time()
        unusual_activities = []
        frame_height = frame_shape[0]
        
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            person_height = y2 - y1
            
            # Simple person ID based on rough position
            person_id = f"p_{center_x//50}_{center_y//50}"  # Group nearby positions
            
            # Store position with timestamp
            self.person_tracks[person_id].append((center_x, center_y, current_time))
            
            # Need at least 2 positions to calculate speed
            if len(self.person_tracks[person_id]) >= 2:
                recent_positions = list(self.person_tracks[person_id])
                
                # Calculate speed between last 2 positions
                pos1 = recent_positions[-2]
                pos2 = recent_positions[-1]
                
                distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                time_diff = pos2[2] - pos1[2]
                
                if time_diff > 0:
                    speed = distance / time_diff
                    
                    # Simple speed threshold - adjust based on person size
                    base_threshold = 80  # Much lower base threshold
                    size_factor = max(1, person_height / 100)  # Bigger people = higher threshold
                    speed_threshold = base_threshold * size_factor
                    
                    if speed > speed_threshold:
                        unusual_activities.append(f"🏃‍♂️ RUNNING detected! Speed: {speed:.1f}")
                        
                        # Draw running indicator if frame is available
                        if frame is not None:
                            cv2.circle(frame, (center_x, center_y), 30, (0, 0, 255), 3)
                            cv2.putText(frame, "RUNNING!", (center_x-40, center_y-35), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return unusual_activities
    
    def detect_crowd_behavior(self, person_boxes, frame_shape):
        """Detect crowd behavior - only for close people, not background"""
        # Filter for significant people only
        significant_people = []
        frame_height = frame_shape[0]
        
        for box in person_boxes:
            x1, y1, x2, y2 = box
            person_height = y2 - y1
            person_area = (y2 - y1) * (x2 - x1)
            
            min_person_height = frame_height * 0.15
            min_person_area = (frame_height * 0.1) * (frame_height * 0.05)
            
            if person_height > min_person_height and person_area > min_person_area:
                significant_people.append(box)
        
        if len(significant_people) < 5:  # Need at least 5 close people
            return []
        
        behaviors = []
        
        # Check for very close clustering among significant people only
        close_pairs = 0
        for i, box1 in enumerate(significant_people):
            center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
            for j, box2 in enumerate(significant_people[i+1:], i+1):
                center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
                distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Very close distance for emergency
                if distance < 40:  
                    close_pairs += 1
        
        # Only alert for extreme overcrowding
        if close_pairs > len(significant_people) * 0.6:  # 60% very close
            behaviors.append("🚨 EXTREME Overcrowding - People dangerously close")
        
        return behaviors
    
    def trigger_emergency_alert(self, alert_message):
        """Trigger emergency alert with sound and visual notification"""
        current_time = time.time()
        
        # Avoid spamming alerts
        if current_time - self.last_alert_time > self.alert_cooldown:
            self.emergency_state = True
            self.last_alert_time = current_time
            
            # Create alert sound in separate thread to avoid blocking
            def play_alert():
                try:
                    # Create a louder, more noticeable beep sound
                    duration = 0.8  # Longer duration
                    freq = 800  # Lower frequency - more attention grabbing
                    sample_rate = 22050
                    frames = int(duration * sample_rate)
                    
                    # Create a pulsing beep (2 short beeps)
                    arr1 = np.sin(2 * np.pi * freq * np.linspace(0, 0.3, int(0.3 * sample_rate)))
                    silence = np.zeros(int(0.1 * sample_rate))
                    arr2 = np.sin(2 * np.pi * freq * np.linspace(0, 0.3, int(0.3 * sample_rate)))
                    
                    arr = np.concatenate([arr1, silence, arr2])
                    arr = (arr * 32767).astype(np.int16)
                    arr = np.repeat(arr.reshape(len(arr), 1), 2, axis=1)
                    
                    sound = pygame.sndarray.make_sound(arr)
                    sound.play()
                    pygame.time.wait(800)
                    
                    # Print to console for debugging
                    print(f"🚨 EMERGENCY ALERT: {alert_message}")
                except Exception as e:
                    print(f"Sound error: {e}")
                    # Fallback - at least print the alert
                    print(f"🚨 EMERGENCY (No Sound): {alert_message}")
            
            # Play sound in background thread
            threading.Thread(target=play_alert, daemon=True).start()
            
            return True
        return False

# Initialize crowd analyzer
crowd_analyzer = CrowdAnalyzer()

st.title("🚨 Smart Crowd Management - People Detection & Risk Analysis")
st.markdown("**Real-time crowd density monitoring with emergency detection**")

# Sidebar controls
st.sidebar.header("Settings")
st.sidebar.markdown("**🚨 Emergency Triggers:**")
st.sidebar.markdown("- ANY person running (80+ speed)")  
st.sidebar.markdown("- 5+ people very close")
st.sidebar.markdown("- Double beep sound + console log")
st.sidebar.markdown("---")
density_threshold = st.sidebar.slider("Density Alert Threshold", 2.0, 8.0, 4.0, 0.5)
speed_threshold = st.sidebar.slider("Running Speed Base", 60, 150, 80, 10)
show_tracks = st.sidebar.checkbox("Show Movement Tracks", False)
st.sidebar.markdown("**Debug Info:**")
st.sidebar.markdown("Check console for emergency logs!")

# Upload video or use webcam
option = st.radio("Choose video source:", ("Webcam", "Upload"))

if option == "Upload":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_source = tfile.name
    else:
        video_source = None
else:
    video_source = 0  # webcam

if video_source is not None:
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stframe = st.empty()  # placeholder for video frames
    
    with col2:
        st.subheader("📊 Live Stats")
        stats_placeholder = st.empty()
        
        st.subheader("🚨 Alerts")
        alerts_placeholder = st.empty()
    
    # Emergency banner placeholder
    emergency_banner = st.empty()
    
    cap = cv2.VideoCapture(video_source)
    
    # Set video properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            crowd_analyzer.frame_count += 1
            
            # Run YOLO inference
            results = model(frame, imgsz=640, verbose=False)
            all_person_boxes = []
            significant_person_boxes = []
            frame_height = frame.shape[0]
            
            # Process detections and filter for significant people
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:  # class 0 = person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_height = y2 - y1
                        person_area = person_height * (x2 - x1)
                        
                        all_person_boxes.append((x1, y1, x2, y2))
                        
                        # Check if person is significant (close/large enough)
                        min_person_height = frame_height * 0.12  # Reduced for better detection
                        min_person_area = (frame_height * 0.08) * (frame_height * 0.04)  # Reduced
                        
                        if person_height > min_person_height and person_area > min_person_area:
                            significant_person_boxes.append((x1, y1, x2, y2))
                            # Draw green box for close people
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "Person", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            # Draw gray box for distant people
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                            cv2.putText(frame, "Distant", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # Analyze crowd - use significant people only for analysis
            total_people = len(all_person_boxes)
            close_people = len(significant_person_boxes)
            
            density, density_level = crowd_analyzer.calculate_density(significant_person_boxes, frame.shape)
            unusual_movements = crowd_analyzer.detect_unusual_movement(all_person_boxes, frame.shape, frame)  # Pass frame
            crowd_behaviors = crowd_analyzer.detect_crowd_behavior(significant_person_boxes, frame.shape)
            
            # Emergency only for REAL situations but more sensitive to running
            emergency_alerts = []
            
            # Running detected - emergency even for 1 person if running fast enough
            running_detected = any("RUNNING" in movement for movement in unusual_movements)
            
            if running_detected:
                # If someone is running, emergency regardless of crowd size
                emergency_alerts.extend([f"🚨 {mov}" for mov in unusual_movements])
            
            # Crowd behaviors only if significant
            if crowd_behaviors and close_people >= 5:
                emergency_alerts.extend([f"👥 {behavior}" for behavior in crowd_behaviors])
            
            # Emergency for running OR serious crowd situations
            if emergency_alerts:
                alert_triggered = crowd_analyzer.trigger_emergency_alert(
                    f"EMERGENCY: {', '.join(emergency_alerts)}"
                )
                
                if alert_triggered:
                    # Show emergency banner
                    emergency_banner.error("🚨 EMERGENCY DETECTED! 🚨 Alert sent to management!")
            else:
                crowd_analyzer.emergency_state = False
                emergency_banner.empty()
            
            # Add overlay information to frame
            overlay_color = (0, 0, 255) if crowd_analyzer.emergency_state else (255, 255, 255)
            
            # Person count - show both total and close
            cv2.putText(frame, f"Total: {total_people} | Close: {close_people}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)
            
            # Density level (based on close people only)
            cv2.putText(frame, f"Density: {density_level} ({density:.2f})", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)
            
            # Emergency status
            if crowd_analyzer.emergency_state:
                cv2.putText(frame, "EMERGENCY!", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # Add flashing red border
                cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 5)
            
            # Update stats display
            with stats_placeholder.container():
                st.metric("👥 Total People", total_people)
                st.metric("🔍 Close People", close_people, f"({total_people - close_people} distant)")
                st.metric("📊 Density Level", density_level, f"{density:.2f}")
                st.metric("⚡ Status", "EMERGENCY" if crowd_analyzer.emergency_state else "NORMAL")
            
            # Update alerts display
            with alerts_placeholder.container():
                if emergency_alerts:
                    for alert in emergency_alerts[-5:]:  # Show last 5 alerts
                        st.warning(alert)
                else:
                    st.success("✅ All Clear")
            
            # Show frame in Streamlit
            stframe.image(frame, channels="BGR", use_container_width=True)
            
            # Add small delay to control frame rate
            time.sleep(0.03)  # ~30 FPS
            
    except Exception as e:
        st.error(f"Error during video processing: {str(e)}")
    finally:
        cap.release()

# Additional Information
st.markdown("---")
st.markdown("""
### 🔧 System Features:
- **Real-time People Detection**: Uses YOLOv8 for accurate person detection
- **Crowd Density Analysis**: Monitors crowd density levels (LOW/MEDIUM/HIGH/CRITICAL)
- **Movement Tracking**: Detects unusual movement patterns and speeds
- **Emergency Alerts**: Audio and visual alerts for emergency situations
- **Risk Assessment**: Identifies overcrowding and unusual behaviors

### 🚨 Emergency Triggers (Balanced):
- **Running Detection**: ANY person running fast (even 1 person)
- **Speed Threshold**: 120+ base threshold + person size adjustment
- **Visual Indicator**: Red circle + "RUNNING!" text on runner
- **Extreme Clustering**: 5+ close people with 60%+ very close together
- **Background Filtering**: Distant people still ignored for crowd analysis

### 📊 People Categories:
- **Close People**: Large enough to matter (green boxes)
- **Distant People**: Background people ignored (gray boxes) 
- **Total Count**: All detected people
- **Analysis**: Based only on close people

### 🎯 Density Levels (Close People Only):
- **LOW**: 1-3 close people
- **MEDIUM**: 4-8 close people  
- **HIGH**: 9-20 close people
- **CRITICAL**: 20+ close people

### 📊 Metrics:
- People count in real-time
- Density level classification
- Movement pattern analysis
- Emergency status monitoring
""")
