import math
import torch
import cv2
import numpy as np
import easyocr
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

# Load the number plate detection model (adjust based on your pt model)
plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"C:\Users\91750\Downloads\best.pt", force_reload=True)  
reader = easyocr.Reader(['en'])  # Initialize OCR

video_path = r"c:\Users\91750\Downloads\demo (2).mp4"
vehicle_detection = VehicleDetectionTracker()

def calculate_risk_score(distance, speed, vehicle_type):
    """Calculate risk score based on distance, speed, and vehicle type."""
    risk_score = 0
    
    # Distance-based risk (closer = higher risk)
    distance = distance if distance is not None else 0
    if distance < 5:
        risk_score += 50
    elif distance < 10:
        risk_score += 30
    elif distance < 20:
        risk_score += 15
    
    # Speed-based risk (higher speed = higher risk)
    speed = speed if speed is not None else 0
    if speed > 50:
        risk_score += 30
    elif speed > 30:
        risk_score += 20
    elif speed > 10:
        risk_score += 10

    # Vehicle type risk (trucks/buses pose more risk)
    vehicle_risk_factor = {"truck": 30, "bus": 25, "car": 15, "bike": 10, "cyclist": 5}
    risk_score += vehicle_risk_factor.get(vehicle_type.lower(), 10)  # Default risk for unknown vehicles
    
    return min(risk_score, 100)  # Ensure max score is 100

VEHICLE_WIDTHS = {"car": 1.8, "truck": 2.5, "bus": 2.6, "bike": 0.8, "cyclist": 0.6}
FOCAL_LENGTH = 800  # Example value, needs calibration

def calculate_distance(bbox_width, vehicle_type):
    """Calculate distance using bounding box width."""
    if vehicle_type.lower() in VEHICLE_WIDTHS and bbox_width > 0:
        W_real = VEHICLE_WIDTHS[vehicle_type.lower()]
        return (FOCAL_LENGTH * W_real) / bbox_width
    return None  # Return None if invalid input

def detect_number_plate(frame, vehicle_bbox):
    """Crop vehicle region and detect number plate."""
    x, y, w, h = vehicle_bbox
    vehicle_roi = frame[y:y+h, x:x+w]  # Crop vehicle region

    # Run number plate detection
    results = plate_model(vehicle_roi)
    plates = results.pandas().xyxy[0]  # Extract detected plates

    plate_info = []
    for _, row in plates.iterrows():
        x1, y1, x2, y2, conf = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence']
        plate_crop = vehicle_roi[y1:y2, x1:x2]  # Crop plate
        plate_text = reader.readtext(plate_crop, detail=0)  # OCR for text
        plate_info.append({"bbox": [x1, y1, x2, y2], "text": plate_text, "confidence": conf})
    
    return plate_info

def process_frame(frame, result):
    """Process each frame to extract vehicle details and detect plates."""
    processed_vehicles = []
    for vehicle in result['detected_vehicles']:
        vehicle_bbox = (
            int(vehicle["vehicle_coordinates"]["x"]),
            int(vehicle["vehicle_coordinates"]["y"]),
            int(vehicle["vehicle_coordinates"]["width"]),
            int(vehicle["vehicle_coordinates"]["height"])
        )
        distance = calculate_distance(vehicle["vehicle_coordinates"]["width"], vehicle["vehicle_type"])
        plate_info = detect_number_plate(frame, vehicle_bbox)

        processed_vehicles.append({
            "vehicle_type": vehicle["vehicle_type"],
            "speed_info": vehicle["speed_info"]["kph"],
            "Coordinates": vehicle_bbox,
            "distance": distance,
            "risk_score": calculate_risk_score(distance, vehicle["speed_info"]["kph"], vehicle["vehicle_type"]),
            "plate_info": plate_info
        })
    
    return {
        "number_of_vehicles_detected": result["number_of_vehicles_detected"],
        "detected_vehicles": processed_vehicles
    }

def result_callback(result, frame):
    """Callback function for vehicle detection."""
    processed_result = process_frame(frame, result)
    print(processed_result)

vehicle_detection.process_video(video_path, result_callback)
