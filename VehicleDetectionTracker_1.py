# from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

# video_path = r"c:\Users\91750\Downloads\traffic1.mp4"
# vehicle_detection = VehicleDetectionTracker()
# result_callback = lambda result: print({
#     "number_of_vehicles_detected": result["number_of_vehicles_detected"],
#     "detected_vehicles": [
#         {
#             # "vehicle_id": vehicle["vehicle_id"],
#             "vehicle_type": vehicle["vehicle_type"],
#             # "detection_confidence": vehicle["detection_confidence"],
#             # "color_info": vehicle["color_info"],
#             # "model_info": vehicle["model_info"],
#             "speed_info": vehicle["speed_info"]["kph"],
#             "Coordinates": vehicle["vehicle_coordinates"]
#         }
#         for vehicle in result['detected_vehicles']
#     ]
# })
# vehicle_detection.process_video(video_path, result_callback)

import math
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

video_path = r"c:\Users\91750\Downloads\demo (2).mp4"
vehicle_detection = VehicleDetectionTracker()
def calculate_risk_score(distance, speed, vehicle_type):
    """Calculate risk score based on distance, speed, and vehicle type."""
    risk_score = 0
    
    # Distance-based risk (closer = higher risk)
    distance=distance if distance is not None else 0
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
    else:
        risk_score +=0
    
    # Vehicle type risk (trucks/buses pose more risk)
    vehicle_risk_factor = {
        "truck": 30,
        "bus": 25,
        "car": 15,
        "bike": 10,
        "cyclist": 5
    }
    risk_score += vehicle_risk_factor.get(vehicle_type.lower(), 10)  # Default risk for unknown vehicles
    
    return min(risk_score, 100)  # Ensure max score is 100




VEHICLE_WIDTHS = {
    "car": 1.8,
    "truck": 2.5,
    "bus": 2.6,
    "bike": 0.8,
    "cyclist": 0.6
}

# Known camera focal length (in pixels) - Adjust based on calibration
FOCAL_LENGTH = 800  # Example value, needs calibration

def calculate_distance(bbox_width, vehicle_type):
    """Calculate distance using bounding box width."""
    if vehicle_type.lower() in VEHICLE_WIDTHS and bbox_width > 0:
        W_real = VEHICLE_WIDTHS[vehicle_type.lower()]
        return (FOCAL_LENGTH * W_real) / bbox_width
    return None  # Return None if invalid input

result_callback = lambda result: print({
    "number_of_vehicles_detected": result["number_of_vehicles_detected"],
    "detected_vehicles": [
        {
            "vehicle_type": vehicle["vehicle_type"],
            "speed_info": vehicle["speed_info"]["kph"],
            "Coordinates": vehicle["vehicle_coordinates"],
            "distance": (distance := calculate_distance(vehicle["vehicle_coordinates"]["width"],vehicle["vehicle_type"])),
            "risk_score": calculate_risk_score(distance, vehicle["speed_info"]["kph"], vehicle["vehicle_type"])
        }
        for vehicle in result['detected_vehicles']
    ]
})

vehicle_detection.process_video(video_path, result_callback)
