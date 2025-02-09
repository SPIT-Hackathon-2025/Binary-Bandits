import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

class RoadSafetyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.vehicle_detection = VehicleDetectionTracker()
        self.cap = None
        self.timer = QTimer()

    def initUI(self):
        self.setWindowTitle('Road Safety App')
        self.setGeometry(100, 100, 800, 600)
        
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 384)
        
        self.btn_select = QPushButton('Select Video', self)
        self.btn_select.clicked.connect(self.select_video)
        
        self.btn_start = QPushButton('Start Processing', self)
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        
        self.btn_stop = QPushButton('Stop Processing', self)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        self.setLayout(layout)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Video', '', 'Video Files (*.mp4 *.avi *.mov)')
        if file_path:
            self.video_path = file_path
            self.btn_start.setEnabled(True)

    def start_processing(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.vehicle_detection.process_video(self.video_path, self.result_callback)

    def stop_processing(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        sys.exit(app.exec_())

    def result_callback(self, result):
        print({
            "number_of_vehicles_detected": result["number_of_vehicles_detected"],
            "detected_vehicles": [
                {
                    "vehicle_type": vehicle["vehicle_type"],
                    "speed_info": vehicle["speed_info"]["kph"],
                    "Coordinates": vehicle["vehicle_coordinates"],
                    "distance": (distance := self.calculate_distance(vehicle["vehicle_coordinates"]["width"], vehicle["vehicle_type"])),
                    "risk_score": self.calculate_risk_score(distance, vehicle["speed_info"]["kph"], vehicle["vehicle_type"])
                }
                for vehicle in result['detected_vehicles']
            ]
        })
    
    def calculate_risk_score(self, distance, speed, vehicle_type):
        risk_score = 0
        distance = distance if distance is not None else 0
        if distance < 5:
            risk_score += 50
        elif distance < 10:
            risk_score += 30
        elif distance < 20:
            risk_score += 15
        
        speed = speed if speed is not None else 0
        if speed > 50:
            risk_score += 30
        elif speed > 30:
            risk_score += 20
        elif speed > 10:
            risk_score += 10
        
        vehicle_risk_factor = {
            "truck": 30,
            "bus": 25,
            "car": 15,
            "bike": 10,
            "cyclist": 5
        }
        risk_score += vehicle_risk_factor.get(vehicle_type.lower(), 10)
        
        return min(risk_score, 100)
    
    def calculate_distance(self, bbox_width, vehicle_type):
        VEHICLE_WIDTHS = {
            "car": 1.8,
            "truck": 2.5,
            "bus": 2.6,
            "bike": 0.8,
            "cyclist": 0.6
        }
        FOCAL_LENGTH = 800
        
        if vehicle_type.lower() in VEHICLE_WIDTHS and bbox_width > 0:
            W_real = VEHICLE_WIDTHS[vehicle_type.lower()]
            return (FOCAL_LENGTH * W_real) / bbox_width
        return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RoadSafetyApp()
    window.show()
    sys.exit(app.exec_())